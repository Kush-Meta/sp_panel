"""Assemble the point-in-time revenue-growth modeling panel.

Writes:
  data/panel.parquet            one row per (ticker, target_quarter)
  data/panel_dictionary.csv     column -> feature group
  data/panel_coverage.csv       ticker-level coverage/audit summary
  data/revenue_source_audit.csv revenue tag/proxy usage by ticker

Target: ``revenue_yoy_target`` — realized YoY revenue growth for the target
quarter, computed on a CLEAN, calendar-aligned revenue series.

This module was redesigned to fix the data-quality and leakage problems of the
original panel (see improved/README history / git log). The core ideas:

1. CALENDAR-ALIGNED YoY.  Each ticker is reindexed onto a gap-free quarterly
   grid, so YoY[T] always compares the same calendar quarter one year earlier.
   (The old `pct_change(4)` over a gappy frame compared mismatched quarters and
   produced impossible values once Q4s went missing.)

2. CORRUPTION REMOVED, NOT WINSORIZED.  Non-positive revenue is dropped and any
   YoY outside a wide sanity band is set to NaN (excluded from train AND eval),
   instead of being clipped into a plausible-but-wrong number and fed back in as
   a feature/label.

3. STRICT POINT-IN-TIME LAGGING.  Every feature for target quarter T is computed
   from data through quarter ``T - feature_lag`` only (default 1: forecast T's
   growth from everything realized and reported by the end of T-1 — a realistic
   post-earnings, next-quarter forecast). Nothing from quarter T touches the
   feature row, so there is no look-ahead. Macro, market and BEA blocks are
   joined with their own explicit lags / as-of cutoffs.
"""
import argparse

import numpy as np
import pandas as pd

from . import config

# YoY revenue growth outside this band for a large-cap is almost always a data
# artifact (residual unit/tag switch, M&A/divestiture step-change), not signal.
# We blank rather than clip so corrupt rows never enter training or evaluation.
YOY_LO, YOY_HI = -0.95, 3.0

MACRO_LEVELS = [
    "fed_funds", "ust_10y", "ust_2y", "ust_3m", "yield_curve_10y2y",
    "cpi", "core_pce", "unemployment", "ism_pmi_proxy", "retail_sales",
    "consumer_sentiment", "vix", "hy_spread", "usd_index", "wti_oil",
]
MACRO_CHANGES = ["cpi", "retail_sales", "ism_pmi_proxy", "wti_oil", "usd_index"]


def _load(name):
    p = config.DATA_DIR / f"{name}.parquet"
    if not p.exists():
        print(f"[assemble] MISSING {p.name} - skipping dependent features")
        return None
    return pd.read_parquet(p)


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _winsorize(df, cols, lo=0.01, hi=0.99):
    """Clip heavy-tailed *feature* ratios (never the target) to stabilize models."""
    for c in cols:
        if c in df.columns and df[c].notna().any():
            ql, qh = df[c].quantile(lo), df[c].quantile(hi)
            df[c] = df[c].clip(ql, qh)
    return df


# ---------------------------------------------------------------------------
# 1. Clean, calendar-aligned per-ticker quarterly series
# ---------------------------------------------------------------------------
def clean_quarterly(fin):
    """One clean row per (ticker, calendar_quarter) on a gap-free quarterly grid."""
    fin = fin.copy()
    fin["period_end"] = pd.to_datetime(fin["period_end"])
    fin["cq"] = pd.PeriodIndex(fin["period_end"], freq="Q")
    fin = (fin.sort_values(["ticker", "cq", "period_end"])
              .drop_duplicates(["ticker", "cq"], keep="last"))

    ratio_inputs = ["gross_profit", "operating_income", "net_income", "dep_amort",
                    "rd_expense", "sga_expense", "capex", "cfo", "cash", "assets",
                    "assets_current", "liabilities_current", "long_term_debt",
                    "equity", "interest_expense", "inventory", "shares_outstanding"]
    keep = ["ticker", "cik", "revenue"] + [c for c in ratio_inputs if c in fin.columns]

    out = []
    for tk, g in fin.groupby("ticker"):
        g = g.set_index("cq").sort_index()
        full = pd.period_range(g.index.min(), g.index.max(), freq="Q")
        g = g.reindex(full)
        g["ticker"] = tk
        g["cik"] = g["cik"].ffill().bfill()
        g["revenue"] = g["revenue"].where(g["revenue"] > 0)        # positive only
        g["yoy"] = g["revenue"] / g["revenue"].shift(4) - 1.0       # calendar-aligned
        g["qoq"] = g["revenue"] / g["revenue"].shift(1) - 1.0
        g["ttm_revenue"] = g["revenue"].rolling(4).sum()
        bad = (g["yoy"] <= YOY_LO) | (g["yoy"] >= YOY_HI)
        g.loc[bad, "yoy"] = np.nan                                  # blank corrupt growth
        g.loc[(g["qoq"] <= -0.95) | (g["qoq"] >= 5.0), "qoq"] = np.nan
        g = g.reset_index().rename(columns={"index": "cq"})
        cols = [c for c in keep + ["cq", "yoy", "qoq", "ttm_revenue"] if c in g.columns]
        out.append(g[cols])
    clean = pd.concat(out, ignore_index=True)

    # Common-size margins / ratios (point-in-time at each quarter).
    rev = clean["revenue"]
    clean["m_gross"] = _safe_div(clean.get("gross_profit"), rev) if "gross_profit" in clean else np.nan
    clean["m_oper"] = _safe_div(clean.get("operating_income"), rev) if "operating_income" in clean else np.nan
    clean["m_net"] = _safe_div(clean.get("net_income"), rev) if "net_income" in clean else np.nan
    if {"operating_income", "dep_amort"}.issubset(clean.columns):
        clean["m_ebitda"] = _safe_div(clean["operating_income"] + clean["dep_amort"], rev)
    clean["m_rd"] = _safe_div(clean.get("rd_expense"), rev) if "rd_expense" in clean else np.nan
    clean["m_sga"] = _safe_div(clean.get("sga_expense"), rev) if "sga_expense" in clean else np.nan
    clean["m_capex"] = _safe_div(clean.get("capex"), rev) if "capex" in clean else np.nan
    clean["m_cfo"] = _safe_div(clean.get("cfo"), rev) if "cfo" in clean else np.nan
    clean["asset_turnover"] = _safe_div(rev, clean.get("assets")) if "assets" in clean else np.nan
    clean["log_revenue"] = np.log(rev.where(rev > 0))
    return clean.sort_values(["ticker", "cq"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Causal time-series features (shifted to be known by quarter T-feature_lag)
# ---------------------------------------------------------------------------
def _causal_features(g, lag):
    g = g.sort_values("cq").reset_index(drop=True)
    yoy = g["yoy"]
    f = pd.DataFrame(index=g.index)

    # lagged target (persistence + seasonality)
    f["f_lag_yoy_1"] = yoy.shift(lag)          # last known YoY  (== persistence)
    f["f_lag_yoy_2"] = yoy.shift(lag + 1)
    f["f_lag_yoy_3"] = yoy.shift(lag + 2)
    f["f_lag_yoy_4"] = yoy.shift(lag + 3)      # same quarter last year (seasonal)
    f["f_lag_yoy_5"] = yoy.shift(lag + 4)

    # acceleration / deltas
    f["f_accel_1"] = f["f_lag_yoy_1"] - f["f_lag_yoy_2"]
    f["f_accel_seasonal"] = f["f_lag_yoy_1"] - f["f_lag_yoy_5"]

    # rolling mean / volatility of growth
    sy = yoy.shift(lag)
    f["f_roll_mean_4"] = sy.rolling(4, min_periods=2).mean()
    f["f_roll_mean_8"] = sy.rolling(8, min_periods=4).mean()
    f["f_roll_std_4"] = sy.rolling(4, min_periods=2).std()
    f["f_roll_std_8"] = sy.rolling(8, min_periods=4).std()
    f["f_yoy_z"] = _safe_div(f["f_lag_yoy_1"] - f["f_roll_mean_8"], f["f_roll_std_8"])

    # company-level historical trend (expanding, strictly causal)
    f["f_hist_mean"] = sy.expanding(min_periods=4).mean()
    f["f_hist_std"] = sy.expanding(min_periods=4).std()
    f["f_obs_count"] = sy.expanding(min_periods=1).count()

    # QoQ momentum
    sq = g["qoq"].shift(lag)
    f["f_qoq_1"] = sq
    f["f_qoq_mean_4"] = sq.rolling(4, min_periods=2).mean()

    # size + margins + margin changes (known at T-lag)
    for col, name in [("log_revenue", "f_log_revenue"), ("m_gross", "f_m_gross"),
                      ("m_oper", "f_m_oper"), ("m_net", "f_m_net"),
                      ("m_ebitda", "f_m_ebitda"), ("m_rd", "f_m_rd"),
                      ("m_sga", "f_m_sga"), ("m_capex", "f_m_capex"),
                      ("m_cfo", "f_m_cfo"), ("asset_turnover", "f_asset_turnover")]:
        if col in g.columns:
            s = g[col].shift(lag)
            f[name] = s
            if name in ("f_m_gross", "f_m_oper", "f_m_net", "f_m_ebitda"):
                f[f"{name}_chg4"] = s - g[col].shift(lag + 4)

    # lagged raw fundamentals needed by the market block (as-of T-lag)
    for col in ("ttm_revenue", "shares_outstanding", "long_term_debt", "cash"):
        if col in g.columns:
            f[f"asof_{col}"] = g[col].shift(lag)

    f["cq"] = g["cq"].values
    f["ticker"] = g["ticker"].values
    return f


def build_features(clean, feature_lag=1):
    parts = [_causal_features(g, feature_lag) for _, g in clean.groupby("ticker")]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3. Sector aggregates / macro / market / BEA  (all point-in-time)
# ---------------------------------------------------------------------------
def add_sector_features(panel):
    """Sector signals built only from peers' last-known growth (leave-one-out)."""
    grp = panel.groupby(["sector", "target_quarter"])["f_lag_yoy_1"]
    s_sum, s_cnt = grp.transform("sum"), grp.transform("count")
    own = panel["f_lag_yoy_1"].fillna(0.0)
    own_n = panel["f_lag_yoy_1"].notna().astype(int)
    panel["f_sector_yoy_loo"] = np.where(s_cnt > own_n, (s_sum - own) / (s_cnt - own_n), np.nan)
    panel["f_sector_yoy_median"] = grp.transform("median")
    panel["f_sector_yoy_std"] = grp.transform("std")
    panel["f_yoy_vs_sector"] = panel["f_lag_yoy_1"] - panel["f_sector_yoy_median"]
    return panel


def add_macro_features(panel, macro, macro_lag=1):
    if macro is None or macro.empty:
        return panel
    m = macro.copy()
    m["src_q"] = pd.PeriodIndex(pd.to_datetime(m["date"]), freq="Q")
    m = m.sort_values("src_q").set_index("src_q")
    cols = [c for c in MACRO_LEVELS if c in m.columns]
    feat = {f"mac_{c}": m[c] for c in cols}
    for c in MACRO_CHANGES:
        if c in m.columns:
            feat[f"mac_{c}_yoy"] = m[c].pct_change(4, fill_method=None)
            feat[f"mac_{c}_qoq"] = m[c].pct_change(1, fill_method=None)
    mf = pd.DataFrame(feat)
    mf["target_quarter"] = (mf.index + macro_lag).astype(str)   # macro from s informs s+lag
    return panel.merge(mf.reset_index(drop=True), on="target_quarter", how="left")


def add_market_features(panel, prices, risk, benchmarks):
    """As-of price/risk features cut off the day before the forecast origin."""
    if prices is None or prices.empty:
        return panel
    px = prices[["ticker", "date", "adj_close", "volume"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["mkt_ret_63"] = px.groupby("ticker")["adj_close"].pct_change(63, fill_method=None)
    px["mkt_ret_126"] = px.groupby("ticker")["adj_close"].pct_change(126, fill_method=None)
    roll_max = px.groupby("ticker")["adj_close"].transform(lambda s: s.rolling(63, min_periods=21).max())
    px["mkt_drawdown_63"] = px["adj_close"] / roll_max - 1
    if risk is not None and not risk.empty:
        r = risk.copy()
        r["date"] = pd.to_datetime(r["date"])
        px = px.merge(r, on=["ticker", "date"], how="left")

    keep = ["ticker", "date", "adj_close", "mkt_ret_63", "mkt_ret_126",
            "mkt_drawdown_63", "vol_252", "beta_SPX_252"]
    keep = [c for c in keep if c in px.columns]
    px = px[keep].sort_values(["date", "ticker"]).rename(
        columns={"date": "market_asof_date", "adj_close": "px_end"})

    left = panel.copy()
    left["market_cutoff"] = pd.to_datetime(left["forecast_origin"]) - pd.Timedelta(days=1)
    left = left.sort_values(["market_cutoff", "ticker"])
    out = pd.merge_asof(left, px, left_on="market_cutoff", right_on="market_asof_date",
                        by="ticker", direction="backward", allow_exact_matches=True)
    # valuation ratios from as-of price x as-of fundamentals (no look-ahead)
    if {"px_end", "asof_shares_outstanding"}.issubset(out.columns):
        out["mkt_market_cap"] = out["px_end"] * out["asof_shares_outstanding"]
        out["mkt_ps_ratio"] = _safe_div(out["mkt_market_cap"], out.get("asof_ttm_revenue"))
        ev = out["mkt_market_cap"]
        if "asof_long_term_debt" in out:
            ev = ev + out["asof_long_term_debt"].fillna(0)
        if "asof_cash" in out:
            ev = ev - out["asof_cash"].fillna(0)
        out["mkt_ev_to_rev"] = _safe_div(ev, out.get("asof_ttm_revenue"))
    return out.drop(columns=["market_cutoff", "market_asof_date", "px_end"], errors="ignore")


def add_guidance_features(panel, guidance):
    """Point-in-time management-guidance signals from SEC-filing guidance events.

    NOTE: this is *management guidance* (text-extracted from filings), not sell-side
    analyst consensus, and the parsed dollar *midpoints* are unreliable, so we use
    only the robust DIRECTIONAL signal (raise/lower/reaffirm actions) and counts.
    Everything is strictly point-in-time: only guidance issued before the forecast
    origin (the start of the target quarter) is counted, in trailing 90/180/365-day
    windows. Coverage is thin (~13% of rows carry any recent guidance), so these
    features are expected to add only marginal lift — see MODELING.md.
    """
    if guidance is None or guidance.empty:
        return panel
    g = guidance[guidance["metric"] == "revenue"].copy()
    g["guidance_date"] = pd.to_datetime(g["guidance_date"])
    real = g[g["guidance_action"].isin(["raise", "lower", "reaffirm", "initiate", "withdraw"])]

    fo = pd.to_datetime(panel["forecast_origin"])
    out = {k: np.zeros(len(panel)) for k in (
        "f_guid_count_180d", "f_guid_net_dir_180d", "f_guid_net_dir_365d",
        "f_guid_raise_90d", "f_guid_lower_90d")}
    days_since = np.full(len(panel), np.nan)
    latest_raise = np.zeros(len(panel))
    latest_lower = np.zeros(len(panel))

    idx_by_tkr = panel.groupby("ticker").indices
    for tk, idx in idx_by_tkr.items():
        gt = real[real["ticker"] == tk]
        if gt.empty:
            continue
        dates = gt["guidance_date"].values
        is_raise = (gt["guidance_action"] == "raise").values
        is_lower = (gt["guidance_action"] == "lower").values
        order = np.argsort(dates)
        dates, is_raise, is_lower = dates[order], is_raise[order], is_lower[order]
        for i in idx:
            cutoff = np.datetime64(fo.iloc[i])
            before = dates < cutoff
            if not before.any():
                continue
            w90 = before & (dates >= cutoff - np.timedelta64(90, "D"))
            w180 = before & (dates >= cutoff - np.timedelta64(180, "D"))
            w365 = before & (dates >= cutoff - np.timedelta64(365, "D"))
            out["f_guid_count_180d"][i] = w180.sum()
            out["f_guid_net_dir_180d"][i] = is_raise[w180].sum() - is_lower[w180].sum()
            out["f_guid_net_dir_365d"][i] = is_raise[w365].sum() - is_lower[w365].sum()
            out["f_guid_raise_90d"][i] = is_raise[w90].sum()
            out["f_guid_lower_90d"][i] = is_lower[w90].sum()
            last = np.where(before)[0][-1]
            days_since[i] = (cutoff - dates[last]) / np.timedelta64(1, "D")
            latest_raise[i] = float(is_raise[last])
            latest_lower[i] = float(is_lower[last])

    for k, v in out.items():
        panel[k] = v
    panel["f_guid_days_since_last"] = days_since
    panel["f_guid_latest_raise"] = latest_raise
    panel["f_guid_latest_lower"] = latest_lower
    return panel


def add_bea_features(panel, bea, quarter_lag=1):
    """Sector value-added from BEA, lagged one quarter (released with a delay)."""
    if bea is None or bea.empty:
        return panel
    b = bea.copy()
    b["src_q"] = pd.PeriodIndex(b["calendar_quarter"], freq="Q")
    b["target_quarter"] = (b["src_q"] + quarter_lag).astype(str)
    b = b.drop(columns=["calendar_quarter", "src_q"]).rename(columns={
        "bea_va": "bea_sector_va", "bea_va_yoy": "bea_sector_va_yoy",
        "bea_va_qoq": "bea_sector_va_qoq"})
    return panel.merge(b, on=["sector", "target_quarter"], how="left")


# ---------------------------------------------------------------------------
# 4. Assemble
# ---------------------------------------------------------------------------
def build_panel(start="2011Q1", feature_lag=1, macro_lag=1, winsor=(0.01, 0.99),
                with_market=True, with_bea=True, with_guidance=True, min_coverage=0.6):
    fin = _load("financials_quarterly")
    uni = _load("universe_cik")
    if fin is None or uni is None:
        raise SystemExit("financials_quarterly and universe_cik are required.")
    tickers = sorted(uni["ticker"].dropna().unique())
    sector_map = uni.set_index("ticker")["sector"].to_dict()

    clean = clean_quarterly(fin)
    feats = build_features(clean, feature_lag=feature_lag)

    tgt = clean[["ticker", "cq", "yoy"]].rename(columns={"yoy": "revenue_yoy_target"})
    panel = feats.merge(tgt, on=["ticker", "cq"], how="left")
    panel["sector"] = panel["ticker"].map(sector_map)
    panel = panel[panel["sector"].notna()].copy()

    # ids / timing: forecast made at the start of the target quarter, using data
    # through quarter T-feature_lag (a realistic next-quarter forecast).
    panel["target_q"] = panel["cq"]
    panel["target_quarter"] = panel["cq"].astype(str)
    panel["target_period_end"] = panel["cq"].dt.to_timestamp(how="end").dt.normalize()
    panel["forecast_origin"] = panel["cq"].dt.to_timestamp(how="start")
    panel["feature_lag"] = feature_lag
    panel["f_quarter"] = panel["cq"].dt.quarter
    for q in (1, 2, 3, 4):
        panel[f"f_is_q{q}"] = (panel["f_quarter"] == q).astype(float)

    # ready-made baseline columns (causal, known by T-feature_lag)
    panel["bl_persistence"] = panel["f_lag_yoy_1"]
    panel["bl_seasonal"] = panel["f_lag_yoy_4"]
    panel["bl_trailing_mean4"] = panel["f_roll_mean_4"]
    panel["bl_trailing_mean8"] = panel["f_roll_mean_8"]
    panel["bl_company_hist"] = panel["f_hist_mean"]

    panel = add_sector_features(panel)
    panel["bl_sector_median"] = panel["f_sector_yoy_median"]
    panel = add_macro_features(panel, _load("macro_quarterly"), macro_lag=macro_lag)
    if with_market:
        panel = add_market_features(panel, _load("prices_daily"),
                                    _load("risk_daily"), _load("benchmarks_daily"))
    if with_bea:
        # BEA GDP-by-industry is published ~a quarter in arrears, so it needs an
        # extra quarter of lag versus monthly FRED macro to stay point-in-time.
        panel = add_bea_features(panel, _load("bea_sector_quarterly"), quarter_lag=macro_lag + 1)
    if with_guidance:
        panel = add_guidance_features(panel, _load("company_guidance_normalized"))

    # winsorize heavy-tailed FEATURE ratios only (never the target).
    ratio_cols = [c for c in panel.columns
                  if c.startswith(("f_m_", "mac_")) or c in (
                      "f_asset_turnover", "mkt_ps_ratio", "mkt_ev_to_rev",
                      "mkt_ret_63", "mkt_ret_126", "f_yoy_z")]
    panel = _winsorize(panel, ratio_cols, *winsor)

    panel = panel[panel["target_q"] >= pd.Period(start, freq="Q")]
    # trim leading quarters with sparse coverage
    cov = panel.groupby("target_quarter")["ticker"].nunique() / len(tickers)
    keep_q = cov[cov >= min_coverage].index
    if len(keep_q):
        panel = panel[panel["target_quarter"].isin(keep_q)]

    panel = panel.sort_values(["target_q", "ticker"]).reset_index(drop=True)
    _write_audits(panel, clean, tickers)
    return panel


def feature_columns(panel):
    """Model feature columns: engineered f_*, macro mac_*, market mkt_*, BEA bea_*."""
    pref = ("f_", "mac_", "mkt_", "bea_")
    drop = {"f_quarter"}
    return [c for c in panel.columns
            if c.startswith(pref) and c not in drop and not c.startswith("asof_")]


def _write_audits(panel, clean, tickers):
    rows = []
    for ticker, g in panel.groupby("ticker"):
        rows.append({
            "ticker": ticker,
            "rows": len(g),
            "labeled_rows": int(g["revenue_yoy_target"].notna().sum()),
            "first_target_quarter": g["target_quarter"].min(),
            "last_target_quarter": g["target_quarter"].max(),
        })
    cov = pd.DataFrame(rows)
    missing = sorted(set(tickers) - set(cov["ticker"]))
    if missing:
        cov = pd.concat([cov, pd.DataFrame({"ticker": missing, "rows": 0, "labeled_rows": 0})],
                        ignore_index=True)
    cov.to_csv(config.DATA_DIR / "panel_coverage.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Dictionary (column -> feature group), consumed by evaluate.py
# ---------------------------------------------------------------------------
def _dictionary(panel):
    rows = []
    for c in panel.columns:
        if c in ("revenue_yoy_target",):
            grp = "target"
        elif c.startswith("bl_"):
            grp = "baseline_ref"
        elif c.startswith("asof_"):
            grp = "id"
        elif c in ("f_lag_yoy_1", "f_lag_yoy_2", "f_lag_yoy_3", "f_lag_yoy_4",
                   "f_lag_yoy_5", "f_accel_1", "f_accel_seasonal", "f_roll_mean_4",
                   "f_roll_mean_8", "f_roll_std_4", "f_roll_std_8", "f_yoy_z",
                   "f_qoq_1", "f_qoq_mean_4"):
            grp = "growth_dynamics"
        elif c.startswith("f_is_q"):
            grp = "seasonality"
        elif c in ("f_hist_mean", "f_hist_std", "f_obs_count", "f_log_revenue",
                   "f_asset_turnover") or c.startswith("f_m_"):
            grp = "company_behavior"
        elif c.startswith("f_sector_") or c == "f_yoy_vs_sector":
            grp = "sector_behavior"
        elif c.startswith("f_guid_"):
            grp = "guidance"
        elif c.startswith("mac_"):
            grp = "macro"
        elif c.startswith("mkt_"):
            grp = "market_risk"
        elif c.startswith("bea_"):
            grp = "sector_behavior"
        elif c in ("ticker", "cik", "sector", "target_quarter", "target_q", "cq",
                   "target_period_end", "forecast_origin", "feature_lag", "f_quarter"):
            grp = "id"
        else:
            grp = "other"
        rows.append({"column": c, "group": grp})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2011Q1")
    ap.add_argument("--feature-lag", type=int, default=1,
                    help="quarters to lag features behind the target (1 = predict T from T-1)")
    ap.add_argument("--macro-lag", type=int, default=1)
    ap.add_argument("--no-market", action="store_true", help="skip price/risk features")
    ap.add_argument("--no-bea", action="store_true", help="skip BEA sector features")
    ap.add_argument("--no-guidance", action="store_true", help="skip management-guidance features")
    ap.add_argument("--output-name", default="panel",
                    help="base name for panel and dictionary outputs under data/")
    args = ap.parse_args()
    panel = build_panel(start=args.start, feature_lag=args.feature_lag,
                        macro_lag=args.macro_lag, with_market=not args.no_market,
                        with_bea=not args.no_bea, with_guidance=not args.no_guidance)
    out = config.DATA_DIR / f"{args.output_name}.parquet"
    panel.to_parquet(out, index=False)
    _dictionary(panel).to_csv(config.DATA_DIR / f"{args.output_name}_dictionary.csv", index=False)
    feats = feature_columns(panel)
    print(f"[assemble] panel: {panel.shape[0]} rows x {panel.shape[1]} cols, "
          f"{panel['ticker'].nunique()} tickers, {panel['target_quarter'].nunique()} quarters "
          f"({panel['target_quarter'].min()}..{panel['target_quarter'].max()})")
    print(f"[assemble] labeled rows: {int(panel['revenue_yoy_target'].notna().sum())} "
          f"| model features: {len(feats)}")
    print(f"[assemble] target: mean={panel['revenue_yoy_target'].mean():.4f} "
          f"std={panel['revenue_yoy_target'].std():.4f} "
          f"min={panel['revenue_yoy_target'].min():.4f} max={panel['revenue_yoy_target'].max():.4f}")
    print(f"[assemble] -> {out}")


if __name__ == "__main__":
    main()
