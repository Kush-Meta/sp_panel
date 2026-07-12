"""Stages 3-4 of the macro-inclusion analysis (Stage 2 lives in evaluate.py).

Stage 2 established THAT the macro block adds a small, regime-concentrated,
statistically insignificant amount on average. These stages answer the two
follow-ups, with the same discipline (one fixed model, identical walk-forward
splits, paired row-level forecasts, per-quarter loss differentials, quarter-
clustered Diebold-Mariano tests — see evaluate._dm_test for why the quarter is
the unit of independent evidence):

Stage 3 — WHICH macro features carry the signal
  (a) Additive theme ablations: add ONE macro theme at a time to the no-macro
      base and measure the paired delta. Themes group near-duplicate series
      (CPI level vs core PCE vs CPI YoY ...) so signal is not diluted across
      collinear columns the way per-feature importance is.
  (b) Residual-on-macro LASSO: the per-quarter MEAN residual of the no-macro
      model is exactly the common shock the firm features missed. Regressing
      that ~50-observation series on standardized quarterly macro candidates
      asks directly "which series explain what the model misses?" —
      exploratory (in-sample, n≈50), meant to corroborate (a), not to test.

Stage 4 — WHERE macro matters (sector heterogeneity)
  (a) Per-sector paired deltas of the full macro block (from the Stage-2 A/B
      runs): does macro help Energy while hurting Staples? A pooled average
      can hide opposite-signed sector effects.
  (b) Explicit sector x macro interactions: trees CAN discover sector-macro
      interactions internally (sector one-hots are in the design matrix), but
      at depth 4 with a weak macro signal they rarely spend the splits.
      Multiplying one representative series per theme by each sector dummy
      hands the model those interactions for free; a paired test vs the full
      model measures whether that was the missing piece.
  (c) Sector-specialist models: one LightGBM per sector (same hyperparameters,
      same protocol) vs the pooled model, compared ONLY on forecasts both
      made. This is the direct answer to "should we build per-sector models?"
      — specialists see cleaner within-sector structure but train on ~10x
      less data, so the honest comparison is empirical.

Run:
  python -m sp_panel.macro_stages                 # both stages, ~15-25 min
  python -m sp_panel.macro_stages --stage 3
  python -m sp_panel.macro_stages --stage 4
  python -m sp_panel.macro_stages --stage blocks  # new-feature-block ablation

Outputs (data/):
  macro_stage3_themes.csv          additive theme ablations + DM tests
  macro_stage3_residual_lasso.csv  LASSO coefs on the missed common shock
  macro_stage4_by_sector.csv       full-macro-block deltas per sector
  macro_stage4_interactions.csv    sector x macro interactions vs full model
  macro_stage4_sector_models.csv   pooled vs sector-specialist, per sector
  feature_blocks_ablation.csv      accounting / filing-event / industry-demand
  feature_blocks_by_sector.csv     ... and their per-sector deltas
"""
import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from . import assemble, config
from .evaluate import (TARGET, HAS_LGBM, _dm_test, _metrics, make_models,
                       walk_forward)

# Near-duplicate series grouped into economically coherent themes. A theme is
# the unit of decision: "include the inflation block", not "include CPI-YoY
# but not core PCE" (which no ~50-quarter sample can resolve).
MACRO_THEMES = {
    "rates_curve": ["mac_fed_funds", "mac_ust_10y", "mac_ust_2y", "mac_ust_3m",
                    "mac_yield_curve_10y2y"],
    "inflation": ["mac_cpi", "mac_core_pce", "mac_cpi_yoy", "mac_cpi_qoq"],
    "activity_demand": ["mac_unemployment", "mac_ism_pmi_proxy",
                        "mac_ism_pmi_proxy_yoy", "mac_ism_pmi_proxy_qoq",
                        "mac_retail_sales", "mac_retail_sales_yoy",
                        "mac_retail_sales_qoq", "mac_consumer_sentiment"],
    "risk_credit": ["mac_vix", "mac_baa_spread"],
    "commodities_fx": ["mac_wti_oil", "mac_wti_oil_yoy", "mac_wti_oil_qoq",
                       "mac_usd_index", "mac_usd_index_yoy", "mac_usd_index_qoq"],
}

# One headline series per theme for the sector-interaction features (stage 4b).
# Both oil and USD are kept: the Energy/oil and multinational/USD stories are
# the canonical sector-macro channels this stage exists to test.
THEME_REPS = ["mac_yield_curve_10y2y", "mac_cpi_yoy", "mac_ism_pmi_proxy_yoy",
              "mac_baa_spread", "mac_wti_oil_yoy", "mac_usd_index_yoy"]

# New candidate feature blocks (--stage blocks): tested one at a time on top of
# the panel WITHOUT them, then all together. Same paired quarter-clustered
# protocol as the macro ablation.
NEW_BLOCKS = {
    "accounting": "f_acct_",       # deferred revenue, receivables gap, goodwill/M&A
    "filing_event": "f_evt_",      # CARs + abnormal volume around the T-1 filing
    "industry_demand": "f_ind_",   # sector-matched FRED demand series
    "short_interest": "f_si_",     # FINRA short positioning (skipped until data exists)
    "estimates": "f_est_",         # analyst consensus/revisions (skipped until wired)
}


def _one_model(model_name=None):
    name = model_name or ("lightgbm" if HAS_LGBM else "hist_gbm")
    return name, make_models([name])[name]


def _run(panel, cols, warmup, min_train, model_name):
    """One walk-forward run of the fixed model; model rows only."""
    name, mdl = _one_model(model_name)
    preds = walk_forward(panel, cols, {name: mdl}, warmup_quarters=warmup,
                         min_train=min_train, ensemble=False)
    return preds[preds["model"] == name].reset_index(drop=True)


def _pair(a, b):
    """Row-paired loss differentials, oriented so POSITIVE = second run better."""
    key = ["ticker", "target_quarter"]
    m = a[key + ["sector", "y_true", "y_pred"]].merge(
        b[key + ["y_pred"]], on=key, suffixes=("_a", "_b"))
    ea, eb = m["y_true"] - m["y_pred_a"], m["y_true"] - m["y_pred_b"]
    m["d_se"] = ea ** 2 - eb ** 2
    m["d_hit"] = ((np.sign(m["y_true"]) == np.sign(m["y_pred_b"])).astype(float)
                  - (np.sign(m["y_true"]) == np.sign(m["y_pred_a"])).astype(float))
    return m


def _delta_row(m, label):
    """Summary + quarter-clustered DM test for one paired comparison."""
    ea = m["y_true"] - m["y_pred_a"]
    eb = m["y_true"] - m["y_pred_b"]
    per_q = m.groupby("target_quarter")["d_se"].mean()
    stat, p, T = _dm_test(per_q.values)
    return {
        "comparison": label, "n": len(m),
        "rmse_base": float(np.sqrt((ea ** 2).mean())),
        "rmse_alt": float(np.sqrt((eb ** 2).mean())),
        "rmse_delta": float(np.sqrt((ea ** 2).mean()) - np.sqrt((eb ** 2).mean())),
        "dir_acc_delta": float(m["d_hit"].mean()),
        "dm_stat": stat, "p_value": p, "n_quarters": T,
        "quarters_better": int((per_q > 0).sum()),
    }


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------
def stage3_themes(panel, base_cols, preds_a, warmup, min_train, model_name):
    """Additive ablations: no-macro base + one theme at a time."""
    rows = []
    for theme, cols in MACRO_THEMES.items():
        cols = [c for c in cols if c in panel.columns]
        if not cols:
            continue
        print(f"[stage3] walk-forward: base + {theme} ({len(cols)} cols)")
        preds_t = _run(panel, base_cols + cols, warmup, min_train, model_name)
        r = _delta_row(_pair(preds_a, preds_t), theme)
        r["n_cols"] = len(cols)
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("rmse_delta", ascending=False)
    out.to_csv(config.DATA_DIR / "macro_stage3_themes.csv", index=False)
    return out


def stage3_residual_lasso(panel, preds_a):
    """LASSO the missed common shock (per-quarter mean residual of the
    no-macro model) on standardized quarterly macro values.

    Exploratory: n ~= number of test quarters, fitted in-sample. Use it to
    corroborate the theme ablations and to rank candidates for Stage 5
    monitoring, not as a significance test.
    """
    resid = (preds_a.assign(e=preds_a["y_true"] - preds_a["y_pred"])
             .groupby("target_quarter")["e"].mean())
    mac_cols = sorted(c for c in panel.columns if c.startswith("mac_"))
    X = (panel.groupby("target_quarter")[mac_cols].first()
         .reindex(resid.index).astype(float))
    keep = [c for c in mac_cols if X[c].notna().all()]
    X = X[keep]
    Xz = (X - X.mean()) / X.std().replace(0, np.nan)
    Xz = Xz.dropna(axis=1)
    lasso = LassoCV(cv=5, random_state=0, max_iter=50000).fit(Xz.values, resid.values)
    r2 = float(lasso.score(Xz.values, resid.values))
    out = (pd.DataFrame({"feature": Xz.columns, "coef": lasso.coef_})
           .loc[lambda d: d["coef"] != 0]
           .assign(abs_coef=lambda d: d["coef"].abs())
           .sort_values("abs_coef", ascending=False)
           .drop(columns="abs_coef").reset_index(drop=True))
    out.attrs["r2_in_sample"] = r2
    out.to_csv(config.DATA_DIR / "macro_stage3_residual_lasso.csv", index=False)
    return out, r2, len(resid)


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------
def stage4_by_sector(paired_ab):
    """Per-sector deltas of the full macro block (positive = macro helps)."""
    rows = [_delta_row(g, sec) for sec, g in paired_ab.groupby("sector")]
    out = (pd.DataFrame(rows).rename(columns={"comparison": "sector"})
           .sort_values("rmse_delta", ascending=False))
    out.to_csv(config.DATA_DIR / "macro_stage4_by_sector.csv", index=False)
    return out


def stage4_interactions(panel, full_cols, preds_b, warmup, min_train, model_name):
    """Full model + explicit sector x macro interaction columns, vs full model."""
    p2 = panel.copy()
    ix_cols = []
    for rep in THEME_REPS:
        if rep not in p2.columns:
            continue
        for sec in sorted(p2["sector"].dropna().unique()):
            slug = re.sub(r"\W+", "_", sec.lower()).strip("_")
            col = f"ix_{rep[4:]}__{slug}"
            p2[col] = p2[rep] * (p2["sector"] == sec).astype(float)
            ix_cols.append(col)
    print(f"[stage4] walk-forward: full + {len(ix_cols)} sector-interaction cols")
    preds_i = _run(p2, full_cols + ix_cols, warmup, min_train, model_name)
    m = _pair(preds_b, preds_i)
    overall = _delta_row(m, "ALL")
    per_sec = [_delta_row(g, sec) for sec, g in m.groupby("sector")]
    out = pd.DataFrame([overall] + per_sec).rename(columns={"comparison": "scope"})
    out.to_csv(config.DATA_DIR / "macro_stage4_interactions.csv", index=False)
    return out


def stage4_sector_models(panel, full_cols, preds_b, warmup, min_train_sector,
                         model_name):
    """One specialist model per sector vs the pooled model, on common rows.

    The specialist uses the identical feature set and hyperparameters — the
    only difference is that it trains on its own sector's history (~10x less
    data). min_train is scaled down or the specialist could never start.
    """
    rows, all_pairs = [], []
    for sec, sub in panel.groupby("sector"):
        n_lab = int(sub[TARGET].notna().sum())
        if n_lab < 2 * min_train_sector:
            print(f"[stage4] {sec}: only {n_lab} labeled rows — skipped")
            continue
        print(f"[stage4] specialist walk-forward: {sec} ({n_lab} labeled rows)")
        preds_s = _run(sub.reset_index(drop=True), full_cols, warmup,
                       min_train_sector, model_name)
        m = _pair(preds_b[preds_b["sector"] == sec], preds_s)   # common rows only
        if m.empty:
            continue
        r = _delta_row(m, sec)
        r["n_train_labeled"] = n_lab
        rows.append(r)
        all_pairs.append(m)
    out = pd.DataFrame(rows).rename(columns={"comparison": "sector"})
    if all_pairs:
        agg = _delta_row(pd.concat(all_pairs, ignore_index=True), "ALL")
        agg["n_train_labeled"] = int(out["n_train_labeled"].sum())
        out = pd.concat([pd.DataFrame([agg]).rename(columns={"comparison": "sector"}),
                         out], ignore_index=True)
    out = out.rename(columns={"rmse_base": "rmse_pooled", "rmse_alt": "rmse_specialist"})
    out.to_csv(config.DATA_DIR / "macro_stage4_sector_models.csv", index=False)
    return out


def stage_blocks(panel, feat_cols, warmup, min_train, model_name):
    """Additive ablation of the NEW_BLOCKS feature blocks.

    Base = the full current feature set MINUS every new block, so each run
    answers: "does this block improve the model we already trust?" Blocks are
    added one at a time (marginal value in isolation) and all together
    (interactions between the new blocks). Per-sector deltas are reported for
    each comparison — a block like industry_demand can be flat overall while
    mattering in the two sectors its series actually describe.
    """
    prefixes = tuple(NEW_BLOCKS.values())
    base_cols = [c for c in feat_cols if not c.startswith(prefixes)]
    if len(base_cols) == len(feat_cols):
        print("[blocks] no new-block columns in the panel; skipping")
        return pd.DataFrame(), pd.DataFrame()
    print(f"[blocks] base run ({len(base_cols)} features)")
    preds_base = _run(panel, base_cols, warmup, min_train, model_name)

    overall, by_sector, per_q_all = [], [], []
    combos = [(name, [c for c in feat_cols if c.startswith(pref)])
              for name, pref in NEW_BLOCKS.items()]
    combos.append(("ALL_BLOCKS", [c for c in feat_cols if c.startswith(prefixes)]))
    for name, cols in combos:
        if not cols:
            continue
        print(f"[blocks] walk-forward: base + {name} ({len(cols)} cols)")
        preds = _run(panel, base_cols + cols, warmup, min_train, model_name)
        m = _pair(preds_base, preds)
        r = _delta_row(m, name)
        r["n_cols"] = len(cols)
        overall.append(r)
        for sec, g in m.groupby("sector"):
            sr = _delta_row(g, sec)
            sr["block"] = name
            by_sector.append(sr)
        # per-quarter deltas: lets a block with partial history (e.g. short
        # interest, 2018+) be re-tested on its covered quarters only.
        pq = m.groupby("target_quarter")[["d_se", "d_hit"]].mean().reset_index()
        pq["block"] = name
        per_q_all.append(pq)
    overall = (pd.DataFrame(overall).rename(columns={"comparison": "block"})
               .sort_values("rmse_delta", ascending=False))
    by_sector = pd.DataFrame(by_sector).rename(columns={"comparison": "sector"})
    overall.to_csv(config.DATA_DIR / "feature_blocks_ablation.csv", index=False)
    by_sector.to_csv(config.DATA_DIR / "feature_blocks_by_sector.csv", index=False)
    if per_q_all:
        pd.concat(per_q_all, ignore_index=True).to_csv(
            config.DATA_DIR / "feature_blocks_quarterly.csv", index=False)
    return overall, by_sector


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["3", "4", "all", "blocks"], default="all")
    ap.add_argument("--panel", default=str(config.DATA_DIR / "panel.parquet"))
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--min-train", type=int, default=400)
    ap.add_argument("--min-train-sector", type=int, default=150)
    ap.add_argument("--model", default=None, help="ablation model (default lightgbm)")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    feat_cols = assemble.feature_columns(panel)
    macro_cols = [c for c in feat_cols if c.startswith("mac_")]
    base_cols = [c for c in feat_cols if not c.startswith("mac_")]
    themed = {c for cols in MACRO_THEMES.values() for c in cols}
    unthemed = sorted(set(macro_cols) - themed)
    if unthemed:
        print(f"[macro_stages] WARNING: mac_ cols not in any theme: {unthemed}")

    pd.set_option("display.width", 220)
    if args.stage == "blocks":
        # experimental=True so the excluded-by-default candidate blocks are
        # visible to the ablation and can be (re)tested.
        all_cols = assemble.feature_columns(panel, experimental=True)
        overall, by_sector = stage_blocks(panel, all_cols, args.warmup,
                                          args.min_train, args.model)
        if not overall.empty:
            print("\n=== New-feature-block ablation (positive delta = block helps) ===")
            print(overall.round(4).to_string(index=False))
            print("\n--- Per-sector deltas ---")
            print(by_sector.round(4).to_string(index=False))
        return

    # Stages 3/4 compare against the same A (no-macro) and B (full) runs.
    print(f"[macro_stages] A run: no_macro ({len(base_cols)} features)")
    preds_a = _run(panel, base_cols, args.warmup, args.min_train, args.model)
    print(f"[macro_stages] B run: with_macro ({len(feat_cols)} features)")
    preds_b = _run(panel, feat_cols, args.warmup, args.min_train, args.model)
    paired_ab = _pair(preds_a, preds_b)

    if args.stage in ("3", "all"):
        themes = stage3_themes(panel, base_cols, preds_a, args.warmup,
                               args.min_train, args.model)
        print("\n=== Stage 3a: additive theme ablations (positive delta = theme helps) ===")
        print(themes.round(4).to_string(index=False))
        lasso, r2, n_q = stage3_residual_lasso(panel, preds_a)
        print(f"\n=== Stage 3b: LASSO on the missed common shock "
              f"(n={n_q} quarters, in-sample R2={r2:.3f}, exploratory) ===")
        print(lasso.round(4).to_string(index=False) if not lasso.empty
              else "(LASSO selected nothing — the missed shock is not linearly "
                   "spanned by the macro candidates)")

    if args.stage in ("4", "all"):
        by_sec = stage4_by_sector(paired_ab)
        print("\n=== Stage 4a: full macro block, per-sector paired deltas ===")
        print(by_sec.round(4).to_string(index=False))
        inter = stage4_interactions(panel, feat_cols, preds_b, args.warmup,
                                    args.min_train, args.model)
        print("\n=== Stage 4b: + explicit sector x macro interactions (vs full model) ===")
        print(inter.round(4).to_string(index=False))
        spec = stage4_sector_models(panel, feat_cols, preds_b, args.warmup,
                                    args.min_train_sector, args.model)
        print("\n=== Stage 4c: sector-specialist models vs pooled (positive delta = specialist wins) ===")
        print(spec.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
