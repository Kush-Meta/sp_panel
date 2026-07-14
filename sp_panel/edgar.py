"""Pull SEC EDGAR companyfacts and parse them into a tidy quarterly panel.

companyfacts returns, per XBRL concept, a list of facts each tagged with the
period (start/end), fiscal year/period (fy/fp), the form it came from, and the
filing date. The messy parts this module handles:

  1. Tag synonyms      -> coalesced to canonical fields via config.CONCEPTS.
  2. Instant vs flow    -> balance-sheet items are point-in-time; income/cash-flow
                           items span a period.
  3. Restatements       -> the same period appears in multiple filings. We keep the
                           LATEST-filed value ("as most recently reported"). Set
                           as_first_reported=True for point-in-time / no look-ahead.
  4. Q4 derivation      -> 10-Ks report full-year flows, not a standalone Q4. We
                           derive Q4 = FY - (Q1+Q2+Q3) when derive_q4=True.

Output is long format (one row per ticker/concept/period); pivot_wide() turns it
into a quarterly panel with one column per concept.
"""
import json
import numpy as np
import pandas as pd

from . import config
from .utils import sec_session, sec_get_json

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _cache_path(cik_str: str):
    return config.CACHE_DIR / f"companyfacts_{cik_str}.json"


def fetch_companyfacts(session, cik_str: str, refresh: bool = False):
    """Fetch (and cache) the raw companyfacts JSON for one CIK."""
    cp = _cache_path(cik_str)
    if cp.exists() and not refresh:
        return json.loads(cp.read_text())
    url = FACTS_URL.format(cik=cik_str)
    data = sec_get_json(session, url)
    if data is not None:
        cp.write_text(json.dumps(data))
    return data


def _pick_unit(units: dict, concept: str):
    """Choose the right units key for a concept."""
    if concept in ("eps_basic", "eps_diluted"):
        pref = ["USD/shares"]
    elif concept == "shares_outstanding":
        pref = ["shares"]
    else:
        pref = ["USD"]
    for p in pref:
        if p in units:
            return p, units[p]
    # Fallback: first available unit.
    k = next(iter(units))
    return k, units[k]


def _extract_concept(facts_usgaap: dict, concept: str, synonyms: list):
    """Stitch facts across ALL synonym tags, not just the first.

    Companies switch XBRL tags over time (e.g. revenue moved from `Revenues`/
    `SalesRevenueNet` to `RevenueFromContractWithCustomerExcludingAssessedTax`
    at ASC 606 adoption in 2018). The old tags cover the early years and the new
    tag covers recent years, so taking only the first present tag truncates
    history. We take the union; `prio` records synonym priority so the dedup in
    parse_company can prefer the higher-priority tag when periods overlap (the
    dollar figure is the same either way)."""
    out = []
    for prio, tag in enumerate(synonyms):
        if tag not in facts_usgaap:
            continue
        unit_key, arr = _pick_unit(facts_usgaap[tag]["units"], concept)
        for f in arr:
            out.append({
                "concept": concept, "tag": tag, "prio": prio, "unit": unit_key,
                "start": f.get("start"), "end": f.get("end"),
                "val": f.get("val"), "fy": f.get("fy"), "fp": f.get("fp"),
                "form": f.get("form"), "filed": f.get("filed"),
                "frame": f.get("frame"),
            })
    return out


def parse_company(data: dict, ticker: str, derive_q4: bool = True,
                  as_first_reported: bool = False) -> pd.DataFrame:
    """Parse one companyfacts blob into a long tidy frame of quarterly facts."""
    if not data or "facts" not in data or "us-gaap" not in data["facts"]:
        return pd.DataFrame()
    usgaap = data["facts"]["us-gaap"]
    # SEC serves cik as int for most filers but as a zero-padded STRING for
    # some new registrants; a mixed int/str column breaks parquet writes.
    cik = data.get("cik")
    cik = int(cik) if cik is not None else None

    rows = []
    for concept, synonyms in config.CONCEPTS.items():
        rows.extend(_extract_concept(usgaap, concept, synonyms))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["form"].isin(["10-Q", "10-K", "10-K/A", "10-Q/A"])].copy()
    df["filed"] = pd.to_datetime(df["filed"])
    df["end_dt"] = pd.to_datetime(df["end"])
    df["start_dt"] = pd.to_datetime(df["start"], errors="coerce")
    # Instant (balance-sheet) facts have no period start; flows have both.
    df["kind"] = df["start_dt"].isna().map({True: "instant", False: "flow"})
    df["dur_days"] = (df["end_dt"] - df["start_dt"]).dt.days

    instants = df[df["kind"] == "instant"].copy()
    flows = df[df["kind"] == "flow"].copy()

    # Quarterly flows: duration roughly one quarter.
    q_flows = flows[(flows["dur_days"].between(60, 100))].copy()
    # Annual flows (for Q4 derivation): full-year duration.
    a_flows = flows[(flows["dur_days"].between(330, 380))].copy()

    def _dedup(frame):
        if frame.empty:
            return frame
        # Prefer the higher-priority synonym tag (lower prio); within the same
        # tag, prefer latest-filed (or first-filed if as_first_reported). Sorting
        # puts the preferred row last, then keep='last'.
        prio = frame["prio"] if "prio" in frame else 0
        frame = frame.assign(_prio=prio).sort_values(
            ["_prio", "filed"], ascending=[False, not as_first_reported])
        return (frame.drop_duplicates(subset=["concept", "end"], keep="last")
                     .drop(columns="_prio"))

    q_flows, a_flows, instants = _dedup(q_flows), _dedup(a_flows), _dedup(instants)

    # Q4 = annual - (Q1+Q2+Q3). We match the three quarters to the fiscal year by
    # PERIOD-DATE CONTAINMENT, never by the fy/fp labels: companyfacts fiscal-year
    # labels are unreliable and frequently offset (e.g. a calendar-2022 10-K is
    # tagged fy=2024 while its Q1-Q3 10-Qs are tagged fy=2023), which makes
    # label-based matching pair the annual with the wrong quarters and produce
    # impossible standalone-Q4 numbers. Date containment is label-agnostic and
    # works for off-calendar fiscal years too.
    derived = []
    if derive_q4 and not a_flows.empty:
        for concept in a_flows["concept"].unique():
            a_sub = a_flows[a_flows["concept"] == concept]
            q_sub = q_flows[q_flows["concept"] == concept]
            for _, arow in a_sub.iterrows():
                a_start, a_end = arow["start_dt"], arow["end_dt"]
                if pd.isna(a_start) or pd.isna(a_end):
                    continue
                # A real standalone Q4 already present? then don't synthesize one.
                if (q_sub["end_dt"] == a_end).any():
                    continue
                # Quarters whose period falls inside the fiscal year and ends at
                # least ~3 weeks before year-end (i.e. the first three quarters).
                inside = q_sub[
                    (q_sub["start_dt"] >= a_start - pd.Timedelta(days=8))
                    & (q_sub["end_dt"] <= a_end + pd.Timedelta(days=8))
                    & (q_sub["end_dt"] < a_end - pd.Timedelta(days=20))
                ].drop_duplicates(subset=["end"]).sort_values("end_dt")
                if len(inside) != 3:
                    continue
                q4_val = arow["val"] - inside["val"].sum()
                q4_start = inside["end_dt"].max() + pd.Timedelta(days=1)
                derived.append({
                    "concept": concept, "tag": arow["tag"], "unit": arow["unit"],
                    "start": q4_start.strftime("%Y-%m-%d"), "end": arow["end"],
                    "val": q4_val, "fy": arow["fy"], "fp": "Q4", "form": arow["form"],
                    "filed": arow["filed"], "frame": None,
                    "start_dt": q4_start, "end_dt": arow["end_dt"],
                    "kind": "flow", "dur_days": int((a_end - q4_start).days),
                })
    derived_df = pd.DataFrame(derived)

    panel = pd.concat([q_flows, instants, derived_df], ignore_index=True)
    panel["ticker"] = ticker
    panel["cik"] = cik
    cols = ["ticker", "cik", "concept", "tag", "unit", "fy", "fp",
            "start", "end", "end_dt", "val", "form", "filed", "frame"]
    return panel[cols].sort_values(["concept", "end_dt"]).reset_index(drop=True)


def pivot_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide quarterly panel: one row per (ticker, period_end), one column
    per concept. The fiscal label prefers the quarter tag (Q1-Q4) over FY so the
    year-end balance sheet and the derived Q4 flow share a single row."""
    if long_df.empty:
        return long_df
    w = long_df.pivot_table(
        index=["ticker", "cik", "end"],
        columns="concept", values="val", aggfunc="last",
    ).reset_index().rename(columns={"end": "period_end"})
    w["period_end"] = pd.to_datetime(w["period_end"])
    filed = (long_df.groupby(["ticker", "end"], as_index=False)["filed"].max()
                    .rename(columns={"end": "period_end"}))
    filed["period_end"] = pd.to_datetime(filed["period_end"])
    w = w.merge(filed, on=["ticker", "period_end"], how="left")

    # Audit and fallback for financial-services reporters. Some banks/brokers
    # report net revenue as `RevenuesNetOfInterestExpense`; others expose
    # interest-income-net plus noninterest income. Keep row-level provenance so
    # modeling can verify which revenue definition survived into the panel.
    rev_src = long_df[long_df["concept"] == "revenue"].copy()
    if not rev_src.empty:
        rev_src = (rev_src.sort_values(["filed", "tag"])
                          .drop_duplicates(["ticker", "end"], keep="last")
                          [["ticker", "end", "tag"]])
        rev_src["end"] = pd.to_datetime(rev_src["end"])
        rev_src = rev_src.rename(columns={"end": "period_end", "tag": "revenue_tag"})
        w = w.merge(rev_src, on=["ticker", "period_end"], how="left")
    else:
        w["revenue_tag"] = pd.NA
    w["revenue_source"] = np.where(w.get("revenue").notna(), "direct:" + w["revenue_tag"].fillna("unknown"), "missing")
    if {"net_interest_income", "noninterest_income"}.issubset(w.columns):
        fallback = (
            w["revenue"].isna()
            & w["net_interest_income"].notna()
            & w["noninterest_income"].notna()
        )
        w.loc[fallback, "revenue"] = (
            w.loc[fallback, "net_interest_income"]
            + w.loc[fallback, "noninterest_income"]
        )
        w.loc[fallback, "revenue_tag"] = "InterestIncomeExpenseNet+NoninterestIncome"
        w.loc[fallback, "revenue_source"] = "proxy:net_interest_income+noninterest_income"

    # Attach a single fy/fp per (ticker, period_end), favoring quarter over FY.
    order = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "FY": 4}
    lab = long_df.copy()
    lab["ord"] = lab["fp"].map(order).fillna(9)
    lab = (lab.sort_values("ord")
              .drop_duplicates(["ticker", "end"], keep="first")[["ticker", "end", "fy", "fp"]])
    lab["end"] = pd.to_datetime(lab["end"])
    w = w.merge(lab.rename(columns={"end": "period_end"}),
                on=["ticker", "period_end"], how="left")

    front = ["ticker", "cik", "fy", "fp", "period_end", "filed", "revenue_source", "revenue_tag"]
    rest = [c for c in w.columns if c not in front]
    return w[front + rest].sort_values(["ticker", "period_end"]).reset_index(drop=True)


def collect_financials(universe: pd.DataFrame, refresh: bool = False,
                       derive_q4: bool = True, as_first_reported: bool = False):
    """Pull + parse financials for every resolved CIK. Returns (long, wide)."""
    sess = sec_session()
    longs = []
    todo = universe.dropna(subset=["cik_str"])
    for i, row in enumerate(todo.itertuples(), 1):
        tk, cik_str = row.ticker, row.cik_str
        try:
            data = fetch_companyfacts(sess, cik_str, refresh=refresh)
            if data is None:
                print(f"[edgar] {i}/{len(todo)} {tk}: no companyfacts (404)")
                continue
            parsed = parse_company(data, tk, derive_q4=derive_q4,
                                   as_first_reported=as_first_reported)
            longs.append(parsed)
            print(f"[edgar] {i}/{len(todo)} {tk}: {len(parsed)} facts")
        except Exception as e:
            print(f"[edgar] {i}/{len(todo)} {tk}: ERROR {e}")
    long_df = pd.concat(longs, ignore_index=True) if longs else pd.DataFrame()
    return long_df, pivot_wide(long_df)
