"""Point-in-time company guidance and analyst-expectation features.

This module starts with auditable, regex-based extraction. It is deliberately
conservative: only revenue/sales contexts are treated as revenue guidance, and
analyst estimates are used only when their as-of date is before forecast origin.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")
MONEY_RE = re.compile(
    r"\$?\s*(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>billion|bn|million|mm|m)?",
    re.I,
)
REVENUE_WORD_RE = re.compile(r"\b(revenue|revenues|net sales|sales)\b", re.I)
BAD_CONTEXT_RE = re.compile(r"\b(eps|earnings per share|per share|tax|margin|income|profit)\b", re.I)
FORWARD_CONTEXT_RE = re.compile(
    r"\b(guidance|outlook|expect|expects|expected|forecast|project|projects|"
    r"projected|anticipate|anticipates|anticipated|target|targets|will|"
    r"approximately|about|range|between)\b",
    re.I,
)
RANGE_RE = re.compile(
    r"(?P<a>\$?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|bn|million|mm|m)?)"
    r"\s*(?:-|to|and)\s*"
    r"(?P<b>\$?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?:billion|bn|million|mm|m)?)",
    re.I,
)
ACTION_WORDS = {
    "raise": ("raise", "raises", "raised", "raising", "increase", "increased", "increases", "higher"),
    "lower": ("lower", "lowers", "lowered", "lowering", "reduce", "reduced", "reduces", "cut", "cuts", "decrease"),
    "reaffirm": ("reaffirm", "reaffirms", "reaffirmed", "reiterate", "reiterates", "reiterated", "maintain", "maintains"),
    "initiate": ("initiate", "initiates", "initiated", "introduce", "introduced", "provide", "provided"),
    "withdraw": ("withdraw", "withdraws", "withdrew", "withdrawn", "suspend", "suspended"),
}
PERIOD_PATTERNS = [
    re.compile(r"\b(20\d{2})\s*[- ]?\s*Q([1-4])\b", re.I),
    re.compile(r"\bQ([1-4])\s*[- ]?\s*(20\d{2})\b", re.I),
    re.compile(r"\b(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(20\d{2})\b", re.I),
]
ORD_TO_Q = {"first": "1", "second": "2", "third": "3", "fourth": "4"}


def _clean_text(text):
    text = TAG_RE.sub(" ", text or "")
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return SPACE_RE.sub(" ", text).strip()


def _money_to_float(raw, default_unit=None):
    if not raw:
        return np.nan
    m = MONEY_RE.search(raw)
    if not m:
        return np.nan
    val = float(m.group("num").replace(",", ""))
    unit = (m.group("unit") or default_unit or "").lower()
    if unit in {"billion", "bn"}:
        val *= 1_000_000_000
    elif unit in {"million", "mm", "m"}:
        val *= 1_000_000
    return val


def _period_from_context(ctx):
    for pat in PERIOD_PATTERNS:
        m = pat.search(ctx)
        if not m:
            continue
        if pat.pattern.startswith("\\b(20"):
            year, q = m.group(1), m.group(2)
        elif m.group(1).isdigit():
            q, year = m.group(1), m.group(2)
        else:
            q, year = ORD_TO_Q[m.group(1).lower()], m.group(2)
        return f"{year}Q{q}"
    return None


def _action_from_context(ctx):
    low = ctx.lower()
    found = []
    for action, words in ACTION_WORDS.items():
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words):
            found.append(action)
    if not found:
        return "mention"
    if "withdraw" in found:
        return "withdraw"
    if "lower" in found:
        return "lower"
    if "raise" in found:
        return "raise"
    if "reaffirm" in found:
        return "reaffirm"
    return found[0]


def _extract_guidance_mentions(row):
    text = _clean_text(row.text)
    mentions = []
    for match in REVENUE_WORD_RE.finditer(text):
        start = max(0, match.start() - 350)
        end = min(len(text), match.end() + 650)
        ctx = text[start:end]
        if BAD_CONTEXT_RE.search(ctx[:120]) and not re.search(r"\bguidance|outlook|expect|forecast", ctx, re.I):
            continue
        if not FORWARD_CONTEXT_RE.search(ctx):
            continue

        ranges = list(RANGE_RE.finditer(ctx))
        if ranges:
            for r in ranges[:2]:
                default_unit = None
                b_unit = MONEY_RE.search(r.group("b"))
                if b_unit:
                    default_unit = b_unit.group("unit")
                low = _money_to_float(r.group("a"), default_unit=default_unit)
                high = _money_to_float(r.group("b"), default_unit=default_unit)
                if not np.isfinite(low) or not np.isfinite(high):
                    continue
                if low > high:
                    low, high = high, low
                mentions.append(_guidance_record(row, ctx, low, high))
            continue

        if FORWARD_CONTEXT_RE.search(ctx):
            nums = [m.group(0) for m in MONEY_RE.finditer(ctx)]
            vals = [_money_to_float(x) for x in nums]
            vals = [v for v in vals if np.isfinite(v) and v >= 1_000_000]
            if vals:
                mentions.append(_guidance_record(row, ctx, vals[0], vals[0]))
    return mentions


def _guidance_record(row, ctx, low, high):
    midpoint = (low + high) / 2
    width = high - low
    return {
        "ticker": row.ticker,
        "sector": getattr(row, "sector", None),
        "guidance_date": pd.to_datetime(row.filing_date),
        "source_form": row.form,
        "accession_number": row.accession_number,
        "source_url": row.document_url,
        "metric": "revenue",
        "guided_period": _period_from_context(ctx),
        "guidance_action": _action_from_context(ctx),
        "guidance_low": low,
        "guidance_high": high,
        "guidance_midpoint": midpoint,
        "guidance_range_width": width,
        "guidance_range_width_pct": width / midpoint if midpoint else np.nan,
        "guidance_context": ctx[:1000],
    }


def extract_company_guidance(texts):
    if texts is None or texts.empty:
        return pd.DataFrame()
    rows = []
    for row in texts.itertuples():
        rows.extend(_extract_guidance_mentions(row))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates([
        "ticker", "guidance_date", "source_form", "accession_number",
        "guided_period", "guidance_low", "guidance_high", "guidance_action",
    ])
    return out.sort_values(["ticker", "guidance_date"]).reset_index(drop=True)


def _safe_ratio(a, b):
    return a / b if b and np.isfinite(b) and b != 0 else np.nan


def _summarize_guidance_for_row(guidance, row):
    origin = pd.to_datetime(row.forecast_origin)
    sub = guidance[(guidance["ticker"] == row.ticker) & (guidance["guidance_date"] < origin)]
    rec = {"ticker": row.ticker, "target_quarter": row.target_quarter}
    if hasattr(row, "forecast_timing"):
        rec["forecast_timing"] = row.forecast_timing

    for window in (90, 180, 365):
        w = sub[sub["guidance_date"] >= origin - pd.Timedelta(days=window)]
        prefix = f"expect_guidance_{window}d"
        rec[f"{prefix}_count"] = int(len(w))
        if w.empty:
            continue
        for action in ACTION_WORDS:
            rec[f"{prefix}_{action}_count"] = int((w["guidance_action"] == action).sum())
        rec[f"{prefix}_midpoint_mean"] = float(w["guidance_midpoint"].mean())
        rec[f"{prefix}_range_width_pct_mean"] = float(w["guidance_range_width_pct"].mean())
        rec[f"{prefix}_midpoint_to_last_revenue"] = _safe_ratio(
            rec[f"{prefix}_midpoint_mean"], getattr(row, "revenue", np.nan)
        )
        rec[f"{prefix}_midpoint_to_ttm_revenue"] = _safe_ratio(
            rec[f"{prefix}_midpoint_mean"], getattr(row, "ttm_revenue", np.nan)
        )

    latest = sub.sort_values("guidance_date").tail(1)
    if not latest.empty:
        g = latest.iloc[0]
        rec["expect_guidance_latest_days_ago"] = int((origin - g["guidance_date"]).days)
        rec["expect_guidance_latest_midpoint"] = float(g["guidance_midpoint"])
        rec["expect_guidance_latest_range_width_pct"] = float(g["guidance_range_width_pct"])
        rec["expect_guidance_latest_to_last_revenue"] = _safe_ratio(g["guidance_midpoint"], getattr(row, "revenue", np.nan))
        rec["expect_guidance_latest_to_ttm_revenue"] = _safe_ratio(g["guidance_midpoint"], getattr(row, "ttm_revenue", np.nan))
        rec["expect_guidance_latest_action_raise"] = int(g["guidance_action"] == "raise")
        rec["expect_guidance_latest_action_lower"] = int(g["guidance_action"] == "lower")
        rec["expect_guidance_latest_action_reaffirm"] = int(g["guidance_action"] == "reaffirm")
        rec["expect_guidance_latest_action_withdraw"] = int(g["guidance_action"] == "withdraw")

    target_match = sub[sub["guided_period"].astype(str) == str(row.target_quarter)] if "guided_period" in sub else sub.head(0)
    rec["expect_guidance_target_match_count"] = int(len(target_match))
    if not target_match.empty:
        g = target_match.sort_values("guidance_date").tail(1).iloc[0]
        rec["expect_guidance_target_latest_midpoint"] = float(g["guidance_midpoint"])
        rec["expect_guidance_target_latest_range_width_pct"] = float(g["guidance_range_width_pct"])
        rec["expect_guidance_target_days_ago"] = int((origin - g["guidance_date"]).days)
        rec["expect_guidance_target_to_last_revenue"] = _safe_ratio(g["guidance_midpoint"], getattr(row, "revenue", np.nan))
        rec["expect_guidance_target_to_ttm_revenue"] = _safe_ratio(g["guidance_midpoint"], getattr(row, "ttm_revenue", np.nan))
    return rec


def _normalize_estimate_date(x):
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        return None
    return str(pd.Period(dt, freq="Q"))


def _summarize_estimates_for_row(estimates, row):
    rec = {}
    if estimates is None or estimates.empty:
        return rec
    origin = pd.to_datetime(row.forecast_origin)
    est = estimates[
        (estimates["ticker"] == row.ticker)
        & (pd.to_datetime(estimates["estimate_asof_date"]) < origin)
    ].copy()
    if est.empty:
        return rec
    if "target_quarter" not in est.columns:
        est["target_quarter"] = est["fiscal_date_ending"].map(_normalize_estimate_date)
    est = est[est["target_quarter"].astype(str) == str(row.target_quarter)]
    if est.empty:
        return rec
    latest = est.sort_values("estimate_asof_date").tail(1).iloc[0]
    avg = pd.to_numeric(latest.get("revenue_estimate_avg"), errors="coerce")
    high = pd.to_numeric(latest.get("revenue_estimate_high"), errors="coerce")
    low = pd.to_numeric(latest.get("revenue_estimate_low"), errors="coerce")
    count = pd.to_numeric(latest.get("analyst_count"), errors="coerce")
    rec["expect_analyst_estimate_days_ago"] = int((origin - pd.to_datetime(latest["estimate_asof_date"])).days)
    rec["expect_analyst_revenue_avg"] = float(avg) if pd.notna(avg) else np.nan
    rec["expect_analyst_revenue_high"] = float(high) if pd.notna(high) else np.nan
    rec["expect_analyst_revenue_low"] = float(low) if pd.notna(low) else np.nan
    rec["expect_analyst_count"] = float(count) if pd.notna(count) else np.nan
    rec["expect_analyst_dispersion_pct"] = _safe_ratio(high - low, avg) if pd.notna(high) and pd.notna(low) else np.nan
    rec["expect_analyst_avg_to_last_revenue"] = _safe_ratio(avg, getattr(row, "revenue", np.nan))
    rec["expect_analyst_avg_to_ttm_revenue"] = _safe_ratio(avg, getattr(row, "ttm_revenue", np.nan))
    return rec


def build_expectation_features(guidance, estimates, panel):
    targets = panel.drop_duplicates(["ticker", "target_quarter", "forecast_timing"]).copy()
    targets["forecast_origin"] = pd.to_datetime(targets["forecast_origin"])
    if guidance is None:
        guidance = pd.DataFrame()
    if not guidance.empty:
        guidance = guidance.copy()
        guidance["guidance_date"] = pd.to_datetime(guidance["guidance_date"])
    rows = []
    for row in targets.itertuples():
        rec = _summarize_guidance_for_row(guidance, row)
        rec.update(_summarize_estimates_for_row(estimates, row))
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["ticker", "target_quarter", "forecast_timing"]).reset_index(drop=True)


def _merge_existing(path, new):
    if not path.exists() or new.empty or "forecast_timing" not in new.columns:
        return new
    old = pd.read_parquet(path)
    if old.empty or "forecast_timing" not in old.columns:
        return new
    keys = ["ticker", "target_quarter", "forecast_timing"]
    old_idx = old.set_index(keys).index
    new_idx = new.set_index(keys).index
    old = old[~old_idx.isin(new_idx)]
    return pd.concat([old, new], ignore_index=True, sort=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=str(config.DATA_DIR / "sec_filing_texts.parquet"))
    ap.add_argument("--panel", default=str(config.DATA_DIR / "panel.parquet"))
    ap.add_argument("--estimates", default=str(config.DATA_DIR / "analyst_estimates.parquet"))
    ap.add_argument("--refresh-guidance", action="store_true")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    guidance_path = config.DATA_DIR / "company_guidance_raw.parquet"
    if guidance_path.exists() and not args.refresh_guidance:
        guidance = pd.read_parquet(guidance_path)
    else:
        texts = pd.read_parquet(args.texts)
        guidance = extract_company_guidance(texts)
        guidance.to_parquet(guidance_path, index=False)
    normalized_path = config.DATA_DIR / "company_guidance_normalized.parquet"
    guidance.to_parquet(normalized_path, index=False)

    est_path = Path(args.estimates)
    estimates = pd.read_parquet(est_path) if est_path.exists() else pd.DataFrame()
    features = build_expectation_features(guidance, estimates, panel)
    out_path = config.DATA_DIR / "expectations_quarterly_features.parquet"
    features = _merge_existing(out_path, features)
    features.to_parquet(out_path, index=False)
    print(f"[expectations] guidance rows: {len(guidance)} -> {guidance_path}")
    print(f"[expectations] quarterly features: {features.shape} -> {out_path}")


if __name__ == "__main__":
    main()
