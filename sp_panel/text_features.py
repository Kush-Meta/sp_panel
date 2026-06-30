"""Feature engineering for public SEC filing text.

This module intentionally starts with transparent, auditable lexicon features.
It converts raw filing documents into document-level text metrics and then into
point-in-time quarterly features that can be joined to the revenue panel.
"""
import argparse
import re
from collections import Counter

import numpy as np
import pandas as pd

from . import config


POSITIVE = {
    "able", "abundance", "accomplish", "accomplished", "achieve", "achieved",
    "achieves", "achieving", "advancement", "advancements", "advances",
    "beneficial", "benefit", "benefited", "benefiting", "best", "better",
    "boost", "boosted", "breakthrough", "confident", "constructive",
    "delivered", "effective", "efficiency", "efficient", "enhance",
    "enhanced", "excellent", "exceed", "exceeded", "exceeding", "expand",
    "expanded", "expanding", "favorable", "gain", "gained", "gaining",
    "growth", "improve", "improved", "improvement", "improvements",
    "improving", "increase", "increased", "increases", "increasing",
    "leading", "opportunity", "optimistic", "outperform", "positive",
    "progress", "record", "resilient", "robust", "strong", "strength",
    "strengthen", "strengthened", "success", "successful", "upside",
}

NEGATIVE = {
    "adverse", "adversely", "challenge", "challenged", "challenges",
    "challenging", "decline", "declined", "declines", "declining",
    "decrease", "decreased", "decreases", "decreasing", "delay", "delayed",
    "deteriorate", "deteriorated", "difficult", "difficulty", "disruption",
    "disruptions", "downturn", "downside", "impairment", "inflationary",
    "loss", "losses", "negative", "pressure", "pressures", "recession",
    "reduce", "reduced", "reduces", "reducing", "risk", "risks", "slow",
    "slowed", "slowing", "soft", "softness", "uncertain", "uncertainty",
    "unfavorable", "volatile", "volatility", "weak", "weakened", "weakness",
}

UNCERTAINTY = {
    "approximately", "could", "depend", "depends", "estimate", "estimated",
    "estimates", "may", "might", "possible", "possibly", "potential",
    "potentially", "uncertain", "uncertainties", "uncertainty", "variable",
    "vary", "volatile", "volatility", "whether",
}

LITIGIOUS = {
    "claim", "claims", "compliance", "contract", "contracts", "court",
    "damages", "legal", "liability", "litigation", "regulation",
    "regulations", "regulatory", "settlement", "sue", "sued", "tax",
}

FORWARD = {
    "anticipate", "anticipated", "believe", "expect", "expected", "expects",
    "forecast", "forecasted", "forecasts", "future", "guidance", "intend",
    "intends", "may", "outlook", "plan", "planned", "plans", "project",
    "projected", "projects", "should", "target", "targets", "will",
}

DEMAND = {"demand", "orders", "bookings", "traffic", "volume", "volumes", "consumption", "sell-through"}
PRICING = {"price", "prices", "pricing", "promotion", "promotions", "discount", "discounts", "mix"}
MARGIN = {"margin", "margins", "profitability", "cost", "costs", "expense", "expenses", "productivity"}
BACKLOG = {"backlog", "book-to-bill", "pipeline", "remaining", "performance", "obligations"}
INVENTORY = {"inventory", "inventories", "stock", "destocking", "restocking"}
SUPPLY = {"supply", "supplier", "suppliers", "logistics", "freight", "shortage", "shortages", "chain"}
FX = {"currency", "currencies", "foreign", "exchange", "fx", "dollar"}
GUIDANCE = {"guidance", "outlook", "forecast", "expects", "expect", "project", "projected", "target", "targets"}

LEXICONS = {
    "positive": POSITIVE,
    "negative": NEGATIVE,
    "uncertainty": UNCERTAINTY,
    "litigious": LITIGIOUS,
    "forward": FORWARD,
    "demand": DEMAND,
    "pricing": PRICING,
    "margin": MARGIN,
    "backlog": BACKLOG,
    "inventory": INVENTORY,
    "supply_chain": SUPPLY,
    "fx": FX,
    "guidance": GUIDANCE,
}

WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")
SENT_RE = re.compile(r"[.!?]+")


def _words(text):
    return [w.lower().strip("-'") for w in WORD_RE.findall(text or "")]


def _doc_features(row):
    text = row.text or ""
    words = _words(text)
    counts = Counter(words)
    n_words = len(words)
    n_sents = max(1, len([s for s in SENT_RE.split(text) if s.strip()]))
    out = {
        "ticker": row.ticker,
        "sector": row.sector,
        "form": row.form,
        "filing_date": pd.to_datetime(row.filing_date),
        "report_date": pd.to_datetime(row.report_date, errors="coerce"),
        "accession_number": row.accession_number,
        "document_url": row.document_url,
        "char_count": row.char_count,
        "word_count": n_words,
        "sentence_count": n_sents,
        "avg_sentence_words": n_words / n_sents if n_sents else np.nan,
        "numeric_density": sum(ch.isdigit() for ch in text) / max(1, len(text)),
    }
    for name, lex in LEXICONS.items():
        c = sum(counts[w] for w in lex)
        out[f"text_{name}_count"] = c
        out[f"text_{name}_share"] = c / n_words if n_words else np.nan
    pos = out["text_positive_share"]
    neg = out["text_negative_share"]
    out["text_net_tone"] = pos - neg
    out["text_pos_neg_ratio"] = out["text_positive_count"] / max(1, out["text_negative_count"])
    out["is_8k"] = int(row.form == "8-K")
    out["is_10q"] = int(row.form == "10-Q")
    out["is_10k"] = int(row.form == "10-K")
    return out


def build_document_features(texts):
    if texts is None or texts.empty:
        return pd.DataFrame()
    rows = [_doc_features(row) for row in texts.itertuples()]
    return pd.DataFrame(rows).sort_values(["ticker", "filing_date", "form"]).reset_index(drop=True)


def build_quarterly_text_features(doc_features, panel):
    """As-of join document text features to each panel forecast origin.

    Features summarize filings in the 90/180/365 days before forecast_origin.
    """
    if doc_features is None or doc_features.empty:
        return pd.DataFrame()
    target_cols = ["ticker", "target_quarter", "forecast_origin"]
    if "forecast_timing" in panel.columns:
        target_cols.append("forecast_timing")
    targets = panel[target_cols].drop_duplicates().copy()
    targets["forecast_origin"] = pd.to_datetime(targets["forecast_origin"])
    docs = doc_features.copy()
    docs["filing_date"] = pd.to_datetime(docs["filing_date"])
    metric_cols = [
        c for c in docs.columns
        if c.startswith("text_") and c.endswith(("_share", "_ratio", "_tone"))
    ]
    count_cols = [c for c in docs.columns if c.startswith("text_") and c.endswith("_count")]
    rows = []
    for row in targets.itertuples():
        sub = docs[(docs["ticker"] == row.ticker) & (docs["filing_date"] < row.forecast_origin)]
        rec = {"ticker": row.ticker, "target_quarter": row.target_quarter}
        if hasattr(row, "forecast_timing"):
            rec["forecast_timing"] = row.forecast_timing
        for window in (90, 180, 365):
            w = sub[sub["filing_date"] >= row.forecast_origin - pd.Timedelta(days=window)]
            prefix = f"text_{window}d"
            rec[f"{prefix}_doc_count"] = len(w)
            rec[f"{prefix}_8k_count"] = int((w["form"] == "8-K").sum()) if not w.empty else 0
            rec[f"{prefix}_10qk_count"] = int(w["form"].isin(["10-Q", "10-K"]).sum()) if not w.empty else 0
            rec[f"{prefix}_word_count"] = int(w["word_count"].sum()) if not w.empty else 0
            if w.empty:
                continue
            weights = w["word_count"].clip(lower=1)
            for c in metric_cols:
                rec[f"{prefix}_{c}"] = np.average(w[c].fillna(0), weights=weights)
            for c in count_cols:
                rec[f"{prefix}_{c}"] = w[c].sum()
            rec[f"{prefix}_numeric_density"] = np.average(w["numeric_density"].fillna(0), weights=weights)
            rec[f"{prefix}_avg_sentence_words"] = np.average(w["avg_sentence_words"].fillna(0), weights=weights)
        latest = sub.sort_values("filing_date").tail(1)
        if not latest.empty:
            rec["text_latest_filing_date"] = latest.iloc[0]["filing_date"]
            rec["text_latest_form"] = latest.iloc[0]["form"]
            rec["text_latest_net_tone"] = latest.iloc[0]["text_net_tone"]
            rec["text_latest_uncertainty_share"] = latest.iloc[0]["text_uncertainty_share"]
            rec["text_latest_guidance_share"] = latest.iloc[0]["text_guidance_share"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(["ticker", "target_quarter"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=str(config.DATA_DIR / "sec_filing_texts.parquet"))
    ap.add_argument("--panel", default=str(config.DATA_DIR / "panel.parquet"))
    args = ap.parse_args()

    texts = pd.read_parquet(args.texts)
    panel = pd.read_parquet(args.panel)
    docs = build_document_features(texts)
    docs.to_parquet(config.DATA_DIR / "sec_text_doc_features.parquet", index=False)
    q = build_quarterly_text_features(docs, panel)
    qpath = config.DATA_DIR / "sec_text_quarterly_features.parquet"
    if qpath.exists() and "forecast_timing" in q.columns:
        old = pd.read_parquet(qpath)
        if "forecast_timing" in old.columns:
            keys = ["ticker", "target_quarter", "forecast_timing"]
            old_idx = old.set_index(keys).index
            new_idx = q.set_index(keys).index
            old = old[~old_idx.isin(new_idx)]
            q = pd.concat([old, q], ignore_index=True, sort=False)
    q.to_parquet(qpath, index=False)
    print(f"[text] document features: {docs.shape} -> {config.DATA_DIR / 'sec_text_doc_features.parquet'}")
    print(f"[text] quarterly features: {q.shape} -> {config.DATA_DIR / 'sec_text_quarterly_features.parquet'}")


if __name__ == "__main__":
    main()
