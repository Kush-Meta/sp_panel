"""Alternative/textual data collectors.

Public SEC data:
  * sec_filings_index.parquet: filing metadata and document URLs for 10-K/10-Q/8-K.
  * sec_filing_texts.parquet: optional raw filing text for selected forms.

Alpha Vantage data (requires ALPHAVANTAGE_API_KEY):
  * earnings_call_transcripts.parquet: turn-level earnings-call transcript rows.
  * analyst_estimates.parquet: current API-supplied quarterly/annual estimates.
  * av_news_sentiment.parquet: ticker news sentiment feed.
  * av_insider_transactions.parquet: insider transaction feed.
"""
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
import numpy as np

from . import config
from .utils import RateLimiter, sec_get_json, sec_session

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"
ALPHA_URL = "https://www.alphavantage.co/query"

_alpha_limiter = RateLimiter(0.18)  # comfortably under free-tier 5 calls/minute
_alpha_daily_limit_hit = False


def _sec_text_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": config.USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def _alpha_get(params, pause=True):
    global _alpha_daily_limit_hit
    if _alpha_daily_limit_hit:
        return None
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        print("[alternative] ALPHAVANTAGE_API_KEY not set - skipping Alpha Vantage stage")
        return None
    if pause:
        _alpha_limiter.wait()
    params = {**params, "apikey": key}
    r = requests.get(ALPHA_URL, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if any(k in data for k in ("Error Message", "Information", "Note")):
        msg = data.get("Information") or data.get("Note") or data.get("Error Message") or "provider message"
        if "standard API rate limit" in msg or "requests per day" in msg:
            _alpha_daily_limit_hit = True
            print("[alternative] Alpha Vantage daily request limit reached - stopping API calls for this run")
        else:
            print(f"[alternative] Alpha Vantage response: {msg[:160]}")
        return None
    return data


def collect_sec_filings_index(universe, forms=("10-K", "10-Q", "8-K"), since="2014-01-01", refresh=False):
    """Collect SEC submissions metadata and EDGAR document URLs."""
    cache = config.CACHE_DIR / "sec_submissions"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    sess = sec_session()
    since_dt = pd.to_datetime(since)
    forms = set(forms)

    for i, row in enumerate(universe.itertuples(), 1):
        cik = str(row.cik_str).zfill(10)
        cp = cache / f"submissions_{cik}.json"
        try:
            if cp.exists() and not refresh:
                data = json.loads(cp.read_text())
            else:
                data = sec_get_json(sess, SUBMISSIONS_URL.format(cik=cik))
                if data:
                    cp.write_text(json.dumps(data))
            if not data:
                continue
            recent = data.get("filings", {}).get("recent", {})
            n = len(recent.get("accessionNumber", []))
            for j in range(n):
                form = recent.get("form", [None] * n)[j]
                filed = recent.get("filingDate", [None] * n)[j]
                if form not in forms or pd.to_datetime(filed, errors="coerce") < since_dt:
                    continue
                acc = recent["accessionNumber"][j]
                doc = recent.get("primaryDocument", [None] * n)[j]
                rows.append({
                    "ticker": row.ticker,
                    "sector": row.sector,
                    "cik": int(row.cik),
                    "cik_str": cik,
                    "form": form,
                    "filing_date": filed,
                    "report_date": recent.get("reportDate", [None] * n)[j],
                    "acceptance_datetime": recent.get("acceptanceDateTime", [None] * n)[j],
                    "accession_number": acc,
                    "primary_document": doc,
                    "primary_doc_description": recent.get("primaryDocDescription", [None] * n)[j],
                    "filing_detail_url": f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/{acc.replace('-', '')}/",
                    "document_url": ARCHIVES_URL.format(
                        cik_int=int(row.cik), accession_nodash=acc.replace("-", ""), doc=doc
                    ) if doc else None,
                })
            print(f"[alternative] SEC submissions {i}/{len(universe)} {row.ticker}: {len(rows)} cumulative rows")
        except Exception as e:
            print(f"[alternative] SEC submissions {row.ticker}: ERROR {type(e).__name__}: {e}")
    return pd.DataFrame(rows).sort_values(["ticker", "filing_date", "form"]).reset_index(drop=True)


def _select_text_candidates(index_df, forms, max_docs_per_ticker, strategy):
    candidates = index_df[index_df["form"].isin(forms) & index_df["document_url"].notna()].copy()
    candidates["filing_date"] = pd.to_datetime(candidates["filing_date"])
    if strategy == "spaced":
        rows = []
        for _, g in candidates.sort_values("filing_date").groupby("ticker"):
            if len(g) <= max_docs_per_ticker:
                rows.append(g)
                continue
            idx = np.linspace(0, len(g) - 1, max_docs_per_ticker).round().astype(int)
            rows.append(g.iloc[sorted(set(idx))])
        return pd.concat(rows, ignore_index=True) if rows else candidates.head(0)
    return (candidates.sort_values(["ticker", "filing_date"], ascending=[True, False])
            .groupby("ticker", group_keys=False)
            .head(max_docs_per_ticker))


def collect_sec_filing_texts(index_df, forms=("10-K", "10-Q", "8-K"), max_docs_per_ticker=8,
                             strategy="latest", refresh=False):
    """Download a bounded raw-text pool from document URLs in the SEC index."""
    out_cache = config.CACHE_DIR / "sec_filing_texts"
    out_cache.mkdir(parents=True, exist_ok=True)
    sess = _sec_text_session()
    forms = set(forms)
    candidates = _select_text_candidates(index_df, forms, max_docs_per_ticker, strategy)
    rows = []
    for i, row in enumerate(candidates.itertuples(), 1):
        key = f"{row.ticker}_{row.accession_number}_{row.primary_document}".replace("/", "_")
        cp = out_cache / f"{key}.txt"
        try:
            if cp.exists() and not refresh:
                text = cp.read_text(errors="replace")
            else:
                time.sleep(0.12)
                r = sess.get(row.document_url, timeout=60)
                if r.status_code != 200:
                    print(f"[alternative] text {row.ticker} {row.form}: HTTP {r.status_code}")
                    continue
                text = r.text
                cp.write_text(text)
            rows.append({
                "ticker": row.ticker,
                "sector": row.sector,
                "form": row.form,
                "filing_date": row.filing_date,
                "report_date": row.report_date,
                "accession_number": row.accession_number,
                "document_url": row.document_url,
                "text_path": str(cp),
                "char_count": len(text),
                "text": text,
            })
            print(f"[alternative] SEC text {i}/{len(candidates)} {row.ticker} {row.form}: {len(text)} chars")
        except Exception as e:
            print(f"[alternative] SEC text {row.ticker}: ERROR {type(e).__name__}: {e}")
    return pd.DataFrame(rows)


def collect_alpha_transcripts(universe, quarters, max_calls=None, output_path=None, refresh=False):
    """Collect Alpha Vantage quarter-level earnings-call transcripts.

    Alpha Vantage uses calendar-like quarter labels such as 2024Q1. The returned
    transcript is turn-level, so each speaker turn is one row.
    """
    output_path = Path(output_path) if output_path else None
    existing = pd.DataFrame()
    done = set()
    if output_path and output_path.exists() and not refresh:
        existing = pd.read_parquet(output_path)
        if not existing.empty and {"ticker", "quarter"}.issubset(existing.columns):
            done = set(zip(existing["ticker"].astype(str), existing["quarter"].astype(str)))
    rows, calls = [], 0
    for ticker in universe["ticker"].tolist():
        symbol = ticker.replace(".", "-")
        for q in quarters:
            if (ticker, q) in done:
                continue
            if max_calls is not None and calls >= max_calls:
                out = pd.DataFrame(rows)
                if output_path and not out.empty:
                    combined = pd.concat([existing, out], ignore_index=True) if not existing.empty else out
                    combined.to_parquet(output_path, index=False)
                    return combined
                return out
            data = _alpha_get({
                "function": "EARNINGS_CALL_TRANSCRIPT",
                "symbol": symbol,
                "quarter": q,
            })
            calls += 1
            if not data:
                continue
            transcript = data.get("transcript", [])
            for idx, turn in enumerate(transcript):
                raw_sentiment = turn.get("sentiment")
                raw_score = turn.get("sentiment_score")
                score = pd.to_numeric(raw_score, errors="coerce")
                if pd.isna(score):
                    score = pd.to_numeric(raw_sentiment, errors="coerce")
                rows.append({
                    "ticker": ticker,
                    "quarter": data.get("quarter", q),
                    "symbol": data.get("symbol", symbol),
                    "turn_index": idx,
                    "speaker": turn.get("speaker"),
                    "title": turn.get("title"),
                    "content": turn.get("content"),
                    "sentiment": raw_sentiment,
                    "sentiment_score": score,
                })
            print(f"[alternative] transcript {ticker} {q}: {len(transcript)} turns")
            if output_path and rows:
                out = pd.DataFrame(rows)
                combined = pd.concat([existing, out], ignore_index=True) if not existing.empty else out
                combined.to_parquet(output_path, index=False)
    out = pd.DataFrame(rows)
    if output_path and not out.empty:
        return pd.concat([existing, out], ignore_index=True) if not existing.empty else out
    return existing if output_path and not existing.empty else out


def _first_present(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def collect_alpha_earnings_estimates(universe, max_calls=None, output_path=None, refresh=False):
    """Collect Alpha Vantage earnings/revenue estimates.

    These rows are current snapshots from the API unless the provider includes a
    historical as-of field. We store estimate_asof_date and downstream features
    only use rows whose as-of date is before the forecast origin.
    """
    output_path = Path(output_path) if output_path else None
    existing = pd.DataFrame()
    done = set()
    if output_path and output_path.exists() and not refresh:
        existing = pd.read_parquet(output_path)
        if not existing.empty and "ticker" in existing.columns:
            done = set(existing["ticker"].astype(str))
    rows, calls = [], 0
    run_asof = datetime.now(timezone.utc).date().isoformat()
    for ticker in universe["ticker"].tolist():
        if ticker in done:
            continue
        if max_calls is not None and calls >= max_calls:
            out = pd.DataFrame(rows)
            if output_path and not out.empty:
                combined = pd.concat([existing, out], ignore_index=True) if not existing.empty else out
                combined.to_parquet(output_path, index=False)
                return combined
            return out
        data = _alpha_get({
            "function": "EARNINGS_ESTIMATES",
            "symbol": ticker.replace(".", "-"),
        })
        calls += 1
        if not data:
            continue
        blocks = [
            ("quarterly", data.get("quarterlyEstimates") or data.get("quarterlyEarningsEstimates") or []),
            ("annual", data.get("annualEstimates") or data.get("annualEarningsEstimates") or []),
        ]
        count = 0
        for period_type, items in blocks:
            for item in items:
                fiscal_date = _first_present(item, [
                    "fiscalDateEnding", "fiscal_date_ending", "date", "reportedDate",
                ])
                asof = _first_present(item, [
                    "estimateAsOfDate", "estimate_asof_date", "asOfDate", "updatedDate",
                ]) or run_asof
                rows.append({
                    "ticker": ticker,
                    "symbol": data.get("symbol", ticker.replace(".", "-")),
                    "period_type": period_type,
                    "fiscal_date_ending": fiscal_date,
                    "estimate_asof_date": asof,
                    "revenue_estimate_avg": _first_present(item, [
                        "revenueEstimateAverage", "revenueEstimateAvg", "revenue_avg",
                    ]),
                    "revenue_estimate_high": _first_present(item, [
                        "revenueEstimateHigh", "revenue_high",
                    ]),
                    "revenue_estimate_low": _first_present(item, [
                        "revenueEstimateLow", "revenue_low",
                    ]),
                    "eps_estimate_avg": _first_present(item, [
                        "epsEstimateAverage", "epsEstimateAvg", "eps_avg",
                    ]),
                    "eps_estimate_high": _first_present(item, [
                        "epsEstimateHigh", "eps_high",
                    ]),
                    "eps_estimate_low": _first_present(item, [
                        "epsEstimateLow", "eps_low",
                    ]),
                    "analyst_count": _first_present(item, [
                        "numberOfAnalysts", "analystCount", "analysts",
                    ]),
                    "raw": json.dumps(item, sort_keys=True),
                })
                count += 1
        print(f"[alternative] estimates {ticker}: {count} rows")
        if output_path and rows:
            out = pd.DataFrame(rows)
            combined = pd.concat([existing, out], ignore_index=True) if not existing.empty else out
            combined.to_parquet(output_path, index=False)
    out = pd.DataFrame(rows)
    if output_path and not out.empty:
        return pd.concat([existing, out], ignore_index=True) if not existing.empty else out
    return existing if output_path and not existing.empty else out


def collect_alpha_news(universe, limit_per_ticker=1000):
    rows = []
    for ticker in universe["ticker"].tolist():
        data = _alpha_get({
            "function": "NEWS_SENTIMENT",
            "tickers": ticker.replace(".", "-"),
            "limit": limit_per_ticker,
            "sort": "LATEST",
        })
        if not data:
            continue
        for item in data.get("feed", []):
            ts = item.get("time_published")
            ticker_sent = item.get("ticker_sentiment", [])
            own = next((x for x in ticker_sent if x.get("ticker") in {ticker, ticker.replace(".", "-")}), {})
            rows.append({
                "ticker": ticker,
                "time_published": ts,
                "title": item.get("title"),
                "url": item.get("url"),
                "source": item.get("source"),
                "summary": item.get("summary"),
                "overall_sentiment_score": item.get("overall_sentiment_score"),
                "overall_sentiment_label": item.get("overall_sentiment_label"),
                "ticker_relevance_score": own.get("relevance_score"),
                "ticker_sentiment_score": own.get("ticker_sentiment_score"),
                "ticker_sentiment_label": own.get("ticker_sentiment_label"),
            })
        print(f"[alternative] news {ticker}: {len(data.get('feed', []))} feed rows")
    return pd.DataFrame(rows)


def collect_alpha_insider_transactions(universe):
    rows = []
    for ticker in universe["ticker"].tolist():
        data = _alpha_get({
            "function": "INSIDER_TRANSACTIONS",
            "symbol": ticker.replace(".", "-"),
        })
        if not data:
            continue
        tx = data.get("data", [])
        for item in tx:
            rows.append({"ticker": ticker, **item})
        print(f"[alternative] insider {ticker}: {len(tx)} rows")
    return pd.DataFrame(rows)


def quarters_from_panel(path=None, start=None, end=None):
    path = Path(path or (config.DATA_DIR / "panel.parquet"))
    if path.exists():
        q = sorted(pd.read_parquet(path, columns=["target_quarter"])["target_quarter"].dropna().unique())
    else:
        q = []
    if start:
        q = [x for x in q if x >= start]
    if end:
        q = [x for x in q if x <= end]
    return q
