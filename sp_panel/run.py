"""Run the full collection. From the project root:

    python -m sp_panel.run --all

Or pick stages:

    python -m sp_panel.run --cik --financials
    python -m sp_panel.run --prices --macro

Everything is cached, so reruns are cheap and resumable. Outputs land in data/
as parquet (and CIK map + a run manifest as csv/json).
"""
import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from . import config
from .cik_map import resolve_universe
from .edgar import collect_financials
from .prices import fetch_prices, compute_risk
from .macro import fetch_macro, to_quarterly
from .short_interest import fetch_short_interest
from .alternative_data import (
    collect_alpha_earnings_estimates,
    collect_alpha_insider_transactions,
    collect_alpha_news,
    collect_alpha_transcripts,
    collect_sec_filing_texts,
    collect_sec_filings_index,
    quarters_from_panel,
)


def _save(df: pd.DataFrame, name: str):
    if df is None or df.empty:
        print(f"[save] {name}: empty, skipped")
        return
    path = config.DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"[save] {name}: {df.shape[0]} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cik", action="store_true")
    ap.add_argument("--financials", action="store_true")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--macro", action="store_true")
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--sec-filings", action="store_true",
                    help="collect public SEC 10-K/10-Q/8-K filing metadata")
    ap.add_argument("--sec-texts", action="store_true",
                    help="download bounded raw SEC filing text from sec_filings_index")
    ap.add_argument("--earnings-transcripts", action="store_true",
                    help="collect Alpha Vantage earnings-call transcripts (needs ALPHAVANTAGE_API_KEY)")
    ap.add_argument("--av-news", action="store_true",
                    help="collect Alpha Vantage news sentiment (needs ALPHAVANTAGE_API_KEY)")
    ap.add_argument("--av-insider", action="store_true",
                    help="collect Alpha Vantage insider transactions (needs ALPHAVANTAGE_API_KEY)")
    ap.add_argument("--av-estimates", action="store_true",
                    help="collect Alpha Vantage earnings/revenue estimates (needs ALPHAVANTAGE_API_KEY)")
    ap.add_argument("--alt", action="store_true",
                    help="run public alternative-data stages (currently SEC filing index)")
    ap.add_argument("--since", default="2014-01-01",
                    help="start date for SEC filing metadata")
    ap.add_argument("--sec-texts-per-ticker", type=int, default=8,
                    help="max SEC primary documents to download per ticker when --sec-texts is set")
    ap.add_argument("--sec-text-strategy", choices=["latest", "spaced"], default="latest",
                    help="choose latest filings or evenly spaced historical filings per ticker")
    ap.add_argument("--transcript-start", default=None,
                    help="first transcript quarter, e.g. 2021Q1")
    ap.add_argument("--transcript-end", default=None,
                    help="last transcript quarter, e.g. 2025Q4")
    ap.add_argument("--transcript-max-calls", type=int, default=None,
                    help="cap Alpha Vantage transcript API calls for rate-limit management")
    ap.add_argument("--av-estimates-max-calls", type=int, default=None,
                    help="cap Alpha Vantage estimates API calls for rate-limit management")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cache and re-download")
    ap.add_argument("--first-reported", action="store_true",
                    help="use as-first-reported values (no look-ahead) instead of latest")
    args = ap.parse_args()

    if config.CONTACT_EMAIL == "you@example.com":
        raise SystemExit("Set CONTACT_EMAIL in sp_panel/config.py before running "
                         "(the SEC blocks generic User-Agents).")

    do = lambda flag: args.all or getattr(args, flag)
    manifest = {"run_at": datetime.now(timezone.utc).isoformat(), "stages": {}}

    # Stage 1: CIK resolution (everything else joins on this).
    universe = resolve_universe(refresh=args.refresh)
    _save(universe, "universe_cik")
    universe.to_csv(config.DATA_DIR / "universe_cik.csv", index=False)
    manifest["stages"]["cik"] = int(universe["cik"].notna().sum())

    # Stage 2: EDGAR financials.
    if do("financials"):
        long_df, wide = collect_financials(
            universe, refresh=args.refresh,
            derive_q4=True, as_first_reported=args.first_reported)
        _save(long_df, "financials_long")
        _save(wide, "financials_quarterly")
        manifest["stages"]["financials_rows"] = int(len(long_df))

    # Stage 3: prices + risk metrics.
    if do("prices"):
        prices_long, benchmarks = fetch_prices(universe, refresh=args.refresh)
        _save(prices_long, "prices_daily")
        _save(benchmarks, "benchmarks_daily")
        risk = compute_risk(prices_long, benchmarks)
        _save(risk, "risk_daily")
        manifest["stages"]["price_rows"] = int(len(prices_long))

    # Stage 4: macro factors.
    if do("macro"):
        macro_daily = fetch_macro(refresh=args.refresh)
        _save(macro_daily, "macro_daily")
        _save(to_quarterly(macro_daily), "macro_quarterly")
        manifest["stages"]["macro_rows"] = int(len(macro_daily))

    # Stage 5: short interest (optional; needs FINRA creds).
    if do("short"):
        # Request from 2013 (the walk-forward eval starts 2014Q1); FINRA
        # returns whatever history the dataset actually has.
        si = fetch_short_interest(universe["ticker"].tolist(), start_date="2013-01-01")
        _save(si, "short_interest")
        manifest["stages"]["short_rows"] = int(len(si))

    # Stage 6: public/API textual and alternative data.
    if args.alt or args.sec_filings:
        filings = collect_sec_filings_index(universe, since=args.since, refresh=args.refresh)
        _save(filings, "sec_filings_index")
        manifest["stages"]["sec_filings_rows"] = int(len(filings))
    if args.sec_texts:
        idx_path = config.DATA_DIR / "sec_filings_index.parquet"
        filings = pd.read_parquet(idx_path) if idx_path.exists() else collect_sec_filings_index(
            universe, since=args.since, refresh=args.refresh)
        texts = collect_sec_filing_texts(
            filings,
            max_docs_per_ticker=args.sec_texts_per_ticker,
            strategy=args.sec_text_strategy,
            refresh=args.refresh)
        _save(texts, "sec_filing_texts")
        manifest["stages"]["sec_filing_text_rows"] = int(len(texts))
    if args.earnings_transcripts:
        quarters = quarters_from_panel(start=args.transcript_start, end=args.transcript_end)
        transcripts = collect_alpha_transcripts(
            universe,
            quarters=quarters,
            max_calls=args.transcript_max_calls,
            output_path=config.DATA_DIR / "earnings_call_transcripts.parquet",
            refresh=args.refresh,
        )
        _save(transcripts, "earnings_call_transcripts")
        manifest["stages"]["earnings_call_transcript_rows"] = int(len(transcripts))
    if args.av_news:
        news = collect_alpha_news(universe)
        _save(news, "av_news_sentiment")
        manifest["stages"]["av_news_rows"] = int(len(news))
    if args.av_insider:
        insider = collect_alpha_insider_transactions(universe)
        _save(insider, "av_insider_transactions")
        manifest["stages"]["av_insider_rows"] = int(len(insider))
    if args.av_estimates:
        estimates = collect_alpha_earnings_estimates(
            universe,
            max_calls=args.av_estimates_max_calls,
            output_path=config.DATA_DIR / "analyst_estimates.parquet",
            refresh=args.refresh,
        )
        _save(estimates, "analyst_estimates")
        manifest["stages"]["analyst_estimate_rows"] = int(len(estimates))

    (config.DATA_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nDone. Manifest:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
