# sp_panel — macro × financials data collection

Ingestion toolkit for a 100-company panel across 10 GICS sectors. Pulls quarterly
financials (SEC EDGAR), prices/volatility/beta (yfinance), macro factors (FRED),
and optionally short interest (FINRA), and writes a tidy panel to `data/`.

## Setup

```bash
pip install -r requirements.txt
# edit sp_panel/config.py -> set CONTACT_NAME and CONTACT_EMAIL (required by SEC)
```

## Run

```bash
# everything that works without extra credentials:
python -m sp_panel.run --cik --financials --prices --macro

# or just one stage at a time:
python -m sp_panel.run --financials
python -m sp_panel.run --prices
python -m sp_panel.run --short          # needs FINRA creds (see below)

# force re-download instead of using cache:
python -m sp_panel.run --financials --refresh

# use as-first-reported values (avoids look-ahead bias):
python -m sp_panel.run --financials --first-reported
```

Raw `companyfacts` JSON and the SEC ticker map are cached under `data/cache/`, so
reruns are fast and resumable. If a run dies partway, just run it again.

## Outputs (in `data/`)

| File | What it is |
|------|------------|
| `universe_cik.{parquet,csv}` | ticker → sector → CIK, the join key for everything |
| `financials_long.parquet` | one row per company/concept/period (audit trail: tag, form, filed date) |
| `financials_quarterly.parquet` | wide quarterly panel: one row per (ticker, period_end), one column per concept |
| `prices_daily.parquet` | adjusted daily closes + volume |
| `benchmarks_daily.parquet` | Nasdaq-100 (NDX) and S&P 500 (SPX) levels |
| `risk_daily.parquet` | annualized vol + rolling beta vs NDX and SPX |
| `macro_daily.parquet` / `macro_quarterly.parquet` | FRED factors (daily, and resampled to quarter-end with quarter averages) |
| `short_interest.parquet` | FINRA bi-monthly short interest (if creds set) |
| `panel.parquet` | clean point-in-time modeling table for next-quarter YoY revenue growth |
| `panel_dictionary.csv` | column → feature group, consumed by the evaluator |
| `panel_coverage.csv` | ticker-level coverage and label counts for the modeling panel |
| `model_metrics_overall.csv` | walk-forward metrics: every model + baseline (and the old-suite reference) |
| `model_metrics_by_{sector,year,company}.csv` | metric breakdowns |
| `model_predictions.parquet` | per (model, ticker, quarter) walk-forward predictions |
| `model_feature_importance.csv` | LightGBM gain importance for the feature block |
| `sec_filings_index.parquet` | public SEC 10-K/10-Q/8-K metadata and document URLs |
| `sec_filing_texts.parquet` | bounded raw SEC filing text pool for textual feature engineering |
| `earnings_call_transcripts.parquet` | Alpha Vantage turn-level earnings-call transcripts, if API key is set |
| `av_news_sentiment.parquet` | Alpha Vantage news sentiment feed, if API key is set |
| `av_insider_transactions.parquet` | Alpha Vantage insider transactions feed, if API key is set |
| `run_manifest.json` | row counts per stage for the last run |

## Assembling the panel for analysis

Your modeling table is `panel.parquet`, one row per `(ticker, target_quarter)`.
The target is `revenue_yoy_target`: realized YoY revenue growth for the target
quarter, computed on a **clean, calendar-aligned** revenue series. Every feature
is strictly point-in-time — built from data through quarter `T - feature_lag`
(default 1: forecast quarter T from everything realized and reported by the end of
T-1, a realistic next-quarter forecast). Feature blocks:
- **growth dynamics** — lagged YoY (T-1…T-5), acceleration, trailing growth mean
  and volatility, QoQ momentum, expanding company trend, momentum z-score;
- **company** — size and common-size margins (gross/operating/net/EBITDA, R&D,
  SG&A, capex, CFO, asset turnover) and their 4-quarter changes;
- **sector** — point-in-time leave-one-out peer growth mean/median/dispersion and
  growth-vs-sector;
- **macro** — rates/yield-curve, inflation, activity, sentiment, oil, USD, credit
  spreads, plus YoY/QoQ changes (lagged one quarter);
- **market** — as-of price momentum, drawdown, volatility, beta, P/S and EV/Rev;
- **BEA** — sector value-added (extra-lagged for point-in-time safety);
- **seasonality** — quarter-of-year indicators.

Build the panel and run the walk-forward evaluation:

```bash
python -m sp_panel.assemble                       # -> data/panel.parquet (+ dictionary, coverage)
python -m sp_panel.evaluate                        # all models + baselines, expanding walk-forward
python -m sp_panel.evaluate --models lightgbm      # restrict the model zoo
python -m sp_panel.evaluate --feature-lag 1 --warmup 12   # tune the protocol
```

`evaluate` writes `model_metrics_overall.csv` (every model vs the persistence /
trailing-mean / sector-median / expanding-mean baselines, plus the historical
old-suite row for reference), the by-sector / by-year / by-company breakdowns,
`model_predictions.parquet`, and `model_feature_importance.csv`.

The gradient boosters (LightGBM / XGBoost / CatBoost) are optional — see
`requirements-modeling.txt`. If a wheel is missing the zoo degrades gracefully to
the scikit-learn models.

**See [`MODELING.md`](MODELING.md)** for the full root-cause audit of the previous
approach, the data fix (calendar-aligned target + corrected Q4 derivation), the
results table, and recommended next steps.

## Textual / alternative data

Public SEC filing metadata and bounded raw filing text:

```bash
python -m sp_panel.run --sec-filings --since 2014-01-01
python -m sp_panel.run --sec-texts --since 2014-01-01 --sec-texts-per-ticker 4
python -m sp_panel.expectations --panel data/panel.parquet --refresh-guidance
```

`sp_panel.expectations` extracts revenue/sales guidance from downloaded SEC
filing text and writes `company_guidance_raw.parquet`,
`company_guidance_normalized.parquet`, and
`expectations_quarterly_features.parquet`. These text/guidance blocks are
**collected but not currently consumed** by the model: in the previous suite they
added negligible lift, so the redesigned `assemble.py` keeps the feature set focused
on what measurably helps. They are the natural input for the LLM/text
feature-extraction next step described in `MODELING.md`.

Alpha Vantage optional data, after setting `ALPHAVANTAGE_API_KEY`:

```bash
export ALPHAVANTAGE_API_KEY="..."
python -m sp_panel.run --earnings-transcripts --transcript-start 2021Q1 --transcript-end 2026Q1 --transcript-max-calls 500
python -m sp_panel.run --av-estimates --av-estimates-max-calls 25
python -m sp_panel.run --av-news
python -m sp_panel.run --av-insider
```

Use `--transcript-max-calls` and `--av-estimates-max-calls` to respect your
account's rate limits. The free Alpha Vantage tier is slow for 100 tickers x many
quarters, so use a paid key or collect in batches for full transcript/estimate
panels.

## Known limitations / decisions (read before modeling)

- **Restatements / look-ahead.** By default we keep the *latest-filed* value for each
  period ("as most recently reported"). For point-in-time backtests use
  `--first-reported`, which keeps the originally filed number.
- **Q4 derivation.** 10-Ks report full-year flows, not a standalone Q4, so Q4 flow
  items are derived as FY − (Q1+Q2+Q3). The three quarters are matched to the fiscal
  year by **period-date containment**, not by the `fy`/`fp` labels (which SEC
  `companyfacts` reports inconsistently — see `MODELING.md`); this fires whenever the
  three contained quarters are present, including off-calendar fiscal years.
- **Revenue tags.** Revenue lives under different XBRL tags across companies/eras;
  we coalesce a priority list (see `config.CONCEPTS`). Spot-check a few names —
  financials and REITs especially report revenue/“total revenues” idiosyncratically.
- **Sector quirks.** R&D is ~zero for Financials, Energy, Real Estate; banks won't
  populate inventory/COGS; REITs use FFO (not in us-gaap standard tags) — add tags to
  `config.CONCEPTS` if you need them.
- **Foreign-incorporated keepers.** SLB, LIN, ETN file full 10-K/10-Q so they parse
  fine, but carry larger FX translation effects in reported figures.
- **Frequency mismatch.** Prices are daily, short interest bi-monthly, financials
  quarterly, macro mixed — align deliberately (the quarterly resamplers are a start).

## Not yet wired (the NLP / relationship layers)

These were in your original spec but need a provider or modeling step beyond this
scaffold:
- **Management sentiment** — pull earnings-call transcripts (FMP or API Ninjas), then
  score with FinBERT / Loughran-McDonald.
- **Footnote disclosures** — SEC "Financial Statement and Notes Data Sets" (bulk) or
  full-text search (efts.sec.gov).
- **Customer / supplier earnings** — require a supply-chain mapping first (10-K major-
  customer disclosures, or a paid SPLC/Revere dataset).
- **M&A signals** — 8-K items 1.01/2.01 or a news feed.

Say the word and I'll add modules for any of these.
