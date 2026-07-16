# Revenue-growth modeling: data fix, redesign, and results

This documents the modeling layer for `revenue_yoy_target` (next-quarter YoY
revenue growth). The pipeline is now a **single source of truth**:

- **`sp_panel/edgar.py`** — SEC extraction, incl. the fixed Q4 derivation.
- **`sp_panel/assemble.py`** — clean target + leakage-free feature engineering → `data/panel.parquet`.
- **`sp_panel/evaluate.py`** — walk-forward validation, baselines, model zoo, breakdowns.

```bash
python -m sp_panel.run --financials      # (re)build financials from cached SEC JSON
python -m sp_panel.assemble              # -> data/panel.parquet (+ dictionary, coverage)
python -m sp_panel.evaluate              # -> data/model_metrics_*.csv, model_predictions.parquet
```

---

## TL;DR

| | Old suite best | Best simple baseline | **New best (XGBoost / ensemble)** |
|---|---:|---:|---:|
| RMSE | 0.353 | 0.229 | **0.188** |
| MAE  | 0.189 | 0.101 | **0.093** |
| R² (vs mean) | 0.055 | 0.172 | **0.44** |
| Directional accuracy | 0.687 | 0.816 | **0.83** |

The redesign fixes a corrupted target at its source, rebuilds the features to be
strictly point-in-time, and evaluates against honest baselines. RMSE drops ~47%
and R² goes from ~0.06 to ~0.44 versus the original suite, while beating every
simple baseline (RMSE −18% vs the best one).

---

## 1. What was wrong (root-cause audit)

**(a) The target was corrupted — the dominant problem.** YoY was computed with
`pct_change(4)` over a per-ticker frame that was *missing quarters*, so the lag-4
shift compared mismatched calendar quarters. The worst offender was **Q4**: the
old Q4 derivation in `edgar.py` matched the annual figure to its three quarters by
the **fiscal-year label** (`fy`), but SEC `companyfacts` labels are unreliable and
frequently offset — e.g. TSLA's calendar-2022 10-Qs are tagged `fy=2023` while the
2022 10-K is tagged `fy=2024`. The match therefore paired the annual with the
wrong (or zero) quarters, producing **missing Q4s and impossible standalone-Q4
numbers** (TSLA Q4-2022 came out as \$9.9B vs the true ~\$24.3B). Raw YoY ranged
from −411 to +18,000 with a std of 238.

**(b) Winsorization hid (a) instead of removing it.** The assembler clipped the
target to its 1/99 percentile, rewriting corrupt revenue into plausible-but-wrong
numbers and feeding them in as both labels and lag features.

**(c) No baseline comparison.** The suite reported R² vs the mean only — never vs
persistence, a trailing average, or a sector median — so there was no way to tell
whether the models beat trivial heuristics. (They barely did.)

**(d) A 400-column kitchen sink** (incl. 100 ticker dummies, macro variance bloat)
that diluted the genuinely predictive lag/momentum/seasonality structure.

**(e) Stale as-of join.** `financials_quarterly.filed` is the *latest* filing that
mentions a period (median lag 510 days, because the collector keeps "as most
recently reported" values), so the filing-date as-of merge was both leaky-in-spirit
and absurdly stale.

## 2. What changed

**Source fix — Q4 derivation (`edgar.py`).** Q4 = annual − (Q1+Q2+Q3), with the
three quarters matched to the fiscal year by **period-date containment**, never by
the `fy`/`fp` labels. This is label-agnostic and works for off-calendar fiscal
years. It corrected every bad Q4 and **recovered ~330 previously-missing Q4 labels**
(Q4 YoY coverage 1,111 → 1,440; total labeled rows 5,109 → 5,480).

**Clean target (`assemble.clean_quarterly`).** Reindex each ticker onto a gap-free
quarterly grid so YoY always compares the same calendar quarter a year earlier;
drop non-positive revenue; **blank** (don't clip) any YoY outside `[−0.95, 3.0]` as
a data artifact. Clean target std 0.25, sensible range. Residual corruption: 0.4%.

**Leakage-free features (`assemble._causal_features`).** Every feature for quarter
T uses only data through **T−1** (a realistic post-earnings, next-quarter forecast).
A compact, purpose-built block: lagged target (T−1…T−5 → persistence + seasonality),
acceleration, trailing growth **mean and volatility**, expanding company trend,
QoQ momentum, size/margins + their changes, quarter-of-year, point-in-time
leave-one-out **sector aggregates**, lagged **macro**, as-of **market** risk/momentum
(vol, beta, returns, drawdown, P/S, EV/Rev), and (extra-lagged) **BEA** sector
value-added. Max feature↔target correlation is 0.22 — no leakage.

**Honest, time-aware evaluation (`evaluate.py`).** Expanding-window walk-forward
(train on the past, test on the next quarter), six baselines, a model zoo
(ElasticNet, Ridge, RandomForest, ExtraTrees, HistGradientBoosting, **LightGBM,
XGBoost, CatBoost**, + blended ensemble), metrics (MAE/RMSE/R²/directional
accuracy) overall and by sector / year / company.

## 3. Results (walk-forward, ~4,500 forecasts, 2014Q1+)

| model | MAE | RMSE | R² | dir. acc |
|---|---:|---:|---:|---:|
| **xgboost** | **0.093** | **0.188** | **0.441** | 0.823 |
| ensemble (LGBM+XGB+Cat+HGB) | 0.092 | 0.188 | 0.439 | **0.832** |
| hist_gbm | 0.095 | 0.191 | 0.423 | 0.813 |
| lightgbm | 0.095 | 0.191 | 0.421 | 0.828 |
| catboost | 0.095 | 0.196 | 0.392 | 0.822 |
| random_forest | 0.095 | 0.199 | 0.374 | 0.822 |
| elasticnet | 0.101 | 0.199 | 0.373 | 0.808 |
| extra_trees | 0.106 | 0.210 | 0.302 | 0.778 |
| baseline: persistence | 0.101 | 0.229 | 0.172 | 0.816 |
| baseline: sector median | 0.126 | 0.240 | 0.090 | 0.754 |
| baseline: trailing mean (4q) | 0.126 | 0.246 | 0.044 | 0.753 |
| baseline: company expanding mean | 0.138 | 0.246 | 0.040 | 0.711 |
| baseline: expanding global mean | 0.142 | 0.252 | −0.003 | 0.714 |
| baseline: seasonal (same q last yr) | 0.186 | 0.347 | −0.906 | 0.646 |
| _OLD suite best (hgb/M4) — corrupted target_ | _0.189_ | _0.353_ | _0.055_ | _0.687_ |

**Robustness.** R² is positive in **every sector** (0.24–0.52) and **every year**
2014–2026 (0.21–0.56), including the COVID-disrupted period. The ensemble beats the
expanding-mean baseline (RMSE) for **97/100** companies and persistence for 77/100.
Hardest sector: Energy (commodity-driven, RMSE 0.27 but R² 0.49); easiest: Consumer
Staples (RMSE 0.07).

**Best model:** XGBoost and the ensemble are tied; the ensemble is marginally more
directionally accurate and more stable. **LightGBM** is the simplest near-best
single model. **Top features** (LightGBM gain): persistence (22%), QoQ momentum
(21%), same-quarter-last-year (9%), the rest of the lag ladder, trailing
volatility, operating margin, growth-vs-sector — i.e. momentum + seasonality +
mean-reversion + sector context (financial block ≈ 84% of total gain; macro 7%,
market 5%, BEA 2%).

> Comparability note: the old numbers are on the corrupted+winsorized target
> (different scale/test window), so RMSE is not perfectly apples-to-apples. R²,
> directional accuracy, and skill-over-baselines are the rigorous comparisons, and
> all move strongly in the new approach's favor.

## 3b. Expectations/guidance benchmark & uncertainty intervals

Two additions aimed at the biggest gaps versus professional practice (an
expectations benchmark, and calibrated uncertainty):

**Management-guidance features (`assemble.add_guidance_features`).** Point-in-time,
leakage-free signals from SEC-filing guidance events (`company_guidance_normalized`):
net raise/lower direction and counts over trailing 90/180/365-day windows before
the forecast origin, recency, and latest action. Important caveats found in the
data: this is *management guidance*, not sell-side consensus; coverage is thin
(**~13% of rows** have any recent guidance); and the text-extracted dollar
*midpoints* are unreliable (only **2 rows** yield a usable guidance-implied revenue
level), so only the robust *directional* signal is used.

Controlled A/B (LightGBM, with vs without the 8 `f_guid_*` features):

| config | MAE | RMSE | R² | dir. acc |
|---|---:|---:|---:|---:|
| without guidance | 0.0947 | 0.1912 | 0.421 | 0.828 |
| with guidance | 0.0946 | 0.1905 | 0.426 | 0.823 |

Net: **≈0 overall lift** (RMSE −0.0007). Even restricted to the 705 covered rows,
guidance only trims RMSE 0.177→0.173 (~2%) and is marginally *worse* directionally.
This is the honest, expected result given the coverage/quality — and it pinpoints
the fix: a real **sell-side consensus feed** (IBES / Visible Alpha, or Alpha
Vantage estimates → `analyst_estimates.parquet` via the existing `run.py` hook),
which would be dense, numeric, and far more informative than parsed guidance.

**Quantile intervals (`evaluate.quantile_walk_forward`).** LightGBM quantile
regression (p10/p50/p90) in the same walk-forward gives calibrated predictive
intervals instead of a bare point estimate:

| nominal | empirical coverage | mean width | pinball p50 | p50 MAE | p50 dir. acc |
|---|---:|---:|---:|---:|---:|
| 80% (p10–p90) | **0.71** | 0.216 (±11pp) | 0.045 | 0.090 | 0.83 |

The 80% interval **under-covers (71%)** — the intervals are a touch too narrow
out-of-sample, a known tendency of quantile GBMs under regime shifts. A conformal
calibration wrapper would restore nominal coverage and is the recommended next step
for production use. Outputs: `model_quantile_calibration.csv`,
`model_quantile_predictions.parquet`, `model_guidance_ablation.csv`.

## 3c. Mixture-of-experts gate (tested, not adopted)

`sp_panel/moe.py` prototypes a feature-gated MoE: three block experts (company /
macro / sector), blended by a softmax gate conditioned on company context (size,
growth volatility, sector, and each expert's trailing per-company accuracy), trained
walk-forward and leakage-free. Head-to-head on identical rows:

| method | RMSE | R² | dir. acc |
|---|---:|---:|---:|
| monolithic full | **0.193** | 0.437 | 0.823 |
| company expert alone | 0.195 | 0.430 | 0.819 |
| global static stack | 0.195 | 0.426 | 0.820 |
| MoE gate | 0.199 | 0.402 | 0.813 |
| macro expert alone | 0.247 | 0.079 | 0.719 |

**The gate underperforms** the monolithic model and even a static stack. Why: the
company block alone ≈ the full model (the macro/sector experts are weak and
redundant), so there is nothing to arbitrate; learned weights are near-uniform
(per-company volatility 0.03) and the gate helps the two noisiest sectors slightly
while hurting the stable ones. Trees already do conditional weighting internally,
and a gate cannot see the cross-block interactions a monolithic model exploits.

The architecture is sound but **starved of differentiated signal** on this feature
set — it's the right thing to revisit once the experts are individually strong and
genuinely complementary (alt-data demand signals, segment revenue, analyst
consensus). Run: `python -m sp_panel.moe`.

## 3d. Macro ablation: does the `mac_*` block earn its place?

Feature-importance percentages cannot answer "should we include macro?" — macro
has no cross-sectional variation (every company sees the same CPI), so its
effective sample is the ~50 test *quarters*, not the ~4,500 company-quarters, and
tree importances structurally under-weight it. The honest instrument is a paired
ablation: one fixed model (LightGBM), identical walk-forward, three feature sets —
**A** `no_macro`, **B** `with_macro`, **C** `macro_only` (+ sector one-hot). B−A is
paired row-by-row, collapsed to per-quarter mean loss differentials (errors within
a quarter share the common shock, so the quarter is the unit of independent
evidence), and tested with a Diebold–Mariano / Newey–West t-test (HLN-corrected).

Run: `python -m sp_panel.evaluate --macro-ab-only` (also runs inside the full
evaluate; skip with `--no-macro-ab`). Inputs were upgraded first: `FRED_START`
1990 (fills 2011–2014 macro that was previously NaN and gives transforms decades
of context), redundant `_qvar` columns dropped, and `hy_spread` replaced by
`baa_spread` (BAA10Y) because the keyless FRED CSV endpoint caps the licensed ICE
BofA series at ~3 years of history.

Result (2026-07, 49 test quarters, 4,529 paired forecasts):

| config | RMSE | R² | dir. acc |
|---|---:|---:|---:|
| no_macro (A) | 0.1918 | 0.418 | **0.826** |
| with_macro (B) | **0.1907** | **0.424** | 0.824 |
| macro_only (C) | 0.2358 | 0.120 | 0.738 |

- **B−A is positive but insignificant**: mean quarterly ΔSE +0.0004, DM p = 0.43,
  macro better in only 24/49 quarters; directional accuracy is *slightly worse*.
- **The gain is regime-concentrated**: ~91% of the net improvement comes from a
  single quarter (2021Q2, the COVID base-effect rebound), with the rest of the
  positive delta in the 2022–2023Q1 inflation/rate-shock quarters (+0.008 RMSE
  in 2021 and 2022). In normal years macro is a coin flip or a small drag.
- **C beats naive baselines but not persistence** (0.236 vs 0.229): common-shock
  information alone is worth something, but less than a firm's own last YoY.

**Verdict: macro is tail insurance, not a steady-state improvement.** Keep the
block (it is nearly free, and it pays at turning points exactly when forecasts
matter most), but do not expect average-accuracy gains. Note FRED serves revised
(not vintage) data, which flatters macro; the "insignificant on average" verdict
is therefore conservative.
Outputs: `model_macro_ablation{,_tests,_by_year,_quarterly}.csv`.

## 3e. Which macro features, and where (Stages 3–4)

`python -m sp_panel.macro_stages` (same protocol as §3d: one fixed LightGBM,
identical walk-forward splits, paired forecasts, quarter-clustered DM tests).

**Stage 3a — themes, added one at a time to the no-macro base.** Grouping the 25
`mac_*` columns into 5 economically coherent themes stops collinear dilution:

| theme (added alone) | ΔRMSE | DM p | verdict |
|---|---:|---:|---|
| activity_demand (retail, INDPRO, sentiment, unemployment) | +0.0015 | 0.33 | best, insignificant |
| risk_credit (VIX, Baa spread) | +0.0010 | 0.29 | weak positive |
| commodities_fx (oil, USD) | +0.0004 | 0.70 | ~nothing |
| inflation (CPI, core PCE) | −0.0003 | 0.86 | nothing |
| rates_curve (FF, USTs, slope) | −0.0011 | 0.11 | borderline **hurts** |

Rates/curve *levels* trend, so walk-forward trees extrapolate them badly — if the
block is slimmed, drop these first. **Stage 3b** (LASSO of the no-macro model's
per-quarter mean residual — the missed common shock — on standardized macro)
corroborates: it selects a single variable, `mac_retail_sales_qoq`, with a tiny
coefficient. The macro information the firm features miss is (barely) demand
momentum, not rates or inflation.

**Stage 4a — the pooled macro delta by sector** confirms the theoretical ordering:
cyclical sectors gain (Energy +0.0090, Consumer Discretionary +0.0047 p=0.10 —
the only sector <0.10), defensive/idiosyncratic sectors pay a small noise tax
(Industrials −0.0036 p=0.07, Real Estate −0.0047, Staples −0.0025). The
near-zero pooled average of §3d is opposite-signed sector effects cancelling.

**Stage 4b — explicit sector×macro interactions** (6 theme representatives ×
sector dummies, added to the full model): ALL +0.0007 (p=0.41). Energy improves
+0.0062 (p=0.15) — the oil×Energy channel is real but sub-significant — while
Technology worsens. Depth-4 trees with sector one-hots already capture what
little interaction structure exists.

**Stage 4c — sector-specialist models** (same features/hyperparameters, trained
per sector, compared to the pooled model on identical rows): **specialists lose
in all 10 sectors** and significantly overall (RMSE 0.207 vs 0.194, ΔRMSE
−0.0131, DM p = 0.014; directional accuracy −3.9pp; significant individual
losses in Financials p=0.02, Industrials p=0.03, Technology p=0.03). Even
Energy's specialist loses (−0.0014). With ~10 tickers/sector, specialists train
on ~150–500 rows vs the pooled model's up-to-5,000; the cross-sector cyclical
structure the pooled model borrows is worth far more than sector purity costs.
This triangulates with the MoE result (§3c): three architectures — hard split,
learned gate, explicit interactions — all fail to beat one pooled model at this
data scale. Revisit specialists only with ~5× more names per sector (full S&P
500) or genuinely sector-specific features (commodity curves, same-store sales,
loan-growth data).

Outputs: `macro_stage3_{themes,residual_lasso}.csv`,
`macro_stage4_{by_sector,interactions,sector_models}.csv`.

## 3f. New feature blocks: accounting, filing-event CARs, industry demand

Three literature-motivated blocks were built and ablated
(`python -m sp_panel.macro_stages --stage blocks`, same paired quarter-clustered
protocol; base = current model without the block):

- **accounting** (`f_acct_*`): deferred-revenue change/level (leads future
  sales), receivables-vs-revenue growth gap (Sloan-style red flag), goodwill
  jump + trailing M&A flag (marks inorganic YoY). New XBRL concepts
  `deferred_revenue`/`receivables`/`goodwill` extracted from the already-cached
  companyfacts. Coverage 42–100% of rows.
- **filing_event** (`f_evt_*`): CARs and abnormal volume around the T−1
  report's filing from `filing_event_study.parquet` (PEAD-motivated; the
  earnings 8-K precedes the 10-Q by a median 2 days here, so the −5..+5 window
  spans the announcement). Coverage ~52%.
- **industry_demand** (`f_ind_*`): sector-matched FRED demand growth (autos →
  Discretionary, semiconductor IP → Tech, oil & gas extraction → Energy, C&I
  loans → Financials, ...; `config.SECTOR_SERIES`), one cross-sectionally
  varying column instead of ten pooled ones.

Result (49 quarters, ~4,525 paired forecasts, LightGBM):

| block | ΔRMSE | DM p | verdict |
|---|---:|---:|---|
| accounting | +0.0003 | 0.73 | flat; best in Energy (+0.0061) |
| short_interest | −0.0005 | 0.37 | no lift (see below) |
| industry_demand | −0.0010 | 0.17 | no lift |
| filing_event | −0.0012 | 0.12 | borderline drag |
| all four | +0.0001 | 0.88 | nothing |

**Short interest** (`f_si_*`: SI/shares, days-to-cover, 1m/1q positioning
change; FINRA `ConsolidatedShortInterest` via `--short`, bi-monthly with a
14-day publication lag) deserves its own note because the literature prior was
strongest. The FINRA API only has history from ~2018, so the block was retested
on its covered window: 2018Q1+ ΔRMSE −0.0009 (p = 0.34, better in 15/33
quarters, dir-acc −0.75pp) — no lift even where the data exists
(`feature_blocks_si_2018.csv`). Per sector it drags nearly everywhere
(Comm Svcs −0.0021 p=0.07, Cons Disc −0.0030 p=0.09); only Technology (+0.0030,
n.s.) leans positive. Plausible reading: shorts predict *misses* for the
troubled tail of the market, and this 100-name mega-cap universe has too few
such names for the signal to pay for its noise. Data-quality gotcha: FINRA's
`EquityShortInterest` dataset, despite the name, is OTC-only and silently
returns zero rows for listed tickers — `short_interest.py` now uses
`ConsolidatedShortInterest`.

**None earn a place.** The information they carry is largely subsumed: the
model already sees T−1's actual revenue surprise directly (lag/accel features),
so the announcement CAR adds only the market's read *beyond* the numbers;
deferred revenue tracks the revenue trajectory already encoded in the lag
ladder; industry demand overlaps sector LOO aggregates and BEA value-added.
Default feature set therefore **keeps `f_acct_*`** (only positive block, and
the M&A flag has data-hygiene value) and **excludes `f_evt_*`/`f_ind_*`**
(`assemble.EXPERIMENTAL_PREFIXES`; pass `feature_columns(panel,
experimental=True)` to re-test). Honest-negative caveats: CAR coverage is only
~52% and starts 2013Q4; a denser announcement-dated feed (8-K based) could be
retested. Outputs: `feature_blocks_ablation.csv`, `feature_blocks_by_sector.csv`.

Still missing (requires ALPHAVANTAGE_API_KEY, parked for now): sell-side
estimate revisions (`--av-estimates`) — the one untested block with a strong
prior; wire as `f_est_*` (prefix already reserved in `EXPERIMENTAL_PREFIXES`
and `macro_stages.NEW_BLOCKS`) and re-run `--stage blocks` when the parquet
exists.

## 3g. Annual (TTM) growth target

`revenue_annual_target` = the year starting at T vs the year ending at T−1
(`ttm[T+3]/ttm[T−1] − 1`), features as-of T−1 as usual. Because the label spans
T..T+3, walk-forward training is **purged** (train only on origins whose label
window closed before the test origin; `walk_forward(purge_quarters=3)`), and
DM tests use Newey–West lags = 4 (overlapping windows). Annual baselines:
TTM-growth persistence / expanding company mean / sector median. Run:
`python -m sp_panel.evaluate --annual` (outputs `*_annual.csv`).

Results (47 test quarters, ~4,040 forecasts): annual growth is much harder —
best model RandomForest RMSE 0.227, R² 0.12, dir-acc 0.76 (vs R² 0.44
quarterly), beating the best baseline by 6.7%. Notably the baseline hierarchy
flips: quarterly persistence is nearly unbeatable per-quarter but is the WORST
annual baseline (R² −0.81); the expanding grand mean is the best naive rule —
at a 1-year horizon, mean reversion dominates persistence. Boosting's edge
also fades (RF > ensemble > GBMs): with a noisier target, variance reduction
beats sequential bias-chasing.

**Macro finally clears significance at this horizon** (`macro_annual_*.csv`):
the full block is still ~0 pooled (+0.0007, p=0.60, dir-acc +1.2pp), but
- **activity_demand added alone: +0.0024 RMSE, DM p = 0.044**, better in 29/47
  quarters, dir-acc +1.7pp — the demand-momentum theme (INDPRO, retail sales,
  unemployment, sentiment) is genuinely predictive one year out;
- **inflation actively hurts: −0.0019, p = 0.034**; rates_curve levels also
  negative (−0.0013) — same nonstationarity problem as the quarterly horizon;
- **Energy is the first sector with significant macro value: +0.0061,
  p = 0.043** (30/46 quarters), with the other cyclicals (Cons Disc +0.0045,
  Materials +0.0043, Health Care +0.0036) consistently positive.

Recommended annual-model macro config: keep activity_demand (and harmless
risk_credit), drop inflation and rates-level columns. Tree importance broadly
agrees on activity (INDPRO alone = 21% of macro gain) but also ranks USD/WTI
highly even though the commodities theme ablates to ~0 — one more reminder
that importance ≠ incremental value.

## 3h. Margin targets: next-year gross margin and EBITDA margin

`gm_annual_target` / `em_annual_target` = TTM gross / EBITDA margin over the
year starting at T (ratio of 4-quarter sums), same purged walk-forward as the
annual revenue target. Coverage fixes that made these viable: gross profit
falls back to `revenue − cost_of_revenue` when the `GrossProfit` tag is absent
(coverage 1.8k → 3.2k rows) and `DepreciationAndAmortization` was added as a
`dep_amort` synonym (EBITDA rows 0.9k → 1.9k). Financials/REITs largely lack
these concepts, so margin targets skip them. Sign-based directional accuracy
is vacuous for margins (~0.99 — margins are almost always positive); the
summary CSVs add `chg_dir_acc`: direction of margin *change* vs the trailing
TTM margin. Run: `python -m sp_panel.evaluate --target gross-margin` /
`--target ebitda-margin`.

**Gross margin (2,540 forecasts): persistence wins.** Last year's margin
predicts next year's with R² 0.861 / RMSE 0.085 / median error 1.6 margin
points, and **no model beats it** (best: XGBoost 0.090). Gross margin is
structural — pricing model and cost structure — and barely moves year to
year; the models pay a shrinkage cost on a target that doesn't want to be
shrunk. Models do call the *direction of change* better than chance (~56%),
which persistence by definition cannot. Practical guidance: forecast = the
trailing TTM margin; use the model only for change-direction flags.

**EBITDA margin (1,446 forecasts): the models earn their keep.** CatBoost
RMSE 0.104 vs persistence 0.117 (−11%), R² 0.782 vs 0.726, change-direction
~59%. EBITDA margin moves with operating leverage and cost discipline, so
there is forecastable variation beyond stickiness — and the familiar pattern
recurs: persistence still wins the *median* quarter (2.4 vs 3.5 points), the
model wins the tails. Outputs: `model_metrics_*_{gm,em}_annual.csv`,
`model_metrics_full_summary_{gm,em}_annual.csv`.

## 3i. Macro lag structure: which delay carries the signal?

The default panel takes each macro variable at one fixed lag (quarter T−1).
To test whether that throws information away, `add_macro_features` gained an
`extra_lags` option that adds each macro *level* at T−2/T−3/T−4
(`mac_<name>_l<k>`, 45 columns; `build_panel(macro_extra_lags=(2,3,4))`,
panel saved as `panel_maclag.parquet`). Same paired quarter-clustered
protocol; annual purged with NW lags 4.

| target | RMSE lag-1 only | + lag ladder | Δ | DM p | quarters better |
|---|---:|---:|---:|---:|---:|
| quarterly | 0.1889 | 0.1880 | +0.0009 | 0.22 | 25/49 |
| **annual** | 0.2367 | 0.2348 | **+0.0018** | **0.047** | **30/47** |

**Deeper macro lags significantly help the annual model** — the second macro
result to clear 5% significance at that horizon (after activity_demand, §3g) —
and do nothing for the quarterly one. The lag-usage breakdown (mean |SHAP|
share of the macro block) explains why:

| lag bucket | quarterly | annual |
|---|---:|---:|
| lag 1 — changes (momentum) | **40.5%** | 10.9% |
| lag 1 — levels | 11.9% | 10.4% |
| lag 2 | 19.7% | **45.8%** |
| lag 3 | 12.7% | 18.3% |
| lag 4 | 15.2% | 14.7% |

The quarterly model wants *fresh macro momentum* (top macro features: oil and
retail-sales changes, all lag-1); the annual model wants the *macro state from
2–4 quarters back* (top features: industrial production and oil levels at l2–l4,
10y yield at l2 — nearly 80% of its macro attention sits at lag ≥ 2). That is
the classic transmission-delay picture: activity indicators lead revenue with a
~2-quarter delay that only becomes visible at the annual horizon.
**Recommendation:** keep the single-lag block for the quarterly model; add the
lag ladder to the annual configuration. Outputs: `macro_lag_ablation.csv`,
`macro_lag_importance.csv`.

## 4. Should you use / fine-tune an LLM?

**No, not for the prediction.** This is a structured-numeric forecasting problem;
gradient-boosted trees are the right tool and already extract strong signal. The
repo's text is far too sparse to fine-tune an LLM (~2k filings, ~1.2k transcript
turns), and the original engineered text block added ≈0 lift. The defensible use is
**text → numeric feature extraction** (management tone from MD&A/earnings calls via
FinBERT or a zero-shot classifier; explicit guidance direction; risk-factor
deltas), fed point-in-time to the tabular model — an increment on top of this
pipeline, not a replacement.

## 5. Recommended next steps

1. **Real sell-side consensus.** The single biggest accuracy lever and the missing
   industry benchmark. Management guidance (§3b) is too sparse/noisy to help; pull a
   dense numeric estimate feed (IBES / Visible Alpha, or Alpha Vantage estimates →
   `analyst_estimates.parquet`) and model the *surprise vs consensus*.
2. **Conformal-calibrate the quantile intervals** (§3b) — the 80% band currently
   under-covers (71%); a split-conformal wrapper restores nominal coverage.
3. **`--first-reported` re-pull** so features can use T−1 data the moment it is
   actually released (a true post-earnings nowcast) instead of a conservative lag.
4. **Sector-specialist models / commodity features** for the hard sectors (Energy).
5. **Light time-series-CV hyperparameter tuning** (defaults are sensible but untuned).
6. **Then** revisit text as feature extraction (§4) and measure lift honestly
   against these baselines.

_Done in this iteration:_ quantile interval forecasts and a point-in-time
guidance/expectations benchmark (§3b).

## 6. Limitations
- Intrinsically low-signal target; R² ≈ 0.44 is good for revenue nowcasting but
  large residual variance remains.
- The fixed 1-quarter feature lag is conservative; a true post-earnings nowcast
  would likely score higher.
- Macro history now starts in 1990 (`FRED_START`), so every panel row has full
  macro coverage; BEA is extra-lagged for point-in-time safety. FRED serves
  *revised* macro values, not real-time vintages — this flatters macro features
  slightly, so a negative macro-ablation verdict (§3d) is conservative evidence.

## 7. Model glossary — what each model in the suite actually does

All models see the same design matrix (the engineered features + sector
one-hot dummies) and the same walk-forward splits; they differ only in how
they turn features into a prediction. Linear models get median imputation and
standardization; tree models handle raw scales, and the gradient boosters
handle missing values natively (a NaN is just routed to whichever side of a
split fits it best — no imputation guesswork).

**The baselines (sanity rules, not models).** `persistence` repeats the last
known growth; `seasonal` repeats the same quarter last year; the trailing
means average the last 4/8 known values; `company_hist` is the firm's own
long-run average; `sector_median` is the median of sector peers' last growth;
`expanding_mean` is the grand mean of all past targets. Any model that cannot
beat these is decoration — that comparison is the whole reason they exist.

**Ridge** — plain linear regression plus a penalty on large coefficients (L2).
Every feature gets a weight; the penalty shrinks them all toward zero so no
single noisy feature can dominate. Penalty strength chosen by cross-validation
inside the training window. Its weakness here: revenue growth's feature
interactions are nonlinear, and a straight line through them underfits.

**ElasticNet** — same idea with a mixed penalty (L1+L2). The L1 part can push
weights exactly to zero, so it doubles as automatic feature selection. Useful
as an interpretable floor: whatever ElasticNet can do is achievable with a
sparse linear rule.

**RandomForest** — hundreds of decision trees, each trained on a bootstrap
resample of the data and restricted to a random subset of features at every
split, then averaged. Individual deep trees overfit wildly; averaging many
*decorrelated* overfitters cancels their errors (variance reduction). Robust,
hard to tune badly, and notably the winner on the noisy annual target.

**ExtraTrees** — RandomForest with one extra dose of randomness: split
thresholds are drawn at random instead of optimized. Trees get individually
worse but even less correlated; sometimes that trade wins, here it doesn't.

**Gradient boosting (the family: hist_gbm, LightGBM, XGBoost, CatBoost)** —
instead of averaging independent trees, build *small* trees sequentially,
each one fit to the errors the ensemble has made so far, and add it with a
small step size (learning rate 0.03). The ensemble creeps toward the signal,
correcting itself as it goes — bias reduction rather than variance reduction.
The four implementations differ in the details:
- **hist_gbm** (scikit-learn) bins features into histograms for speed and
  stops adding trees when an internal 15% validation slice stops improving.
- **LightGBM** grows trees leaf-wise — it always extends whichever leaf cuts
  the loss most, giving lopsided but efficient trees (capped at 15 leaves,
  depth 4 here). Fast; used as the fixed workhorse in every ablation.
- **XGBoost** grows level-wise with a regularized objective; the steadiest
  performer on the quarterly target.
- **CatBoost** uses "ordered boosting" — each tree's error estimates are
  computed only from rows that come earlier in a random ordering, which
  fights a subtle self-contamination in the gradients.

**Ensemble** — the simple average of the four boosters' predictions. They make
similar but not identical mistakes; averaging cancels the non-shared part.
Reliably as good as, or slightly better than, the best member — the usual
free lunch of blending.

**Quantile LightGBM** (§3b) — the same LightGBM machinery trained with the
pinball loss at p10/p50/p90 instead of squared error, producing an honest
uncertainty band around the point forecast rather than a single number.

**Why the winner changes with the horizon:** boosting chases residual signal;
when the target is comparatively predictable (quarterly, R² ≈ 0.44) that bias
reduction wins. When the target is mostly noise (annual, R² ≈ 0.12), chasing
residuals means chasing noise, and RandomForest's variance-averaging is the
better temperament. That flip is itself evidence the pipeline is honest.

## 8. Feature importance — what the models actually use

**Method.** LightGBM *gain* importance: for every split in every tree, the
training-loss reduction is credited to the feature that made the split;
credits are summed and normalized to 100%. Computed on a full-sample fit per
target (`model_feature_importance{,_annual,_gm_annual,_em_annual}.csv`;
grouped view in `feature_importance_grouped.csv`).

**Read it with the §3d caveat in mind.** Importance measures what the model
*consulted*, not what improves forecasts: it splits across correlated columns,
starves common-shock features (macro's effective sample is ~50 quarters), and
can credit features that ablate to zero — at the annual horizon the trees
lean on USD/WTI columns even though the commodities theme adds nothing out of
sample, while the activity theme is both used AND significant. Importance
describes the model; **ablation (§3d–3g) decides inclusion.**

Grouped gain importance, % of total, per target:

| feature group | quarterly rev | annual rev | gross margin | EBITDA margin |
|---|---:|---:|---:|---:|
| own history (lags/momentum/vol) | **70.8** | **41.9** | 4.7 | 5.7 |
| fundamentals & margins | 8.9 | 21.7 | **71.8** | **82.8** |
| macro | 6.1 | 5.5 | 0.7 | 2.4 |
| sector aggregates (peers) | 4.3 | 3.1 | 0.2 | 0.7 |
| market (price/risk/valuation) | 4.1 | 12.3 | 1.9 | 2.4 |
| accounting indicators | 3.9 | 6.0 | 1.0 | 1.8 |
| BEA sector value-added | 1.6 | 4.8 | 3.8 | 1.9 |
| guidance | 0.2 | 4.0 | 0.1 | 0.2 |
| sector identity (one-hot) | 0.0 | 0.7 | 15.7 | 2.1 |
| seasonality | 0.1 | 0.0 | 0.0 | 0.0 |

Top individual features: quarterly = recent QoQ momentum (21%) + last YoY
(21%) + seasonal lag (9%); annual = the same history features at half the
weight plus valuation ratios (EV/revenue, P/S ≈ 7%); gross margin = the
current gross margin itself (38%) + cost-structure ratios + sector identity;
EBITDA margin = the current EBITDA margin (66%).

**The pattern across targets is the real insight.** Growth targets are
history-dominated (what you grew is what you'll grow, fading with horizon as
valuation/fundamentals rise); margin targets are level-dominated (what you
earn is what you'll keep earning — sector identity matters for gross margin
because margin *levels* are sector-structural). Macro never exceeds ~6%
anywhere — consistent with the ablation verdicts: a real but small, shock-
concentrated, horizon-dependent contributor.

## 8b. Per-feature importance table (individual features)

§8 groups importance into blocks; this is the full per-feature breakdown from
the same LightGBM gain method (full-sample fit per target, gain % of total).
Every feature carrying non-zero weight in at least one target is listed, ranked
by its weight in the primary quarterly model. Sector one-hots appear as
`sec_*`. The complete 96-row export (including all-zero features) is
`data/model_feature_importance_all_targets.csv`; per-target files remain at
`model_feature_importance{,_annual,_gm_annual,_em_annual}.csv`. Read it with
the §8 caveat: gain shows what the trees *split on*, which splits across
correlated columns and can rank a feature the ablation shows adds nothing —
it describes the model, it does not by itself justify inclusion.

| feature | group | quarterly | annual | gross margin | EBITDA margin |
|---|---|---:|---:|---:|---:|
| `f_qoq_mean_4` | own history | 21.37 | 7.99 | 0.34 | 0.37 |
| `f_lag_yoy_1` | own history | 20.76 | 8.05 | 0.16 | 0.20 |
| `f_lag_yoy_4` | own history | 9.45 | 1.80 | 0.04 | 0.07 |
| `f_roll_std_4` | own history | 3.88 | 2.92 | 0.12 | 0.30 |
| `f_lag_yoy_2` | own history | 3.61 | 1.15 | 0.25 | 0.15 |
| `f_m_oper` | fundamentals & margins | 2.14 | 1.58 | 3.01 | 5.64 |
| `f_yoy_vs_sector` | sector aggregates | 1.98 | 1.76 | 0.09 | 0.13 |
| `f_qoq_1` | own history | 1.84 | 4.07 | 0.09 | 0.26 |
| `f_acct_recv_gap` | accounting indicators | 1.51 | 1.30 | 0.23 | 0.25 |
| `mkt_ret_63` | market | 1.40 | 0.59 | 0.04 | 0.05 |
| `f_roll_mean_4` | own history | 1.32 | 0.97 | 0.18 | 0.09 |
| `f_lag_yoy_5` | own history | 1.32 | 0.79 | 0.07 | 0.11 |
| `f_m_net` | fundamentals & margins | 1.30 | 1.95 | 3.71 | 2.72 |
| `mkt_ret_126` | market | 1.25 | 2.48 | 0.07 | 0.39 |
| `f_m_sga` | fundamentals & margins | 1.22 | 3.31 | 8.05 | 0.43 |
| `f_yoy_z` | own history | 1.20 | 0.68 | 0.05 | 0.13 |
| `f_sector_yoy_loo` | sector aggregates | 1.19 | 0.25 | 0.02 | 0.12 |
| `f_roll_std_8` | own history | 1.13 | 1.52 | 0.14 | 0.28 |
| `f_accel_1` | own history | 1.06 | 0.57 | 0.10 | 0.15 |
| `f_m_oper_chg4` | fundamentals & margins | 1.02 | 0.98 | 0.18 | 1.08 |
| `f_acct_defrev_chg4` | accounting indicators | 0.95 | 2.88 | 0.09 | 0.50 |
| `f_acct_gw_jump` | accounting indicators | 0.93 | 0.58 | 0.30 | 0.19 |
| `f_lag_yoy_3` | own history | 0.92 | 2.00 | 0.07 | 0.09 |
| `f_hist_std` | own history | 0.88 | 4.21 | 1.33 | 1.98 |
| `mac_retail_sales_yoy` | macro | 0.84 | 0.49 | 0.02 | 0.07 |
| `f_sector_yoy_median` | sector aggregates | 0.80 | 0.70 | 0.07 | 0.24 |
| `bea_sector_va_qoq` | BEA sector value-added | 0.77 | 2.20 | 0.05 | 0.25 |
| `mac_wti_oil_yoy` | macro | 0.77 | 0.39 | 0.04 | 0.14 |
| `f_hist_mean` | own history | 0.77 | 2.33 | 1.35 | 0.79 |
| `f_accel_seasonal` | own history | 0.74 | 1.05 | 0.05 | 0.27 |
| `mac_retail_sales_qoq` | macro | 0.68 | 0.26 | 0.01 | 0.07 |
| `f_m_net_chg4` | fundamentals & margins | 0.67 | 1.60 | 0.10 | 0.50 |
| `f_m_ebitda` | fundamentals & margins | 0.59 | 2.53 | 1.55 | 66.31 |
| `mac_ism_pmi_proxy_qoq` | macro | 0.58 | 0.06 | 0.01 | 0.04 |
| `mkt_market_cap` | market | 0.57 | 2.03 | 0.92 | 0.89 |
| `f_roll_mean_8` | own history | 0.53 | 1.40 | 0.22 | 0.27 |
| `f_acct_defrev_to_rev` | accounting indicators | 0.51 | 1.28 | 0.37 | 0.86 |
| `mac_wti_oil_qoq` | macro | 0.50 | 0.18 | 0.02 | 0.06 |
| `f_asset_turnover` | fundamentals & margins | 0.48 | 2.75 | 9.06 | 3.15 |
| `bea_sector_va` | BEA sector value-added | 0.39 | 1.99 | 3.69 | 1.57 |
| `bea_sector_va_yoy` | BEA sector value-added | 0.39 | 0.57 | 0.06 | 0.12 |
| `mac_yield_curve_10y2y` | macro | 0.39 | 0.08 | 0.06 | 0.06 |
| `f_log_revenue` | fundamentals & margins | 0.37 | 3.44 | 3.44 | 0.75 |
| `f_sector_yoy_std` | sector aggregates | 0.36 | 0.42 | 0.02 | 0.21 |
| `mkt_ps_ratio` | market | 0.35 | 3.49 | 0.37 | 0.49 |
| `f_m_gross_chg4` | fundamentals & margins | 0.33 | 0.40 | 0.43 | 0.13 |
| `mac_wti_oil` | macro | 0.31 | 0.48 | 0.04 | 0.04 |
| `mac_consumer_sentiment` | macro | 0.29 | 0.15 | 0.03 | 0.11 |
| `f_m_ebitda_chg4` | fundamentals & margins | 0.28 | 1.39 | 0.07 | 1.04 |
| `mkt_drawdown_63` | market | 0.27 | 0.18 | 0.04 | 0.12 |
| `mac_cpi_yoy` | macro | 0.26 | 0.08 | 0.03 | 0.61 |
| `mkt_ev_to_rev` | market | 0.26 | 3.53 | 0.44 | 0.47 |
| `mac_usd_index_yoy` | macro | 0.21 | 0.54 | 0.01 | 0.07 |
| `f_m_rd` | fundamentals & margins | 0.21 | 1.08 | 4.11 | 0.03 |
| `mac_usd_index_qoq` | macro | 0.18 | 0.03 | 0.01 | 0.01 |
| `mac_vix` | macro | 0.16 | 0.18 | 0.02 | 0.06 |
| `f_m_gross` | fundamentals & margins | 0.16 | 0.52 | 38.01 | 0.92 |
| `mac_fed_funds` | macro | 0.15 | 0.12 | 0.02 | 0.12 |
| `mac_cpi_qoq` | macro | 0.15 | 0.12 | 0.01 | 0.05 |
| `f_guid_days_since_last` | guidance | 0.14 | 2.86 | 0.11 | 0.15 |
| `mac_ism_pmi_proxy_yoy` | macro | 0.13 | 0.15 | 0.06 | 0.07 |
| `mac_unemployment` | macro | 0.12 | 0.10 | 0.00 | 0.02 |
| `mac_ism_pmi_proxy` | macro | 0.10 | 1.17 | 0.11 | 0.11 |
| `f_m_cfo` | fundamentals & margins | 0.08 | 0.01 | 0.05 | 0.05 |
| `mac_ust_3m` | macro | 0.08 | 0.10 | 0.02 | 0.22 |
| `f_is_q1` | seasonality | 0.07 | 0.01 | 0.00 | 0.04 |
| `f_obs_count` | own history | 0.07 | 0.42 | 0.19 | 0.16 |
| `mac_ust_2y` | macro | 0.05 | 0.25 | 0.05 | 0.01 |
| `mac_cpi` | macro | 0.04 | 0.05 | 0.07 | 0.15 |
| `mac_baa_spread` | macro | 0.04 | 0.07 | 0.01 | 0.06 |
| `mac_ust_10y` | macro | 0.04 | 0.09 | 0.02 | 0.17 |
| `mac_usd_index` | macro | 0.03 | 0.15 | 0.07 | 0.02 |
| `sec_Energy` | sector identity | 0.02 | 0.13 | 0.42 | 0.01 |
| `f_guid_latest_lower` | guidance | 0.02 | 0.01 | 0.01 | 0.00 |
| `mac_core_pce` | macro | 0.02 | 0.08 | 0.01 | 0.01 |
| `f_m_capex` | fundamentals & margins | 0.01 | 0.13 | 0.06 | 0.03 |
| `f_guid_latest_raise` | guidance | 0.01 | 1.03 | 0.00 | 0.00 |
| `f_guid_raise_90d` | guidance | 0.01 | 0.00 | 0.00 | 0.00 |
| `sec_Industrials` | sector identity | 0.00 | 0.06 | 1.55 | 0.00 |
| `f_acct_ma_recent` | accounting indicators | 0.00 | 0.00 | 0.02 | 0.00 |
| `f_is_q2` | seasonality | 0.00 | 0.00 | 0.00 | 0.01 |
| `mac_retail_sales` | macro | 0.00 | 0.08 | 0.01 | 0.04 |
| `sec_Consumer Discretionary` | sector identity | 0.00 | 0.00 | 0.75 | 0.03 |
| `f_guid_net_dir_365d` | guidance | 0.00 | 0.07 | 0.00 | 0.00 |
| `sec_Communication Services` | sector identity | 0.00 | 0.25 | 0.21 | 0.14 |
| `sec_Consumer Staples` | sector identity | 0.00 | 0.00 | 0.18 | 0.00 |
| `sec_Financials` | sector identity | 0.00 | 0.01 | 0.00 | 0.12 |
| `sec_Health Care` | sector identity | 0.00 | 0.01 | 8.18 | 0.00 |
| `sec_Materials` | sector identity | 0.00 | 0.00 | 2.20 | 0.00 |
| `sec_Real Estate` | sector identity | 0.00 | 0.21 | 0.01 | 1.81 |
| `sec_Technology` | sector identity | 0.00 | 0.03 | 2.18 | 0.00 |

## 8c. Importance three ways: gain vs SHAP vs permutation

Gain is one lens; the two industry-standard alternatives were computed on the
production quarterly model (`feature_importance_methods.csv`):

- **TreeSHAP (mean |SHAP|)** — for every prediction, exactly attribute the
  output to each feature (Shapley values, computed natively by LightGBM via
  `pred_contrib`). The de-facto standard for model explanation in
  finance/credit-risk validation. Measures *prediction-time* influence, where
  gain measures *training-time* loss reduction.
- **Permutation importance** — shuffle one column on a chronological holdout
  (train ≤2021Q4, test 2022+, 10 repeats) and measure the RMSE damage. Fully
  model-agnostic and the most "out-of-sample" of the three, but it punishes
  correlated features: shuffling one macro column while its near-duplicates
  survive leaks the signal through, so redundant blocks score near zero.

**The ranking is robust where it matters.** The top-3 are identical under all
three methods (`f_lag_yoy_1`, `f_qoq_mean_4`, `f_lag_yoy_4` — last-known
growth, recent momentum, seasonal lag), 8/10 of the gain top-10 are in the
SHAP top-10, and rank agreement is Spearman 0.96 (gain↔SHAP) / ~0.66
(either↔permutation). Permutation concentrates even harder on the top three
(~71% of all importance) because only features with *unique* information
survive shuffling.

**The macro block under each lens: gain 6.0%, SHAP 7.5%, permutation 1.2%.**
That spread is the §8 caveat made visible — macro columns are consulted by the
trees (gain/SHAP) but carry little unique out-of-sample information
(permutation), which is precisely what the ablations (§3d) concluded by an
independent route. Method guidance: present **SHAP** as the headline ranking
(industry-standard, prediction-grounded), **permutation** as the "unique
information" check, and **ablation** (§3d–3g) as the decision tool for
including/excluding feature groups.

## 9. Out-of-universe validation: do the results hold on small caps?

Every result above comes from one universe — 100 S&P 500 mega caps. To test
whether the findings are properties of *the problem* or artifacts of *large
caps*, the entire pipeline was replicated on two disjoint out-of-sample
universes from the same 10 GICS sectors:

- **SC600** — 100 S&P SmallCap 600 names (definitionally not in the S&P 500;
  ~$1–7B caps). `universe_smallcap600.csv`.
- **R2000** — 100 Russell 2000 names, excluding S&P 500 *and* 600 members (the
  genuinely small tier). `universe_russell2000.csv`.

Construction (`build_validation_universes.py`): S&P 600/500 constituents from
Wikipedia; Russell 2000 membership from Vanguard's VTWO holdings API (iShares'
IWM download is bot-walled) with sectors from Nasdaq's screener; per-sector
seeded random sample of 10 + backups; a coverage swap pass demotes names with
<20 labeled quarters and promotes same-sector backups (audit trail in the
`role` column). Config gained two env overrides — `SP_PANEL_DATA_DIR` (redirect
all outputs; without it the second universe would overwrite `data/`) and
`SP_PANEL_UNIVERSE_CSV` (load the universe from CSV) — so the exact same code
runs on each universe. Universe-independent inputs (BEA, FRED caches) are
copied in, so no API keys or re-pulls. Run per universe:
`SP_PANEL_DATA_DIR=data_sc600 SP_PANEL_UNIVERSE_CSV=universe_smallcap600.csv
python -m sp_panel.run --financials --prices --macro && python -m sp_panel.assemble`,
then `evaluate` for each target; compare with `python -m sp_panel.summarize
--compare "S&P500=data" "SC600=data_sc600" "R2000=data_r2000"`.

**Caveats** (documented, not fatal): (a) survivorship — current constituents
with long histories are survivors, but the S&P 500 universe has the same bias,
so the *comparison* is like-for-like; (b) small caps tag less XBRL, so margin
coverage is thinner (gross margin ~2.7k/2.0k rows vs 3.2k); (c) R2000 sectors
are Nasdaq-screener approximations of GICS, not the official scheme;
(d) guidance/filing-event/short-interest data were not collected for the new
universes (they are off by default anyway).

### Result 1 — the harder-but-same story (quarterly revenue growth)

| universe | tickers | target std | best model RMSE | R² | dir-acc | vs best baseline |
|---|---:|---:|---:|---:|---:|---:|
| S&P500 | 100 | 0.251 | 0.188 | 0.44 | 0.83 | +17.7% |
| SC600 | 100 | 0.358 | 0.291 | 0.37 | 0.78 | +13.2% |
| R2000 | 99 | 0.404 | 0.326 | 0.32 | 0.79 | +13.0% |

Smaller companies have **intrinsically noisier revenue** (target std rises
0.25→0.36→0.40), so absolute error rises and R² falls monotonically with size —
exactly as theory predicts. But the **model still beats every naive baseline by
13–18% in all three universes**, the ensemble of boosters wins each time, and
directional accuracy stays ~0.78–0.83. The *relative* value of the model is
universe-invariant; only the achievable ceiling moves.

### Result 2 — feature-importance structure is universe-invariant

Quarterly gain importance, grouped (% of total), is nearly identical across the
three universes:

| feature group | S&P500 | SC600 | R2000 |
|---|---:|---:|---:|
| own history (lags/momentum/vol) | 70.8 | 65.1 | 66.0 |
| fundamentals & margins | 8.9 | 7.7 | 10.5 |
| macro | 6.1 | 7.8 | 6.8 |
| sector aggregates (peers) | 4.3 | 7.8 | 5.9 |
| market (price/risk/valuation) | 4.1 | 5.8 | 6.2 |
| accounting indicators | 3.9 | 3.3 | 2.9 |

The rank ordering is preserved and the magnitudes barely move — own-history
dominates, macro sits at 6–8%, everything else is single digits. The
history-dominated-growth / level-dominated-margin split (§8) also replicates:
fundamentals carry 55–83% of margin importance in every universe. **The
structure we documented is a property of the forecasting problem, not of the
S&P 500.**

### Result 3 — the macro verdict holds (and is slightly stronger for small caps)

Quarter-clustered macro A/B ablation (with vs without `mac_*`), per universe:

| universe | ΔRMSE | DM p | quarters macro better | best-quarter share |
|---|---:|---:|---:|---:|
| S&P500 | +0.0004 | 0.43 | 24/49 | 0.91 |
| SC600 | +0.0003 | 0.80 | 20/49 | 1.51 |
| R2000 | +0.0010 | 0.23 | 24/46 | 0.32 |

Macro is **insignificant on average in all three** — the tail-insurance verdict
replicates. Two nuances worth noting for the small caps: R2000 has the largest
(still-insignificant) delta and, tellingly, the *least* regime concentration
(0.32 vs 0.91 — the gain is spread across quarters rather than one COVID
rebound), and at the **annual** horizon macro importance *rises* with
smallness (5.5 → 9.1 → 10.7%). Reading: small, domestically-exposed firms are
somewhat more macro-sensitive than mega caps, consistent with intuition — but
not enough to clear significance at the quarterly horizon.

### One genuine divergence

R2000 **gross-margin** importance shifts the accounting block from ~1% (large
caps) to **16.6%**, and margin R² degrades most for the smallest tier (EBITDA
margin 0.78 → 0.51 → 0.39). Small-cap margins are less sticky and lean more on
balance-sheet accounting signal — a place where a size-specific model *might*
pay, unlike the growth targets. Comparison artifacts:
`data/validation_universe_comparison_{importance_*,metrics,macro}.csv`.

**Bottom line for the deck:** the three headline conclusions — models beat
baselines, own-history dominates with macro as a minor shock-concentrated
contributor, and margins are level-driven — all replicate on companies 10–100×
smaller. The findings are robust to company size; small caps are simply a
noisier version of the same structure.

## 10. Pooled training: does adding all 300 companies help?

The three universes were concatenated into one 300-company panel (299 after CIK
resolution; tickers are disjoint by construction, no overlap leakage) and the
quarterly zoo re-run on it (`data_pool300/`, outputs via `SP_PANEL_DATA_DIR`).
The guidance block (`f_guid_*`, S&P 500 only, 0.2% importance) is NaN for small
caps — trees handle it natively. The honest test is **not** the aggregate
metric but a matched-row comparison: identical (ticker, quarter) rows predicted
by a model trained on 100 (standalone) vs 300 (pooled) companies.

**The aggregate-metric trap.** Pooled-300 aggregate RMSE 0.265 / R² 0.39 looks
*worse* than S&P-500-alone (0.188 / 0.44) — but that is entirely **test-set
composition** (noisier small caps enter the average), not model degradation.
Inside the pooled model the S&P 500 rows still score 0.189 / 0.435. A naive read
of the aggregate would conclude "pooling hurts" — the opposite of the truth.

**Matched-row result** (`data/pool300_vs_standalone.csv`, ensemble; LightGBM
identical pattern):

| tier | standalone RMSE | pooled-300 RMSE | % change | R² standalone→pooled |
|---|---:|---:|---:|---:|
| S&P 500 (data-rich) | 0.188 | 0.189 | +0.4% (flat) | 0.440 → 0.435 |
| S&P 600 (mid) | 0.291 | 0.278 | **−4.4%** | 0.370 → 0.425 |
| Russell 2000 (smallest) | 0.326 | 0.322 | −1.2% | 0.320 → 0.330 |

### Key insights

1. **The flagship model is at its data ceiling, not its sample-size ceiling.**
   Tripling the training data did nothing for the S&P 500 (±0.5%, noise). R²≈0.44
   is a *structural* limit set by how intrinsically unpredictable quarterly
   revenue is — not a "need more data" problem. Future mega-cap gains must come
   from better *signal* (orthogonal features like analyst consensus, §3f/§5),
   not more rows.
2. **Transfer flows toward the data-poor.** All the benefit accrued to the tiers
   that were starved — S&P 600 most (−4–5% RMSE, +0.05 R²), Russell 2000 less,
   mega caps zero. Pooling is a *small-cap coverage strategy*, not a mega-cap
   improvement strategy.
3. **That pooling helps at all is a deep robustness result.** If size tiers had
   different revenue dynamics, mixing would cause negative transfer and hurt
   everywhere. Instead it helped small caps and was neutral for large — proof
   the same machinery (momentum/seasonality/mean-reversion/sector) governs
   revenue growth across the whole size spectrum. Independent confirmation of §9:
   there the per-tier models *looked* the same; here they *share training data
   productively*.
4. **The middle tier wins most, not the smallest.** S&P 600 sits closest to the
   center of the pooled distribution and borrows from both neighbors; Russell
   2000 is the most idiosyncratic and borrows less cleanly. Pooling value tracks
   how *representative* a tier is, not just how data-poor.
5. **Discipline lesson.** The aggregate said "worse"; the matched comparison said
   "neutral-to-better." Same theme as quarter-clustering (§3d) and purging
   (§3g) — the headline number is a trap; the honest answer needs the comparison
   held fixed.

**Caveat:** pooling *narrows* the mega-vs-small gap, it doesn't close it. Small
caps stay harder (RMSE ~0.28–0.32 vs 0.19) because their revenue is genuinely
noisier — a property of the companies, not the model. Artifacts:
`data/pool300_zoo_aggregate.csv`, `data/pool300_vs_standalone.csv`.
