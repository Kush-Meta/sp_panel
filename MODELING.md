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
- Macro features begin in 2014 (FRED pull start); BEA is extra-lagged for
  point-in-time safety.
