# Macro features: impact, lags, and whether they belong in the model

*Prepared 2026-07-15 · underlying analyses in MODELING.md §3d–3i, §8–§8c ·
every number regenerable from the artifact files listed in the appendix.*

---

## 1. The question and the answer

**Question: does it make statistical sense to include macroeconomic features
in the revenue model?**

**Answer: it depends on the forecast horizon — and the evidence supports a
precise configuration, not a blanket yes/no.**

- **Next-quarter model: NO on statistical grounds, KEEP on risk grounds.** The
  macro block does not significantly improve quarterly forecasts (p = 0.43;
  replicated as insignificant on two independent small-cap universes, p = 0.80
  and 0.23). ~91% of its small net benefit came from a single quarter (the
  2021Q2 COVID rebound). It is tail insurance: nearly free in normal times,
  helpful exactly at turning points.
- **Annual model: YES — two independent results clear 5% significance.**
  (1) The demand-activity theme (industrial production, retail sales,
  unemployment, sentiment) added alone improves RMSE with p = 0.044.
  (2) Deepening macro lags to T−2..T−4 improves RMSE with p = 0.047, better in
  30/47 quarters. Meanwhile inflation features significantly *hurt* (p = 0.034)
  and rate levels trend-drag. The statistically defensible annual macro block
  is: **demand-activity + oil, at lags 2–4, excluding inflation and rate
  levels.**
- **Margin models: no tested evidence.** Macro's descriptive share is small
  (2.6% of the gross-margin model, 8.7% of EBITDA-margin). The EBITDA-margin
  model's interest in the rates complex (fed funds / 3m / 10y at lags 2–4) is
  an economically sensible hypothesis (financing costs pass through with
  delay) but has not been ablation-tested — treat as untested, not proven.

---

## 2. Highest-impact macro features, and at what lag

Impact = mean |SHAP| share of the *entire* model (TreeSHAP on the lag-ladder
panel; descriptive — see §6.2 for why inference comes from ablations instead).

**Next-quarter model (macro block = 10.6% of model):**

| rank | variable | lag | % of model |
|---:|---|---|---:|
| 1 | WTI oil — YoY change | T−1 | 0.90 |
| 2 | WTI oil — QoQ change | T−1 | 0.82 |
| 3 | Retail sales — QoQ change | T−1 | 0.77 |
| 4 | Retail sales — YoY change | T−1 | 0.67 |
| 5 | 3-month Treasury — level | T−2 | 0.60 |
| 6 | Industrial production — level | T−3 | 0.41 |

**Annual model (macro block = 13.0% of model):**

| rank | variable | lag | % of model |
|---:|---|---|---:|
| 1 | **Industrial production — level** | **T−2** | **2.97** |
| 2 | WTI oil — level | T−2 | 1.35 |
| 3 | Industrial production — level | T−3 | 1.29 |
| 4 | WTI oil — level | T−4 | 0.97 |
| 5 | 10-year Treasury — level | T−2 | 0.95 |
| 6 | Industrial production — level | T−1 | 0.45 |

The pattern: the **quarterly model wants fresh momentum** (T−1 *changes* in
oil and retail sales = 40% of its macro attention); the **annual model wants
the macro state from 2–4 quarters back** (~80% of its macro attention at
lag ≥ 2, led by industrial production at T−2 — the largest single macro
feature in any model). This is a transmission-delay picture: macro conditions
take roughly two quarters to propagate into corporate revenue, visible only
at the annual horizon.

---

## 3. Feature × lag × model table

Top macro features per model (SHAP % of that model; full 200-row table:
`data/macro_feature_lag_target_shap.csv`).

| variable | lag | strongest in | % of that model | ablation-backed? |
|---|---|---|---:|---|
| Industrial production (level) | T−2 | annual | 2.97 | ✅ activity theme p=.044 + ladder p=.047 |
| WTI oil (level) | T−2 | annual | 1.35 | ✅ ladder p=.047 |
| Industrial production (level) | T−3 | annual | 1.29 | ✅ both |
| WTI oil (level) | T−4 | annual | 0.97 | ✅ ladder |
| 10y Treasury (level) | T−2 | annual | 0.95 | ⚠️ rates theme insignificant |
| WTI oil (YoY chg) | T−1 | quarterly | 0.90 | ❌ block p=.43 (tail insurance) |
| WTI oil (QoQ chg) | T−1 | quarterly | 0.82 | ❌ block p=.43 |
| Retail sales (QoQ chg) | T−1 | quarterly | 0.77 | ❌ block p=.43 |
| 3m Treasury (level) | T−2 | quarterly / EBITDA margin | 0.60 / 0.63 | ❌ / untested |
| Fed funds (level) | T−4 | EBITDA margin | 0.52 | untested |
| CPI (YoY chg) | T−1 | EBITDA margin | 0.37 | ❌ inflation *hurts* annual rev (p=.034) |
| Consumer sentiment (level) | T−4 | quarterly | 0.28 | ❌ block p=.43 |

Macro block share per model: quarterly 10.6% · annual 13.0% · gross margin
2.6% · EBITDA margin 8.7%.

---

## 4. The statistical evidence (every inclusion test run)

Positive Δ = macro helps. All tests: paired walk-forward forecasts, per-quarter
loss differentials, quarter-clustered Diebold–Mariano (see §6.3).

| test | horizon | ΔRMSE | p | verdict |
|---|---|---:|---:|---|
| Full macro block (S&P 500) | quarterly | +0.0011 | 0.43 | not significant; 91% of gain from 2021Q2 |
| Full macro block (S&P 600 replication) | quarterly | +0.0003 | 0.80 | not significant |
| Full macro block (Russell 2000 replication) | quarterly | +0.0010 | 0.23 | not significant |
| **Activity/demand theme alone** | **annual** | **+0.0024** | **0.044** | **significant — include** |
| **Deeper lags (T−2..T−4)** | **annual** | **+0.0018** | **0.047** | **significant — include** |
| Inflation theme alone | annual | −0.0019 | 0.034 | significantly *hurts* — exclude |
| Rates/curve theme alone | annual | −0.0013 | 0.25 | drag — exclude |
| Commodities/FX theme alone | annual | −0.0002 | 0.91 | nothing |
| Risk/credit theme alone | annual | −0.0002 | 0.86 | nothing |
| Full macro block, Energy sector | annual | +0.0061 | 0.043 | significant — strongest sector |
| Macro-only model (ceiling) | quarterly | RMSE 0.236 | — | worse than naive persistence (0.229) |

---

## 5. Actionable recommendations (each defensible from the table above)

1. **Annual model: adopt a slim macro block** — industrial production, retail
   sales, unemployment, sentiment, plus oil levels, each at lags 1–4.
   Evidence: p = 0.044 (theme) and p = 0.047 (lags), independent tests,
   consistent direction. Expected effect ≈ 0.2–0.4 RMSE points and +1.7pp
   directional accuracy.
2. **Annual model: drop CPI/PCE and rate-level columns.** Inflation features
   significantly damage forecasts (p = 0.034); trending rate levels drag
   because walk-forward trees cannot extrapolate beyond the training range.
3. **Quarterly model: keep the current single-lag block unchanged, with
   calibrated expectations.** No average gain is defensible from the data;
   the case for keeping it is asymmetric risk (it paid at the 2021–22 turning
   points, which is when forecasts face the most scrutiny). Do not expand it —
   deeper lags tested insignificant at this horizon (p = 0.22).
4. **If forecasting Energy names specifically, weight macro higher** — the one
   sector where the block is significant on its own (p = 0.043, annual).
5. **Do not sell macro as a headline driver.** Under the strictest importance
   method (permutation), macro's unique contribution to the quarterly model is
   1.2%. The model's engine is each company's own history (~50% of importance,
   top-3 features identical under all three methods).
6. **Untested hypothesis worth one future ablation:** the EBITDA-margin
   model's attention to policy rates at lags 2–4 (financing-cost
   pass-through). Cheap to test with the existing harness.

---

## 6. Methodology deep dive

### 6.1 Data and point-in-time construction
Quarterly fundamentals for 100 S&P 500 companies (SEC EDGAR XBRL, Q4 derived
as FY−(Q1+Q2+Q3) by period-date containment), 19 FRED macro series from 1990,
prices, BEA sector value-added. Every feature for target quarter T uses only
data through T−1; macro joins at an explicit publication lag (monthly series
lag 1 quarter; BEA lag 2). Targets: next-quarter YoY revenue growth; annual =
TTM(T..T+3)/TTM(T−4..T−1)−1; next-year TTM gross and EBITDA margins. One
honest caveat: FRED serves *revised* macro values, not real-time vintages,
which flatters macro slightly — so the insignificant verdicts are
conservative.

### 6.2 Measuring impact: three methods, and why impact ≠ inclusion
- **Gain** (LightGBM): training-loss reduction credited to each splitting
  feature. Fast, standard, but train-time and splits across correlated
  columns.
- **TreeSHAP** (headline ranking here): every prediction exactly decomposed
  into per-feature contributions (Shapley values); features ranked by mean
  absolute contribution. Industry standard in finance/credit-risk validation.
- **Permutation**: shuffle one column on a chronological 2022+ holdout,
  measure RMSE damage, 10 repeats. Most out-of-sample, but punishes
  correlated features (a shuffled column's near-duplicates leak the signal
  back in).

Agreement: top-3 features identical under all three; Spearman 0.96
(gain↔SHAP), ~0.66 (either↔permutation). The macro block scores 6.0% (gain),
7.5% (SHAP), **1.2% (permutation)** — the model *consults* macro but macro
carries little *unique* information. That is why no importance number, from
any method, answers the inclusion question: importance describes what a
fitted model uses; **inclusion is decided by ablation** — remove the block,
re-run honestly, test the difference.

### 6.3 The ablation protocol (how every p-value above was produced)
1. **Walk-forward evaluation**: for each test quarter q, train on all data
   strictly before q, predict q; ~49 test quarters, 2014–2026. No shuffling —
   random splits leak the future in panel data.
2. **Purging (annual targets)**: an annual label at origin T spans T..T+3, so
   training for test origin q excludes origins with unrealized label windows
   (train on target_q < q−3). Without purging, training labels contain the
   test period's revenue.
3. **Paired comparison**: model A (without the block) and B (with it) are the
   *same* LightGBM on the *same* splits, differing only in feature columns;
   forecasts are paired per (company, quarter), so the accuracy difference is
   attributable to the features alone.
4. **Quarter-clustered inference — the critical step**: the ~4,500 paired
   forecasts are not independent (all companies in a quarter share the macro
   shock — COVID is one lesson, not one hundred). Loss differentials are
   averaged within each quarter, giving ~49 independent observations, then
   tested with a Diebold–Mariano statistic using a Newey–West long-run
   variance (serial correlation across quarters; lags = 4 for overlapping
   annual windows) and the Harvey–Leybourne–Newbold small-sample correction,
   with t-distribution p-values. A row-level t-test would overstate the
   sample by ~100× and manufacture false significance.
5. **Regime diagnostics**: share of the net gain contributed by the single
   best quarter (0.91 for the quarterly macro block → tail-insurance
   verdict), and win-rate across quarters.

### 6.4 The lag ladder
`add_macro_features(extra_lags=(2,3,4))` adds each macro *level* at T−2, T−3,
T−4 (45 columns; changes are not laddered — a lagged change is nearly a
difference of laddered levels). The ladder's inclusion was itself decided by
the §6.3 protocol (annual: p = 0.047 → adopt; quarterly: p = 0.22 → reject),
and lag attribution within the ladder uses SHAP shares.

### 6.5 Robustness
The quarterly macro verdict was replicated on two fully disjoint universes
(100 S&P SmallCap 600; 100 Russell 2000 ex-S&P) with the same protocol —
insignificant in all three, and the grouped importance structure is
near-identical across universes (own-history 65–71%, macro 6–8%). Pooling all
300 companies into one training set left mega-cap accuracy unchanged and
improved small-cap accuracy — evidence the same structure governs the whole
size spectrum.

### 6.6 Power and limitations
With ~49 quarters, block-level tests are the smallest unit with reasonable
power; single-feature ablations would be underpowered, which is why per-
feature statements in §2–3 are SHAP-descriptive while inclusion decisions are
theme/ladder-level. Two macro shock episodes (2020, 2022) exist in the
evaluation window; macro's tail-insurance value is estimated from n≈2 events.
Multiple themes were tested; the two significant annual results are mutually
consistent and economically directional, which mitigates (does not eliminate)
multiple-comparison risk.

---

## Appendix: artifacts

| file | contents |
|---|---|
| `data/macro_feature_lag_target_shap.csv` | every macro feature × lag × model, SHAP share |
| `data/macro_lag_ablation.csv` / `_importance.csv` | lag-ladder tests and lag-bucket usage |
| `data/macro_stage3_themes.csv`, `macro_annual_themes.csv` | theme ablations, both horizons |
| `data/model_macro_ablation*.csv` | quarterly block ablation + by-year/quarter |
| `data/validation_universe_comparison_macro.csv` | 3-universe replication |
| `data/feature_importance_methods.csv` | gain vs SHAP vs permutation, all features |
| `data/panel_maclag.parquet` | lag-ladder panel for reproduction |
