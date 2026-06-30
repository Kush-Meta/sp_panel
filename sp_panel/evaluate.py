"""Time-aware (walk-forward) evaluation of revenue-growth models vs baselines.

Predicts ``revenue_yoy_target`` from the clean point-in-time panel assembled by
``sp_panel.assemble``.

Validation protocol
-------------------
Expanding-window walk-forward: for each test quarter q (after a warm-up), train
on *all* rows strictly before q and predict q. Train and test never share a
quarter, so this is a train-on-past / test-on-future split with no shuffling and
no leakage — the correct protocol for a time-series panel. (A random KFold split,
which would leak the future into the past, is never used.)

Everything is scored against simple, honest baselines:
  * persistence         - last available YoY growth
  * seasonal            - same quarter one year earlier
  * trailing mean (4/8) - rolling mean of past growth
  * company history     - expanding mean of the company's own past growth
  * sector median       - median of peers' last-known growth, point-in-time
  * expanding global mean - mean of all training-set targets

Models: ElasticNet, Ridge, RandomForest, ExtraTrees, HistGradientBoosting,
LightGBM, XGBoost, CatBoost, and a blended ensemble of the gradient boosters.

Metrics: MAE, RMSE, R² (vs target mean), directional accuracy — reported overall
and broken down by sector, year, and company.

Outputs (data/):
  model_metrics_overall.csv     models + baselines (+ historical old-suite best)
  model_metrics_by_sector.csv
  model_metrics_by_year.csv
  model_metrics_by_company.csv
  model_predictions.parquet     per (model, ticker, quarter) walk-forward preds
  model_feature_importance.csv  LightGBM gain importance (full-sample fit)
"""
import argparse
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import (ExtraTreesRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)

from . import config
from . import assemble

warnings.filterwarnings("ignore")

TARGET = "revenue_yoy_target"

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False

ENSEMBLE_MEMBERS = ["lightgbm", "xgboost", "catboost", "hist_gbm"]


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------
def make_models(which=None):
    m = {}
    m["elasticnet"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], n_alphas=40, cv=3,
                             max_iter=20000, random_state=0)),
    ])
    m["ridge"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("reg", RidgeCV(alphas=np.logspace(-2, 4, 13))),
    ])
    m["random_forest"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("reg", RandomForestRegressor(n_estimators=300, max_depth=9,
                                      min_samples_leaf=10, max_features="sqrt",
                                      n_jobs=-1, random_state=0)),
    ])
    m["extra_trees"] = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("reg", ExtraTreesRegressor(n_estimators=400, max_depth=12,
                                    min_samples_leaf=10, max_features="sqrt",
                                    n_jobs=-1, random_state=0)),
    ])
    m["hist_gbm"] = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.03, max_leaf_nodes=15, max_depth=4,
        min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=0)
    if HAS_LGBM:
        m["lightgbm"] = LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=15, max_depth=4,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=0,
            n_jobs=-1, verbose=-1)
    if HAS_XGB:
        m["xgboost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=0, n_jobs=-1, verbosity=0)
    if HAS_CAT:
        m["catboost"] = CatBoostRegressor(
            iterations=500, learning_rate=0.03, depth=5, l2_leaf_reg=3.0,
            loss_function="RMSE", random_seed=0, verbose=False)
    if which:
        m = {k: v for k, v in m.items() if k in which}
    return m


def design_matrix(panel, feat_cols):
    """Engineered features + sector one-hot, as a numeric float matrix."""
    sec = pd.get_dummies(panel["sector"], prefix="sec").astype(float)
    X = pd.concat([panel[feat_cols].astype(float).reset_index(drop=True),
                   sec.reset_index(drop=True)], axis=1)
    return X, list(X.columns)


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------
BASELINE_COLS = {
    "persistence": "bl_persistence",
    "seasonal": "bl_seasonal",
    "trailing_mean4": "bl_trailing_mean4",
    "trailing_mean8": "bl_trailing_mean8",
    "company_hist": "bl_company_hist",
    "sector_median": "bl_sector_median",
}


def walk_forward(panel, feat_cols, models, warmup_quarters=12, min_train=400,
                 ensemble=True):
    d = panel[panel[TARGET].notna()].copy().reset_index(drop=True)
    X_all, _ = design_matrix(d, feat_cols)
    quarters = sorted(d["target_q"].unique())
    test_quarters = quarters[warmup_quarters:]

    rows = []
    for qi, q in enumerate(test_quarters, 1):
        tr = (d["target_q"] < q).values
        te = (d["target_q"] == q).values
        if tr.sum() < min_train or te.sum() == 0:
            continue
        Xtr = X_all[tr]
        ytr = d.loc[tr, TARGET].values
        Xte = X_all[te]
        base = d.loc[te, ["ticker", "sector", "target_quarter", TARGET]].reset_index(drop=True)
        base = base.rename(columns={TARGET: "y_true"})

        # baselines (precomputed causal columns; fall back to expanding mean)
        bl = {name: d.loc[te, col].values for name, col in BASELINE_COLS.items()}
        bl["expanding_mean"] = np.full(te.sum(), ytr.mean())
        for name, pred in bl.items():
            r = base.copy()
            r["model"] = f"baseline:{name}"
            r["y_pred"] = pd.Series(pred).fillna(ytr.mean()).values
            rows.append(r)

        # models
        member = {}
        for name, mdl in models.items():
            mdl.fit(Xtr, ytr)
            yhat = mdl.predict(Xte)
            member[name] = yhat
            r = base.copy()
            r["model"] = name
            r["y_pred"] = yhat
            rows.append(r)

        if ensemble:
            mem = [member[k] for k in ENSEMBLE_MEMBERS if k in member]
            if len(mem) >= 2:
                r = base.copy()
                r["model"] = "ensemble"
                r["y_pred"] = np.mean(mem, axis=0)
                rows.append(r)

        if qi % 10 == 0:
            print(f"[walk_forward] {qi}/{len(test_quarters)} quarters "
                  f"(test {q}, train n={tr.sum()})")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(g):
    y, yhat = g["y_true"].values, g["y_pred"].values
    e = y - yhat
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(g)),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "r2": float(1 - (e ** 2).sum() / sst) if sst > 0 else np.nan,
        "dir_acc": float((np.sign(y) == np.sign(yhat)).mean()),
    }


def score(preds):
    rows = [{"model": m, "scope": "ALL", **_metrics(g)} for m, g in preds.groupby("model")]
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def score_by(preds, col):
    rows = []
    for (m, key), g in preds.groupby(["model", col]):
        if len(g) >= 5:
            rows.append({"model": m, col: key, **_metrics(g)})
    return pd.DataFrame(rows)


def _pinball(y, q_pred, alpha):
    e = y - q_pred
    return float(np.mean(np.maximum(alpha * e, (alpha - 1) * e)))


def quantile_walk_forward(panel, feat_cols, quantiles=(0.1, 0.5, 0.9),
                          warmup_quarters=12, min_train=400):
    """Walk-forward LightGBM quantile regression -> calibrated p10/p50/p90 intervals.

    Trees fit a separate quantile objective per level. Gives an honest predictive
    interval (and a quantile-loss point forecast at p50) instead of a bare number.
    """
    if not HAS_LGBM:
        print("[quantile] lightgbm unavailable; skipping quantile forecasts")
        return pd.DataFrame()
    d = panel[panel[TARGET].notna()].copy().reset_index(drop=True)
    X_all, _ = design_matrix(d, feat_cols)
    quarters = sorted(d["target_q"].unique())
    rows = []
    for q in quarters[warmup_quarters:]:
        tr = (d["target_q"] < q).values
        te = (d["target_q"] == q).values
        if tr.sum() < min_train or te.sum() == 0:
            continue
        Xtr, ytr, Xte = X_all[tr], d.loc[tr, TARGET].values, X_all[te]
        base = d.loc[te, ["ticker", "sector", "target_quarter", TARGET]].reset_index(drop=True)
        base = base.rename(columns={TARGET: "y_true"})
        for a in quantiles:
            mdl = LGBMRegressor(objective="quantile", alpha=a, n_estimators=400,
                                learning_rate=0.03, num_leaves=15, max_depth=4,
                                min_child_samples=20, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, reg_lambda=1.0, random_state=0,
                                n_jobs=-1, verbose=-1).fit(Xtr, ytr)
            base[f"q{int(a*100)}"] = mdl.predict(Xte)
        rows.append(base)
    qp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # enforce monotone quantiles (guard against crossing)
    if not qp.empty:
        qcols = sorted([c for c in qp.columns if c.startswith("q")], key=lambda c: int(c[1:]))
        qp[qcols] = np.sort(qp[qcols].values, axis=1)
    return qp


def quantile_report(qp):
    if qp.empty:
        return pd.DataFrame()
    y = qp["y_true"].values
    lo, mid, hi = qp["q10"].values, qp["q50"].values, qp["q90"].values
    return pd.DataFrame([{
        "interval": "p10-p90 (nominal 80%)",
        "empirical_coverage": float(((y >= lo) & (y <= hi)).mean()),
        "mean_width": float(np.mean(hi - lo)),
        "median_width": float(np.median(hi - lo)),
        "pinball_p10": _pinball(y, lo, 0.10),
        "pinball_p50": _pinball(y, mid, 0.50),
        "pinball_p90": _pinball(y, hi, 0.90),
        "p50_mae": float(np.abs(y - mid).mean()),
        "p50_rmse": float(np.sqrt(((y - mid) ** 2).mean())),
        "p50_dir_acc": float((np.sign(y) == np.sign(mid)).mean()),
        "n": int(len(qp)),
    }])


def old_suite_best():
    """Historical best of the original corrupted-target suite, for context."""
    p = config.DATA_DIR / "model_suite_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    m = pd.read_csv(p)
    if "scope" not in m.columns or "directional_accuracy" not in m.columns:
        return pd.DataFrame()
    m = m[m["scope"] == "ALL"].sort_values("rmse")
    if m.empty:
        return pd.DataFrame()
    b = m.iloc[0]
    return pd.DataFrame([{
        "model": f"OLD suite best ({b.get('model_family','?')}/{b.get('feature_set','?')})",
        "scope": "ALL", "n": int(b["n"]), "mae": b["mae"], "rmse": b["rmse"],
        "r2": b["r2"], "dir_acc": b["directional_accuracy"],
        "note": "corrupted+winsorized target; RMSE not directly comparable",
    }])


def feature_importance(panel, feat_cols):
    if not HAS_LGBM:
        return pd.DataFrame()
    d = panel[panel[TARGET].notna()]
    X, cols = design_matrix(d, feat_cols)
    mdl = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15,
                        max_depth=4, min_child_samples=20, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                        random_state=0, n_jobs=-1, verbose=-1).fit(X, d[TARGET].values)
    imp = pd.DataFrame({"feature": cols, "gain": mdl.booster_.feature_importance("gain")})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    return imp.sort_values("gain", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
def guidance_ablation(panel, feat_cols, warmup=12, min_train=400):
    """Controlled A/B: one model (LightGBM) with vs without the f_guid_* features.

    Isolates the marginal value of the management-guidance block. Guidance coverage
    is thin (~13% of rows), so a small or zero delta is the expected, honest result.
    """
    if not HAS_LGBM:
        return pd.DataFrame()
    guid = [c for c in feat_cols if c.startswith("f_guid_")]
    if not guid:
        return pd.DataFrame()
    no_guid = [c for c in feat_cols if not c.startswith("f_guid_")]
    rows = []
    for label, cols in [("with_guidance", feat_cols), ("without_guidance", no_guid)]:
        preds = walk_forward(panel, cols, {"lightgbm": make_models(["lightgbm"])["lightgbm"]},
                             warmup_quarters=warmup, min_train=min_train, ensemble=False)
        m = preds[preds["model"] == "lightgbm"]
        rows.append({"config": label, "n_guidance_feats": len(cols) - len(no_guid), **_metrics(m)})
    out = pd.DataFrame(rows)
    return out


def evaluate(panel_path=None, feature_lag=1, warmup=12, min_train=400,
             start="2011Q1", which=None, quantiles=True, guidance_ab=True):
    if panel_path:
        panel = pd.read_parquet(panel_path)
        feat_cols = assemble.feature_columns(panel)
    else:
        panel = assemble.build_panel(start=start, feature_lag=feature_lag)
        feat_cols = assemble.feature_columns(panel)
    models = make_models(which)
    print(f"[evaluate] panel rows={len(panel)} labeled={panel[TARGET].notna().sum()} "
          f"features={len(feat_cols)} (+sector one-hot)")
    print(f"[evaluate] models: {list(models.keys())} | lgbm={HAS_LGBM} xgb={HAS_XGB} cat={HAS_CAT}")

    preds = walk_forward(panel, feat_cols, models, warmup_quarters=warmup, min_train=min_train)
    preds["year"] = preds["target_quarter"].str[:4]
    preds.to_parquet(config.DATA_DIR / "model_predictions.parquet", index=False)

    overall = score(preds)
    table = pd.concat([overall, old_suite_best()], ignore_index=True)
    table = table.sort_values("rmse", na_position="last").reset_index(drop=True)
    table.to_csv(config.DATA_DIR / "model_metrics_overall.csv", index=False)
    score_by(preds, "sector").to_csv(config.DATA_DIR / "model_metrics_by_sector.csv", index=False)
    score_by(preds, "year").to_csv(config.DATA_DIR / "model_metrics_by_year.csv", index=False)
    score_by(preds, "ticker").to_csv(config.DATA_DIR / "model_metrics_by_company.csv", index=False)
    fi = feature_importance(panel, feat_cols)
    if not fi.empty:
        fi.to_csv(config.DATA_DIR / "model_feature_importance.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Walk-forward comparison: models vs baselines (overall) ===")
    print(table[["model", "n", "mae", "rmse", "r2", "dir_acc"]].round(4).to_string(index=False))

    base = overall[overall["model"].str.startswith("baseline:")]
    best = overall[~overall["model"].str.startswith("baseline:")].iloc[0]
    print(f"\nbest baseline: RMSE={base['rmse'].min():.4f} MAE={base['mae'].min():.4f}")
    print(f"best model: {best['model']} RMSE={best['rmse']:.4f} "
          f"({100*(base['rmse'].min()-best['rmse'])/base['rmse'].min():+.1f}% vs best baseline) "
          f"MAE={best['mae']:.4f} R2={best['r2']:.4f} dir_acc={best['dir_acc']:.3f}")

    # --- guidance A/B: marginal value of the management-guidance block ---
    if guidance_ab:
        ab = guidance_ablation(panel, feat_cols, warmup=warmup, min_train=min_train)
        if not ab.empty:
            ab.to_csv(config.DATA_DIR / "model_guidance_ablation.csv", index=False)
            print("\n=== Guidance A/B (LightGBM, with vs without f_guid_* features) ===")
            print(ab[["config", "n_guidance_feats", "n", "mae", "rmse", "r2", "dir_acc"]].round(4).to_string(index=False))
            d_rmse = ab.loc[ab.config == "with_guidance", "rmse"].iloc[0] - \
                     ab.loc[ab.config == "without_guidance", "rmse"].iloc[0]
            print(f"guidance RMSE delta: {d_rmse:+.4f} "
                  f"({'helps' if d_rmse < 0 else 'no improvement'}; coverage ~13% of rows)")

    # --- quantile intervals (calibrated uncertainty) ---
    if quantiles:
        qp = quantile_walk_forward(panel, feat_cols, warmup_quarters=warmup, min_train=min_train)
        if not qp.empty:
            qp.to_parquet(config.DATA_DIR / "model_quantile_predictions.parquet", index=False)
            qr = quantile_report(qp)
            qr.to_csv(config.DATA_DIR / "model_quantile_calibration.csv", index=False)
            print("\n=== Quantile forecast calibration (LightGBM p10/p50/p90) ===")
            print(qr.round(4).to_string(index=False))
            print("(empirical_coverage should be ~0.80 if the 80% interval is well calibrated)")
    return table, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=None, help="panel parquet path (default: build fresh)")
    ap.add_argument("--feature-lag", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--min-train", type=int, default=400)
    ap.add_argument("--start", default="2011Q1")
    ap.add_argument("--models", default=None,
                    help="comma list to restrict the model zoo (default: all available)")
    ap.add_argument("--no-quantiles", action="store_true", help="skip quantile interval forecasts")
    ap.add_argument("--no-guidance-ab", action="store_true", help="skip the guidance A/B")
    args = ap.parse_args()
    which = [s.strip() for s in args.models.split(",")] if args.models else None
    evaluate(panel_path=args.panel, feature_lag=args.feature_lag, warmup=args.warmup,
             min_train=args.min_train, start=args.start, which=which,
             quantiles=not args.no_quantiles, guidance_ab=not args.no_guidance_ab)


if __name__ == "__main__":
    main()
