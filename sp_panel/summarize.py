"""Six-metric model summaries and cross-universe comparison (MODELING.md §9).

Per-universe summaries (reads config.DATA_DIR, honoring SP_PANEL_DATA_DIR):

    python -m sp_panel.summarize                  # every target with predictions
    python -m sp_panel.summarize --suffix _annual # one target

Cross-universe comparison (three data dirs -> side-by-side tables):

    python -m sp_panel.summarize --compare "S&P500=data" \
        "SC600=data_sc600" "R2000=data_r2000"

Metrics: median/mean absolute error, RMSE, mean/median percentage error
(computed only where |actual| >= 1% — the growth targets cross zero, where
percentage error is undefined-ish), R^2, directional accuracy, and — for the
margin targets, whose sign-accuracy is vacuous — the direction of margin
CHANGE vs the trailing persistence baseline (chg_dir_acc).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

SUFFIXES = {"": "quarterly rev growth", "_annual": "annual rev growth",
            "_gm_annual": "gross margin (1y)", "_em_annual": "EBITDA margin (1y)"}

GROUPS = [
    ("own history (lags/momentum/vol)", ("f_lag_", "f_accel", "f_roll_", "f_yoy_z",
                                         "f_hist_", "f_obs_count", "f_qoq_")),
    ("fundamentals & margins", ("f_m_", "f_log_revenue", "f_asset_turnover")),
    ("accounting indicators", ("f_acct_",)),
    ("sector aggregates (peers)", ("f_sector_", "f_yoy_vs_sector")),
    ("market (price/risk/valuation)", ("mkt_",)),
    ("macro", ("mac_",)),
    ("BEA sector value-added", ("bea_",)),
    ("guidance", ("f_guid_",)),
    ("seasonality", ("f_is_q",)),
    ("sector identity (one-hot)", ("sec_",)),
]


def _group_of(feature):
    for name, prefixes in GROUPS:
        if feature.startswith(prefixes):
            return name
    return "other"


def six_metrics(g, persist=None):
    y, yhat = g["y_true"].values, g["y_pred"].values
    e = y - yhat
    sst = ((y - y.mean()) ** 2).sum()
    mask = np.abs(y) >= 0.01
    pe = 100 * e[mask] / y[mask] if mask.any() else np.array([np.nan])
    out = {"n": len(g), "median_ae": np.median(np.abs(e)), "mean_ae": np.abs(e).mean(),
           "rmse": np.sqrt((e ** 2).mean()), "mpe_pct": pe.mean(),
           "mdpe_pct": np.median(pe), "r2": 1 - (e ** 2).sum() / sst if sst > 0 else np.nan,
           "dir_acc": (np.sign(y) == np.sign(yhat)).mean()}
    if persist is not None and persist.notna().any():
        ok = persist.notna().values
        dy, dyh = y[ok] - persist.values[ok], yhat[ok] - persist.values[ok]
        out["chg_dir_acc"] = float((np.sign(dy) == np.sign(dyh)).mean())
    return pd.Series(out)


def summarize(suffix="", data_dir=None, save=True):
    """Six-metric table per model from model_predictions{suffix}.parquet."""
    data_dir = Path(data_dir or config.DATA_DIR)
    path = data_dir / f"model_predictions{suffix}.parquet"
    if not path.exists():
        return None
    preds = pd.read_parquet(path)
    is_margin = "gm" in suffix or "em" in suffix
    if is_margin:
        bl = (preds[preds["model"] == "baseline:persistence"]
              [["ticker", "target_quarter", "y_pred"]]
              .rename(columns={"y_pred": "bl_persist"}))
        preds = preds.merge(bl, on=["ticker", "target_quarter"], how="left")
    tab = (preds.groupby("model")
           .apply(lambda g: six_metrics(g, g["bl_persist"] if is_margin else None),
                  include_groups=False)
           .sort_values("rmse"))
    tab.index = [i.replace("baseline:", "bl: ") for i in tab.index]
    if save:
        tab.to_csv(data_dir / f"model_metrics_full_summary{suffix}.csv")
    return tab


def grouped_importance(suffix="", data_dir=None):
    data_dir = Path(data_dir or config.DATA_DIR)
    path = data_dir / f"model_feature_importance{suffix}.csv"
    if not path.exists():
        return None
    fi = pd.read_csv(path)
    fi["group"] = fi["feature"].map(_group_of)
    return fi.groupby("group")["gain_pct"].sum()


def compare(dirs):
    """Side-by-side tables across universes. `dirs` = {label: path}."""
    out_dir = Path(config.DATA_DIR)
    pd.set_option("display.width", 220)

    # 1. grouped importance per target
    for sfx, tname in SUFFIXES.items():
        cols = {}
        for label, d in dirs.items():
            g = grouped_importance(sfx, d)
            if g is not None:
                cols[label] = g
        if not cols:
            continue
        tab = pd.DataFrame(cols).fillna(0)
        tab = tab.sort_values(list(cols)[0], ascending=False)
        fname = f"validation_universe_comparison_importance{sfx or '_quarterly'}.csv"
        tab.to_csv(out_dir / fname)
        print(f"\n=== grouped importance, {tname} (% of total gain) ===")
        print(tab.round(1).to_string())

    # 2. headline metrics: best model vs best baseline, quarterly + annual
    rows = []
    for label, d in dirs.items():
        for sfx, tname in SUFFIXES.items():
            tab = summarize(sfx, d, save=False)
            if tab is None:
                continue
            is_bl = tab.index.str.startswith("bl: ")
            best_m, best_b = tab[~is_bl].iloc[0], tab[is_bl].sort_values("rmse").iloc[0]
            rows.append({"universe": label, "target": tname,
                         "best_model": tab[~is_bl].index[0],
                         "model_rmse": best_m["rmse"], "model_r2": best_m["r2"],
                         "model_dir_acc": best_m["dir_acc"],
                         "best_baseline": tab[is_bl].sort_values("rmse").index[0],
                         "baseline_rmse": best_b["rmse"],
                         "rmse_vs_baseline_pct": 100 * (best_b["rmse"] - best_m["rmse"]) / best_b["rmse"]})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "validation_universe_comparison_metrics.csv", index=False)
    print("\n=== best model vs best baseline per universe/target ===")
    print(metrics.round(4).to_string(index=False))

    # 3. macro ablation (quarterly, squared-error row of the DM tests)
    rows = []
    for label, d in dirs.items():
        p = Path(d) / "model_macro_ablation_tests.csv"
        if p.exists():
            t = pd.read_csv(p)
            se = t[t["loss"] == "squared_error"].iloc[0].to_dict()
            rows.append({"universe": label, **{k: se[k] for k in
                        ("mean_quarterly_delta", "dm_stat", "p_value", "n_quarters",
                         "quarters_macro_better", "share_from_best_quarter")}})
    if rows:
        macro = pd.DataFrame(rows)
        macro.to_csv(out_dir / "validation_universe_comparison_macro.csv", index=False)
        print("\n=== quarterly macro ablation (with vs without mac_*) ===")
        print(macro.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default=None, help="one target suffix (default: all found)")
    ap.add_argument("--compare", nargs="+", default=None,
                    metavar="LABEL=DIR", help="cross-universe mode")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    if args.compare:
        dirs = dict(kv.split("=", 1) for kv in args.compare)
        compare(dirs)
        return
    for sfx, tname in SUFFIXES.items():
        if args.suffix is not None and sfx != args.suffix:
            continue
        tab = summarize(sfx)
        if tab is not None:
            print(f"\n===== {tname} ({config.DATA_DIR}) =====")
            print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
