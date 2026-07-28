"""Tail-guard for zero-shot TimesFM forecasts (MODELING.md §11).

TimesFM beats the feature ensemble at every absolute-error percentile up to
the 99th, yet loses on RMSE because ~0.24% of forecasts blow up — it predicts
+300% growth for companies whose reported revenue series contain scale breaks
(REITs, restructured industrials). A univariate model cannot see that the
series is broken; the fix is a plausibility check against the company's own
history.

The rule (deliberately crude, causal, and parameter-light — the kind of filter
any practitioner would write BEFORE seeing which rows failed):

    ceiling_i = max(k * max|historical YoY growth of company i before q|,
                    floor)
    if |implied growth| > ceiling_i:  fall back to persistence (last known YoY)

`floor` protects companies with near-flat history from having a ceiling of ~0.
Both the history and the fallback use only data available before quarter q, so
the guard inherits the pipeline's point-in-time discipline.

Honesty note: the guard was designed after observing S&P 500 failures, so its
S&P 500 numbers are in-window. `--universe` runs it unchanged on the S&P 600
and Russell 2000 panels, which it was never designed against — that is the
real test, alongside the 2024+ window.

Run: python -m sp_panel.timesfm_guard [--universe data|data_sc600|data_r2000]
"""
import argparse

import numpy as np
import pandas as pd

from . import assemble, config

K_GRID = (2.0, 3.0, 5.0)
FLOOR_GRID = (0.5, 1.0)


def _history_ceiling(data_dir):
    """max |YoY| observed strictly before each quarter, per (ticker, quarter)."""
    fin = pd.read_parquet(data_dir / "financials_quarterly.parquet")
    clean = assemble.clean_quarterly(fin)
    clean["cq"] = pd.PeriodIndex(clean["cq"], freq="Q")
    yoy = clean.pivot_table(index="cq", columns="ticker", values="yoy", aggfunc="last")
    yoy = yoy.sort_index()
    # expanding max of |yoy|, then shift so quarter q sees only < q
    hist = yoy.abs().expanding(min_periods=1).max().shift(1)
    out = hist.stack().rename("max_hist_yoy").reset_index()
    out.columns = ["cq", "ticker", "max_hist_yoy"]
    out["target_quarter"] = out["cq"].astype(str)
    return out[["ticker", "target_quarter", "max_hist_yoy"]]


def apply_guard(tf, ceiling, persistence, k, floor):
    """Return a copy of `tf` with guarded predictions + a `guarded` flag."""
    d = tf.merge(ceiling, on=["ticker", "target_quarter"], how="left")
    d = d.merge(persistence, on=["ticker", "target_quarter"], how="left")
    lim = np.maximum(k * d["max_hist_yoy"].fillna(0.0), floor)
    bad = d["y_pred"].abs() > lim
    # only guard where a fallback actually exists
    bad &= d["persistence"].notna()
    d["guarded"] = bad
    d["y_pred_guarded"] = np.where(bad, d["persistence"], d["y_pred"])
    return d


def _score(y, yhat, label, n_guarded=None):
    e = np.asarray(y) - np.asarray(yhat)
    sst = ((y - np.mean(y)) ** 2).sum()
    row = {"model": label, "n": len(e), "median_ae": float(np.median(np.abs(e))),
           "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "r2": float(1 - (e ** 2).sum() / sst),
           "dir_acc": float((np.sign(y) == np.sign(yhat)).mean())}
    if n_guarded is not None:
        row["n_guarded"] = int(n_guarded)
    return row


def run(universe="data", arm="log"):
    data_dir = config.ROOT / universe
    tf = pd.read_parquet(data_dir / f"model_predictions_timesfm_{arm}.parquet")
    tf["y_pred"] = tf["y_pred"].clip(-0.95, 3.0)
    panel = pd.read_parquet(data_dir / "panel.parquet")
    persistence = (panel[["ticker", "target_quarter", "bl_persistence"]]
                   .rename(columns={"bl_persistence": "persistence"}))
    ens = pd.read_parquet(data_dir / "model_predictions.parquet")
    ens = ens[ens["model"] == "ensemble"][["ticker", "target_quarter", "y_pred"]]
    ens = ens.rename(columns={"y_pred": "y_ens"})
    ceiling = _history_ceiling(data_dir)

    rows, frames = [], {}
    base = tf.merge(ens, on=["ticker", "target_quarter"], how="inner")
    rows.append(_score(base["y_true"], base["y_ens"], "ensemble (86 feats)"))
    rows.append(_score(base["y_true"], base["y_pred"], "timesfm raw"))
    for k in K_GRID:
        for floor in FLOOR_GRID:
            g = apply_guard(tf, ceiling, persistence, k, floor)
            g = g.merge(ens, on=["ticker", "target_quarter"], how="inner")
            rows.append(_score(g["y_true"], g["y_pred_guarded"],
                               f"timesfm guarded (k={k}, floor={floor})",
                               n_guarded=int(g["guarded"].sum())))
            frames[(k, floor)] = g
    # blend the middle-of-grid guarded model with the ensemble
    g = frames[(3.0, 0.5)]
    rows.append(_score(g["y_true"], 0.5 * g["y_pred_guarded"] + 0.5 * g["y_ens"],
                       "blend 50/50 (guarded k=3 + ensemble)"))
    tab = pd.DataFrame(rows)
    tab.to_csv(config.DATA_DIR / f"timesfm_guard_{universe}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n=== {universe} ({arm} arm, n={len(base)}) ===")
    print(tab.round(4).to_string(index=False))
    return tab, frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="data")
    ap.add_argument("--arm", default="log")
    args = ap.parse_args()
    run(universe=args.universe, arm=args.arm)
