"""Guard/blend robustness checks for the TimesFM comparison (MODELING.md §11).

Answers four questions about the tail-guard + blend result:
  1. does the blend need the guard at all?
  2. how sensitive is it to the guard threshold k?
  3. how sensitive is it to the blend weight?
  4. does it hold in the 2024+ window (post-TimesFM-pretraining)?

The historical-growth ceiling is derived from the panel's own target column
(expanding max |YoY| shifted one quarter) — strictly causal.

Run: python -m sp_panel.tfm_robust [universe ...]
"""
import sys

import numpy as np
import pandas as pd

from . import config
from .evaluate import _dm_test


def _rmse(y, p):
    return float(np.sqrt(((np.asarray(y) - np.asarray(p)) ** 2).mean()))


def analyse(universe="data", arm="log"):
    d = config.ROOT / universe
    tf = pd.read_parquet(d / f"model_predictions_timesfm_{arm}.parquet")
    tf["y_pred"] = tf["y_pred"].clip(-0.95, 3.0)
    panel = pd.read_parquet(d / "panel.parquet")
    ens = pd.read_parquet(d / "model_predictions.parquet")
    ens = ens[ens.model == "ensemble"][["ticker", "target_quarter", "y_pred"]].rename(
        columns={"y_pred": "y_ens"})

    p = panel.sort_values(["ticker", "target_quarter"])
    p["max_hist"] = (p.groupby("ticker")["revenue_yoy_target"]
                     .transform(lambda s: s.abs().expanding(min_periods=1).max().shift(1)))
    keep = p[["ticker", "target_quarter", "max_hist", "bl_persistence"]].rename(
        columns={"bl_persistence": "persistence"})
    j = (tf.merge(keep, on=["ticker", "target_quarter"], how="left")
           .merge(ens, on=["ticker", "target_quarter"], how="inner"))

    def dm(frame, a, b):
        dd = (frame.y_true - frame[a]) ** 2 - (frame.y_true - frame[b]) ** 2
        return _dm_test(dd.groupby(frame.target_quarter).mean().values)

    print(f"\n{'='*78}\n{universe}  (n={len(j)}, arm={arm})\n{'='*78}")
    print(f"ensemble        RMSE {_rmse(j.y_true, j.y_ens):.4f}  "
          f"MAE {np.abs(j.y_true - j.y_ens).mean():.4f}")
    print(f"timesfm raw     RMSE {_rmse(j.y_true, j.y_pred):.4f}  "
          f"MAE {np.abs(j.y_true - j.y_pred).mean():.4f}")
    b = 0.5 * j.y_pred + 0.5 * j.y_ens
    s, pv, _ = dm(j.assign(bl=b), "y_ens", "bl")
    print(f"blend, NO guard RMSE {_rmse(j.y_true, b):.4f}  DM {s:+.2f} p={pv:.4f}")

    print("\nguard k sensitivity:")
    store = {}
    for k in (2.0, 3.0, 5.0, 10.0):
        lim = np.maximum(k * j["max_hist"].fillna(0.0), 0.5)
        bad = (j.y_pred.abs() > lim) & j.persistence.notna()
        gp = np.where(bad, j.persistence, j.y_pred)
        bl = 0.5 * gp + 0.5 * j.y_ens
        s1, p1, _ = dm(j.assign(x=gp), "y_ens", "x")
        s2, p2, _ = dm(j.assign(x=bl), "y_ens", "x")
        store[k] = gp
        print(f"  k={k:<5} guarded={int(bad.sum()):>3}  "
              f"guarded-only {_rmse(j.y_true, gp):.4f} (DM {s1:+.2f} p={p1:.3f})  "
              f"blend {_rmse(j.y_true, bl):.4f} (DM {s2:+.2f} p={p2:.3f})")

    gp3 = store[3.0]
    print("\nblend weight sensitivity (k=3): " + "  ".join(
        f"w={w}:{_rmse(j.y_true, w * gp3 + (1 - w) * j.y_ens):.4f}"
        for w in (0.3, 0.4, 0.5, 0.6, 0.7)))

    j2 = j.assign(gp=gp3, bl=0.5 * gp3 + 0.5 * j.y_ens)
    j2["yr"] = j2.target_quarter.str[:4].astype(int)
    for lbl, sub in (("2014-2023", j2[j2.yr < 2024]), ("2024+ clean", j2[j2.yr >= 2024])):
        if len(sub) < 50:
            continue
        s, pv, T = dm(sub, "y_ens", "bl")
        print(f"{lbl:14s} ensemble {_rmse(sub.y_true, sub.y_ens):.4f} | "
              f"blend {_rmse(sub.y_true, sub.bl):.4f} | DM {s:+.2f} p={pv:.4f} "
              f"n={len(sub)} q={T}")


if __name__ == "__main__":
    for u in sys.argv[1:] or ["data"]:
        try:
            analyse(u)
        except FileNotFoundError as e:
            print(f"\n[{u}] not ready: {e}")
