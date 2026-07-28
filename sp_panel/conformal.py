"""Conformal calibration of the quantile forecasts (MODELING.md §12).

The raw LightGBM p10/p90 band under-covers: it claims 80% and delivers ~71%,
because quantile regression fits quantiles on TRAINING data and nothing forces
those quantiles to hold out of sample. Conformal prediction fixes this without
retraining: measure how far the realized values actually fell outside the band
on data the model already forecast, then widen the band by exactly that much.

Method — Conformalized Quantile Regression (Romano/Patterson/Candes 2019):

    conformity score   E_i = max(lo_i - y_i, y_i - hi_i)
        (negative when y is inside the band, positive when outside — the
         signed distance to the nearest edge)
    adjustment         Q = the ceil((n+1)(1-alpha))-th smallest E in the
                       calibration set (the finite-sample-corrected quantile)
    calibrated band    [lo - Q, hi + Q]

If the raw band under-covers, most E are positive, Q > 0, and the band widens
until it covers 1-alpha of calibration points; if it over-covers, Q < 0 and the
band TIGHTENS. That two-sidedness is why this is calibration, not padding.

Time-series discipline (the part standard conformal libraries get wrong here):
the coverage guarantee assumes calibration and test scores are exchangeable,
which fails across regime changes. So every method below calibrates ONLY on
quarters strictly before the test quarter — the same walk-forward rule as the
rest of the pipeline — and the rolling/ACI variants additionally adapt to
recent conditions.

Methods (--method):
  raw            no calibration (the current p10/p90 band; the thing to beat)
  cqr_expanding  calibrate on every past quarter
  cqr_rolling    calibrate on the last `window` quarters (adapts to regime)
  cqr_sector     Mondrian: a separate adjustment per sector, so Energy's band
                 is honest for Energy rather than only on average
  aci            Adaptive Conformal Inference (Gibbs & Candes 2021): carry a
                 running alpha, nudged up when recent quarters over-cover and
                 down when they miss — self-correcting under drift

Run: python -m sp_panel.conformal            (all methods, writes comparison)
"""
import argparse

import numpy as np
import pandas as pd

from . import config

ALPHA = 0.20                    # nominal 80% interval (p10-p90)
MIN_CAL = 150                   # min calibration rows before trusting an adjustment
DEFAULT_WINDOW = 8              # quarters in the rolling calibration window
ACI_GAMMA = 0.02                # ACI learning rate


def _scores(y, lo, hi):
    """CQR conformity score: signed distance to the nearest band edge."""
    return np.maximum(lo - y, y - hi)


def _conformal_q(scores, alpha):
    """ceil((n+1)(1-alpha))-th smallest score; inf when n is too small to
    certify the level (guarantee is vacuous, band becomes uninformative)."""
    n = len(scores)
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return np.inf
    return float(np.sort(scores)[k - 1])


def calibrate(qp, method="cqr_rolling", window=DEFAULT_WINDOW, alpha=ALPHA,
              min_cal=MIN_CAL, gamma=ACI_GAMMA):
    """Add lo_cal/hi_cal columns. Every adjustment uses only past quarters."""
    d = qp.sort_values("target_quarter").reset_index(drop=True).copy()
    d["score"] = _scores(d["y_true"].values, d["q10"].values, d["q90"].values)
    quarters = sorted(d["target_quarter"].unique())
    lo_cal = d["q10"].astype(float).copy()
    hi_cal = d["q90"].astype(float).copy()
    alpha_t = alpha                                    # ACI running level
    applied = []

    for q in quarters:
        te = (d["target_quarter"] == q).values
        past = d["target_quarter"] < q
        if method == "cqr_rolling":
            keep = [p for p in quarters if p < q][-window:]
            past = past & d["target_quarter"].isin(keep)
        cal = d.loc[past]
        if len(cal) < min_cal:                          # warm-up: leave raw
            applied.append({"target_quarter": q, "n_cal": len(cal), "Q": 0.0,
                            "alpha_used": alpha_t})
            continue

        lvl = alpha_t if method == "aci" else alpha
        if method == "cqr_sector":
            gq = {s: _conformal_q(g["score"].values, lvl)
                  for s, g in cal.groupby("sector") if len(g) >= min_cal // 4}
            glob = _conformal_q(cal["score"].values, lvl)
            adj = d.loc[te, "sector"].map(gq).fillna(glob).values
        else:
            adj = _conformal_q(cal["score"].values, lvl)
        lo_cal[te] = d.loc[te, "q10"].values - adj
        hi_cal[te] = d.loc[te, "q90"].values + adj
        applied.append({"target_quarter": q, "n_cal": len(cal),
                        "Q": float(np.mean(adj)), "alpha_used": alpha_t})

        if method == "aci":                             # observe, then adapt
            y = d.loc[te, "y_true"].values
            err = float(np.mean((y < lo_cal[te]) | (y > hi_cal[te])))
            alpha_t = float(np.clip(alpha_t + gamma * (alpha - err), 0.01, 0.60))

    d["lo_cal"], d["hi_cal"] = lo_cal, hi_cal
    d["method"] = method
    return d, pd.DataFrame(applied)


def report(d, lo_col="lo_cal", hi_col="hi_cal", label="", alpha=ALPHA):
    """Coverage, width, and the Winkler interval score.

    Winkler is the proper scoring rule for intervals — width plus a penalty of
    (2/alpha) x the distance of any miss. Coverage alone is gameable (an
    infinitely wide band covers everything); Winkler is not.
    """
    y = d["y_true"].values
    lo, hi = d[lo_col].values, d[hi_col].values
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    winkler = width + (2 / alpha) * (np.clip(lo - y, 0, None) + np.clip(y - hi, 0, None))
    per_q = (pd.DataFrame({"q": d["target_quarter"], "in": inside})
             .groupby("q")["in"].mean())
    return {
        "method": label or d.get("method", pd.Series(["?"])).iloc[0],
        "coverage": float(inside.mean()),
        "target": 1 - alpha,
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "winkler": float(np.mean(winkler)),
        "worst_quarter_cov": float(per_q.min()),
        "quarterly_cov_std": float(per_q.std()),
        "n": int(len(d)),
    }


def run(window=DEFAULT_WINDOW, alpha=ALPHA):
    qp = pd.read_parquet(config.DATA_DIR / "model_quantile_predictions.parquet")
    rows, per_sector, frames = [], [], {}

    rows.append(report(qp.assign(method="raw"), "q10", "q90", "raw", alpha))
    for method in ("cqr_expanding", "cqr_rolling", "cqr_sector", "aci"):
        d, sched = calibrate(qp, method=method, window=window, alpha=alpha)
        frames[method] = d
        rows.append(report(d, label=method, alpha=alpha))
        sched.to_csv(config.DATA_DIR / f"conformal_schedule_{method}.csv", index=False)

    # conditional coverage: does each sector get an honest band?
    best = frames["cqr_rolling"]
    for label, frame, lo, hi in (("raw", qp, "q10", "q90"),
                                 ("cqr_rolling", best, "lo_cal", "hi_cal"),
                                 ("cqr_sector", frames["cqr_sector"], "lo_cal", "hi_cal")):
        for sec, g in frame.groupby("sector"):
            cov = ((g["y_true"] >= g[lo]) & (g["y_true"] <= g[hi])).mean()
            per_sector.append({"method": label, "sector": sec,
                               "coverage": float(cov), "n": len(g)})

    tab = pd.DataFrame(rows)
    sec_tab = pd.DataFrame(per_sector).pivot(index="sector", columns="method",
                                             values="coverage")
    tab.to_csv(config.DATA_DIR / "conformal_calibration.csv", index=False)
    sec_tab.to_csv(config.DATA_DIR / "conformal_by_sector.csv")
    for m, d in frames.items():
        d.to_parquet(config.DATA_DIR / f"model_quantile_calibrated_{m}.parquet", index=False)

    pd.set_option("display.width", 200)
    print("=== Conformal calibration of the p10-p90 band (target 0.80) ===")
    print(tab.round(4).to_string(index=False))
    print("\n=== Per-sector coverage (conditional honesty) ===")
    print(sec_tab.round(3).to_string())
    return tab, sec_tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()
    run(window=args.window, alpha=args.alpha)
