"""Zero-shot TimesFM 2.5 on the walk-forward revenue task (MODELING.md §11).

For each test quarter q (same 49 quarters as evaluate.py), each labeled
company's clean revenue history STRICTLY before q is fed to TimesFM
(google/timesfm-2.5-200m-pytorch), which forecasts the next quarter's revenue
level; implied growth = forecast / revenue(q-4) - 1. TimesFM is univariate —
it sees one series and nothing else — so the fair rungs are the naive
baselines and the own-history-only GBM, not the full feature model.

Arms (--arm):
  level  forecast the revenue level, derive YoY               (primary)
  log    forecast log-revenue, exponentiate, derive YoY       (robustness)
  yoy    forecast the YoY-growth series directly              (robustness)

Outputs: data/model_predictions_timesfm_<arm>.parquet in the standard
prediction schema (+ q10/q50/q90 implied-growth quantile columns).
Zero-shot honesty is free: no training, so truncating context at q-1 is the
entire walk-forward protocol.
"""
import argparse
import time

import numpy as np
import pandas as pd

from . import assemble, config

MIN_CONTEXT = 8          # quarters of contiguous history required to forecast
MAX_CONTEXT = 64

# TimesFM 2.5 quantile head layout: [mean, q10, q20, ..., q90]
QIDX = {"q10": 1, "q50": 5, "q90": 9}


def _load_model():
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=MAX_CONTEXT, max_horizon=4, normalize_inputs=True,
        use_continuous_quantile_head=True, fix_quantile_crossing=True))
    return model


def _contiguous_suffix(values):
    """Longest run of non-NaN values ending at the last observation."""
    v = np.asarray(values, float)
    if len(v) == 0 or np.isnan(v[-1]):
        return np.array([])
    bad = np.where(np.isnan(v))[0]
    start = (bad[-1] + 1) if len(bad) else 0
    return v[start:]


def run(arm="level"):
    model = _load_model()
    panel = pd.read_parquet(config.DATA_DIR / "panel.parquet")
    lab = panel[panel["revenue_yoy_target"].notna()]
    sector = panel.drop_duplicates("ticker").set_index("ticker")["sector"]

    clean = assemble.clean_quarterly(assemble._load("financials_quarterly"))
    clean["cq"] = pd.PeriodIndex(clean["cq"], freq="Q")
    rev = clean.pivot_table(index="cq", columns="ticker", values="revenue",
                            aggfunc="last").sort_index()
    yoy = clean.pivot_table(index="cq", columns="ticker", values="yoy",
                            aggfunc="last").sort_index()

    quarters = sorted(lab["target_q"].unique())
    test_quarters = quarters[12:]                      # same warmup as evaluate
    rows, t0 = [], time.time()
    for qi, q in enumerate(test_quarters, 1):
        todo = lab[lab["target_q"] == q]
        ctxs, meta = [], []
        for _, r in todo.iterrows():
            tk = r["ticker"]
            if tk not in rev.columns:
                continue
            hist = (yoy if arm == "yoy" else rev)[tk]
            ctx = _contiguous_suffix(hist[hist.index < q].values)[-MAX_CONTEXT:]
            if len(ctx) < MIN_CONTEXT:
                continue
            rev4 = rev[tk].get(q - 4, np.nan)
            if arm != "yoy" and (not np.isfinite(rev4) or rev4 <= 0):
                continue
            if arm == "log":
                ctx = np.log(ctx)
            ctxs.append(np.asarray(ctx, float))
            meta.append((tk, str(q), float(r["revenue_yoy_target"]), rev4))
        if not ctxs:
            continue
        point, quant = model.forecast(horizon=1, inputs=ctxs)
        for (tk, tq, ytrue, rev4), p, qq in zip(meta, point[:, 0], quant[:, 0, :]):
            if arm == "yoy":
                yhat, qs = float(p), {k: float(qq[i]) for k, i in QIDX.items()}
            else:
                lvl = np.exp(p) if arm == "log" else p
                qlv = np.exp(qq) if arm == "log" else qq
                yhat = float(lvl / rev4 - 1.0)
                qs = {k: float(qlv[i] / rev4 - 1.0) for k, i in QIDX.items()}
            rows.append({"ticker": tk, "sector": sector.get(tk), "target_quarter": tq,
                         "y_true": ytrue, "model": f"timesfm_{arm}", "y_pred": yhat, **qs})
        if qi % 10 == 0:
            print(f"[timesfm:{arm}] {qi}/{len(test_quarters)} quarters "
                  f"({len(rows)} forecasts, {time.time()-t0:.0f}s)")
    out = pd.DataFrame(rows)
    path = config.DATA_DIR / f"model_predictions_timesfm_{arm}.parquet"
    out.to_parquet(path, index=False)
    e = out["y_true"] - out["y_pred"]
    print(f"[timesfm:{arm}] {len(out)} forecasts -> {path.name} | "
          f"RMSE={np.sqrt((e**2).mean()):.4f} MAE={e.abs().mean():.4f} "
          f"dir_acc={(np.sign(out.y_true) == np.sign(out.y_pred)).mean():.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="level", choices=["level", "log", "yoy"])
    args = ap.parse_args()
    run(arm=args.arm)
