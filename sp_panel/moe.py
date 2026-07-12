"""Feature-gated mixture-of-experts (MoE) layer on top of the revenue-growth suite.

Idea (the "knob" approach): instead of one monolithic model, train block experts —
each seeing only one *view* of the world — and blend them with a GATE that decides,
per observation, how much to trust each expert:

    experts:  company  (growth dynamics + company fundamentals + seasonality + guidance)
              macro    (macro + market/micro factors)
              sector   (sector aggregates + BEA + sector identity)

    gate(context) -> softmax weights over the 3 experts -> blended prediction

The gate is conditioned on company CONTEXT (size, growth volatility, sector, and —
crucially — each expert's *trailing accuracy for that specific company*), so a
company that has historically been macro-driven learns to up-weight the macro
expert, and an idiosyncratic name up-weights the company expert. Parameters are
shared globally (data-efficient); per-company behaviour emerges from the context,
not from per-company free weights (which would overfit ~55 quarters/company).

Everything is strictly walk-forward and leakage-free:
  * experts are trained on quarters < q and predict q (out-of-sample);
  * the gate is trained only on the experts' accumulated *out-of-sample* predictions
    from quarters earlier than the one being predicted.

Compared head-to-head against: the monolithic full-feature model, a static
equal-weight blend, and a global (context-free) non-negative stack.

Run:  python -m sp_panel.moe
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from lightgbm import LGBMRegressor

from . import assemble, config
from .evaluate import _metrics

LGB = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, max_depth=4,
           min_child_samples=20, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=-1, verbose=-1)

EXPERTS3 = ("company", "macro", "sector")
CONTEXT_STATE = ["f_log_revenue", "f_roll_std_8", "f_yoy_z"]


# ---------------------------------------------------------------------------
# Expert feature views
# ---------------------------------------------------------------------------
def expert_feature_sets(panel):
    dic = assemble._dictionary(panel).set_index("column")["group"].to_dict()
    cols = assemble.feature_columns(panel)
    grp = {c: dic.get(c, "other") for c in cols}
    company = [c for c in cols if grp[c] in ("growth_dynamics", "company_behavior", "seasonality", "guidance")]
    macro = [c for c in cols if grp[c] in ("macro", "market_risk")]
    sector = [c for c in cols if grp[c] == "sector_behavior"]
    return {"company": company, "macro": macro, "sector": sector, "full": cols}


def _expert_matrix(d, cols, add_sector):
    X = d[cols].astype(float).copy()
    if add_sector:
        X = pd.concat([X.reset_index(drop=True),
                       pd.get_dummies(d["sector"], prefix="sec").astype(float).reset_index(drop=True)], axis=1)
    return X.values


# ---------------------------------------------------------------------------
# Level-0: expert out-of-sample predictions (walk-forward)
# ---------------------------------------------------------------------------
def expert_oos(panel, experts, warmup=12, min_train=400):
    d = panel[panel["revenue_yoy_target"].notna()].copy().reset_index(drop=True)
    d["q"] = d["target_q"]
    mats = {name: _expert_matrix(d, cols, add_sector=(name in ("sector", "full")))
            for name, cols in experts.items()}
    y = d["revenue_yoy_target"].values
    quarters = sorted(d["q"].unique())
    out = []
    for q in quarters[warmup:]:
        tr = (d["q"] < q).values
        te = (d["q"] == q).values
        if tr.sum() < min_train or te.sum() == 0:
            continue
        r = d.loc[te, ["ticker", "sector", "target_quarter"] + CONTEXT_STATE].reset_index(drop=True)
        r["y_true"] = y[te]
        r["q"] = q
        for name in experts:
            mdl = LGBMRegressor(**LGB).fit(mats[name][tr], y[tr])
            r[f"pred_{name}"] = mdl.predict(mats[name][te])
        out.append(r)
    oos = pd.concat(out, ignore_index=True)
    # trailing per-ticker accuracy of each expert (strictly past, expanding)
    oos = oos.sort_values(["ticker", "q"]).reset_index(drop=True)
    for name in EXPERTS3:
        ae = (oos["y_true"] - oos[f"pred_{name}"]).abs()
        oos[f"terr_{name}"] = ae.groupby(oos["ticker"]).transform(
            lambda s: s.shift(1).expanding().mean())
    return oos


# ---------------------------------------------------------------------------
# The softmax gate (numpy + Adam)
# ---------------------------------------------------------------------------
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_gate(C, P, y, l2=1e-2, iters=500, lr=0.05):
    """Learn W so that softmax(C@W) blends expert preds P to best fit y (MSE+L2)."""
    n, d = C.shape
    K = P.shape[1]
    W = np.zeros((d, K))
    mW = np.zeros_like(W); vW = np.zeros_like(W)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        w = _softmax(C @ W)
        yhat = (w * P).sum(axis=1)
        r = yhat - y
        # d loss / d logits_k = (2 r / n) * w_k * (P_k - yhat)
        dlogits = (2 * r / n)[:, None] * w * (P - yhat[:, None])
        g = C.T @ dlogits + 2 * l2 * W
        mW = b1 * mW + (1 - b1) * g
        vW = b2 * vW + (1 - b2) * (g * g)
        W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
    return W


def _prep_context(train_df, test_df, ctx_cols):
    """Median-impute + standardize on train stats; append bias column."""
    med = train_df[ctx_cols].median()
    tr = train_df[ctx_cols].fillna(med)
    te = test_df[ctx_cols].fillna(med)
    mu, sd = tr.mean(), tr.std().replace(0, 1.0)
    tr = (tr - mu) / sd
    te = (te - mu) / sd
    tr["bias"], te["bias"] = 1.0, 1.0
    return tr.values, te.values


# ---------------------------------------------------------------------------
# Level-1: walk-forward over the OOS table — gate + comparison blenders
# ---------------------------------------------------------------------------
def blend_walk_forward(oos, gate_warmup=6, l2=1e-2):
    oos = oos.copy()
    sec = pd.get_dummies(oos["sector"], prefix="sec").astype(float)
    oos = pd.concat([oos.reset_index(drop=True), sec.reset_index(drop=True)], axis=1)
    ctx_cols = CONTEXT_STATE + [f"terr_{e}" for e in EXPERTS3] + list(sec.columns)
    Pcols = [f"pred_{e}" for e in EXPERTS3]
    quarters = sorted(oos["q"].unique())

    rows = []
    for i, q in enumerate(quarters):
        if i < gate_warmup:
            continue
        tr = (oos["q"] < q).values
        te = (oos["q"] == q).values
        if tr.sum() < 200 or te.sum() == 0:
            continue
        Ptr, Pte = oos.loc[tr, Pcols].values, oos.loc[te, Pcols].values
        ytr = oos.loc[tr, "y_true"].values
        base = oos.loc[te, ["ticker", "sector", "target_quarter", "y_true"]].reset_index(drop=True)
        base["q"] = q

        # --- gated MoE ---
        Ctr, Cte = _prep_context(oos.loc[tr], oos.loc[te], ctx_cols)
        W = fit_gate(Ctr, Ptr, ytr, l2=l2)
        gw = _softmax(Cte @ W)
        base["pred_moe_gate"] = (gw * Pte).sum(axis=1)
        for j, e in enumerate(EXPERTS3):
            base[f"w_{e}"] = gw[:, j]

        # --- static equal-weight blend ---
        base["pred_equal"] = Pte.mean(axis=1)

        # --- global (context-free) non-negative stack ---
        w_nnls, _ = nnls(Ptr, ytr)
        if w_nnls.sum() > 0:
            w_nnls = w_nnls / w_nnls.sum()
        base["pred_global_stack"] = Pte @ w_nnls

        # --- references carried straight through ---
        base["pred_full"] = oos.loc[te, "pred_full"].values
        for e in EXPERTS3:
            base[f"pred_{e}"] = oos.loc[te, f"pred_{e}"].values
        rows.append(base)
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Scoring + diagnostics
# ---------------------------------------------------------------------------
def score_methods(res):
    methods = {
        "MoE gate (A)": "pred_moe_gate",
        "monolithic full": "pred_full",
        "global stack": "pred_global_stack",
        "equal-weight blend": "pred_equal",
        "expert: company": "pred_company",
        "expert: macro": "pred_macro",
        "expert: sector": "pred_sector",
    }
    rows = []
    for name, col in methods.items():
        g = res[["y_true"]].copy(); g["y_pred"] = res[col]
        rows.append({"method": name, **_metrics(g)})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def score_by_sector(res, col_a="pred_moe_gate", col_b="pred_full"):
    rows = []
    for sec, g in res.groupby("sector"):
        a = g[["y_true"]].copy(); a["y_pred"] = g[col_a]
        b = g[["y_true"]].copy(); b["y_pred"] = g[col_b]
        ma, mb = _metrics(a), _metrics(b)
        rows.append({"sector": sec, "n": ma["n"], "rmse_moe": ma["rmse"],
                     "rmse_full": mb["rmse"], "rmse_delta": ma["rmse"] - mb["rmse"],
                     "dir_moe": ma["dir_acc"], "dir_full": mb["dir_acc"]})
    return pd.DataFrame(rows).sort_values("rmse_delta")


def company_weights(res):
    g = res.groupby("ticker")
    out = g[["w_company", "w_macro", "w_sector"]].mean()
    out["n"] = g.size()
    # stability: average over time of the per-quarter weight std for the company
    stab = res.groupby("ticker")[["w_company", "w_macro", "w_sector"]].std().mean(axis=1)
    out["weight_volatility"] = stab
    return out.reset_index().sort_values("w_macro", ascending=False)


def sector_weights(res):
    return res.groupby("sector")[["w_company", "w_macro", "w_sector"]].mean().round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default=str(config.DATA_DIR / "panel.parquet"))
    ap.add_argument("--warmup", type=int, default=12, help="expert walk-forward warmup quarters")
    ap.add_argument("--gate-warmup", type=int, default=6, help="extra quarters before the gate activates")
    ap.add_argument("--l2", type=float, default=1e-2, help="gate L2 regularization")
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel)
    experts = expert_feature_sets(panel)
    print("[moe] expert feature counts: " +
          "  ".join(f"{k}={len(v)}" for k, v in experts.items()))
    oos = expert_oos(panel, experts, warmup=args.warmup)
    print(f"[moe] expert OOS predictions: {len(oos)} rows over "
          f"{oos['q'].nunique()} quarters")
    res = blend_walk_forward(oos, gate_warmup=args.gate_warmup, l2=args.l2)
    print(f"[moe] gate-active evaluation rows: {len(res)} "
          f"({res['q'].min()}..{res['q'].max()})")

    overall = score_methods(res)
    by_sec = score_by_sector(res)
    cw = company_weights(res)
    sw = sector_weights(res)

    overall.to_csv(config.DATA_DIR / "moe_metrics_overall.csv", index=False)
    by_sec.to_csv(config.DATA_DIR / "moe_metrics_by_sector.csv", index=False)
    cw.to_csv(config.DATA_DIR / "moe_company_weights.csv", index=False)
    res.to_parquet(config.DATA_DIR / "moe_predictions.parquet", index=False)

    pd.set_option("display.width", 200)
    print("\n=== MoE gate vs references (same gate-active test rows) ===")
    print(overall.round(4).to_string(index=False))
    moe = overall.loc[overall.method == "MoE gate (A)"].iloc[0]
    full = overall.loc[overall.method == "monolithic full"].iloc[0]
    print(f"\nMoE vs monolithic full: RMSE {moe.rmse:.4f} vs {full.rmse:.4f} "
          f"({100*(full.rmse-moe.rmse)/full.rmse:+.1f}%)  |  "
          f"dir_acc {moe.dir_acc:.3f} vs {full.dir_acc:.3f}")

    print("\n=== avg gate weights by sector (does the vision hold?) ===")
    print(sw.to_string())
    print("\n=== where MoE helps/hurts vs full, by sector ===")
    print(by_sec.round(4).to_string(index=False))
    print(f"\n[moe] mean per-company weight volatility (0=stable, high=noisy gate): "
          f"{cw['weight_volatility'].mean():.3f}")
    print("[moe] companies leaning most on the MACRO expert:")
    print(cw.head(6)[["ticker", "w_company", "w_macro", "w_sector", "n"]].round(3).to_string(index=False))
    print("[moe] -> wrote moe_metrics_overall.csv, moe_metrics_by_sector.csv, "
          "moe_company_weights.csv, moe_predictions.parquet")


if __name__ == "__main__":
    main()
