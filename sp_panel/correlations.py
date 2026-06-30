"""Correlation analysis: which macro variables move with which sector's revenue,
and how collinear the macro variables are with each other.

This is the simple, visual companion to the models. The models asked "does macro
*improve prediction* over the company's own momentum?" (mostly no). This asks the
plainer question "is macro *related* to revenue at all?" (often yes). Both are
true, and together they're the finding: macro relates to revenue, but momentum
already captures most of it.

Outputs (in data/):
  corr_sector_macro.csv / .png   sector x macro: corr(revenue_yoy, each macro)
  corr_macro_macro.csv  / .png   macro x macro collinearity

Correlations use YoY measures on both sides (revenue YoY vs macro YoY change) so
they're apples-to-apples and not spuriously inflated by shared trends. Caveat: the
macro variables are correlated with EACH OTHER, so a column lighting up does not
prove that variable is THE cause — read the macro-macro map alongside.

Run:  python -m sp_panel.correlations
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

REV = "revenue_yoy"


def _macro_cols(panel):
    cols = [c for c in panel.columns if c.endswith("_chg_yoy")]
    cols += [c for c in ("bea_va_yoy",) if c in panel.columns]
    return [c for c in cols if panel[c].notna().sum() > 50]


def _label(c):
    return c.replace("_chg_yoy", "").replace("bea_va_yoy", "industry_va")


def sector_macro_corr(panel, macro_cols):
    rows = {sec: {m: g[REV].corr(g[m]) for m in macro_cols}
            for sec, g in panel.groupby("sector")}
    return pd.DataFrame(rows).T[macro_cols]


def _heatmap(mat, title, path, figsize, annot=True):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values.astype(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels([_label(c) for c in mat.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([_label(c) for c in mat.index], fontsize=8)
    if annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(v) > 0.55 else "black")
    ax.set_title(title, fontsize=11, pad=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    panel = pd.read_parquet(config.DATA_DIR / "panel.parquet")
    macro_cols = _macro_cols(panel)
    if not macro_cols:
        raise SystemExit("No macro change columns found in panel — run assemble first.")

    sm = sector_macro_corr(panel, macro_cols)
    mm = panel[macro_cols].corr()
    sm.to_csv(config.DATA_DIR / "corr_sector_macro.csv")
    mm.to_csv(config.DATA_DIR / "corr_macro_macro.csv")

    n = len(macro_cols)
    _heatmap(sm, "Revenue growth vs macro change, by sector",
             config.DATA_DIR / "corr_sector_macro.png", (max(8, n * 0.55), 6))
    _heatmap(mm, "Macro vs macro (collinearity)",
             config.DATA_DIR / "corr_macro_macro.png", (max(8, n * 0.55), max(7, n * 0.5)))

    print(f"[corr] sector x macro {sm.shape} -> corr_sector_macro.{{csv,png}}")
    print(f"[corr] macro x macro {mm.shape} -> corr_macro_macro.{{csv,png}}")
    s = sm.stack().dropna()
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(15)
    print("\n[corr] strongest sector <-> macro relationships:")
    for (sec, m), v in top.items():
        print(f"   {sec:24s} {_label(m):20s} {v:+.3f}")


if __name__ == "__main__":
    main()
