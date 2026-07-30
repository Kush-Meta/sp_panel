"""Presentation charts (PNG) for the macro/revenue findings.

Palette is black / red / blue: a validated diverging pair (red = positive,
blue = negative, warm/cool poles reading as opposite) with a neutral midpoint,
plus near-black ink. Red and blue arms are stepped at matched OKLab lightness
so neither pole visually dominates.

Run: python make_charts.py   ->  charts/*.png
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

OUT = "charts"
DPI = 200

# --- palette -------------------------------------------------------------
RED = ["#fad6d2", "#f1aea8", "#e4857e", "#d75853", "#c74845", "#9e3432", "#762221"]
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
NEUTRAL = "#f0efec"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
R_MAIN, B_MAIN = "#c74845", "#2a78d6"
DIVERGE = LinearSegmentedColormap.from_list(
    "rb", BLUE[::-1] + [NEUTRAL] + RED, N=256)

mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "font.size": 11, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def title(ax, t, sub=None, y=1.10):
    """Title above, subtitle beneath it — both clear of the plot area."""
    ax.text(0, y + (0.055 if sub else 0), t, transform=ax.transAxes,
            fontsize=15, fontweight="600", color=INK, va="bottom", ha="left")
    if sub:
        ax.text(0, y - 0.012, sub, transform=ax.transAxes, fontsize=10.5,
                color=MUTED, va="bottom", ha="left")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=DPI)
    plt.close(fig)
    print("wrote", f"{OUT}/{name}.png")


D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "charts", "chart_data.json")))

# =========================================================== 1. heatmap
def chart_heatmap():
    c = D["corr"]
    M = np.array(c["v"])
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    im = ax.imshow(M, cmap=DIVERGE, vmin=-0.9, vmax=0.9, aspect="auto")
    SHORT = {"Oil price (YoY)": "Oil\nprice", "Retail sales (YoY)": "Retail\nsales",
             "US dollar (YoY)": "US\ndollar", "Industrial prod (YoY)": "Industrial\nproduction",
             "Credit spread (Baa)": "Credit\nspread", "CPI inflation (YoY)": "CPI\ninflation",
             "Consumer sentiment": "Consumer\nsentiment", "Yield curve 10y-2y": "Yield\ncurve",
             "Fed funds rate": "Fed funds\nrate", "VIX": "VIX", "Unemployment": "Unemploy-\nment"}
    ax.set_xticks(range(len(c["macros"])))
    ax.set_xticklabels([SHORT.get(m, m) for m in c["macros"]],
                       fontsize=9.5, color=INK2, linespacing=1.35)
    ax.set_yticks(range(len(c["sectors"])))
    ax.set_yticklabels(c["sectors"], fontsize=10.5, color=INK)
    ax.set_xticks(np.arange(-.5, len(c["macros"]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(c["sectors"]), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.4)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}".replace("+", "+").replace("-", "−"),
                    ha="center", va="center", fontsize=9.5,
                    color="white" if abs(v) >= 0.50 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, color=MUTED)
    cb.set_label("correlation", fontsize=9.5, color=MUTED)
    title(ax, "Which sectors move with which macro variables",
          "Correlation of sector-median revenue growth with each macro series, "
          "2011Q1–2026Q1 · red = moves together, blue = moves opposite",
          y=1.055)
    fig.text(0.008, -0.012,
             "Sectors and variables both ordered by average absolute correlation. "
             "Correlation is descriptive — it does not mean the variable improves forecasts.",
             fontsize=9, color=MUTED)
    save(fig, "1_sector_macro_heatmap")


# =========================================================== 2a. leaderboard
def chart_leaderboard():
    d = sorted(D["lead"], key=lambda x: -x["rmse"])
    names = [m["name"] for m in d]
    vals = [m["rmse"] for m in d]
    cols = [R_MAIN if m["base"] else B_MAIN for m in d]
    best = min(range(len(d)), key=lambda i: d[i]["rmse"] if not d[i]["base"] else 9)
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    ax.barh(names, vals, color=cols, height=0.66)
    ax.barh(names[best], vals[best], color=cols[best], height=0.66,
            edgecolor=INK, linewidth=1.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=10,
                color=INK2, family="monospace")
    ax.set_xlim(0, max(vals) * 1.12)
    ax.set_xlabel("Forecast error (RMSE) — shorter is better", fontsize=10.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=B_MAIN, label="Machine-learning model"),
                       Patch(facecolor=R_MAIN, label="Naive benchmark")],
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=10.5)
    title(ax, "What works, what doesn't — next-quarter revenue growth",
          "4,525 out-of-sample forecasts · outlined bar = best model")
    fig.text(0.008, -0.16,
             "The best model cuts error 18% below the strongest naive rule. Linear models trail the "
             "tree ensembles;\nrepeating last year's same quarter is worse than useless.",
             fontsize=9, color=MUTED)
    save(fig, "2a_model_leaderboard")


# =========================================================== 2b. robustness
def chart_robustness():
    d = D["uni"]
    labs = [f"{x['u']}\n{x['t']}" for x in d]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11.5, 7.4))
    ax.barh(y + 0.19, [x["model"] for x in d], height=0.34, color=B_MAIN)
    ax.barh(y - 0.19, [x["base"] for x in d], height=0.34, color=R_MAIN)
    for i, x in enumerate(d):
        better = x["gain"] > 0
        ax.text(max(x["model"], x["base"]) + 0.008, i,
                f"{'−' if better else '+'}{abs(x['gain']):.0f}%",
                va="center", fontsize=10, family="monospace",
                color=INK if better else MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=10)
    ax.set_xlabel("Forecast error (RMSE) — shorter is better", fontsize=10.5)
    ax.set_xlim(0, max(max(x["model"], x["base"]) for x in d) * 1.15)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=B_MAIN, label="Best model"),
                       Patch(facecolor=R_MAIN, label="Best naive benchmark")],
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=10.5)
    title(ax, "Does it hold up? Three universes × four targets",
          "100 S&P 500 mega caps · 100 S&P SmallCap 600 · 100 Russell 2000")
    fig.text(0.008, -0.15,
             "The model beats the benchmark in 10 of 12 combinations. Both exceptions are gross margin, "
             "which is so stable\nthat “next year equals this year” is hard to beat.",
             fontsize=9, color=MUTED)
    save(fig, "2b_robustness_universes")


# =========================================================== 2c. blend
def chart_blend():
    d = D["blend"]
    labs = [f"{x['u']}\n{x['h']}" for x in d]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.barh(y + 0.19, [x["ens"] for x in d], height=0.34, color=R_MAIN)
    ax.barh(y - 0.19, [x["bl"] for x in d], height=0.34, color=B_MAIN)
    for i, x in enumerate(d):
        g = 100 * (x["ens"] - x["bl"]) / x["ens"]
        p = "<.001" if x["p"] < 0.001 else f"{x['p']:.3f}"
        ax.text(x["ens"] + 0.006, i, f"−{g:.1f}%   p={p}", va="center",
                fontsize=10, family="monospace", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=10)
    ax.set_xlabel("Forecast error (RMSE) — shorter is better", fontsize=10.5)
    ax.set_xlim(0, max(x["ens"] for x in d) * 1.30)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(facecolor=R_MAIN, label="Our model alone"),
                       Patch(facecolor=B_MAIN, label="Blended with TimesFM")],
              loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=2, frameon=False, fontsize=10.5)
    title(ax, "Best result: blending with a pretrained forecaster",
          "Wins all six tests · −2.5% to −14.4% error, significant everywhere")
    fig.text(0.008, -0.20,
             "Google's TimesFM sees only revenue history; our model sees 86 features. They make different "
             "mistakes, so averaging\ncancels them. Strongest on the two universes the method was never tuned on.",
             fontsize=9, color=MUTED)
    save(fig, "2c_blend_result")


# =========================================================== 3a. themes
def chart_themes():
    d = D["themes"]
    order = sorted(d, key=lambda x: x["da"])
    names = [x["n"] for x in order]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=True)
    for ax, (k, pk, ttl) in zip(axes, [("dq", "pq", "Next quarter"),
                                       ("da", "pa", "One year ahead")]):
        vals = [x[k] for x in order]
        ps = [x[pk] for x in order]
        cols = [(R_MAIN if v >= 0 else B_MAIN) for v in vals]
        alphas = [1.0 if p < 0.05 else 0.45 for p in ps]
        for i, (v, c, a) in enumerate(zip(vals, cols, alphas)):
            ax.barh(i, v, color=c, alpha=a, height=0.6)
        for i, (v, p) in enumerate(zip(vals, ps)):
            off = 0.00016 if v >= 0 else -0.00016
            ax.text(v + off, i, f"p={p:.3f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9.5,
                    family="monospace", color=INK if p < 0.05 else MUTED)
        ax.axvline(0, color=INK, linewidth=1.0)
        ax.set_title(ttl, loc="left", fontsize=12, color=INK, fontweight="600", pad=8)
        ax.set_xlim(-0.0034, 0.0044)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticklabels([])
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=11)
    axes[0].invert_yaxis()
    fig.suptitle("Which macro variables actually improve forecasts",
                 x=0.008, y=1.15, ha="left", fontsize=15, fontweight="600", color=INK)
    fig.text(0.008, 1.055,
             "Change in forecast error when each group is added alone · "
             "solid = statistically significant (p<0.05)",
             fontsize=10.5, color=MUTED)
    fig.text(0.5, -0.015, "<-- makes forecasts worse    |    improves forecasts -->",
             ha="center", fontsize=10, color=MUTED)
    fig.text(0.008, -0.10,
             "Only demand-activity indicators earn their place, and only at the one-year horizon. "
             "Inflation actively hurts.\nQuarter-clustered Diebold–Mariano test across 49 quarters.",
             fontsize=9, color=MUTED)
    save(fig, "3a_macro_importance")


# =========================================================== 3b. lags
def chart_lags():
    B = D["buckets"]
    labs = ["lag 1 (changes)", "lag 1 (levels)", "lag 2", "lag 3", "lag 4"]
    nice = ["1 qtr ago\n(change)", "1 qtr ago\n(level)", "2 qtrs ago", "3 qtrs ago", "4 qtrs ago"]
    q = [next(b["s"] for b in B if b["b"] == L and b["t"] == "quarterly") for L in labs]
    a = [next(b["s"] for b in B if b["b"] == L and b["t"] == "annual") for L in labs]
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.bar(x - 0.19, q, width=0.36, color=R_MAIN, label="Next-quarter model")
    ax.bar(x + 0.19, a, width=0.36, color=B_MAIN, label="One-year model")
    for i, (vq, va) in enumerate(zip(q, a)):
        ax.text(i - 0.19, vq + 0.9, f"{vq:.0f}%", ha="center", fontsize=10,
                family="monospace", color=INK2)
        ax.text(i + 0.19, va + 0.9, f"{va:.0f}%", ha="center", fontsize=10,
                family="monospace", color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(nice, fontsize=10.5)
    ax.set_ylabel("share of the model's macro attention", fontsize=10.5)
    ax.set_ylim(0, 52)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.13), ncol=2)
    title(ax, "How far back the useful macro signal sits",
          "Next-quarter forecasting wants fresh momentum; one-year forecasting wants "
          "the economy's state 2–4 quarters ago")
    fig.text(0.008, -0.24,
             "That delay is how long macro conditions take to reach company revenue — which is why "
             "deeper lags\nsignificantly improve the one-year model but do nothing for the next-quarter model.",
             fontsize=9, color=MUTED)
    save(fig, "3b_macro_lag_structure")


if __name__ == "__main__":
    chart_heatmap()
    chart_leaderboard()
    chart_robustness()
    chart_blend()
    chart_themes()
    chart_lags()
    print("\nAll charts ->", OUT + "/")
