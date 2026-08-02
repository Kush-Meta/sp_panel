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
def chart_heatmap(values=True, highlights=True, name="1_sector_macro_heatmap"):
    """values: print the correlation in each cell. highlights: box the four
    callout cells. Both off gives the pure colour field."""
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
                       fontsize=11.5, color=INK, fontweight="600", linespacing=1.35)
    ax.set_yticks(range(len(c["sectors"])))
    ax.set_yticklabels(c["sectors"], fontsize=11.5, color=INK, fontweight="600")
    ax.set_xticks(np.arange(-.5, len(c["macros"]), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(c["sectors"]), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.4)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # Four callouts: one per economic channel (commodity, consumer demand, FX,
    # industrial activity) so the eye lands on a story rather than 110 numbers.
    KEY = {("Energy", "Oil price (YoY)"): "1",
           ("Consumer Discretionary", "Retail sales (YoY)"): "2",
           ("Consumer Staples", "US dollar (YoY)"): "3",
           ("Industrials", "Industrial prod (YoY)"): "4"}
    keyed = ({(c["sectors"].index(s), c["macros"].index(m)): n
              for (s, m), n in KEY.items()} if highlights else {})
    if values:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                hot = (i, j) in keyed
                ax.text(j, i, f"{v:+.2f}".replace("-", "−"),
                        ha="center", va="center",
                        fontsize=13 if hot else 10.5,
                        fontweight="700" if hot else "normal",
                        color="white" if abs(v) >= 0.50 else INK)
    for (i, j), n in keyed.items():
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                   edgecolor=INK, linewidth=3.0, zorder=5))
        # with no printed values the box has room for the number in the middle
        if values:
            ax.text(j - .40, i - .34, n, ha="center", va="center", fontsize=10,
                    fontweight="700", color=INK, zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec=INK, lw=1.4))
        else:
            ax.text(j, i, n, ha="center", va="center", fontsize=12,
                    fontweight="700", color=INK, zorder=6,
                    bbox=dict(boxstyle="circle,pad=0.22", fc="white", ec=INK, lw=1.6))
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, color=MUTED)
    cb.set_label("correlation", fontsize=9.5, color=MUTED)
    title(ax, "Which sectors move with which macro variables",
          "Correlation of sector-median revenue growth with each macro series, "
          "2011Q1–2026Q1 · red = moves together, blue = moves opposite",
          y=1.055)
    foot_y = -0.155 if highlights else -0.075
    if highlights:
        fig.text(0.008, -0.085,
                 "Four channels worth calling out:    "
                 "1 Energy tracks oil almost one-for-one (+0.89)    "
                 "2 Consumer Discretionary follows retail sales (+0.79)\n"
                 "3 A stronger dollar depresses Consumer Staples (−0.65)    "
                 "4 Industrials move with industrial production (+0.60)",
                 fontsize=11.5, color=INK, linespacing=1.6)
    fig.text(0.008, foot_y,
             "Sectors and variables both ordered by average absolute correlation. "
             "Correlation is descriptive — it does not mean the variable improves forecasts.",
             fontsize=9.5, color=MUTED)
    save(fig, name)


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
    """Individual macro variables, ranked by how much the model uses them.

    Bar length is per-variable (mean |SHAP|). Colour carries the *group*
    ablation verdict — red where the variable's group significantly improves
    forecasts, blue where it significantly hurts, grey where its group showed
    no significant effect. Significance is group-level by design: with 49
    quarters a per-variable ablation is underpowered, so the honest split is
    'this variable is heavily used' (length) versus 'its group is proven'
    (colour).
    """
    th = {t["n"]: t for t in D["themes"]}
    KEYMAP = {"activity_demand": "Activity & demand", "inflation": "Inflation",
              "rates_curve": "Rates & curve", "risk_credit": "Risk & credit",
              "commodities_fx": "Commodities & FX"}

    def colour(theme, panel):
        t = th[KEYMAP[theme]]
        delta, p = (t["dq"], t["pq"]) if panel == "q" else (t["da"], t["pa"])
        if p >= 0.05:
            return "#b9b7b0", 1.0            # group not proven either way
        return (R_MAIN if delta > 0 else B_MAIN), 1.0

    used = set()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 7.0))
    for ax, (key, panel, ttl) in zip(axes, [("quarterly", "q", "Next quarter"),
                                            ("annual", "a", "One year ahead")]):
        d = list(reversed(D["feat"][key]))
        used.update(colour(x["th"], panel)[0] for x in d)
        labs = [f"{x['v']}  ·  {x['lag']} qtr{'' if x['lag']==1 else 's'} back" for x in d]
        vals = [x["s"] for x in d]
        cols = [colour(x["th"], panel)[0] for x in d]
        ax.barh(np.arange(len(d)), vals, color=cols, height=0.68)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.018, i, f"{v:.2f}%", va="center",
                    fontsize=12, color=INK, family="monospace")
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels(labs, fontsize=12, color=INK)
        ax.set_xlim(0, max(vals) * 1.20)
        ax.set_title(ttl, loc="left", fontsize=14, color=INK, fontweight="600", pad=10)
        ax.set_xlabel("share of what the model uses", fontsize=11.5, color=INK2)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", labelsize=11, colors=INK2)
    fig.subplots_adjust(wspace=0.62)
    # only label categories that actually appear (no inflation variable reaches
    # either top 12, so the "hurts" swatch would otherwise be an empty promise)
    entries = [(R_MAIN, "Group significantly improves forecasts"),
               (B_MAIN, "Group significantly hurts forecasts"),
               ("#b9b7b0", "Group shows no significant effect")]
    handles = [Patch(facecolor=c, label=l) for c, l in entries if c in used]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.035),
               ncol=len(handles), frameon=False, fontsize=12)
    fig.suptitle("Which individual macro variables the model relies on",
                 x=0.008, y=1.10, ha="left", fontsize=17, fontweight="600", color=INK)
    fig.text(0.008, 1.035,
             "Top 12 variables by contribution, with how far back the reading comes from · "
             "colour = whether that variable's group passed the ablation test",
             fontsize=12, color=INK2)
    fig.text(0.008, -0.075,
             "Next quarter, no macro group is statistically significant — the model leans on oil and "
             "retail-sales momentum, but none of it provably helps.\nAt one year, industrial production "
             "two quarters back is the single largest macro input, and its group is significant "
             "(p=0.044); inflation significantly hurts (p=0.034).",
             fontsize=10.5, color=MUTED, linespacing=1.5)
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


def chart_features_annual():
    """Annual panel only — the quarterly side is entirely grey (nothing
    significant), so this is the version that carries a finding."""
    th = {t["n"]: t for t in D["themes"]}
    KEYMAP = {"activity_demand": "Activity & demand", "inflation": "Inflation",
              "rates_curve": "Rates & curve", "risk_credit": "Risk & credit",
              "commodities_fx": "Commodities & FX"}
    d = list(reversed(D["feat"]["annual"]))
    labs = [f"{x['v']}  ·  {x['lag']} qtr{'' if x['lag']==1 else 's'} back" for x in d]
    vals = [x["s"] for x in d]

    def col(theme):
        t = th[KEYMAP[theme]]
        return "#b9b7b0" if t["pa"] >= 0.05 else (R_MAIN if t["da"] > 0 else B_MAIN)

    cols = [col(x["th"]) for x in d]
    fig, ax = plt.subplots(figsize=(11.6, 6.8))
    ax.barh(np.arange(len(d)), vals, color=cols, height=0.7)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, i, f"{v:.2f}%", va="center",
                fontsize=13, color=INK, family="monospace")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(labs, fontsize=13, color=INK)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel("share of what the model uses", fontsize=12.5, color=INK2)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11.5, colors=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    used = set(cols)
    entries = [(R_MAIN, "Group significantly improves forecasts"),
               (B_MAIN, "Group significantly hurts forecasts"),
               ("#b9b7b0", "Group shows no significant effect")]
    handles = [Patch(facecolor=c, label=l) for c, l in entries if c in used]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=len(handles), frameon=False, fontsize=12)
    title(ax, "Which macro variables matter for the one-year forecast",
          "Top 12 by contribution, with how far back the reading comes from",
          y=1.045)
    fig.text(0.008, -0.20,
             "Industrial production two quarters back is the single largest macro input, and its group "
             "is the one that significantly\nimproves forecasts (p=0.044). The ~2-quarter delay is how "
             "long macro conditions take to reach company revenue.",
             fontsize=10.5, color=MUTED, linespacing=1.5)
    save(fig, "3c_macro_importance_annual_only")


def chart_macro_lift():
    """The plain answer to 'how much does macro help?' — percentage reduction
    in forecast error, each configuration against the same model without that
    macro component. Percentages because the raw RMSE gap (0.1918 -> 0.1907)
    is invisible on a zero-based bar.
    """
    rows = [  # label, base RMSE, with-macro RMSE, p
        ("Next quarter\nfull macro block", 0.191814, 0.190685, 0.428),
        ("One year\nfull macro block", 0.236562, 0.235906, 0.601),
        ("One year\n+ deeper lags (2-4 qtrs back)", 0.2367, 0.2348, 0.047),
        ("One year\ndemand indicators only", 0.2366, 0.2341, 0.044),
    ]
    labs = [r[0] for r in rows]
    pct = [100 * (r[1] - r[2]) / r[1] for r in rows]
    ps = [r[3] for r in rows]
    cols = [R_MAIN if p < 0.05 else "#b9b7b0" for p in ps]

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.barh(np.arange(len(rows)), pct, color=cols, height=0.62)
    for i, (v, p) in enumerate(zip(pct, ps)):
        sig = "significant" if p < 0.05 else "not significant"
        ax.text(v + 0.035, i, f"{v:.2f}%   ({sig}, p={p:.3f})", va="center",
                fontsize=12.5, color=INK if p < 0.05 else INK2)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labs, fontsize=12.5, color=INK, linespacing=1.5)
    ax.set_xlim(0, max(pct) * 1.85)
    ax.set_xlabel("reduction in forecast error from adding macro data",
                  fontsize=12, color=INK2)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=1))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11.5, colors=INK2)
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    title(ax, "How much does macro data actually improve the model?",
          "Each bar compares the same model with and without that macro component",
          y=1.07)
    fig.text(0.008, -0.20,
             "For scale: the model as a whole beats the best naive benchmark by 18%. Macro is a small "
             "contributor on top of that.\nIt is not statistically significant at the next-quarter "
             "horizon; at one year, demand indicators and deeper lags both are.",
             fontsize=10.5, color=MUTED, linespacing=1.5)
    save(fig, "4_macro_lift")


def chart_macro_share():
    """How much of the one-year model the macro block accounts for.

    Composition, not improvement: mean |SHAP| share of every feature group,
    computed on the lag-ladder panel (macro at 1-4 quarters back), so the
    macro figure is the 13.0% total across all its variables and lags.
    """
    g = [("Company's own history", 38.9), ("Fundamentals & margins", 21.3),
         ("Macro", 13.0), ("Market / valuation", 11.2),
         ("Accounting signals", 4.9), ("Sector value-added", 3.8),
         ("Sector peers", 3.4), ("Sector identity", 2.0), ("Guidance", 1.6)]
    labs = [x[0] for x in g][::-1]
    vals = [x[1] for x in g][::-1]
    cols = [R_MAIN if l == "Macro" else "#c9c7c0" for l in labs]

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    ax.barh(np.arange(len(g)), vals, color=cols, height=0.66)
    for i, (v, l) in enumerate(zip(vals, labs)):
        ax.text(v + 0.6, i, f"{v:.1f}%", va="center",
                fontsize=13 if l == "Macro" else 12,
                fontweight="700" if l == "Macro" else "normal",
                color=INK if l == "Macro" else INK2)
    ax.set_yticks(np.arange(len(g)))
    ax.set_yticklabels(labs, fontsize=13, color=INK,
                       fontweight=["700" if l == "Macro" else "normal" for l in labs][0])
    for t, l in zip(ax.get_yticklabels(), labs):
        t.set_fontweight("700" if l == "Macro" else "normal")
    ax.set_xlim(0, max(vals) * 1.16)
    ax.set_xlabel("share of what the model uses", fontsize=12, color=INK2)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11.5, colors=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    title(ax, "Macro data is 13% of the one-year model",
          "Every macro variable, at every lag, added together", y=1.07)
    fig.text(0.008, -0.15,
             "Share of the model's total feature contribution (LightGBM SHAP). "
             "The same block is 10.6% of the next-quarter model.",
             fontsize=10.5, color=MUTED)
    save(fig, "4_macro_share")


def chart_source_split():
    """Two blocks: what the companies filed with the SEC, versus everything
    sourced from outside those filings. Same SHAP shares as 4_macro_share,
    just collapsed to their origin."""
    # Plot area holds bars, labels and numbers only. Every line of prose sits
    # below the axis in one block, so the figure crops cleanly to the chart.
    rows = [("Company financial\nindicators", 70.11, B_MAIN),
            ("Outside indicators", 29.89, R_MAIN)]
    fig, ax = plt.subplots(figsize=(11.0, 3.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.55)
    for yy, r in zip(y, rows):
        ax.text(r[1] + 1.5, yy, f"{r[1]:.0f}%", va="center", fontsize=20,
                fontweight="700", color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=14, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(decimals=0))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11.5, colors=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0, 1.30, "Where the one-year model's information comes from",
            transform=ax.transAxes, fontsize=16, fontweight="600", color=INK,
            va="bottom", ha="left")
    fig.text(0.008, -0.34,
             "Company financial indicators (SEC EDGAR): revenue history · margins & fundamentals · accounting signals · "
             "sector peers' filings · management guidance\n"
             "Outside indicators: macro / FRED 13.0% · market & valuation 11.2% · "
             "sector value-added / BEA 3.8% · sector identity 2.0%\n"
             "Share of the model's total feature contribution (LightGBM SHAP). "
             "At the next-quarter horizon the split is 82% / 18%.",
             fontsize=10.5, color=MUTED, linespacing=1.7)
    save(fig, "5_data_source_split")


if __name__ == "__main__":
    chart_source_split()
    chart_macro_share()
    chart_macro_lift()
    chart_heatmap()
    chart_heatmap(values=False, highlights=True,
                  name="1b_heatmap_no_numbers_highlights")
    chart_heatmap(values=False, highlights=False,
                  name="1c_heatmap_no_numbers_clean")
    chart_heatmap(values=True, highlights=False,
                  name="1d_heatmap_numbers_no_highlights")
    chart_features_annual()
    chart_leaderboard()
    chart_robustness()
    chart_blend()
    chart_themes()
    chart_lags()
    print("\nAll charts ->", OUT + "/")
