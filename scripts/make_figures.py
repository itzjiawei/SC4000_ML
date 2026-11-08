"""Report figures from the saved shared-CV results (run after run_baselines, blend and final_ablation).

    python -m scripts.make_figures
Writes results/figures/fig1_models.png ... fig4_per_season.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import cv

FIG = cv.OUT_DIR.parent / "figures"
OOF = cv.OUT_DIR / "oof"

# Reference palette (dataviz skill, light mode): surface, text inks, categorical slots 1-2, recessive grid
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e89", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 9.5, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})

LABELS = {
    "dt_tuned": "Decision tree, tuned (Matt)",
    "rf_owner": "Random forest (Preesha)",
    "xgb_owner": "XGBoost, original (JW)",
    "lr_owner": "Logistic regression, original (Kat)",
    "nb_owner": "Naive Bayes, original (Divisha)",
    "lr_s5": "LR + LogMassey (Kat, S5)",
    "nb_v2": "Naive Bayes + new features",
    "xgb_v2": "XGBoost v2 (reg. + monotone + symmetric + new features)",
    "lr_v2": "LR + new features",
    "final_blend": "Final: 0.6 LR v2 + 0.4 XGBoost v2",
}


def load(name):
    return pd.read_csv(OOF / f"{name}.csv")


def dot_ci(ax, labels, diff, lo, hi, highlight):
    y = np.arange(len(labels))
    for i in y:
        c = BLUE if highlight[i] else INK2
        ax.plot([lo[i], hi[i]], [i, i], color=c, lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(diff[i], i, "o", ms=7, color=c, mec=SURFACE, mew=1.5, zorder=3)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    for t, h in zip(ax.get_yticklabels(), highlight):
        t.set_color(INK if h else INK2)
        t.set_fontweight("bold" if h else "normal")


def fig1_models():
    ref = load("ref_lr_seed_only")
    rows = []
    for name, label in LABELS.items():
        o = load(name)
        b = cv.paired_bootstrap(o, ref)
        rep = o[o.Season.isin(cv.REPORT_SEASONS)]
        rows.append((label, b["diff"], b["ci_low"], b["ci_high"], cv.brier(rep.Target, rep.Pred), name == "final_blend"))
    rows.sort(key=lambda r: -r[1])
    lab, d, lo, hi, br, hl = map(list, zip(*rows))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    dot_ci(ax, lab, d, lo, hi, hl)
    for i, b in enumerate(br):
        ax.annotate(f"{b:.4f}", (1.0, i), xycoords=("axes fraction", "data"), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK if hl[i] else INK2, fontsize=9)
    ax.set_ylim(-0.6, len(lab) - 0.2)
    ax.annotate("Brier", (1.0, len(lab) - 0.35), xycoords=("axes fraction", "data"), xytext=(8, 4),
                textcoords="offset points", va="bottom", color=MUTED, fontsize=8.5)
    ax.set_xticks([-0.01, -0.005, 0, 0.005, 0.01])
    ax.set_xlabel("Change in Brier vs LR on seed difference only (negative = better), 95% CI")
    fig.suptitle("Every model vs the seed-only baseline, 2017–2025 tournaments (1,055 games)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.95, 1))
    fig.savefig(FIG / "fig1_models.png", dpi=200)


def fig2_ablation():
    t = pd.read_csv(cv.OUT_DIR / "final_ablation.csv").dropna(subset=["change"]).sort_values("change")
    labels = [e.replace("- ", "remove ") for e in t.Experiment]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    dot_ci(ax, labels, t.change.to_numpy(), t.ci_low.to_numpy(), t.ci_high.to_numpy(), [False] * len(t))
    ax.set_xlabel("Change in Brier when removed (positive = the component helps), 95% CI")
    full = pd.read_csv(cv.OUT_DIR / "final_ablation.csv").iloc[0].report_combined
    fig.suptitle(f"Ablation of the final system (full system Brier {full:.4f}; other settings held fixed)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_ablation.png", dpi=200)


def fig3_reliability():
    o = load("final_blend")
    o = o[o.Season.isin(cv.REPORT_SEASONS)]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.1), sharey=True)
    for ax, g in zip(axes, ["men", "women"]):
        og = o[o.Gender == g].copy()
        og["bin"] = pd.qcut(og.Pred, 10, labels=False, duplicates="drop")
        b = og.groupby("bin").agg(p=("Pred", "mean"), y=("Target", "mean"), n=("Target", "size"))
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.plot(b.p, b.y, color=BLUE, lw=1.6, zorder=2)
        ax.plot(b.p, b.y, "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5, zorder=3)
        ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.set_aspect("equal")
        ax.set_title(f"{g.capitalize()}  (Brier {cv.brier(og.Target, og.Pred):.4f}, {len(og)} games)", fontsize=10)
        ax.set_xlabel("Predicted P(lower-ID team wins)")
        ax.text(0.96, 0.06, "perfect calibration", rotation=45, ha="right", va="bottom", color=MUTED, fontsize=8,
                transform=ax.transAxes)
    axes[0].set_ylabel("Observed win rate (10 equal-size bins)")
    fig.suptitle("Final system is well calibrated: 2017–2025 out-of-fold predictions", x=0.02, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_reliability.png", dpi=200)


def fig4_per_season():
    series = [("final_blend", "Final system", BLUE), ("xgb_owner", "XGBoost, original (JW)", ORANGE),
              ("ref_lr_seed_only", "LR on seed only", INK2)]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    ax.axvspan(2011.5, 2016.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.text(2014, 0.1005, "tune seasons (settings chosen here,\nso these scores are optimistic)",
            ha="center", va="bottom", color=INK2, fontsize=8)
    ax.text(2021, 0.1005, "report seasons", ha="center", va="bottom", color=INK2, fontsize=8.5)
    for name, label, c in series:
        o = load(name)
        s = o.assign(SE=(o.Pred - o.Target) ** 2).groupby("Season").SE.mean()
        x = s.index.to_numpy()
        # break the line at 2020 (no tournament)
        for seg in (x[x < 2020], x[x > 2020]):
            ax.plot(seg, s.loc[seg], color=c, lw=1.8, zorder=3)
        ax.plot(x, s, "o", ms=4.5, color=c, mec=SURFACE, mew=1, zorder=4, label=label)
    ax.set_xticks([y for y in range(2012, 2026) if y != 2020])
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_ylim(0.10, None)
    ax.set_ylabel("Brier (men + women)")
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5, bbox_to_anchor=(0, 1.02))
    ax.set_title("Brier by tournament season (lower is better)", pad=18)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_per_season.png", dpi=200)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig1_models, fig2_ablation, fig3_reliability, fig4_per_season):
        f()
        print("wrote", f.__name__)
