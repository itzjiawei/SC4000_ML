"""Ablation of the FINAL system (0.6 * lr_v2 + 0.4 * xgb_v2): remove one component at a time,
keep every other setting fixed (no re-tuning), and compare with the full system on the same
2017-2025 report games (95% paired bootstrap; positive = removing it hurts).

    python -m scripts.final_ablation
Writes results/shared_cv/final_ablation.csv
"""
import pandas as pd

from src import cv
from src.features import BASE4
from src.models import STRENGTH, XGB_MEN, XGB_V2_MEN, XGB_V2_WOMEN, XGB_WOMEN, logreg, xgb, xgb_v2

W_LR, W_XGB = 0.6, 0.4
FULL = {"men": BASE4 + ["LogMasseyRankDiff"] + STRENGTH, "women": BASE4 + STRENGTH}
XGB_PARAMS = {"men": XGB_V2_MEN, "women": XGB_V2_WOMEN}
KEY = ["Gender", "Season", "Team1", "Team2", "Target"]


def run(name, feats=FULL, lr_C=0.3, xgb_make=None, w_lr=W_LR):
    """CV both parts with the given changes and blend them with the final weights."""
    xgb_make = xgb_make or {g: xgb_v2(**XGB_PARAMS[g]) for g in ("men", "women")}
    parts = []
    if w_lr > 0:
        parts.append((w_lr, cv.evaluate(f"abl_lr", {g: logreg(lr_C) for g in feats}, feats, save=False).oof))
    if w_lr < 1:
        parts.append((1 - w_lr, cv.evaluate(f"abl_xgb", xgb_make, feats, save=False).oof))
    oof = parts[0][1][KEY].copy()
    oof["Pred"] = 0.0
    for w, o in parts:
        oof["Pred"] += w * oof[KEY].merge(o, on=KEY, how="left", validate="one_to_one").Pred.to_numpy()
    return name, oof


def drop(cols):
    return {g: [c for c in FULL[g] if c not in cols] for g in FULL}


EXPERIMENTS = [
    lambda: run("Full system (0.6 LR + 0.4 XGB)"),
    lambda: run("- XGBoost (LR only)", w_lr=1.0),
    lambda: run("- logistic regression (XGB only)", w_lr=0.0),
    lambda: run("- all new strength/form features", feats=drop(STRENGTH)),
    lambda: run("- improved Elo", feats=drop(["EloV2Diff"])),
    lambda: run("- strength of schedule", feats=drop(["SOSDiff"])),
    lambda: run("- AdjustedStrength", feats=drop(["AdjustedStrengthDiff"])),
    lambda: run("- last-10 margin", feats=drop(["Last10MarginDiff"])),
    lambda: run("- LogMassey (men)", feats=drop(["LogMasseyRankDiff"])),
    lambda: run("- SeedDiff", feats=drop(["SeedDiff"])),
    lambda: run("- symmetric training (XGB part)",
                xgb_make={g: xgb_v2(symmetric=False, **XGB_PARAMS[g]) for g in ("men", "women")}),
    lambda: run("- monotone constraints (XGB part)",
                xgb_make={g: xgb_v2(monotone=False, **XGB_PARAMS[g]) for g in ("men", "women")}),
    lambda: run("- XGB regularisation (JW's original settings)",
                xgb_make={"men": xgb(**XGB_MEN), "women": xgb(**XGB_WOMEN)}),
]

if __name__ == "__main__":
    results = [e() for e in EXPERIMENTS]
    full_name, full = results[0]
    full.assign(Model="final_ablation_full").to_csv(cv.OUT_DIR / "oof" / "final_ablation_full.csv", index=False)
    rows = []
    for name, oof in results:
        s = cv.summarise(oof)
        row = {"Experiment": name, "report_combined": s["report_combined"],
               "report_men": s["report_men"], "report_women": s["report_women"]}
        if name != full_name:
            b = cv.paired_bootstrap(oof, full)
            row.update(change=b["diff"], ci_low=b["ci_low"], ci_high=b["ci_high"])
        rows.append(row)
        print(f"{name:48s} report={row['report_combined']:.4f}"
              + (f"  change {row['change']:+.4f} [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]" if "change" in row else ""))
    pd.DataFrame(rows).to_csv(cv.OUT_DIR / "final_ablation.csv", index=False)
