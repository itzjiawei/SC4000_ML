"""Day-2 XGBoost study on the shared features and CV. Each stage adds ONE thing to the previous stage:

  A0  JW's tuned XGBoost (notebooks/xgboost.ipynb), his features           (no re-tuning)
  A1  + regularised grid chosen on the 2012-16 tune seasons               (same features)
  A2  + monotone constraints
  A3  + symmetric training data (each game in both team orders)
  A4  + new features: AdjustedStrength, improved Elo, last-10 margin, strength of schedule
  A5  one model for men + women with an IsWomen flag (instead of two models)

Hyperparameters are re-chosen at every stage from GRID using ONLY the tune seasons. Then
leave-one-out feature ablations on A4 keep A4's settings fixed.

    python -m scripts.xgb_study
Writes results/shared_cv/xgb_study.csv, xgb_ablation.csv and OOF files oof/xgb_<stage>.csv
"""
import itertools
import json

import numpy as np
import pandas as pd

from src import cv, features
from src.features import BASE4
from src.models import XGB_MEN, XGB_WOMEN, xgb, xgb_v2

GRID = [dict(max_depth=d, n_estimators=n, learning_rate=0.03, min_child_weight=m, reg_lambda=l,
             subsample=0.8, colsample_bytree=0.8)
        for d, n, m, l in itertools.product([1, 2, 3], [100, 200, 400], [5, 20], [1, 10])]

OLD = {"men": BASE4 + ["LogMasseyRankDiff"], "women": BASE4}
STRENGTH = ["AdjustedStrengthDiff", "EloV2Diff", "Last10MarginDiff", "SOSDiff"]
NEW = {"men": BASE4 + ["LogMasseyRankDiff"] + STRENGTH, "women": BASE4 + STRENGTH}
ABLATIONS = {
    "- SeedDiff": ["SeedDiff"],
    "- LogMassey (men)": ["LogMasseyRankDiff"],
    "- AdjustedStrength": ["AdjustedStrengthDiff"],
    "- improved Elo": ["EloV2Diff"],
    "- last-10 margin": ["Last10MarginDiff"],
    "- strength of schedule": ["SOSDiff"],
    "- original box-score 3 (margin, win%, eFG)": ["AvgScoringMarginDiff", "WinRateDiff", "EFGDifferentialDiff"],
}

GAMES = {g: features.load(g)[1] for g in ("men", "women")}


def pooled_cv(make, feats, seasons):
    """One model for both genders (IsWomen flag). Men-only columns are NaN for women (XGBoost handles NaN)."""
    both = pd.concat([GAMES["men"], GAMES["women"]], ignore_index=True)
    both["IsWomen"] = (both.Gender == "women").astype(int)
    return cv.cross_validate(make, both, feats + ["IsWomen"], seasons)


def tune(make_from_params, run):
    """Pick params by pooled Brier over the tune seasons. run(make, seasons) -> OOF."""
    scores = []
    for p in GRID:
        oof = run(make_from_params(p), cv.TUNE_SEASONS)
        scores.append(cv.brier(oof.Target, oof.Pred))
    i = int(np.argmin(scores))
    return GRID[i], scores[i]


def separate(stage_feats, make_from_params, fixed=None):
    """Tune (unless fixed) and run separate men / women models. Returns OOF and chosen params."""
    oofs, chosen = [], {}
    for g in ("men", "women"):
        run = lambda make, seasons, g=g: cv.cross_validate(make, GAMES[g], stage_feats[g], seasons)
        p = fixed[g] if fixed else tune(make_from_params, run)[0]
        chosen[g] = p
        oofs.append(run(make_from_params(p), cv.ALL_SEASONS))
    return pd.concat(oofs, ignore_index=True), chosen


def describe(name, oof, chosen, ref=None, prev=None):
    s = cv.summarise(oof)
    row = {"Stage": name, "tune": s["tune_combined"], "report_men": s["report_men"],
           "report_women": s["report_women"], "report_combined": s["report_combined"],
           "params": json.dumps(chosen)}
    for label, other in [("vs_prev", prev), ("vs_A0", ref)]:
        if other is not None:
            b = cv.paired_bootstrap(oof, other)
            row[label] = f"{b['diff']:+.4f} [{b['ci_low']:+.4f},{b['ci_high']:+.4f}]"
    print(f"{name:48s} tune={row['tune']:.4f} report={row['report_combined']:.4f} "
          f"(M {row['report_men']:.4f} W {row['report_women']:.4f})  "
          f"{row.get('vs_prev', '')}  vsA0 {row.get('vs_A0', '')}")
    return row


def save(name, oof):
    (cv.OUT_DIR / "oof").mkdir(parents=True, exist_ok=True)
    oof.assign(Model=name).to_csv(cv.OUT_DIR / "oof" / f"{name}.csv", index=False)


if __name__ == "__main__":
    rows, oofs = [], {}

    a0, _ = separate(OLD, lambda p: p, fixed={"men": xgb(**XGB_MEN), "women": xgb(**XGB_WOMEN)})
    oofs["A0"] = a0
    rows.append(describe("A0 JW tuned XGBoost (baseline)", a0, "JW params"))

    stages = [
        ("A1 + regularised grid", OLD, lambda p: xgb_v2(monotone=False, symmetric=False, **p)),
        ("A2 + monotone constraints", OLD, lambda p: xgb_v2(monotone=True, symmetric=False, **p)),
        ("A3 + symmetric training data", OLD, lambda p: xgb_v2(monotone=True, symmetric=True, **p)),
        ("A4 + new strength/form features", NEW, lambda p: xgb_v2(monotone=True, symmetric=True, **p)),
    ]
    prev, a4_params = a0, None
    for name, feats_, make in stages:
        oof, chosen = separate(feats_, make)
        key = name.split()[0]
        oofs[key] = oof
        rows.append(describe(name, oof, chosen, ref=a0, prev=prev))
        prev = oof
        if key == "A4":
            a4_params = chosen

    # A5: one pooled model with a gender flag, same features as A4 (Massey NaN for women)
    make5 = lambda p: xgb_v2(monotone=True, symmetric=True, **p)
    feats5 = NEW["men"]
    p5, _ = tune(make5, lambda m, s: pooled_cv(m, feats5, s))
    a5 = pooled_cv(make5(p5), feats5, cv.ALL_SEASONS)
    oofs["A5"] = a5
    rows.append(describe("A5 one model for both genders + IsWomen flag", a5, p5, ref=a0, prev=oofs["A4"]))

    for k, v in oofs.items():
        save(f"xgb_{k}", v)
    pd.DataFrame(rows).to_csv(cv.OUT_DIR / "xgb_study.csv", index=False)

    # Feature ablations on A4 (settings fixed at A4's choice)
    print("\nAblations of A4 (settings fixed):")
    abl = []
    for name, drop in ABLATIONS.items():
        f = {g: [c for c in NEW[g] if c not in drop] for g in NEW}
        if f == NEW:
            continue
        oof, _ = separate(f, lambda p: xgb_v2(monotone=True, symmetric=True, **p), fixed=a4_params)
        b = cv.paired_bootstrap(oof, oofs["A4"])
        s = cv.summarise(oof)
        abl.append({"Ablation": name, "report_combined": s["report_combined"], "report_men": s["report_men"],
                    "report_women": s["report_women"], "vs_A4": b["diff"], "ci_low": b["ci_low"],
                    "ci_high": b["ci_high"]})
        print(f"  {name:45s} report={s['report_combined']:.4f}  vs A4 {b['diff']:+.4f} "
              f"[{b['ci_low']:+.4f},{b['ci_high']:+.4f}]  (positive = removing it hurts)")
    pd.DataFrame(abl).to_csv(cv.OUT_DIR / "xgb_ablation.csv", index=False)
