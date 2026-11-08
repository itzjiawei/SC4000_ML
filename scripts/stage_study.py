"""Feature-stage studies for Logistic Regression (Kat) and Decision Tree (Matt) on the
shared features and shared CV.

For every stage and gender:
  1. pick hyperparameters by pooled Brier on TUNE_SEASONS (2012-2016) only
  2. re-run CV with those settings on all seasons and report REPORT_SEASONS (2017-2025)
  3. paired-bootstrap difference vs S1 on the same report games (negative = better)

    python -m scripts.stage_study logreg
    python -m scripts.stage_study tree

Writes results/shared_cv/stages_<model>.csv and OOF files results/shared_cv/oof/<model>_<stage>_<gender>.csv
"""
import itertools
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src import cv, features
from src.features import BASE4
from src.models import logreg

STUDIES = {
    # Kat's stages, numbered as in notebooks/logistic_regression.ipynb
    "logreg": {
        "grid": [{"C": c} for c in [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 100]],
        "make": lambda p: logreg(p["C"]),
        "untuned": {"men": {"C": 0.1}, "women": {"C": 0.1}},
        "stages": {
            "men": {
                "S1 base4": BASE4,
                "S2 +Elo": BASE4 + ["EloDiff"],
                "S3 +Massey+Elo": BASE4 + ["MasseyRankDiff", "EloDiff"],
                "S4 +Massey": BASE4 + ["MasseyRankDiff"],
                "S5 +LogMassey (log1p, Kat's script)": BASE4 + ["Log1pMasseyRankDiff"],
                "S5b +LogMassey (log, README)": BASE4 + ["LogMasseyRankDiff"],
            },
            "women": {
                "S1 base4": BASE4,
                "S2 +Elo": BASE4 + ["EloDiff"],
            },
        },
    },
    # Matt's stages, numbered as in decision_tree.ipynb (S5 added: LogMassey was never tried)
    "tree": {
        "grid": [{"max_depth": d, "min_samples_leaf": m}
                 for d, m in itertools.product([2, 3, 4, 5, 6, 8], [1, 10, 20, 40, 80])],
        "make": lambda p: (lambda: DecisionTreeClassifier(random_state=42, **p)),
        "untuned": {"men": {"max_depth": 5, "min_samples_leaf": 1},
                    "women": {"max_depth": 5, "min_samples_leaf": 1}},
        "stages": {
            "men": {
                "S1 base4": BASE4,
                "S2 +Massey": BASE4 + ["MasseyRankDiff"],
                "S3 +Elo": BASE4 + ["EloDiff"],
                "S4 +Massey+Elo": BASE4 + ["MasseyRankDiff", "EloDiff"],
                "S5 +LogMassey (new)": BASE4 + ["LogMasseyRankDiff"],
            },
            "women": {
                "S1 base4": BASE4,
                "S2 +Elo": BASE4 + ["EloDiff"],
            },
        },
    },
}


def slug(s: str) -> str:
    return s.split()[0]


def run(study_name: str) -> pd.DataFrame:
    study = STUDIES[study_name]
    rows = []
    for g, stages in study["stages"].items():
        _, games = features.load(g)
        s1_oof = None
        for stage, feats in stages.items():
            # 1. tune on 2012-2016 only
            scores = []
            for p in study["grid"]:
                oof = cv.cross_validate(study["make"](p), games, feats, cv.TUNE_SEASONS)
                scores.append(cv.brier(oof.Target, oof.Pred))
            best = study["grid"][int(np.argmin(scores))]
            # 2. full CV with the chosen settings, plus the owner's original (untuned) settings
            oof = cv.cross_validate(study["make"](best), games, feats, cv.ALL_SEASONS)
            oof_untuned = cv.cross_validate(study["make"](study["untuned"][g]), games, feats, cv.ALL_SEASONS)
            (cv.OUT_DIR / "oof").mkdir(parents=True, exist_ok=True)
            oof.assign(Model=f"{study_name}_{slug(stage)}").to_csv(
                cv.OUT_DIR / "oof" / f"{study_name}_{slug(stage)}_{g}.csv", index=False)
            rep = oof[oof.Season.isin(cv.REPORT_SEASONS)]
            rep_u = oof_untuned[oof_untuned.Season.isin(cv.REPORT_SEASONS)]
            row = {"Gender": g, "Stage": stage, "Chosen": str(best), "TuneBrier": min(scores),
                   "ReportBrier": cv.brier(rep.Target, rep.Pred),
                   "ReportBrier_untuned": cv.brier(rep_u.Target, rep_u.Pred)}
            # 3. compare with S1 on identical games
            if s1_oof is None:
                s1_oof = oof
            else:
                b = cv.paired_bootstrap(oof, s1_oof)
                row.update(vsS1=b["diff"], vsS1_ci_low=b["ci_low"], vsS1_ci_high=b["ci_high"])
            rows.append(row)
            print(f"{g:5s} {stage:38s} {str(best):42s} tune={min(scores):.4f} report={row['ReportBrier']:.4f}"
                  f" untuned={row['ReportBrier_untuned']:.4f}"
                  + (f"  vs S1 {row['vsS1']:+.4f} [{row['vsS1_ci_low']:+.4f},{row['vsS1_ci_high']:+.4f}]"
                     if "vsS1" in row else ""))
    table = pd.DataFrame(rows)
    table.to_csv(cv.OUT_DIR / f"stages_{study_name}.csv", index=False)
    return table


def tree_baseline_confound() -> pd.DataFrame:
    """What Matt's notebook actually compared: S1 with a RandomForest vs S2-S4 with a DecisionTree."""
    rows = []
    for g in ("men", "women"):
        _, games = features.load(g)
        for label, make in [
            ("S1 as coded: RandomForest(n=100, depth=5)",
             lambda: RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ("S1 corrected: DecisionTree(depth=5)", lambda: DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]:
            oof = cv.cross_validate(make, games, BASE4, cv.ALL_SEASONS)
            rep = oof[oof.Season.isin(cv.REPORT_SEASONS)]
            rows.append({"Gender": g, "Model": label, "ReportBrier": cv.brier(rep.Target, rep.Pred)})
            print(f"{g:5s} {label:45s} report={rows[-1]['ReportBrier']:.4f}")
    t = pd.DataFrame(rows)
    t.to_csv(cv.OUT_DIR / "tree_s1_confound.csv", index=False)
    return t


if __name__ == "__main__":
    name = sys.argv[1]
    run(name)
    if name == "tree":
        tree_baseline_confound()
