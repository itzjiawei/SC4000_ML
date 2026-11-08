"""Each teammate's current model configuration, written as factories, so the same
definitions are used for CV (scripts/run_baselines.py) and for the 2026
submission (scripts/predict_2026.py).

Settings copied from each owner's notebook:
  XGBoost        JW       notebooks/xgboost.ipynb (final_men_params / final_women_params)
  LogReg         Kat      notebooks/logistic_regression.ipynb (best C per stage)
  RandomForest   Preesha  notebooks/random_forest.ipynb on branch rf-model (tuned overrides)
  DecisionTree   Matt     decision_tree.ipynb on branch decision_tree (max_depth=5)
  NaiveBayes     Divisha  notebooks/naive_bayes.ipynb on branch nb-model (PCA + GaussianNB)
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features import BASE4


def xgb(**params):
    def make():
        from xgboost import XGBClassifier  # imported lazily so sklearn-only users are unaffected
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, **params)
    return make


# Direction in which each feature should move P(Team1 wins): +1 increases, -1 decreases, 0 free.
# Lower seed / lower rank = better team, hence -1.
MONOTONE_SIGN = {
    "SeedDiff": -1, "MasseyRankDiff": -1, "LogMasseyRankDiff": -1, "Log1pMasseyRankDiff": -1,
    "AvgScoringMarginDiff": 1, "WinRateDiff": 1, "EFGDifferentialDiff": 1, "NetRatingDiff": 1,
    "AdjustedStrengthDiff": 1, "EloDiff": 1, "EloV2Diff": 1, "Last10WinRateDiff": 1, "Last10MarginDiff": 1,
    "SOSDiff": 0, "IsWomen": 0,
}
# Columns that describe the game rather than Team1 - Team2; not negated when the teams are swapped
NOT_ANTISYMMETRIC = {"IsWomen"}


class XGBv2:
    """XGBoost with two optional additions (both new for our project):
    monotone=True   constrain each feature's effect to the sign in MONOTONE_SIGN
    symmetric=True  train on every game twice (teams swapped: features negated, target flipped)
                    and predict (p(A,B) + 1 - p(B,A)) / 2, so P(A beats B) = 1 - P(B beats A) exactly.
    Works with any feature list; the column order is read at fit time."""

    def __init__(self, monotone: bool = True, symmetric: bool = True, **params):
        self.monotone, self.symmetric, self.params = monotone, symmetric, params
        self.classes_ = [0, 1]

    def _swap(self, X):
        X2 = X.copy()
        cols = [c for c in X.columns if c not in NOT_ANTISYMMETRIC]
        X2[cols] = -X2[cols]
        return X2

    def fit(self, X, y):
        import pandas as pd
        from xgboost import XGBClassifier
        params = dict(objective="binary:logistic", eval_metric="logloss", random_state=42, **self.params)
        if self.monotone:
            params["monotone_constraints"] = "(" + ",".join(str(MONOTONE_SIGN.get(c, 0)) for c in X.columns) + ")"
        if self.symmetric:
            X, y = pd.concat([X, self._swap(X)], ignore_index=True), pd.concat([y, 1 - y], ignore_index=True)
        self.model_ = XGBClassifier(**params).fit(X, y)
        return self

    def predict_proba(self, X):
        import numpy as np
        p = self.model_.predict_proba(X)[:, 1]
        if self.symmetric:
            p = (p + 1 - self.model_.predict_proba(self._swap(X))[:, 1]) / 2
        return np.column_stack([1 - p, p])


def xgb_v2(monotone: bool = True, symmetric: bool = True, **params):
    return lambda: XGBv2(monotone=monotone, symmetric=symmetric, **params)


def logreg(C: float):
    return lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="lbfgs", max_iter=3000))


def rf(**params):
    return lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, **params)


def tree(**params):
    return lambda: DecisionTreeClassifier(random_state=42, **params)


def pca_nb(k: int):
    return lambda: make_pipeline(StandardScaler(), PCA(n_components=k, whiten=True), GaussianNB())


# xgb_v2 settings, chosen on the tune seasons in scripts/xgb_study.py (stage A4)
XGB_V2_MEN = dict(max_depth=1, n_estimators=200, learning_rate=0.03, min_child_weight=20,
                  reg_lambda=10, subsample=0.8, colsample_bytree=0.8)
XGB_V2_WOMEN = dict(max_depth=3, n_estimators=100, learning_rate=0.03, min_child_weight=5,
                    reg_lambda=1, subsample=0.8, colsample_bytree=0.8)
STRENGTH = ["AdjustedStrengthDiff", "EloV2Diff", "Last10MarginDiff", "SOSDiff"]  # Day-2 strength/form features
XGB_MEN = dict(max_depth=5, learning_rate=0.01, n_estimators=300, subsample=0.7, reg_lambda=1, reg_alpha=0)
XGB_WOMEN = dict(max_depth=1, learning_rate=0.30, n_estimators=15, subsample=0.5, reg_lambda=1, reg_alpha=0)

# name -> (factory per gender, features per gender). None factory = constant 0.5.
CONFIGS = {
    "ref_coinflip": ({"men": None, "women": None}, {"men": BASE4, "women": BASE4}),
    "ref_lr_seed_only": ({"men": logreg(1.0), "women": logreg(1.0)}, {"men": ["SeedDiff"], "women": ["SeedDiff"]}),

    # Same four base features for every model: the like-for-like comparison
    "xgb_base4": ({"men": xgb(**XGB_MEN), "women": xgb(**XGB_WOMEN)}, {"men": BASE4, "women": BASE4}),
    "lr_base4": ({"men": logreg(0.1), "women": logreg(0.1)}, {"men": BASE4, "women": BASE4}),
    "rf_base4": ({"men": rf(max_depth=6, min_samples_leaf=20, max_features=3),
                  "women": rf(max_depth=2, min_samples_leaf=40, max_features=3)}, {"men": BASE4, "women": BASE4}),
    "dt_base4": ({"men": tree(max_depth=5), "women": tree(max_depth=5)}, {"men": BASE4, "women": BASE4}),
    "nb_base4": ({"men": pca_nb(4), "women": pca_nb(4)}, {"men": BASE4, "women": BASE4}),

    # Each owner's own chosen configuration
    "xgb_owner": ({"men": xgb(**XGB_MEN), "women": xgb(**XGB_WOMEN)},
                  {"men": BASE4 + ["LogMasseyRankDiff"], "women": BASE4}),
    "lr_owner": ({"men": logreg(0.1), "women": logreg(0.5)},
                 {"men": BASE4 + ["MasseyRankDiff", "EloDiff"], "women": BASE4 + ["EloDiff"]}),
    "rf_owner": ({"men": rf(max_depth=6, min_samples_leaf=20, max_features=3),
                  "women": rf(max_depth=2, min_samples_leaf=40, max_features=3)},
                 {"men": BASE4 + ["LogMasseyRankDiff"], "women": BASE4}),
    # Kat's S5: the men's LogMassey stage, now with sourced settings (scripts/stage_study.py logreg).
    # C=0.1 chosen on the 2012-16 tune seasons; the log vs log1p rank transform makes no real difference.
    "lr_s5": ({"men": logreg(0.1), "women": logreg(0.5)},
              {"men": BASE4 + ["LogMasseyRankDiff"], "women": BASE4 + ["EloDiff"]}),
    # Matt's tree with depth / leaf size chosen by Brier on the tune seasons (scripts/stage_study.py tree)
    "dt_tuned": ({"men": tree(max_depth=3, min_samples_leaf=20), "women": tree(max_depth=3, min_samples_leaf=20)},
                 {"men": BASE4 + ["LogMasseyRankDiff"], "women": BASE4}),
    # Day 2 (scripts/xgb_study.py stage A4): regularised + monotone + symmetric XGBoost, new features.
    # Settings chosen on the 2012-16 tune seasons only.
    "xgb_v2": ({"men": xgb_v2(**XGB_V2_MEN), "women": xgb_v2(**XGB_V2_WOMEN)},
               {"men": BASE4 + ["LogMasseyRankDiff"] + STRENGTH, "women": BASE4 + STRENGTH}),
    # Kat's LR with the Day-2 features; C=0.3 had the best tune-season Brier among {0.03, 0.1, 0.3, 1}
    "lr_v2": ({"men": logreg(0.3), "women": logreg(0.3)},
              {"men": BASE4 + ["LogMasseyRankDiff"] + STRENGTH, "women": BASE4 + STRENGTH}),
    # Divisha's PCA + Gaussian NB with the Day-2 features; components chosen on the tune seasons (men 5, women 2)
    "nb_v2": ({"men": pca_nb(5), "women": pca_nb(2)},
              {"men": BASE4 + ["LogMasseyRankDiff"] + STRENGTH, "women": BASE4 + STRENGTH}),
    "nb_owner": ({"men": pca_nb(4), "women": pca_nb(3)},
                 {"men": BASE4 + ["LogMasseyRankDiff", "AdjustedStrengthDiff"], "women": BASE4 + ["AdjustedStrengthDiff"]}),
}
