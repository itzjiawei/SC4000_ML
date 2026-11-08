"""Shared cross-validation for every model in the project.

Protocol (expanding window, never trains on the future):
  for each validation season S: train on all tournament games from seasons < S,
  predict the season-S tournament.

  TUNE_SEASONS   2012-2016   -> use these (and only these) to choose features,
                                hyperparameters, calibration, ensemble weights
  REPORT_SEASONS 2017-2025   -> report these in the paper; do not tune on them
  (2020 had no tournament. Women's data starts in 2010, so 2012 is the first
   season with at least two earlier tournaments to train on.)

Scores:
  per-season Brier, per-gender pooled Brier, and a COMBINED pooled Brier over
  men's + women's games together, which is how Kaggle scores a submission.

Usage:
    from src.cv import evaluate
    res = evaluate("my_lr", {"men": make_lr, "women": make_lr},
                   {"men": BASE4, "women": BASE4})
    res.summary   # one-row DataFrame
    res.oof       # one row per game, saved to results/shared_cv/oof/my_lr.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src import features

TUNE_SEASONS = [2012, 2013, 2014, 2015, 2016]
REPORT_SEASONS = [2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
ALL_SEASONS = TUNE_SEASONS + REPORT_SEASONS
OUT_DIR = features.PROJECT_ROOT / "results" / "shared_cv"

ModelFactory = Callable[[], object]  # returns a fresh, unfitted model with fit / predict_proba


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def cross_validate(make_model: ModelFactory | None, games: pd.DataFrame, feats: list[str],
                   seasons: list[int]) -> pd.DataFrame:
    """Out-of-fold predictions for one gender. make_model=None predicts 0.5 everywhere."""
    parts = []
    for s in seasons:
        train, valid = games[games.Season < s], games[games.Season == s]
        if valid.empty:
            continue
        assert train.Season.max() < s
        if make_model is None:
            p = np.full(len(valid), 0.5)
        else:
            model = make_model()
            model.fit(train[feats], train.Target)
            p = model.predict_proba(valid[feats])[:, 1]
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise AssertionError(f"invalid probabilities in season {s}")
        parts.append(valid[["Gender", "Season", "Team1", "Team2", "Target"]].assign(Pred=p, TrainGames=len(train)))
    return pd.concat(parts, ignore_index=True)


def summarise(oof: pd.DataFrame) -> dict:
    """Brier on tune / report / all seasons, per gender and combined."""
    row = {}
    for label, seasons in [("tune", TUNE_SEASONS), ("report", REPORT_SEASONS), ("all", ALL_SEASONS)]:
        part = oof[oof.Season.isin(seasons)]
        row[f"{label}_combined"] = brier(part.Target, part.Pred)
        for g in ("men", "women"):
            pg = part[part.Gender == g]
            row[f"{label}_{g}"] = brier(pg.Target, pg.Pred) if len(pg) else np.nan
    return row


def per_season(oof: pd.DataFrame) -> pd.DataFrame:
    return (oof.assign(SE=(oof.Pred - oof.Target) ** 2)
            .groupby(["Gender", "Season"]).SE.mean().unstack("Gender"))


@dataclass
class Result:
    name: str
    oof: pd.DataFrame
    summary: pd.DataFrame


def evaluate(name: str, factories: dict[str, ModelFactory | None], feats: dict[str, list[str]],
             seasons: list[int] = ALL_SEASONS, save: bool = True) -> Result:
    """Run CV for both genders (or whichever keys are given) and optionally save OOF predictions."""
    oofs = []
    for g, make in factories.items():
        _, games = features.load(g)
        oofs.append(cross_validate(make, games, feats[g], seasons))
    oof = pd.concat(oofs, ignore_index=True).assign(Model=name)
    summary = pd.DataFrame([{"Model": name, **summarise(oof),
                             "features_men": "+".join(feats.get("men", [])),
                             "features_women": "+".join(feats.get("women", []))}])
    if save:
        (OUT_DIR / "oof").mkdir(parents=True, exist_ok=True)
        oof.to_csv(OUT_DIR / "oof" / f"{name}.csv", index=False)
    return Result(name, oof, summary)


def paired_bootstrap(oof_a: pd.DataFrame, oof_b: pd.DataFrame, seasons: list[int] = REPORT_SEASONS,
                     n_boot: int = 5000, seed: int = 0) -> dict:
    """Brier(A) - Brier(B) on the same games, with a 95% bootstrap interval (negative = A better).
    Follows the paired-bootstrap comparison in notebooks/random_forest.ipynb (Preesha)."""
    key = ["Gender", "Season", "Team1", "Team2"]
    m = oof_a[key + ["Target", "Pred"]].merge(oof_b[key + ["Pred"]], on=key, suffixes=("_a", "_b"),
                                               validate="one_to_one")
    m = m[m.Season.isin(seasons)]
    d = ((m.Pred_a - m.Target) ** 2 - (m.Pred_b - m.Target) ** 2).to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return {"diff": d.mean(), "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
            "games": len(d)}
