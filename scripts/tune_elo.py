"""Choose the improved-Elo settings (K, season carry-over, margin of victory) WITHOUT tournament labels:
score Elo's own pre-game win probability on regular-season games from the first detailed season to 2016.
Later seasons and all tournament results are not used for this choice.

    python -m scripts.tune_elo
Copy the winning settings into ELO_V2_PARAMS in src/features.py.
"""
import itertools

import numpy as np
import pandas as pd

from src import cv, features

GRID = list(itertools.product([20, 32, 45, 60, 80, 100], [0.5, 0.65, 0.75, 0.85, 1.0], [False, True]))

if __name__ == "__main__":
    rows = []
    for g in ("men", "women"):
        for k, keep, mov in GRID:
            _, games = features.elo_v2(g, k=k, keep=keep, mov=mov, return_games=True)
            sel = games[games.Season.between(features.FIRST_SEASON[g], 2016)]
            rows.append({"Gender": g, "k": k, "keep": keep, "mov": mov, "games": len(sel),
                         "RegSeasonBrier": float(np.mean((1 - sel.PWinner) ** 2))})
    t = pd.DataFrame(rows)
    cv.OUT_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(cv.OUT_DIR / "elo_v2_tuning.csv", index=False)
    for g, part in t.groupby("Gender"):
        print(part.sort_values("RegSeasonBrier").head(5).round(5).to_string(index=False), "\n")
