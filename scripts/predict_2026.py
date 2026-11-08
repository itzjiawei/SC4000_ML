"""Train a configuration from src/models.py on all tournaments up to 2025 and predict
every 2026 matchup between seeded teams (the only games Kaggle can score).

    python -m scripts.predict_2026 lr_owner --out results/predictions_2026/lr_owner.csv

Output columns: Season, Team1, Team2, Pred  (Pred = P(lower TeamID wins)).
Pairs involving an unseeded team are not predicted; fill them with
scripts/make_submission.py --fill-missing 0.5 (those games are never played).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import features
from src.models import CONFIGS


def predict(name: str, season: int = 2026) -> pd.DataFrame:
    factories, feats = CONFIGS[name]
    out = []
    for g, make in factories.items():
        teams, games = features.load(g)
        train = games[games.Season < season]
        pairs = features.submission_pairs(g)
        pairs = features.make_matchups(pairs[pairs.Season == season], teams)
        pairs = pairs[pairs.Team1Seed.notna() & pairs.Team2Seed.notna()]
        X = pairs[feats[g]]
        if X.isna().any().any():
            raise AssertionError(f"{g}: missing 2026 features for seeded teams: {X.columns[X.isna().any()].tolist()}")
        if make is None:
            p = np.full(len(X), 0.5)
        else:
            model = make()
            model.fit(train[feats[g]], train.Target)
            p = model.predict_proba(X)[:, 1]
        out.append(pairs[["Season", "Team1", "Team2"]].assign(Pred=p))
        print(f"{g}: trained on {len(train)} games ({train.Season.min()}-{train.Season.max()}), "
              f"predicted {len(X)} seeded 2026 pairs, Pred range {p.min():.3f}-{p.max():.3f}")
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=list(CONFIGS))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    out = a.out or features.PROJECT_ROOT / "results" / "predictions_2026" / f"{a.config}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    predict(a.config).to_csv(out, index=False)
    print("wrote", out)
