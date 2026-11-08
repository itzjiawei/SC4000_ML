# Shared pipeline (use this for all final results)

Every model in the report should be trained and scored with this code, so the
comparison tables are fair: same features, same folds, same metric.

## Setup (once)
```bash
pip install -r requirements.txt
# Kaggle CSVs either in data/raw/ or point to them:
export MMLM_RAW_DIR=/path/to/march-machine-learning-mania-2026
python -m src.features        # builds data/processed/shared/{men,women}_{team_seasons,tourney_games}.csv
```
Run everything from the repo root (`python -m ...`), not from `notebooks/`.
In a notebook, add the repo root to `sys.path` first, then `from src import features, cv`.

## Features (`src/features.py`)
- `features.load("men")` returns `(team_seasons, tourney_games)`.
- Columns available in `tourney_games`: `SeedDiff, AvgScoringMarginDiff, WinRateDiff,
  EFGDifferentialDiff, NetRatingDiff, AdjustedStrengthDiff, EloDiff`, and for men also
  `MasseyRankDiff, LogMasseyRankDiff`. They are identical (up to rounding) to the old CSVs in `data/processed/`.
- `features.BASE4` is the original four features.
- `features.make_matchups(pairs, team_seasons)` builds the same features for any pairs, e.g. 2026.

## Cross-validation (`src/cv.py`)
Expanding window: for each season S, train on all earlier tournaments and predict S.
- **Tune seasons 2012–2016**: choose features and hyperparameters here only.
- **Report seasons 2017–2025** (no 2020): put these numbers in the report.
- The main number is `report_combined`, Brier over men's and women's games pooled, like Kaggle.

```python
from src import cv, features
from sklearn.linear_model import LogisticRegression
make = lambda: LogisticRegression(C=0.1, max_iter=3000)
res = cv.evaluate("kat_lr_test", {"men": make, "women": make},
                  {"men": features.BASE4, "women": features.BASE4})
res.summary                       # tune/report/all Brier, per gender and combined
cv.paired_bootstrap(res.oof, other_oof)   # is the difference real? (CI excludes 0)
```
OOF predictions are saved to `results/shared_cv/oof/<name>.csv`. **Keep them**, because the ensemble needs them.

## Your model in the shared tables
Add your final configuration to `CONFIGS` in `src/models.py`, then:
```bash
python -m scripts.run_baselines                 # writes results/shared_cv/baseline_table.csv
python -m scripts.predict_2026 <config_name>    # 2026 predictions for seeded pairs
python scripts/make_submission.py --sample $MMLM_RAW_DIR/SampleSubmissionStage2.csv \
    --out submissions/<name>.csv --fill-missing 0.5 results/predictions_2026/<config_name>.csv
```
`make_submission.py` checks row count (132,133), duplicates, missing IDs, range [0,1] and the
exact ID order of the sample file. It refuses to write a broken file.
