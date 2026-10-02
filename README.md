# March Machine Learning Mania 2026

SC4000 Machine Learning Group Project.

## Project Structure

- `data/raw/` - Original Kaggle competition data
- `data/processed/` - Feature-engineered datasets
- `notebooks/` - Feature engineering and model experiments
- `submissions/` - Kaggle submission files

## Current Modelling Progress

### 1. Shared Feature Dataset

The initial processed tournament datasets contain four engineered matchup
features:

1. `SeedDiff`
2. `AvgScoringMarginDiff`
3. `WinRateDiff`
4. `EFGDifferentialDiff`

Each row represents one historical NCAA tournament matchup.

- `Team1` = team with the lower TeamID
- `Team2` = team with the higher TeamID
- `Target = 1` if Team1 won
- `Target = 0` if Team2 won

The men's dataset begins from 2003 and the women's dataset begins from 2010
because these are the periods covered by the detailed regular-season data
used to construct the features.

---

### 2. Validation Strategy

Models should be compared using expanding-window temporal cross-validation
rather than a random train/test split.

Current validation seasons:

- 2017
- 2018
- 2019
- 2021

For each validation season, the model is trained using all available earlier
seasons.

For example:

| Validation Season | Men's Training Seasons |
|---|---|
| 2017 | 2003–2016 |
| 2018 | 2003–2017 |
| 2019 | 2003–2018 |
| 2021 | 2003–2019 |

2020 is excluded because the NCAA tournament was cancelled.

Please use the same temporal validation setup when comparing models or feature
sets so that results remain directly comparable across group members.

The primary evaluation metric is Brier score, where lower is better.

---

### 3. XGBoost Four-Feature Baseline

Using the original four features, the current tuned XGBoost temporal CV
results are:

| Dataset | Mean CV Brier |
|---|---:|
| Men | 0.191705 |
| Women | 0.146503 |

These provide a reference point when evaluating additional features.

---

### 4. Elo Feature Engineering

Elo ratings were generated separately for men's and women's regular-season
games.

Rather than selecting the K-factor using the same full historical dataset,
candidate K values were first compared using earlier development seasons and
then evaluated on more recent validation seasons.

Selected K-factors:

- Men: `K = 60`
- Women: `K = 100`

The resulting Elo files contain the final regular-season Elo rating for each
team in each season.

Elo feature ablation using XGBoost:

| Feature Set | Men Mean CV Brier | Women Mean CV Brier |
|---|---:|---:|
| Original 4 | 0.191705 | 0.146503 |
| Elo only | 0.228909 | 0.203368 |
| Original 4 + Elo | 0.198060 | 0.149180 |

Although Elo alone contained predictive information, adding `EloDiff` to the
existing four-feature XGBoost model increased validation Brier for both men
and women.

Elo is therefore not currently retained in the candidate XGBoost feature
sets.

---

### 5. Massey Ordinals Feature Engineering

Massey Ordinals are available for the men's competition.

A pre-tournament consensus ranking was constructed using five ranking systems
with full historical coverage:

- COL
- DOL
- MOR
- POM
- WLK

For each team and season, the latest available ranking on or before Day 133
was obtained from each system.

The five ordinal rankings were averaged:

`ConsensusRank = mean(COL, DOL, MOR, POM, WLK)`

All historical NCAA tournament team-seasons used by the model had rankings
from all five systems.

The matchup-level feature is:

`MasseyRankDiff = Team1MasseyRank - Team2MasseyRank`

Because a lower ordinal ranking represents a stronger team:

- Negative `MasseyRankDiff` → Team1 is ranked stronger
- Positive `MasseyRankDiff` → Team2 is ranked stronger

Men's feature ablation:

| Feature Set | Mean CV Brier |
|---|---:|
| Original 4 | 0.191705 |
| Elo only | 0.228909 |
| Massey only | 0.197365 |
| Original 4 + Elo | 0.198060 |
| Original 4 + Massey | 0.189886 |
| Original 4 + Massey + Elo | 0.197026 |

The current candidate men's feature set therefore includes
`MasseyRankDiff` but excludes `EloDiff`.

The improvement from Massey is modest and varies across validation seasons,
so it should not be interpreted as a large or universal improvement.

---

## Current Processed Files

All processed datasets are stored under:

`data/processed/`

### `men_processed.csv`

Men's historical NCAA tournament matchup dataset containing the original
four engineered features.

Each row represents one tournament game.

Main columns:

- `Season`
- `Team1`
- `Team2`
- `SeedDiff`
- `AvgScoringMarginDiff`
- `WinRateDiff`
- `EFGDifferentialDiff`
- `Target`

Use this file if you want to train or evaluate a men's model using the
original shared four-feature baseline.

---

### `women_processed.csv`

Women's historical NCAA tournament matchup dataset containing the same
original four engineered features.

Each row represents one tournament game.

Main columns:

- `Season`
- `Team1`
- `Team2`
- `SeedDiff`
- `AvgScoringMarginDiff`
- `WinRateDiff`
- `EFGDifferentialDiff`
- `Target`

Use this file if you want to train or evaluate a women's model using the
original shared four-feature baseline.

---

### `men_elo.csv`

Team-season-level men's Elo ratings.

Main columns:

- `Season`
- `TeamID`
- `FinalElo`

This is NOT directly a tournament matchup dataset.

To use Elo in a model:

1. Merge `FinalElo` onto the matchup dataset using `Season` and `Team1`.
2. Merge it again using `Season` and `Team2`.
3. Calculate:

   `EloDiff = Team1Elo - Team2Elo`

Positive `EloDiff` means Team1 has the higher Elo rating.

The selected men's Elo K-factor is `K = 60`.

---

### `women_elo.csv`

Team-season-level women's Elo ratings.

Main columns:

- `Season`
- `TeamID`
- `FinalElo`

This is NOT directly a tournament matchup dataset.

To use Elo:

1. Merge `FinalElo` onto the matchup dataset for Team1.
2. Merge it again for Team2.
3. Calculate:

   `EloDiff = Team1Elo - Team2Elo`

Positive `EloDiff` means Team1 has the higher Elo rating.

The selected women's Elo K-factor is `K = 100`.

---

### `men_massey.csv`

Men's historical NCAA tournament matchup dataset with the consensus Massey
feature already merged.

It contains the original four features together with:

- `Team1MasseyRank`
- `Team2MasseyRank`
- `MasseyRankDiff`

This file can therefore be used directly for models that want to test the
Massey feature.

There is currently no equivalent women's Massey feature in our modelling
pipeline.

---

## Which Dataset Should I Use?

For the shared four-feature baseline:

| Task | File |
|---|---|
| Men's model | `men_processed.csv` |
| Women's model | `women_processed.csv` |

For experiments involving Elo:

| Task | Base File | Additional File |
|---|---|---|
| Men's model + Elo | `men_processed.csv` | `men_elo.csv` |
| Women's model + Elo | `women_processed.csv` | `women_elo.csv` |

Remember that the Elo files are team-season-level files and must first be
merged onto Team1 and Team2 to construct `EloDiff`.

For men's experiments involving Massey:

`men_massey.csv`

already contains the original four features and `MasseyRankDiff`, so no
additional Massey merge is required.

For fair model comparisons, start from the same processed data and use the
same temporal validation seasons.

---

## Current Candidate XGBoost Feature Sets

### Men

Current candidate features:

- `SeedDiff`
- `AvgScoringMarginDiff`
- `WinRateDiff`
- `EFGDifferentialDiff`
- `MasseyRankDiff`

Current mean temporal CV Brier: `0.189886`

### Women

Current candidate features:

- `SeedDiff`
- `AvgScoringMarginDiff`
- `WinRateDiff`
- `EFGDifferentialDiff`

Current mean temporal CV Brier: `0.146503`

Elo is currently excluded from both candidate XGBoost feature sets based on
the temporal CV ablation results.

---

## Feature Engineering Notebooks

The feature construction can be reproduced from:

- `notebooks/feature_engineering.ipynb`
  - Original four features
  - Historical tournament matchup datasets

- `notebooks/elo_feature_engineering.ipynb`
  - Elo K-factor evaluation
  - Men's and women's final team-season Elo ratings

- `notebooks/massey_feature_engineering.ipynb`
  - Pre-tournament Massey rankings
  - Five-system consensus ranking
  - `MasseyRankDiff`

The XGBoost experiments and feature ablations are contained in:

- `notebooks/xgboost.ipynb`

---

## Next Steps

1. Perform a small hyperparameter sensitivity check for the five-feature
   men's XGBoost model.
2. Finalize the men's and women's XGBoost configurations.
3. Train the selected configurations using the appropriate historical data.
4. Construct the same features for the 2026 matchup prediction set.
5. Generate 2026 matchup probabilities.
6. Compare XGBoost against the other group members' models using the same
   temporal validation procedure.
7. Investigate an ensemble only if combining models improves validation
   Brier.