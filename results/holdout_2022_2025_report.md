# Historical holdout: verification status

Incomplete: the merged `notebooks/logistic_regression.ipynb` contains 28 cells (indices 0–27), with men's S1–S4 only. It has no LogMasseyRankDiff/S5 implementation. The saved men's S5 OOF file exists, but does not identify C or pipeline parameters. The men's S5 source/configuration is required before running the final experiment. No existing outputs were overwritten and no ensemble holdout results have been published.

## Completed verification

- Women's S2: original four features + EloDiff; StandardScaler followed by LogisticRegression(C=0.5, penalty="l2", solver="lbfgs"). Notebook temporal CV uses max_iter=1000; final factory uses 3000. Factory reproduction matches every saved OOF probability within 4.44e-16; mean seasonal CV Brier = 0.14599602439257164.
- Men's diagnostic: original four + ln(Team1MasseyRank) − ln(Team2MasseyRank), using the notebook factory with C=0.1 gives CV Brier 0.18967606965825476, not the requested 0.187985. Maximum difference from saved S5 OOF probabilities is 0.06185275081463654. This is a rejected diagnostic, not a selected model. No C search was performed.
- Men's fixed XGBoost holdout benchmark reproduced: 0.2091868370771408 (expected approximately 0.209187), training on 1181 games.
- Women's fixed XGBoost holdout benchmark reproduced: 0.14533676207065582 (expected approximately 0.145336776), training on 693 games. Difference ~1.4e-8 is below the 1e-6 benchmark tolerance.
- Both holdouts contain 268 games, 67 in each of 2022–2025.

## Prepared implementation

`scripts/tournament_holdout_2022_2025.py` is independent of notebook kernels and resolves paths relative to its own file. It requires `--men-lr-c` from the original S5 source and verifies every saved men's and women's OOF probability (absolute tolerance 1e-10) before any final holdout scoring. If the missing S5 pipeline differs from the notebook factory, update it from source rather than guessing. It checks exact matchup/target alignment, duplicate absence, finite features, Team1 ordering, Elo merge row preservation, training-only scaler means/sample counts, and both XGBoost benchmarks. A fresh output directory is required on reruns; existing files are never replaced.

Original four, all Team1 minus Team2: SeedDiff, AvgScoringMarginDiff, WinRateDiff, EFGDifferentialDiff. Women's Elo is FinalElo(Team1) − FinalElo(Team2), merged many-to-one on Season/TeamID. Men's log Massey uses natural logarithms of existing consensus ranks. Existing feature engineering aggregates regular-season results; Elo uses regular-season games; Massey uses rankings through day 133. No feature engineering is rerun or retuned.

XGBoost men: max_depth=5, learning_rate=0.01, n_estimators=300, subsample=0.7. Women: max_depth=1, learning_rate=0.30, n_estimators=15, subsample=0.5. Both: reg_lambda=1, reg_alpha=0, random_state=42, objective="binary:logistic", eval_metric="logloss". `binary:logistic` is the existing notebook's valid XGBoost objective for binary probabilities; literal `binary` is not its objective name.

Holdout models fit once on pre-2022 tournament games (men 2003 onward, women 2010 onward; excluding 2020). Ensemble is exactly the arithmetic mean of XGB and LR probabilities, all predicting Target=1 (lower-ID Team1 wins). Overall Brier is game-weighted; CV verification uses the notebook's mean of seasonal Briers. Ensemble-minus-model columns are negative when the ensemble improves.

Versions used in diagnostic verification: Python 3.13; NumPy 2.5.4; pandas 3.1.0; scikit-learn 1.9.1; XGBoost 3.4.1. Pinned direct dependencies are in `scripts/holdout-requirements.txt`. On this Mac, XGBoost requires an OpenMP runtime; diagnostics used the installed runtime at `/Applications/MATLAB_R2026b.app/bin/maca64` via DYLD_LIBRARY_PATH.

No 2026 tournament results were accessed. Elo rows beyond 2025 are excluded before numeric parsing. This is a retrospective robustness check because 2022–2025 outcomes have previously been examined; models must not be changed based on these scores.
