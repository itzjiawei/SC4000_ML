# Feature-stage studies on the shared pipeline: Logistic Regression and Decision Tree

Protocol: `src/cv.py`. Hyperparameters are chosen on the 2012–2016 tune seasons only. Brier is reported on
2017–2025 (no 2020): 535 men's and 520 women's games. "vs S1" is a paired bootstrap on the same games;
negative means better, and the result is only clear if the 95% interval excludes 0.
Reproduce with `python -m scripts.stage_study logreg` and `python -m scripts.stage_study tree`.

## Logistic Regression (Kat)

**The missing S5 source has been found.** `results/logreg_men_s5_oof.csv` is produced by
`scripts/generate_logreg_oof.py`. It uses `LogMasseyRankDiff = log1p(Team1Rank) − log1p(Team2Rank)` and **C = 100**
(StandardScaler → LogisticRegression, lbfgs, max_iter=1000). Recomputed, it matches all 267 saved probabilities
to within 4e-16, with mean CV Brier 0.187985. The holdout report could not reproduce it because it assumed `log` and C = 0.1.
`scripts/tournament_holdout_2022_2025.py` (line 79) still builds the feature with `np.log`, so its strict
OOF check will fail until it uses `log1p` for the LR feature.

| Gender | Stage | Chosen C | Report Brier | vs S1 [95% CI] |
|---|---|---|---|---|
| Men | S1 base4 | 0.1 | 0.1979 | – |
| Men | S2 +Elo | 0.1 | 0.1973 | −0.0006 [−0.0033, +0.0021] |
| Men | S3 +Massey+Elo | 0.1 | 0.1954 | −0.0025 [−0.0061, +0.0011] |
| Men | S4 +Massey | 0.1 | 0.1958 | −0.0021 [−0.0046, +0.0004] |
| Men | **S5 +LogMassey (log1p)** | 0.1 | **0.1926** | **−0.0053 [−0.0083, −0.0023]** |
| Men | S5b +LogMassey (log) | 0.1 | 0.1927 | −0.0052 [−0.0083, −0.0022] |
| Women | S1 base4 | 2 (0.1: 0.1445) | 0.1464 | – |
| Women | S2 +Elo | 2 (0.1: 0.1439) | 0.1456 | −0.0008 [−0.0034, +0.0018] |

- **LogMassey is the only feature that clearly helps LR.** Raw Massey and Elo do not.
- log vs log1p makes no difference (0.1926 vs 0.1927). Using `log` keeps LR consistent with the other models.
- With the shared protocol C = 0.1 wins on the tune seasons, so the C = 100 in the script was not a tuned value.
  For women, C = 2 won on the tune seasons but did worse on the report seasons than C = 0.1. Over this range C barely matters,
  and the "best" value is noise.
- New team config `lr_s5` (men base4+LogMassey C=0.1, women base4+Elo C=0.5): **combined 0.1690**,
  the best single model so far (Kat's previous config `lr_owner`: 0.1703).

## Decision Tree (Matt)

**Bug confirmed and corrected.** In `decision_tree.ipynb` the S1 cells for men and women train
`RandomForestClassifier(n_estimators=100, max_depth=5)`, while S2–S4 train `DecisionTreeClassifier(max_depth=5)`.
The original conclusion ("S1 is best; Massey and Elo make it worse") compared a forest against single trees:

| Gender | S1 as coded (Random Forest) | S1 corrected (Decision Tree, depth 5) |
|---|---|---|
| Men | 0.2000 | 0.2179 |
| Women | 0.1501 | 0.1822 |

Corrected stages with real decision trees. "Untuned" is Matt's depth 5 with no leaf limit; "tuned" picks depth from {2,3,4,5,6,8}
and min_samples_leaf from {1,10,20,40,80} by Brier on 2012–2016:

| Gender | Stage | Untuned | Tuned (setting) | Tuned vs S1 [95% CI] |
|---|---|---|---|---|
| Men | S1 base4 | 0.2179 | 0.2050 (d4, leaf 80) | – |
| Men | S2 +Massey | 0.2148 | 0.2049 (d3, leaf 1) | −0.0001 [−0.0096, +0.0094] |
| Men | S3 +Elo | 0.2227 | 0.2141 (d2, leaf 1) | **+0.0092 [+0.0025, +0.0156]** (worse) |
| Men | S4 +Massey+Elo | 0.2178 | 0.2024 (d3, leaf 20) | −0.0026 [−0.0116, +0.0058] |
| Men | S5 +LogMassey (new) | 0.2029 | **0.1948** (d3, leaf 20) | −0.0102 [−0.0212, +0.0002] |
| Women | S1 base4 | 0.1822 | **0.1552** (d3, leaf 20) | – |
| Women | S2 +Elo | 0.1741 | 0.1621 (d5, leaf 20) | +0.0068 [−0.0032, +0.0169] |

- With the model held fixed, **Massey does not hurt the tree** (S2 ≈ S1). Elo does hurt men (S3, clearly worse).
- LogMassey, which was never tried for the tree, gives the largest gain (−0.010; the interval just touches 0).
- **Limiting tree size is what matters most.** Choosing depth and leaf size by Brier improves the tree by 0.025 combined
  (`dt_tuned` vs `dt_base4`, CI [−0.036, −0.015]). Unlimited leaves give probabilities near 0 or 1, which Brier punishes.
- Even tuned, the tree (`dt_tuned`, combined 0.1753) is no better than LR on SeedDiff alone (0.1726). For the report it works
  as the "simple, interpretable but high-variance" baseline that motivates RF and XGBoost.
- Depth 5 was originally chosen by **accuracy**. Accuracy ignores how confident a prediction is, so it is the wrong criterion for Brier.
