# Day 2: XGBoost study and new features

Protocol: `src/cv.py`. Every setting is chosen on the 2012–2016 tune seasons only. Results are reported on 2017–2025 (1,055 games;
"combined" pools men and women like Kaggle). Intervals are 95% paired-bootstrap intervals on the same games.
Reproduce with `python -m scripts.tune_elo` and then `python -m scripts.xgb_study`.

## New features (src/features.py)
- **Improved Elo (`EloV2Diff`)**: ratings carry over between seasons (85% kept). A win counts for more the bigger the margin,
  using FiveThirtyEight's published margin multiplier (acknowledge in the report). The rating is the snapshot just before each tournament,
  and that season's tournament games only affect later seasons. Settings come from `scripts/tune_elo.py`, which picks them
  by Elo's own Brier on **regular-season** games up to 2016, so no tournament results are used
  (men K=45, women K=60). Correlation with the tournament result: men 0.39 → 0.49, women 0.47 → 0.61, compared with the old reset-each-season Elo.
- **Last-10 margin (`Last10MarginDiff`)**: average margin over each team's last 10 regular-season games.
- **Strength of schedule (`SOSDiff`)**: average AdjustedStrength of the opponents a team played.
- `AdjustedStrengthDiff` (already in the repo) is included in the same "strength" group.

## Staged study: each row adds one thing to the row above
| Stage | Tune | Report combined | Men | Women | vs previous [95% CI] |
|---|---|---|---|---|---|
| A0 JW's tuned XGBoost* | 0.1788 | 0.1700 | 0.1928 | 0.1466 | – |
| A1 + regularised grid (depth 1–3, min_child_weight, L2) | 0.1725 | 0.1716 | 0.1928 | 0.1499 | +0.0016 [−0.0020, +0.0052] |
| A2 + monotone constraints | 0.1726 | 0.1714 | 0.1934 | 0.1487 | −0.0002 [−0.0022, +0.0018] |
| A3 + symmetric training data | 0.1685 | 0.1706 | 0.1918 | 0.1487 | −0.0008 [−0.0027, +0.0010] |
| **A4 + new strength/form features** (`xgb_v2`) | 0.1619 | **0.1678** | 0.1903 | 0.1448 | −0.0027 [−0.0064, +0.0008] |
| A5 one model for both genders + IsWomen flag | 0.1632 | 0.1674 | 0.1914 | 0.1429 | −0.0004 [−0.0028, +0.0021] |

\*A0's settings were tuned by JW on 2017/18/19/21, which are inside the report seasons, so A0's report score is
optimistic. A1–A5 never see the report seasons when choosing settings. A1 looking slightly worse than A0 is therefore not
evidence against regularisation.

## Ablations of A4 (settings fixed; positive = removing the feature hurts)
| Removed | Report combined | Change [95% CI] |
|---|---|---|
| strength of schedule | 0.1688 | +0.0010 [−0.0013, +0.0032] |
| LogMassey (men) | 0.1688 | +0.0009 [−0.0006, +0.0025] |
| improved Elo | 0.1681 | +0.0003 [−0.0015, +0.0021] |
| SeedDiff | 0.1681 | +0.0002 [−0.0005, +0.0010] |
| last-10 margin | 0.1676 | −0.0002 [−0.0006, +0.0002] |
| margin / win% / eFG (original 3) | 0.1675 | −0.0003 [−0.0009, +0.0002] |
| AdjustedStrength | 0.1670 | −0.0008 [−0.0022, +0.0005] |

## What it means
1. **The gain comes from the features, not the XGBoost tricks.** The new strength features gave the biggest step (A3 → A4).
   They also help logistic regression: `lr_v2` (C = 0.3, chosen on the tune seasons) reaches **0.1673**, the best model so far,
   compared with 0.1690 for `lr_s5`.
2. **The best XGBoost for men is depth 1** (no interactions between features). That is nearly an additive model, like LR. It explains
   why XGBoost does not beat LR here: the tournament signal is mostly "how much stronger is A than B".
3. **Symmetric data** made the largest tune-season gain (0.1726 → 0.1685) but only a small report gain. **Monotone
   constraints** change little in Brier, but they guarantee sensible behaviour (a better seed can never lower the win probability).
4. **One model with a gender flag** (A5) is no better than two models (−0.0004, interval includes 0). We keep two models because it's simpler.
5. **The features are highly redundant.** No single feature removal is significant, because they all measure team strength.
6. Compared with LR on SeedDiff alone: `xgb_v2` −0.0048 [−0.0102, +0.0010], `lr_v2` −0.0053 [−0.0106, +0.0005].
   We are getting close to a clear win, and blending them is the next step (Day 3).
