# Day 3: blending, clipping and calibration

Every choice is made on the 2012–2016 tune seasons using saved out-of-fold predictions. The 2017–2025 report seasons are only measured.
Reproduce with `python -m scripts.blend --predict`. The recipe is in `final_recipe.json`.

## Candidates (all on the shared Day-2 features except RF)
| Model | Tune | Report combined |
|---|---|---|
| `lr_v2` logistic regression (Kat) | 0.1610 | 0.1673 |
| `xgb_v2` XGBoost (JW) | 0.1619 | 0.1678 |
| `nb_v2` PCA + Gaussian NB (Divisha; PCA 5 men / 2 women, chosen on tune) | 0.1638 | 0.1682 |
| `rf_owner` random forest (Preesha, original features) | 0.1819 | 0.1716 |

## Steps
| Step | Chosen on tune seasons | Tune | Report combined |
|---|---|---|---|
| equal-weight average of all 4 | – | 0.1635 | 0.1666 |
| **1. tuned weights** (simplex grid, step 0.05; probability vs logit averaging) | **0.6 LR + 0.4 XGB**, probability averaging | 0.1601 | **0.1666** |
| 2. + clipping to [c, 1−c], c ∈ {0, .005, .01, .02, .03, .05} | c = 0 (no clipping) | 0.1601 | 0.1666 |
| 3. + temperature t ∈ [0.8, 1.2] | t = 1.0 (no change) | 0.1601 | 0.1666 |

Final blend vs others on the report games (95% paired bootstrap):
- vs `lr_v2` alone: −0.0007 [−0.0021, +0.0006]
- vs JW's original XGBoost: −0.0034 [−0.0082, +0.0015]
- **vs LR on SeedDiff alone: −0.0060 [−0.0112, −0.0005]. This is the first configuration that beats the seed baseline with an interval excluding 0.**

## Findings
- **Blending helps a little and consistently.** NB and RF get weight 0: their errors overlap with LR's. XGBoost adds something different.
- **Clipping does not help.** Clipping LR alone at 0.01–0.05 changes report Brier by +0.0000 to +0.0002. The most extreme predictions
  (for example a women's 1-seed vs a 16-seed) are almost always right. The blend is also naturally less extreme
  (2026 range 0.013–0.987, compared with 0.000–1.000 for LR alone), so a single huge upset now costs less.
- **The models are already well calibrated.** Temperature scaling picks t = 1. This matches JW's earlier finding that Platt scaling hurt.
- **A single fixed recipe is used for everything.** The same weights go into the 2026 file, which was chosen by CV before seeing any leaderboard score for it.

2026 file: `submissions/2026_blend_lr60_xgb40.csv` (validated: 132,133 rows, sample ID order).
