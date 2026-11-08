"""Re-run every teammate's current model on the shared features and shared CV.

    python -m scripts.run_baselines            # all configs in src/models.py
    python -m scripts.run_baselines lr_owner   # just one

Writes results/shared_cv/baseline_table.csv, per_season_brier.csv and oof/<name>.csv.
"""
import sys

import pandas as pd

from src import cv
from src.models import CONFIGS


def main(names):
    rows, seasons = [], []
    for name in names:
        factories, feats = CONFIGS[name]
        res = cv.evaluate(name, factories, feats)
        rows.append(res.summary)
        seasons.append(cv.per_season(res.oof).add_prefix(f"{name}:"))
        s = res.summary.iloc[0]
        print(f"{name:18s} tune={s.tune_combined:.4f}  report={s.report_combined:.4f} "
              f"(men {s.report_men:.4f}, women {s.report_women:.4f})")
    table = pd.concat(rows, ignore_index=True).sort_values("report_combined")
    cv.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(names) == len(CONFIGS):
        table.to_csv(cv.OUT_DIR / "baseline_table.csv", index=False)
        pd.concat(seasons, axis=1).to_csv(cv.OUT_DIR / "per_season_brier.csv")
    return table


if __name__ == "__main__":
    main(sys.argv[1:] or list(CONFIGS))
