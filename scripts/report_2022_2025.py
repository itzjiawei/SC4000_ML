"""2022-2025 tournament results from the shared CV. Replaces scripts/tournament_holdout_2022_2025.py.

Uses the saved out-of-fold predictions in results/shared_cv/oof/, so every model and blend
is scored on the same games with the same features. Season S is always predicted by a model
trained only on tournaments before S (2023 is trained through 2022, and so on).
The old script instead trained once on games up to 2021.

    python -m scripts.report_2022_2025 xgb_owner lr_s5 --blend xgb_owner lr_s5

Writes results/shared_cv/report_2022_2025.csv
"""
import argparse

import pandas as pd

from src import cv

SEASONS = [2022, 2023, 2024, 2025]
KEY = ["Gender", "Season", "Team1", "Team2", "Target"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(cv.OUT_DIR / "oof" / f"{name}.csv")[KEY + ["Pred"]]


def blend(names: list[str]) -> pd.DataFrame:
    parts = [load(n).rename(columns={"Pred": n}) for n in names]
    m = parts[0]
    for p in parts[1:]:
        m = m.merge(p, on=KEY, validate="one_to_one")
    return m[KEY].assign(Pred=m[names].mean(axis=1))


def table(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, oof in models.items():
        o = oof[oof.Season.isin(SEASONS)]
        row = {"Model": name}
        for g in ("men", "women"):
            og = o[o.Gender == g]
            for s in SEASONS:
                og_s = og[og.Season == s]
                row[f"{g}_{s}"] = cv.brier(og_s.Target, og_s.Pred)
            row[f"{g}_2022_2025"] = cv.brier(og.Target, og.Pred)
        row["combined_2022_2025"] = cv.brier(o.Target, o.Pred)
        row["games"] = len(o)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--blend", nargs="+", help="names to average with equal weights")
    a = ap.parse_args()
    models = {n: load(n) for n in a.models}
    if a.blend:
        models["blend(" + "+".join(a.blend) + ")"] = blend(a.blend)
    t = table(models)
    t.to_csv(cv.OUT_DIR / "report_2022_2025.csv", index=False)
    cols = ["Model", "men_2022_2025", "women_2022_2025", "combined_2022_2025", "games"]
    print(t[cols].round(4).to_string(index=False))
    if a.blend:
        b = models["blend(" + "+".join(a.blend) + ")"]
        for n in a.blend:
            r = cv.paired_bootstrap(b, models[n], seasons=SEASONS)
            print(f"blend - {n}: {r['diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] on {r['games']} games")
