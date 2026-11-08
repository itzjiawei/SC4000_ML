"""Turn any model's predictions into a validated Kaggle submission for
March Machine Learning Mania 2026.

Input predictions CSV: one of these column layouts
  (a) ID, Pred                      ID = "2026_<lowerID>_<higherID>"
  (b) Season, Team1, Team2, Pred    Pred = P(Team1 beats Team2)
      Team1/Team2 may be in either order; rows with Team1 > Team2 are
      flipped automatically (IDs swapped, Pred -> 1 - Pred).
Men's (1xxx) and women's (3xxx) predictions can be in one file or in several
files given one after another.

Usage
  python make_submission.py --sample path/to/SampleSubmissionStage2.csv \
      --out submissions/xgb_lr_blend.csv  men_preds.csv women_preds.csv \
      [--clip 0.02 0.98] [--fill-missing 0.5]

By default the script REFUSES to write a file if any sample ID has no prediction.
--fill-missing fills those gaps with a constant and prints how many it filled.
"""
import argparse
import sys

import numpy as np
import pandas as pd


def to_id_pred(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if {"ID", "Pred"}.issubset(df.columns):
        out = df[["ID", "Pred"]].copy()
        parts = out["ID"].str.split("_", expand=True).astype(int)
        flip = parts[1] > parts[2]
        if flip.any():
            print(f"[{source}] flipping {flip.sum()} IDs written higher_lower")
            out.loc[flip, "ID"] = (parts[0].astype(str) + "_" + parts[2].astype(str) + "_" + parts[1].astype(str))[flip]
            out.loc[flip, "Pred"] = 1 - out.loc[flip, "Pred"]
        return out
    if {"Season", "Team1", "Team2", "Pred"}.issubset(df.columns):
        d = df[["Season", "Team1", "Team2", "Pred"]].copy()
        flip = d["Team1"] > d["Team2"]
        if flip.any():
            print(f"[{source}] flipping {flip.sum()} rows where Team1 > Team2")
            d.loc[flip, ["Team1", "Team2"]] = d.loc[flip, ["Team2", "Team1"]].to_numpy()
            d.loc[flip, "Pred"] = 1 - d.loc[flip, "Pred"]
        d["ID"] = d["Season"].astype(int).astype(str) + "_" + d["Team1"].astype(int).astype(str) + "_" + d["Team2"].astype(int).astype(str)
        return d[["ID", "Pred"]]
    raise ValueError(f"{source}: need columns ID,Pred or Season,Team1,Team2,Pred; got {list(df.columns)}")


def validate(sub: pd.DataFrame, sample: pd.DataFrame) -> list[str]:
    errors = []
    if list(sub.columns) != ["ID", "Pred"]:
        errors.append(f"columns must be exactly ['ID','Pred'], got {list(sub.columns)}")
    if len(sub) != len(sample):
        errors.append(f"row count {len(sub)} != sample {len(sample)}")
    if sub["ID"].duplicated().any():
        errors.append(f"{sub['ID'].duplicated().sum()} duplicate IDs")
    if sub["Pred"].isna().any():
        errors.append(f"{sub['Pred'].isna().sum()} missing Pred values")
    if not np.isfinite(sub["Pred"].to_numpy(dtype=float)).all():
        errors.append("non-finite Pred values")
    if ((sub["Pred"] < 0) | (sub["Pred"] > 1)).any():
        errors.append(f"{((sub['Pred'] < 0) | (sub['Pred'] > 1)).sum()} Pred values outside [0,1]")
    missing = set(sample["ID"]) - set(sub["ID"])
    extra = set(sub["ID"]) - set(sample["ID"])
    if missing:
        errors.append(f"{len(missing)} sample IDs missing, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"{len(extra)} IDs not in sample, e.g. {sorted(extra)[:3]}")
    if not errors and not (sub["ID"].to_numpy() == sample["ID"].to_numpy()).all():
        errors.append("ID order differs from sample (should not happen after reindexing)")
    return errors


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", help="one or more prediction CSVs")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", nargs=2, type=float, metavar=("LO", "HI"))
    ap.add_argument("--fill-missing", type=float, default=None)
    a = ap.parse_args(argv)

    sample = pd.read_csv(a.sample)
    preds = pd.concat([to_id_pred(pd.read_csv(p), p) for p in a.preds], ignore_index=True)

    dup = preds["ID"].duplicated(keep=False)
    if dup.any():
        sys.exit(f"ERROR: {preds.loc[dup, 'ID'].nunique()} IDs predicted more than once across input files")

    extra = set(preds["ID"]) - set(sample["ID"])
    if extra:
        sys.exit(f"ERROR: {len(extra)} predicted IDs are not in the sample file, e.g. {sorted(extra)[:3]}")

    sub = sample[["ID"]].merge(preds, on="ID", how="left")  # keeps exact sample order
    n_missing = sub["Pred"].isna().sum()
    if n_missing:
        if a.fill_missing is None:
            sys.exit(f"ERROR: {n_missing} sample IDs have no prediction (use --fill-missing 0.5 to fill)")
        print(f"WARNING: filled {n_missing} missing predictions with {a.fill_missing}")
        sub["Pred"] = sub["Pred"].fillna(a.fill_missing)

    if a.clip:
        lo, hi = a.clip
        changed = ((sub["Pred"] < lo) | (sub["Pred"] > hi)).sum()
        sub["Pred"] = sub["Pred"].clip(lo, hi)
        print(f"clipped {changed} predictions to [{lo}, {hi}]")

    errors = validate(sub, sample)
    if errors:
        sys.exit("VALIDATION FAILED:\n  - " + "\n  - ".join(errors))

    sub.to_csv(a.out, index=False)
    ids = sub["ID"].str.split("_", expand=True).astype(int)
    print(f"OK: wrote {a.out}")
    print(f"  rows={len(sub)}  men={(ids[1] < 3000).sum()}  women={(ids[1] >= 3000).sum()}")
    print(f"  Pred min={sub['Pred'].min():.4f} mean={sub['Pred'].mean():.4f} max={sub['Pred'].max():.4f}")


if __name__ == "__main__":
    main()
