"""Day 3: blend models, clip, calibrate, all chosen on the 2012-16 tune seasons, and build the final 2026 file.

Steps (each is kept only if it lowers the TUNE-season Brier; report seasons are just measured):
  1. blend weights over CANDIDATES (grid on the simplex, step 0.05), averaging in probability or logit space
  2. clipping to [c, 1-c]
  3. temperature: p' = sigmoid(t * logit(p)); t < 1 softens, t > 1 sharpens

    python -m scripts.blend            # study + writes results/shared_cv/final_recipe.json
    python -m scripts.blend --predict  # also applies the recipe to results/predictions_2026/*.csv
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd

from src import cv, features

CANDIDATES = ["lr_v2", "xgb_v2", "nb_v2", "rf_owner"]
KEY = ["Gender", "Season", "Team1", "Team2"]
CLIPS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
TEMPS = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
RECIPE = cv.OUT_DIR / "final_recipe.json"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def combine(P: np.ndarray, w: np.ndarray, space: str) -> np.ndarray:
    return P @ w if space == "prob" else sigmoid(logit(P) @ w)


def apply(P, recipe):
    p = combine(P, np.array(recipe["weights"]), recipe["space"])
    p = sigmoid(recipe["temperature"] * logit(p))
    c = recipe["clip"]
    return np.clip(p, c, 1 - c) if c > 0 else p


def weight_grid(n, step=0.05):
    k = int(round(1 / step))
    for combo in itertools.product(range(k + 1), repeat=n - 1):
        if sum(combo) <= k:
            yield np.array(list(combo) + [k - sum(combo)]) / k


def load_oof(names):
    m = None
    for n in names:
        o = pd.read_csv(cv.OUT_DIR / "oof" / f"{n}.csv")[KEY + ["Target", "Pred"]].rename(columns={"Pred": n})
        m = o if m is None else m.merge(o.drop(columns="Target"), on=KEY, validate="one_to_one")
    return m


def score(df, p, seasons):
    mask = df.Season.isin(seasons).to_numpy()
    return cv.brier(df.Target[mask], p[mask])


def study():
    df = load_oof(CANDIDATES)
    P = df[CANDIDATES].to_numpy()
    tune = lambda p: score(df, p, cv.TUNE_SEASONS)
    report = lambda p: score(df, p, cv.REPORT_SEASONS)
    rows = []

    def log(step, recipe):
        p = apply(P, recipe)
        oof = df[KEY + ["Target"]].assign(Pred=p)
        rows.append({"Step": step, "tune": tune(p), "report": report(p), **{k: str(v) for k, v in recipe.items()},
                     "oof": oof})
        print(f"{step:42s} tune={tune(p):.4f} report={report(p):.4f}  {json.dumps(recipe)}")

    base = {"weights": [1.0 if c == "lr_v2" else 0.0 for c in CANDIDATES], "space": "prob", "temperature": 1.0, "clip": 0.0}
    for i, c in enumerate(CANDIDATES):
        log(f"single: {c}", {**base, "weights": list(np.eye(len(CANDIDATES))[i])})
    log("equal-weight average", {**base, "weights": [1 / len(CANDIDATES)] * len(CANDIDATES)})

    # 1. weights (and averaging space) on tune seasons
    best = None
    for space in ("prob", "logit"):
        for w in weight_grid(len(CANDIDATES)):
            r = {**base, "weights": [round(float(x), 2) for x in w], "space": space}
            s = tune(apply(P, r))
            if best is None or s < best[0]:
                best = (s, r)
    recipe = best[1]
    log("1. tuned blend weights", recipe)

    # 2. clipping, 3. temperature: each chosen on tune seasons given the previous steps
    for key, values, label in [("clip", CLIPS, "2. + clipping"), ("temperature", TEMPS, "3. + temperature")]:
        recipe = min(({**recipe, key: v} for v in values), key=lambda r: tune(apply(P, r)))
        log(label, recipe)

    # How much does clipping matter for the most extreme single model?
    for c in CLIPS[1:]:
        r = {**base, "clip": c}
        print(f"   lr_v2 alone clipped at {c}: tune={tune(apply(P, r)):.4f} report={report(apply(P, r)):.4f}")

    # Uncertainty vs the best single model and vs the seed-only reference
    final = rows[-1]["oof"]
    for ref in ["lr_v2", "ref_lr_seed_only", "xgb_owner"]:
        b = cv.paired_bootstrap(final, pd.read_csv(cv.OUT_DIR / "oof" / f"{ref}.csv"))
        print(f"final - {ref:17s} {b['diff']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")

    final.assign(Model="final_blend").to_csv(cv.OUT_DIR / "oof" / "final_blend.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "oof"} for r in rows]).to_csv(cv.OUT_DIR / "blend_study.csv", index=False)
    RECIPE.write_text(json.dumps({"models": CANDIDATES, **recipe}, indent=2))
    print("recipe ->", RECIPE)


def predict():
    recipe = json.loads(RECIPE.read_text())
    m = None
    for n in recipe["models"]:
        p = pd.read_csv(features.PROJECT_ROOT / "results" / "predictions_2026" / f"{n}.csv").rename(columns={"Pred": n})
        m = p if m is None else m.merge(p, on=["Season", "Team1", "Team2"], validate="one_to_one")
    m["Pred"] = apply(m[recipe["models"]].to_numpy(), recipe)
    out = features.PROJECT_ROOT / "results" / "predictions_2026" / "final_blend.csv"
    m[["Season", "Team1", "Team2", "Pred"]].to_csv(out, index=False)
    print(f"wrote {out}: {len(m)} pairs, Pred {m.Pred.min():.3f}-{m.Pred.max():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    study()
    if a.predict:
        predict()
