"""Fixed historical holdout; run from any directory with Python 3.13.

First verifies reconstructed LR against every saved OOF prediction. No tuning.
See results/holdout_2022_2025_report.md for source provenance and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
BASE = ['SeedDiff', 'AvgScoringMarginDiff', 'WinRateDiff', 'EFGDifferentialDiff']
KEYS = ['Season', 'Team1', 'Team2']
CV = [2017, 2018, 2019, 2021]
HOLDOUT = [2022, 2023, 2024, 2025]
XGB_COMMON = dict(reg_lambda=1, reg_alpha=0, random_state=42,
                  objective='binary:logistic', eval_metric='logloss')
CONFIG = {
    'men': dict(start=2003, C=None, lr_features=BASE + ['LogMasseyRankDiff'],
                xgb_features=BASE + ['LogMasseyRankDiff'], stage='s5',
                xgb=dict(max_depth=5, learning_rate=0.01, n_estimators=300, subsample=0.7),
                cv_expected=0.187985, holdout_expected=0.209187),
    'women': dict(start=2010, C=0.5, lr_features=BASE + ['EloDiff'],
                  xgb_features=BASE.copy(), stage='s2',
                  xgb=dict(max_depth=1, learning_rate=0.30, n_estimators=15, subsample=0.5),
                  cv_expected=0.145996, holdout_expected=0.145336776),
}


def read_eligible(path: Path, start: int) -> pd.DataFrame:
    # Filter on the textual Season field BEFORE parsing any later-season values.
    # This also excludes 2026 Elo ratings; no 2026 results files are opened.
    import csv
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if start <= int(row['Season']) <= 2025
                and int(row['Season']) != 2020]
    return pd.DataFrame(rows).apply(pd.to_numeric)


def validate(df: pd.DataFrame, features: list[str]) -> None:
    assert not df.empty
    assert not df.duplicated(KEYS).any(), 'Duplicate matchup'
    assert (df.Team1 < df.Team2).all(), 'Team1 must be the lower ID'
    assert df.Target.isin([0, 1]).all()
    assert np.isfinite(df[KEYS + features + ['Target']].to_numpy()).all()


def exact_match(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    assert not left.duplicated(KEYS).any() and not right.duplicated(KEYS).any()
    merged = left.merge(right, on=KEYS, how='outer', validate='one_to_one',
                        indicator=True, suffixes=('', '_saved'))
    assert len(merged) == len(left) == len(right)
    assert merged['_merge'].eq('both').all(), 'Matchup sets differ'
    assert merged.Target.eq(merged.Target_saved).all(), 'Targets differ'
    return merged


def load_data(sex: str) -> pd.DataFrame:
    cfg = CONFIG[sex]
    path = ROOT / 'data/processed' / ('men_massey.csv' if sex == 'men' else 'women_processed.csv')
    df = read_eligible(path, cfg['start'])
    original = df.copy()
    if sex == 'men':
        assert (df[['Team1MasseyRank', 'Team2MasseyRank']] > 0).all().all()
        df['LogMasseyRankDiff'] = np.log(df.Team1MasseyRank) - np.log(df.Team2MasseyRank)
    else:
        elo = read_eligible(ROOT / 'data/processed/women_elo.csv', cfg['start'])
        assert not elo.duplicated(['Season', 'TeamID']).any()
        for side in [1, 2]:
            df = df.merge(elo.rename(columns={'TeamID': f'Team{side}',
                                              'FinalElo': f'Team{side}Elo'}),
                          on=['Season', f'Team{side}'], how='left',
                          validate='many_to_one', sort=False)
        df['EloDiff'] = df.Team1Elo - df.Team2Elo
    pd.testing.assert_frame_equal(df[original.columns], original, check_exact=True)
    validate(df, list(dict.fromkeys(cfg['lr_features'] + cfg['xgb_features'])))
    assert set(df.Season) == set(range(cfg['start'], 2026)) - {2020}
    assert df[df.Season.isin(HOLDOUT)].groupby('Season').size().eq(67).all()
    return df


def make_lr(C: float) -> Pipeline:
    # Exact factory in logistic_regression.ipynb cell 25.
    # Men S5 configuration must be supplied from the missing source.
    return Pipeline([
        ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True)),
        ('logreg', LogisticRegression(C=C, penalty='l2', solver='lbfgs',
                                      max_iter=3000, tol=1e-4, fit_intercept=True,
                                      intercept_scaling=1, class_weight=None,
                                      random_state=None, dual=False, verbose=0,
                                      warm_start=False, l1_ratio=0)),
    ])


def fit_lr(train: pd.DataFrame, features: list[str], C: float) -> Pipeline:
    model = make_lr(C)
    model.fit(train[features], train.Target)
    scaler = model.named_steps['scaler']
    assert scaler.n_samples_seen_ == len(train)
    np.testing.assert_allclose(scaler.mean_, train[features].mean().to_numpy(), atol=1e-12)
    assert (model.named_steps['logreg'].n_iter_ < 3000).all()
    return model


def probability(model, frame, features):
    np.testing.assert_array_equal(model.classes_, [0, 1])
    p = model.predict_proba(frame[features])[:, 1]
    assert len(p) == len(frame) and np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    return p


def verify_lr(sex: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cfg = CONFIG[sex]
    folds = []
    for year in CV:
        train, valid = df[df.Season < year], df[df.Season == year]
        assert train.Season.max() < year
        model = fit_lr(train, cfg['lr_features'], cfg['C'])
        out = valid[KEYS + ['Target']].copy()
        out['LRProbability'] = probability(model, valid, cfg['lr_features'])
        folds.append(out)
    oof = pd.concat(folds, ignore_index=True)
    saved = pd.read_csv(ROOT / f'results/logreg_{sex}_{cfg["stage"]}_oof.csv')
    assert set(saved.Season) == set(CV)
    assert (saved.TrainingThroughSeason < saved.Season).all()
    aligned = exact_match(oof, saved)
    delta = float(abs(aligned.LRProbability - aligned.PredictedProbability).max())
    # Stronger than reproducing a rounded aggregate; do not silently continue.
    np.testing.assert_allclose(aligned.LRProbability, aligned.PredictedProbability,
                               atol=1e-10, rtol=0)
    scores = oof.assign(Error=(oof.LRProbability - oof.Target)**2).groupby('Season').Error.mean()
    mean = float(scores.mean())  # notebook uses equally weighted seasons
    assert abs(mean - cfg['cv_expected']) < 1e-6
    print(f'{sex} LR OOF mean={mean:.12f}; max saved probability difference={delta:.3g}', flush=True)
    return oof, dict(mean_season_brier=mean, pooled_brier=float(brier_score_loss(oof.Target, oof.LRProbability)),
                     max_saved_probability_difference=delta, by_season=scores.to_dict())


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, frame in [*predictions.groupby('Season'), ('Overall', predictions)]:
        row = dict(Season=season, Games=len(frame))
        for model in ['XGB', 'LR', 'Ensemble']:
            row[f'{model}Brier'] = brier_score_loss(frame.Target, frame[f'{model}Probability'])
        row['EnsembleMinusXGB'] = row['EnsembleBrier'] - row['XGBBrier']
        row['EnsembleMinusLR'] = row['EnsembleBrier'] - row['LRBrier']
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--men-lr-c', type=float, help='Exact S5 C from its original source; never tune on holdout')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    if args.men_lr_c is None:
        parser.error('Men S5 source missing from merged notebook. Supply documented --men-lr-c; C=.1 failed OOF reproduction. No holdout scores calculated.')
    CONFIG['men']['C'] = args.men_lr_c
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = [f'{s}_fixed_2022_2025_{kind}.csv' for s in CONFIG
             for kind in ['predictions', 'brier_comparison']]
    names += ['fixed_2022_2025_manifest.json', 'fixed_2022_2025_lr_oof_verification.csv']
    for name in names:
        if (args.output_dir / name).exists():
            raise FileExistsError(f'Refusing to overwrite {name}; use a fresh --output-dir')
    data = {sex: load_data(sex) for sex in CONFIG}
    # Both OOF gates complete before any holdout scoring.
    verified = {sex: verify_lr(sex, df) for sex, df in data.items()}
    outputs, audit = {}, {}
    for sex, df in data.items():
        cfg = CONFIG[sex]
        train, test = df[df.Season <= 2021], df[df.Season.isin(HOLDOUT)]
        assert set(train.Season) == set(range(cfg['start'], 2022)) - {2020}
        lr = fit_lr(train, cfg['lr_features'], cfg['C'])
        xgb = XGBClassifier(**XGB_COMMON, **cfg['xgb'])
        xgb.fit(train[cfg['xgb_features']], train.Target)
        out = test[KEYS + ['Target']].copy()
        out['XGBProbability'] = probability(xgb, test, cfg['xgb_features'])
        out['LRProbability'] = probability(lr, test, cfg['lr_features'])
        out['EnsembleProbability'] = 0.5 * out.XGBProbability + 0.5 * out.LRProbability
        out['TrainingThroughSeason'] = 2021
        out['TrainingGames'] = len(train)
        # Validate benchmark matchups and predictions too, not only rounded score.
        previous = pd.read_csv(ROOT / ('results/men_logmassey_holdout_predictions.csv'
                                if sex == 'men' else 'results/women_xgboost_holdout_predictions.csv'))
        print(f'{sex} existing XGB columns: {list(previous.columns)}', flush=True)
        aligned = exact_match(out, previous)
        saved_column = 'Probability' if sex == 'men' else 'XGBProbability_saved'
        np.testing.assert_allclose(aligned.XGBProbability, aligned[saved_column], atol=2e-6, rtol=0)
        scores = score_predictions(out)
        actual = float(scores.iloc[-1].XGBBrier)
        assert abs(actual - cfg['holdout_expected']) < 1e-6, (
            f'{sex} XGB benchmark mismatch {actual}; investigate environment/data before publishing')
        print(f'\n{sex}\n{scores.to_string(index=False, float_format=lambda x: f"{x:.9f}")}', flush=True)
        outputs[f'{sex}_fixed_2022_2025_predictions.csv'] = out
        outputs[f'{sex}_fixed_2022_2025_brier_comparison.csv'] = scores
        audit[sex] = dict(config=cfg, training_games=len(train), training_seasons=sorted(train.Season.unique().tolist()),
                          oof=verified[sex][1], lr_params=lr.named_steps['logreg'].get_params(),
                          xgb_params=xgb.get_params(), xgb_booster_config=json.loads(xgb.get_booster().save_config()),
                          eligible_data_sha256=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest())
    # No final artifacts written unless all benchmark gates passed.
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False, mode='x')
    pd.concat([oof.assign(Gender=sex) for sex, (oof, _) in verified.items()]).to_csv(
        args.output_dir / 'fixed_2022_2025_lr_oof_verification.csv', index=False, mode='x')
    manifest = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    sklearn=sklearn.__version__, xgboost=xgboost.__version__, models=audit,
                    xgb_common=XGB_COMMON, ensemble_weights=[0.5, 0.5],
                    interpretation='Retrospective robustness check; outcomes previously examined. No retuning.',
                    lr_provenance='Notebook cell 25 pipeline; women C=.5 from cells 20/26. Men S5 absent from notebook: user-supplied C must match every saved S5 OOF probability.',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (args.output_dir / 'fixed_2022_2025_manifest.json').open('x') as handle:
        json.dump(manifest, handle, indent=2, default=str)


if __name__ == '__main__':
    main()
