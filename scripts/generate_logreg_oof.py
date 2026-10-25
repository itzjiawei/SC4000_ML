from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
VALIDATION_SEASONS = (2017, 2018, 2019, 2021)
BASE_FEATURES = [
    "SeedDiff",
    "AvgScoringMarginDiff",
    "WinRateDiff",
    "EFGDifferentialDiff",
]


def merge_elo(matchups: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    team1_elo = elo.rename(
        columns={"TeamID": "Team1", "FinalElo": "Team1Elo"}
    )
    team2_elo = elo.rename(
        columns={"TeamID": "Team2", "FinalElo": "Team2Elo"}
    )
    merged = matchups.merge(
        team1_elo[["Season", "Team1", "Team1Elo"]],
        on=["Season", "Team1"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        team2_elo[["Season", "Team2", "Team2Elo"]],
        on=["Season", "Team2"],
        how="left",
        validate="many_to_one",
    )
    merged["EloDiff"] = merged["Team1Elo"] - merged["Team2Elo"]
    if merged[["Team1Elo", "Team2Elo", "EloDiff"]].isna().any().any():
        raise ValueError("The Elo source is missing a matchup team rating.")
    return merged


def generate_predictions(
    data: pd.DataFrame,
    features: list[str],
    C: float,
    model_name: str,
    shared_data: pd.DataFrame,
    expected_rows: int,
) -> tuple[pd.DataFrame, list[tuple[int, int, int, float]]]:
    prediction_rows = []
    season_scores = []

    for validation_season in VALIDATION_SEASONS:
        train = data.loc[data["Season"] < validation_season]
        validation = data.loc[data["Season"] == validation_season]
        if train.empty or validation.empty:
            raise ValueError(f"Missing train or validation rows for {validation_season}.")
        if train["Season"].max() >= validation_season:
            raise AssertionError("A validation or later season entered training.")

        X_train = train[features]
        X_validation = validation[features]
        if not np.isfinite(X_train.to_numpy()).all():
            raise ValueError(f"Training features are not finite for {validation_season}.")
        if not np.isfinite(X_validation.to_numpy()).all():
            raise ValueError(f"Validation features are not finite for {validation_season}.")

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=C,
                        solver="lbfgs",
                        max_iter=1000,
                    ),
                ),
            ]
        )
        model.fit(X_train, train["Target"])

        scaler = model.named_steps["scaler"]
        if scaler.n_samples_seen_ != len(train):
            raise AssertionError("The scaler was not fitted on training rows alone.")
        classifier = model.named_steps["logreg"]
        if not np.array_equal(classifier.classes_, np.array([0, 1])):
            raise AssertionError("Expected binary target classes 0 and 1.")

        probabilities = model.predict_proba(X_validation)[:, 1]
        if not np.isfinite(probabilities).all() or (
            (probabilities < 0) | (probabilities > 1)
        ).any():
            raise AssertionError("Predicted probabilities must be finite and in [0, 1].")

        for row, probability in zip(validation.itertuples(index=False), probabilities):
            prediction_rows.append(
                {
                    "Season": row.Season,
                    "Team1": row.Team1,
                    "Team2": row.Team2,
                    "Target": row.Target,
                    "PredictedProbability": probability,
                    "TrainingThroughSeason": int(train["Season"].max()),
                    "Model": model_name,
                }
            )

        score = brier_score_loss(validation["Target"], probabilities)
        season_scores.append(
            (validation_season, len(validation), int(train["Season"].max()), score)
        )

    columns = [
        "Season",
        "Team1",
        "Team2",
        "Target",
        "PredictedProbability",
        "TrainingThroughSeason",
        "Model",
    ]
    predictions = pd.DataFrame(prediction_rows, columns=columns)
    if len(predictions) != expected_rows:
        raise AssertionError(
            f"{model_name}: expected {expected_rows} rows, got {len(predictions)}."
        )
    if predictions.duplicated(["Season", "Team1", "Team2"]).any():
        raise AssertionError(f"{model_name}: duplicate matchup predictions.")
    if not (predictions["Team1"] < predictions["Team2"]).all():
        raise AssertionError(f"{model_name}: matchup team IDs are not ordered.")
    if not (predictions["TrainingThroughSeason"] < predictions["Season"]).all():
        raise AssertionError(f"{model_name}: training cutoff is not before validation.")

    expected_labels = shared_data.loc[
        shared_data["Season"].isin(VALIDATION_SEASONS),
        ["Season", "Team1", "Team2", "Target"],
    ]
    label_check = predictions[
        ["Season", "Team1", "Team2", "Target"]
    ].merge(
        expected_labels,
        on=["Season", "Team1", "Team2"],
        how="outer",
        suffixes=("_prediction", "_shared"),
        indicator=True,
        validate="one_to_one",
    )
    if not (label_check["_merge"] == "both").all() or not (
        label_check["Target_prediction"] == label_check["Target_shared"]
    ).all():
        raise AssertionError(f"{model_name}: labels do not match shared processed data.")

    return predictions, season_scores


def report(name: str, season_scores: list[tuple[int, int, int, float]]) -> None:
    print(name)
    for season, count, training_through, score in season_scores:
        print(
            f"  {season}: n={count}, Brier={score:.12f}, "
            f"TrainingThroughSeason={training_through}"
        )
    print(f"  Unweighted mean Brier: {np.mean([s[3] for s in season_scores]):.12f}")


def main() -> None:
    men_shared = pd.read_csv(DATA_DIR / "men_processed.csv")
    men = pd.read_csv(DATA_DIR / "men_massey.csv")
    men_keys = ["Season", "Team1", "Team2", "Target"]
    if not men[men_keys].equals(men_shared[men_keys]):
        raise AssertionError("Massey matchup rows or labels differ from men_processed.csv.")
    if not np.allclose(
        men[BASE_FEATURES].to_numpy(),
        men_shared[BASE_FEATURES].to_numpy(),
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("Massey baseline features differ from men_processed.csv.")
    men["LogMasseyRankDiff"] = (
        np.log1p(men["Team1MasseyRank"])
        - np.log1p(men["Team2MasseyRank"])
    )

    women_shared = pd.read_csv(DATA_DIR / "women_processed.csv")
    women_elo = pd.read_csv(DATA_DIR / "women_elo.csv")
    women = merge_elo(women_shared, women_elo)

    men_predictions, men_scores = generate_predictions(
        data=men,
        features=BASE_FEATURES + ["LogMasseyRankDiff"],
        C=100.0,
        model_name="Men S5",
        shared_data=men_shared,
        expected_rows=267,
    )
    women_predictions, women_scores = generate_predictions(
        data=women,
        features=BASE_FEATURES + ["EloDiff"],
        C=0.5,
        model_name="Women S2",
        shared_data=women_shared,
        expected_rows=252,
    )

    output_files = {
        RESULTS_DIR / "logreg_men_s5_oof.csv": men_predictions,
        RESULTS_DIR / "logreg_women_s2_oof.csv": women_predictions,
    }
    existing = [str(path) for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output file(s): " + ", ".join(existing)
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    for path, predictions in output_files.items():
        predictions.sort_values(
            ["Season", "Team1", "Team2"], kind="stable"
        ).to_csv(path, index=False)

    report("Men S5 (C=100)", men_scores)
    report("Women S2 (C=0.5)", women_scores)
    for path, predictions in output_files.items():
        print(f"Saved {len(predictions)} predictions to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
