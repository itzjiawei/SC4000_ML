"""Shared feature pipeline for March Machine Learning Mania 2026.

One team-season table per gender (every team, every season, INCLUDING 2026),
and one function that turns any list of (Season, Team1, Team2) pairs into
model-ready difference features. Training data and the 2026 submission use the
same code, so the features are identical in both.

Conventions (same as the README):
  Team1 = lower TeamID, Team2 = higher TeamID, Target = 1 if Team1 won.
  Every *Diff feature is Team1 value minus Team2 value.

Feature definitions reproduce the existing notebooks:
  WinRate, AvgScoringMargin, EFGDifferential  -> notebooks/feature_engineering.ipynb
  Seed                                        -> notebooks/feature_engineering.ipynb
  NetRating, AdjustedStrength (ridge, a=20)   -> notebooks/advanced_feature_engineering.ipynb
  Elo (reset each season, K=60 men/100 women) -> notebooks/elo_feature_engineering.ipynb
  MasseyRank (mean of COL/DOL/MOR/POM/WLK,
     latest ranking on or before day 133)     -> notebooks/massey_feature_engineering.ipynb

Raw Kaggle files are read from data/raw/ or from the folder in the
MMLM_RAW_DIR environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "shared"

GENDER_PREFIX = {"men": "M", "women": "W"}
FIRST_SEASON = {"men": 2003, "women": 2010}  # first season with detailed box scores
ELO_K = {"men": 60, "women": 100}
MASSEY_SYSTEMS = ["COL", "DOL", "MOR", "POM", "WLK"]
MASSEY_LAST_DAY = 133

# team-level column -> matchup feature name
TEAM_FEATURES = {
    "Seed": "SeedDiff",
    "AvgScoringMargin": "AvgScoringMarginDiff",
    "WinRate": "WinRateDiff",
    "EFGDifferential": "EFGDifferentialDiff",
    "NetRating": "NetRatingDiff",
    "AdjustedStrength": "AdjustedStrengthDiff",
    "Elo": "EloDiff",
    "MasseyRank": "MasseyRankDiff",
    "LogMasseyRank": "LogMasseyRankDiff",
    # log(1 + rank): the variant Kat's scripts/generate_logreg_oof.py uses for men's S5
    "Log1pMasseyRank": "Log1pMasseyRankDiff",
    # Day 2 features
    "EloV2": "EloV2Diff",
    "Last10WinRate": "Last10WinRateDiff",
    "Last10Margin": "Last10MarginDiff",
    "SOS": "SOSDiff",
}
NEW_FEATURES = ["EloV2Diff", "Last10WinRateDiff", "Last10MarginDiff", "SOSDiff"]
BASE4 = ["SeedDiff", "AvgScoringMarginDiff", "WinRateDiff", "EFGDifferentialDiff"]


# ----------------------------------------------------------------------------
# Raw data
# ----------------------------------------------------------------------------
def raw_dir() -> Path:
    path = Path(os.environ.get("MMLM_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
    if not (path / "MTeams.csv").exists():
        raise FileNotFoundError(
            f"Kaggle data not found in {path}. Put the competition CSVs in data/raw/ "
            "or set MMLM_RAW_DIR=/path/to/march-machine-learning-mania-2026"
        )
    return path


def read_raw(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(raw_dir() / f"{name}.csv", **kwargs)


# ----------------------------------------------------------------------------
# Team-season features
# ----------------------------------------------------------------------------
def _box_score_stats(detailed: pd.DataFrame) -> pd.DataFrame:
    """WinRate, AvgScoringMargin, EFGDifferential, NetRating per team-season."""
    d = detailed
    poss = ((d.WFGA - d.WOR + d.WTO + 0.475 * d.WFTA) + (d.LFGA - d.LOR + d.LTO + 0.475 * d.LFTA)) / 2

    def side(us: str, them: str, won: int) -> pd.DataFrame:
        return pd.DataFrame({
            "Season": d.Season, "TeamID": d[f"{us}TeamID"], "Won": won,
            "Margin": d[f"{us}Score"] - d[f"{them}Score"],
            "PF": d[f"{us}Score"], "PA": d[f"{them}Score"], "Poss": poss,
            "FGM": d[f"{us}FGM"], "FGA": d[f"{us}FGA"], "FGM3": d[f"{us}FGM3"],
            "OppFGM": d[f"{them}FGM"], "OppFGA": d[f"{them}FGA"], "OppFGM3": d[f"{them}FGM3"],
        })

    g = pd.concat([side("W", "L", 1), side("L", "W", 0)], ignore_index=True)
    s = g.groupby(["Season", "TeamID"]).agg(
        Games=("Won", "size"), Wins=("Won", "sum"), AvgScoringMargin=("Margin", "mean"),
        PF=("PF", "sum"), PA=("PA", "sum"), Poss=("Poss", "sum"),
        FGM=("FGM", "sum"), FGA=("FGA", "sum"), FGM3=("FGM3", "sum"),
        OppFGM=("OppFGM", "sum"), OppFGA=("OppFGA", "sum"), OppFGM3=("OppFGM3", "sum"),
    ).reset_index()
    s["WinRate"] = s.Wins / s.Games
    s["EFGDifferential"] = (s.FGM + 0.5 * s.FGM3) / s.FGA - (s.OppFGM + 0.5 * s.OppFGM3) / s.OppFGA
    s["NetRating"] = 100 * (s.PF - s.PA) / s.Poss
    return s[["Season", "TeamID", "Games", "WinRate", "AvgScoringMargin", "EFGDifferential", "NetRating"]]


def _adjusted_strength(detailed: pd.DataFrame, alpha: float = 20.0) -> pd.DataFrame:
    """Ridge regression of winning margin on +1 winner / -1 loser / home indicator, per season."""
    out = []
    for season, g in detailed.groupby("Season"):
        teams = np.sort(np.unique(np.concatenate([g.WTeamID, g.LTeamID])))
        idx = {t: i for i, t in enumerate(teams)}
        n, k = len(g), len(teams)
        loc = g.WLoc.map({"H": 1.0, "A": -1.0, "N": 0.0}).to_numpy()
        rows = np.concatenate([np.arange(n)] * 3)
        cols = np.concatenate([g.WTeamID.map(idx), g.LTeamID.map(idx), np.full(n, k)])
        vals = np.concatenate([np.ones(n), -np.ones(n), loc])
        X = csr_matrix((vals, (rows, cols)), shape=(n, k + 1))
        y = (g.WScore - g.LScore).to_numpy(dtype=float)
        coef = Ridge(alpha=alpha, fit_intercept=False, solver="lsqr").fit(X, y).coef_
        out.append(pd.DataFrame({"Season": season, "TeamID": teams, "AdjustedStrength": coef[:k]}))
    return pd.concat(out, ignore_index=True)


def _legacy_elo(compact: pd.DataFrame, k: float, home: float = 60, scale: float = 400,
                start: float = 1500) -> pd.DataFrame:
    """End-of-regular-season Elo; ratings reset every season (the team's current version)."""
    out = []
    for season, g in compact.groupby("Season"):
        r: dict[int, float] = {}
        for w, l, loc in g.sort_values("DayNum", kind="stable")[["WTeamID", "LTeamID", "WLoc"]].itertuples(index=False):
            rw, rl = r.get(w, start), r.get(l, start)
            aw = rw + (home if loc == "H" else 0)
            al = rl + (home if loc == "A" else 0)
            p = 1 / (1 + 10 ** ((al - aw) / scale))
            r[w], r[l] = rw + k * (1 - p), rl - k * (1 - p)
        out.append(pd.DataFrame({"Season": season, "TeamID": list(r), "Elo": list(r.values())}))
    return pd.concat(out, ignore_index=True)


def elo_v2(gender: str, k: float, keep: float, mov: bool, home: float = 60, scale: float = 400,
           start: float = 1500, return_games: bool = False):
    """Improved Elo (new in this project):
      - ratings carry over between seasons: start = 1500 + keep * (last season's rating - 1500)
      - optional margin-of-victory multiplier, ((|margin| + 3) ** 0.8) / (7.5 + 0.006 * winner's Elo edge),
        the form published by FiveThirtyEight for their NBA Elo (acknowledged in the report)
      - uses every season of compact results from the first year available (1985 men / 1998 women)
      - snapshot = rating after the last regular-season game, BEFORE that season's NCAA tournament;
        tournament games then update ratings, so they only affect LATER seasons
    Returns team-season snapshots (Season, TeamID, EloV2); with return_games=True also the pre-game
    probability for every regular-season game (used to choose k / keep / mov without tournament labels)."""
    p = GENDER_PREFIX[gender]
    reg = read_raw(f"{p}RegularSeasonCompactResults").assign(Tourney=0)
    trn = read_raw(f"{p}NCAATourneyCompactResults").assign(Tourney=1)
    games = pd.concat([reg, trn], ignore_index=True).sort_values(["Season", "Tourney", "DayNum"], kind="stable")
    r: dict[int, float] = {}
    snaps, pregame = [], []
    for season, g in games.groupby("Season", sort=True):
        r = {t: start + keep * (v - start) for t, v in r.items()}
        for tourney_part, gg in g.groupby("Tourney", sort=True):
            if tourney_part == 1:  # regular season finished: take the pre-tournament snapshot
                snaps.append(pd.DataFrame({"Season": season, "TeamID": list(r), "EloV2": list(r.values())}))
            for w, l, loc, ws, ls in gg[["WTeamID", "LTeamID", "WLoc", "WScore", "LScore"]].itertuples(index=False):
                rw, rl = r.get(w, start), r.get(l, start)
                edge = (rw + (home if loc == "H" else 0)) - (rl + (home if loc == "A" else 0))
                pw = 1 / (1 + 10 ** (-edge / scale))
                mult = ((ws - ls + 3) ** 0.8) / (7.5 + 0.006 * edge) if mov else 1.0
                delta = k * mult * (1 - pw)
                r[w], r[l] = rw + delta, rl - delta
                if return_games and tourney_part == 0:
                    pregame.append((season, pw))
        if not (g.Tourney == 1).any():  # season without tournament results (2020, or the current season)
            snaps.append(pd.DataFrame({"Season": season, "TeamID": list(r), "EloV2": list(r.values())}))
    snap = pd.concat(snaps, ignore_index=True)
    if return_games:
        return snap, pd.DataFrame(pregame, columns=["Season", "PWinner"])
    return snap


# Chosen by scripts/tune_elo.py: lowest Brier on regular-season games 2003/2010-2016 (no tournament labels)
ELO_V2_PARAMS = {"men": dict(k=45, keep=0.85, mov=True), "women": dict(k=60, keep=0.85, mov=True)}


def _form_and_schedule(compact: pd.DataFrame, strength: pd.DataFrame, n_last: int = 10) -> pd.DataFrame:
    """Recent form (last n regular-season games: win rate and average margin) and strength of
    schedule (average opponent AdjustedStrength over the season)."""
    def side(us, them, won):
        return pd.DataFrame({"Season": compact.Season, "DayNum": compact.DayNum, "TeamID": compact[f"{us}TeamID"],
                             "OppID": compact[f"{them}TeamID"], "Won": won,
                             "Margin": compact[f"{us}Score"] - compact[f"{them}Score"]})
    g = pd.concat([side("W", "L", 1), side("L", "W", 0)], ignore_index=True).sort_values(
        ["Season", "TeamID", "DayNum"], kind="stable")
    last = g.groupby(["Season", "TeamID"]).tail(n_last).groupby(["Season", "TeamID"]).agg(
        Last10WinRate=("Won", "mean"), Last10Margin=("Margin", "mean")).reset_index()
    opp = g.merge(strength.rename(columns={"TeamID": "OppID", "AdjustedStrength": "OppStrength"}),
                  on=["Season", "OppID"], how="left")
    sos = opp.groupby(["Season", "TeamID"]).OppStrength.mean().rename("SOS").reset_index()
    return last.merge(sos, on=["Season", "TeamID"], how="outer")


def _massey(seasons: set[int]) -> pd.DataFrame:
    m = read_raw("MMasseyOrdinals")
    m = m[m.SystemName.isin(MASSEY_SYSTEMS) & (m.RankingDayNum <= MASSEY_LAST_DAY) & m.Season.isin(seasons)]
    latest = m.sort_values("RankingDayNum").groupby(["Season", "SystemName", "TeamID"]).tail(1)
    c = latest.groupby(["Season", "TeamID"]).agg(MasseyRank=("OrdinalRank", "mean"),
                                                 MasseySystems=("SystemName", "nunique")).reset_index()
    c["LogMasseyRank"] = np.log(c.MasseyRank)
    c["Log1pMasseyRank"] = np.log1p(c.MasseyRank)
    return c


def _seeds(gender: str) -> pd.DataFrame:
    s = read_raw(f"{GENDER_PREFIX[gender]}NCAATourneySeeds")
    s["Seed"] = s.Seed.str.extract(r"(\d+)")[0].astype(int)
    return s[["Season", "TeamID", "Seed"]]


def build_team_seasons(gender: str) -> pd.DataFrame:
    """One row per (Season, TeamID) for every season with detailed results, 2026 included.
    Seed is NaN for teams not in that season's tournament; Massey columns are NaN for women."""
    p = GENDER_PREFIX[gender]
    detailed = read_raw(f"{p}RegularSeasonDetailedResults")
    detailed = detailed[detailed.Season >= FIRST_SEASON[gender]]
    compact = read_raw(f"{p}RegularSeasonCompactResults")
    compact = compact[compact.Season >= FIRST_SEASON[gender]]

    t = _box_score_stats(detailed)
    strength = _adjusted_strength(detailed)
    t = t.merge(strength, on=["Season", "TeamID"], how="left", validate="one_to_one")
    t = t.merge(_legacy_elo(compact, ELO_K[gender]), on=["Season", "TeamID"], how="left", validate="one_to_one")
    t = t.merge(elo_v2(gender, **ELO_V2_PARAMS[gender]), on=["Season", "TeamID"], how="left", validate="one_to_one")
    t = t.merge(_form_and_schedule(compact, strength), on=["Season", "TeamID"], how="left", validate="one_to_one")
    t = t.merge(_seeds(gender), on=["Season", "TeamID"], how="left", validate="one_to_one")
    if gender == "men":
        t = t.merge(_massey(set(t.Season)), on=["Season", "TeamID"], how="left", validate="one_to_one")
    else:
        t["MasseyRank"] = t["LogMasseyRank"] = t["Log1pMasseyRank"] = t["MasseySystems"] = np.nan
    t.insert(0, "Gender", gender)
    return t.sort_values(["Season", "TeamID"]).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Matchups
# ----------------------------------------------------------------------------
def make_matchups(pairs: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Attach Team1/Team2 team-season values and Team1-minus-Team2 differences.
    `pairs` needs Season, Team1, Team2 with Team1 < Team2 (other columns are kept)."""
    if not (pairs.Team1 < pairs.Team2).all():
        raise ValueError("make_matchups expects Team1 < Team2 in every row")
    cols = ["Season", "TeamID"] + list(TEAM_FEATURES)
    out = pairs.copy()
    for side in ("Team1", "Team2"):
        renamed = teams[cols].rename(columns={"TeamID": side, **{c: f"{side}{c}" for c in TEAM_FEATURES}})
        out = out.merge(renamed, on=["Season", side], how="left", validate="many_to_one")
    for col, diff in TEAM_FEATURES.items():
        out[diff] = out[f"Team1{col}"] - out[f"Team2{col}"]
    assert len(out) == len(pairs)
    return out


def tournament_games(gender: str, teams: pd.DataFrame) -> pd.DataFrame:
    """Historical NCAA tournament games with features (the training/validation data)."""
    t = read_raw(f"{GENDER_PREFIX[gender]}NCAATourneyCompactResults")
    t = t[t.Season >= FIRST_SEASON[gender]]
    pairs = pd.DataFrame({
        "Season": t.Season.values,
        "Team1": np.minimum(t.WTeamID, t.LTeamID).values,
        "Team2": np.maximum(t.WTeamID, t.LTeamID).values,
    })
    pairs["Target"] = (t.WTeamID.values == pairs.Team1.values).astype(int)
    games = make_matchups(pairs, teams)
    needed = BASE4 + ["NetRatingDiff", "AdjustedStrengthDiff", "EloDiff"] + NEW_FEATURES
    if gender == "men":
        needed += ["MasseyRankDiff", "LogMasseyRankDiff", "Log1pMasseyRankDiff"]
    missing = games[needed].isna().any(axis=1)
    if missing.any():
        raise AssertionError(f"{gender}: {missing.sum()} tournament games have missing features")
    games.insert(0, "Gender", gender)
    return games


def submission_pairs(gender: str, sample_name: str = "SampleSubmissionStage2") -> pd.DataFrame:
    """All (Season, Team1, Team2) pairs in the Kaggle sample file for one gender."""
    ids = read_raw(sample_name).ID.str.split("_", expand=True).astype(int)
    ids.columns = ["Season", "Team1", "Team2"]
    is_men = ids.Team1 < 3000
    return ids[is_men if gender == "men" else ~is_men].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Cached entry point
# ----------------------------------------------------------------------------
def load(gender: str, rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (team_seasons, tournament_games) for a gender, cached in data/processed/shared/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tpath, gpath = CACHE_DIR / f"{gender}_team_seasons.csv", CACHE_DIR / f"{gender}_tourney_games.csv"
    if rebuild or not (tpath.exists() and gpath.exists()):
        teams = build_team_seasons(gender)
        games = tournament_games(gender, teams)
        teams.to_csv(tpath, index=False)
        games.to_csv(gpath, index=False)
    return pd.read_csv(tpath), pd.read_csv(gpath)


if __name__ == "__main__":
    for g in ("men", "women"):
        teams, games = load(g, rebuild=True)
        print(f"{g}: {len(teams)} team-seasons ({teams.Season.min()}-{teams.Season.max()}, "
              f"{(teams.Season == 2026).sum()} in 2026); {len(games)} tournament games "
              f"({games.Season.min()}-{games.Season.max()})")
