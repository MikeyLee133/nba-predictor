"""
model.py
--------
Pure prediction logic. No I/O, no scraping, no display.

Takes clean DataFrames and returns structured prediction results.
"""

from dataclasses import dataclass
import pandas as pd
from nba_predictor.config import (
    TEAM_STAT_WEIGHTS,
    PLAYER_STAT_WEIGHTS,
    INVERT_STATS,
    TEAM_SCORE_WEIGHT,
    PLAYER_SCORE_WEIGHT,
    HOME_COURT_MULTIPLIER,
    PLAYER_SCORE_SCALE,
    TOP_PLAYERS_PER_TEAM,
    ABBR_TO_FULL,
)


@dataclass
class SeriesPrediction:
    """Holds the result of a single series prediction."""
    home: str               # team abbreviation
    away: str               # team abbreviation
    home_win_pct: float     # 0–100
    away_win_pct: float     # 0–100
    predicted_winner: str   # team abbreviation
    label: str              # human-readable series label


# ── Normalization ─────────────────────────────────────────────────────────────

def _min_max_normalize(series: pd.Series, invert: bool = False) -> pd.Series:
    """
    Normalize a Series to [0, 1].
    If invert=True, flip so that lower raw values score higher.
    Returns 0.5 for all values when the range is zero (all teams identical).
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    normed = (series - mn) / (mx - mn)
    return (1 - normed) if invert else normed


# ── Score builders ────────────────────────────────────────────────────────────

def build_team_scores(team_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute a composite score (0–100) for every team in team_df.

    Weights and inversion rules come from config.TEAM_STAT_WEIGHTS /
    config.INVERT_STATS so the model is fully configurable without
    touching this function.

    Returns: {full_team_name: score}
    """
    composite = pd.Series(0.0, index=team_df.index)

    for stat, weight in TEAM_STAT_WEIGHTS.items():
        if stat not in team_df.columns:
            continue
        invert = stat in INVERT_STATS
        composite += _min_max_normalize(team_df[stat], invert=invert) * weight

    return dict(zip(team_df["team"], composite * 100))


def build_player_scores(player_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute a star-power score for each team based on its top N players by PER.

    Returns: {team_abbr: raw_score}
    Raw scores are not yet normalized to 0–100 (scaling happens in predict_series).
    """
    team_scores: dict[str, float] = {}

    for team_abbr, group in player_df.groupby("team_id"):
        valid = group.dropna(subset=["per", "pts_per_g"])
        if valid.empty:
            team_scores[team_abbr] = 0.0
            continue

        top_n = valid.nlargest(TOP_PLAYERS_PER_TEAM, "per")
        score = sum(
            top_n[stat].mean() * weight
            for stat, weight in PLAYER_STAT_WEIGHTS.items()
            if stat in top_n.columns
        )
        team_scores[team_abbr] = score

    return team_scores


# ── Prediction ────────────────────────────────────────────────────────────────

def _blended_score(
    abbr: str,
    team_scores: dict[str, float],
    player_scores: dict[str, float],
) -> float:
    """
    Combine team and player scores into one blended value for a single team.
    Player scores are scaled to [0, 100] before blending.
    """
    full_name = ABBR_TO_FULL.get(abbr, abbr)
    ts = team_scores.get(full_name, 50.0)
    ps_raw = player_scores.get(abbr, 0.0)
    ps_norm = min(ps_raw * PLAYER_SCORE_SCALE, 100.0)
    return TEAM_SCORE_WEIGHT * ts + PLAYER_SCORE_WEIGHT * ps_norm


def predict_series(
    home: str,
    away: str,
    label: str,
    team_scores: dict[str, float],
    player_scores: dict[str, float],
) -> SeriesPrediction:
    """
    Predict the outcome of a playoff series between home and away teams.

    Home-court advantage is applied as a multiplier to the home team's
    blended score (see config.HOME_COURT_MULTIPLIER).
    """
    home_score = _blended_score(home, team_scores, player_scores) * HOME_COURT_MULTIPLIER
    away_score = _blended_score(away, team_scores, player_scores)

    total = home_score + away_score or 1.0
    home_win_pct = round(home_score / total * 100, 1)
    away_win_pct = round(100 - home_win_pct, 1)
    predicted_winner = home if home_win_pct >= 50 else away

    return SeriesPrediction(
        home=home,
        away=away,
        home_win_pct=home_win_pct,
        away_win_pct=away_win_pct,
        predicted_winner=predicted_winner,
        label=label,
    )


def predict_all(
    matchups: list[tuple[str, str, str]],
    team_scores: dict[str, float],
    player_scores: dict[str, float],
) -> list[SeriesPrediction]:
    """Run predictions for a list of (home, away, label) matchup tuples."""
    return [
        predict_series(home, away, label, team_scores, player_scores)
        for home, away, label in matchups
    ]
