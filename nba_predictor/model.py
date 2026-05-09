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
    SERIES_COMEBACK_RATES,
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

def build_team_scores(team_df: pd.DataFrame, weights: dict | None = None) -> dict[str, float]:
    """
    Compute a composite score (0–100) for every team in team_df.

    Weights and inversion rules come from config.TEAM_STAT_WEIGHTS /
    config.INVERT_STATS so the model is fully configurable without
    touching this function.

    Returns: {full_team_name: score}
    """
    if weights is None:
        weights = TEAM_STAT_WEIGHTS
    composite = pd.Series(0.0, index=team_df.index)

    for stat, weight in weights.items():
        if stat not in team_df.columns:
            continue
        invert = stat in INVERT_STATS
        composite += _min_max_normalize(team_df[stat], invert=invert) * weight

    return dict(zip(team_df["team"], composite * 100))


def build_player_scores(
    player_df: pd.DataFrame,
    weights: dict | None = None,
    unavailable: set[str] | None = None,
) -> dict[str, float]:
    """
    Compute a star-power score for each team based on its top N players by PER.

    Returns: {team_abbr: raw_score}
    Raw scores are not yet normalized to 0–100 (scaling happens in predict_series).
    """
    if weights is None:
        weights = PLAYER_STAT_WEIGHTS
    if unavailable:
        player_df = player_df[~player_df["player"].isin(unavailable)]
    team_scores: dict[str, float] = {}

    for team_abbr, group in player_df.groupby("team_id"):
        valid = group.dropna(subset=["per", "pts_per_g"])
        if valid.empty:
            team_scores[team_abbr] = 0.0
            continue

        top_n = valid.nlargest(TOP_PLAYERS_PER_TEAM, "per")
        score = sum(
            top_n[stat].mean() * weight
            for stat, weight in weights.items()
            if stat in top_n.columns
        )
        team_scores[team_abbr] = score

    return team_scores


# ── Prediction ────────────────────────────────────────────────────────────────

def _blended_score(
    abbr: str,
    team_scores: dict[str, float],
    player_scores: dict[str, float],
    team_w: float | None = None,
    player_w: float | None = None,
) -> float:
    """
    Combine team and player scores into one blended value for a single team.
    Player scores are scaled to [0, 100] before blending.
    """
    if team_w is None:
        team_w = TEAM_SCORE_WEIGHT
    if player_w is None:
        player_w = PLAYER_SCORE_WEIGHT
    full_name = ABBR_TO_FULL.get(abbr, abbr)
    ts = team_scores.get(full_name, 50.0)
    ps_raw = player_scores.get(abbr, 0.0)
    ps_norm = min(ps_raw * PLAYER_SCORE_SCALE, 100.0)
    return team_w * ts + player_w * ps_norm


def predict_series(
    home: str,
    away: str,
    label: str,
    team_scores: dict[str, float],
    player_scores: dict[str, float],
    team_w: float | None = None,
    player_w: float | None = None,
    home_mult: float | None = None,
) -> SeriesPrediction:
    """
    Predict the outcome of a playoff series between home and away teams.

    Home-court advantage is applied as a multiplier to the home team's
    blended score (see config.HOME_COURT_MULTIPLIER).
    """
    if home_mult is None:
        home_mult = HOME_COURT_MULTIPLIER
    home_score = _blended_score(home, team_scores, player_scores, team_w, player_w) * home_mult
    away_score = _blended_score(away, team_scores, player_scores, team_w, player_w)

    total = max(home_score + away_score, 1.0)
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


def adjust_for_series_score(
    prediction: SeriesPrediction,
    home_wins: int,
    away_wins: int,
) -> SeriesPrediction:
    """
    Blend the model's win probability with historical NBA series comeback rates.
    At 0-0 the prediction is unchanged. As the series progresses, historical
    survival rates pull the probability toward the current leader.
    """
    if home_wins == 0 and away_wins == 0:
        return prediction

    if home_wins > away_wins:
        historical_home_pct = SERIES_COMEBACK_RATES.get((home_wins, away_wins), 0.5) * 100
    else:
        trailer_rate = SERIES_COMEBACK_RATES.get((away_wins, home_wins), 0.5)
        historical_home_pct = (1 - trailer_rate) * 100

    new_home_pct = round((prediction.home_win_pct + historical_home_pct) / 2, 1)
    new_away_pct = round(100 - new_home_pct, 1)
    return SeriesPrediction(
        home=prediction.home,
        away=prediction.away,
        home_win_pct=new_home_pct,
        away_win_pct=new_away_pct,
        predicted_winner=prediction.home if new_home_pct >= 50 else prediction.away,
        label=prediction.label,
    )


def predict_all(
    matchups: list[tuple[str, str, str]],
    team_scores: dict[str, float],
    player_scores: dict[str, float],
    team_w: float | None = None,
    player_w: float | None = None,
    home_mult: float | None = None,
) -> list[SeriesPrediction]:
    """Run predictions for a list of (home, away, label) matchup tuples."""
    return [
        predict_series(home, away, label, team_scores, player_scores, team_w, player_w, home_mult)
        for home, away, label in matchups
    ]
