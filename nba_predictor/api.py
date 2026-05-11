"""
api.py
------
FastAPI REST interface for the NBA playoff predictor.
Run with:  uvicorn nba_predictor.api:app --reload

Interactive docs: http://localhost:8000/docs
"""

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from nba_predictor.config import ABBR_TO_FULL, PLAYOFF_MATCHUPS, SEASON
from nba_predictor.fetcher import fetch_player_df, fetch_team_df
from nba_predictor.model import build_player_scores, build_team_scores, predict_all

# ── App metadata ──────────────────────────────────────────────────────────────

_DESCRIPTION = """
Predicts NBA playoff series outcomes using live stats from the official NBA Stats API.

Stats are fetched per request and cached locally for 24 hours.

## Endpoints

- **`/predictions`** — win probabilities for all active playoff matchups
- **`/predictions/{home}/{away}`** — prediction for a specific series
- **`/teams`** — composite scores for all playoff teams
- **`/health`** — API status and current season
"""

_TAGS = [
    {
        "name": "predictions",
        "description": "Win probability predictions for active playoff series, "
                       "blending team composite scores (60%) and player star-power scores (40%).",
    },
    {
        "name": "teams",
        "description": "Playoff team composite scores derived from min-max normalised "
                       "season statistics across all 30 NBA teams.",
    },
    {
        "name": "system",
        "description": "Health check and API metadata.",
    },
]

app = FastAPI(
    title="NBA Playoff Predictor API",
    description=_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "Source code",
        "url": "https://github.com/MikeyLee133/nba-predictor",
    },
    openapi_tags=_TAGS,
)


# ── Response models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., description="Always 'ok' when the API is running", examples=["ok"])
    season: str = Field(..., description="Current NBA season", examples=["2025-26"])


class PredictionResponse(BaseModel):
    series_label:          str   = Field(..., description="Human-readable series label",
                                         examples=["West Semis: OKC vs Dallas"])
    home:                  str   = Field(..., description="Home team abbreviation", examples=["OKC"])
    home_team:             str   = Field(..., description="Home team full name",
                                         examples=["Oklahoma City Thunder"])
    away:                  str   = Field(..., description="Away team abbreviation", examples=["DAL"])
    away_team:             str   = Field(..., description="Away team full name",
                                         examples=["Dallas Mavericks"])
    home_win_pct:          float = Field(..., description="Home team win probability (0–100)",
                                         examples=[61.4])
    away_win_pct:          float = Field(..., description="Away team win probability (0–100)",
                                         examples=[38.6])
    predicted_winner:      str   = Field(..., description="Predicted winner abbreviation",
                                         examples=["OKC"])
    predicted_winner_full: str   = Field(..., description="Predicted winner full name",
                                         examples=["Oklahoma City Thunder"])


class TeamScoreResponse(BaseModel):
    abbreviation:    str   = Field(..., description="Team abbreviation", examples=["OKC"])
    full_name:       str   = Field(..., description="Team full name",
                                   examples=["Oklahoma City Thunder"])
    composite_score: float = Field(...,
                                   description="Composite score (0–100) from normalised season stats",
                                   examples=[74.3])


# ── Dependencies ──────────────────────────────────────────────────────────────

def get_team_data() -> pd.DataFrame:
    return fetch_team_df()


def get_player_data() -> pd.DataFrame:
    return fetch_player_df()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _predictions(team_df: pd.DataFrame, player_df: pd.DataFrame) -> list[PredictionResponse]:
    team_scores   = build_team_scores(team_df)
    player_scores = build_player_scores(player_df)
    return [
        PredictionResponse(
            series_label=p.label,
            home=p.home,
            home_team=ABBR_TO_FULL.get(p.home, p.home),
            away=p.away,
            away_team=ABBR_TO_FULL.get(p.away, p.away),
            home_win_pct=p.home_win_pct,
            away_win_pct=p.away_win_pct,
            predicted_winner=p.predicted_winner,
            predicted_winner_full=ABBR_TO_FULL.get(p.predicted_winner, p.predicted_winner),
        )
        for p in predict_all(PLAYOFF_MATCHUPS, team_scores, player_scores)
    ]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Health check",
)
def health():
    """Returns API status and the current NBA season being tracked."""
    return HealthResponse(status="ok", season=SEASON)


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    tags=["predictions"],
    summary="All series predictions",
)
def get_all_predictions(
    team_df:   pd.DataFrame = Depends(get_team_data),
    player_df: pd.DataFrame = Depends(get_player_data),
):
    """
    Returns win probabilities for every active playoff matchup.

    Predictions use full-season stats. The model blends a team composite
    score (net rating, offensive/defensive ratings, pace, points, assists,
    3-pointers) with a player star-power score (top 3 players by PIE).
    """
    return _predictions(team_df, player_df)


@app.get(
    "/predictions/{home}/{away}",
    response_model=PredictionResponse,
    tags=["predictions"],
    summary="Single series prediction",
    responses={404: {"description": "No active matchup found for the given team pair"}},
)
def get_series_prediction(
    home:      str,
    away:      str,
    team_df:   pd.DataFrame = Depends(get_team_data),
    player_df: pd.DataFrame = Depends(get_player_data),
):
    """
    Returns the win probability prediction for a specific series.

    Use team abbreviations (e.g. `OKC`, `DAL`). Case-insensitive.
    Returns 404 if the matchup is not in the current active playoff round.
    """
    home, away = home.upper(), away.upper()
    if not any(h == home and a == away for h, a, _ in PLAYOFF_MATCHUPS):
        raise HTTPException(status_code=404, detail=f"No active matchup found for {home} vs {away}")
    return next(p for p in _predictions(team_df, player_df) if p.home == home and p.away == away)


@app.get(
    "/teams",
    response_model=list[TeamScoreResponse],
    tags=["teams"],
    summary="Playoff team scores",
)
def get_teams(
    team_df: pd.DataFrame = Depends(get_team_data),
):
    """
    Returns composite scores for all current playoff teams.

    Scores are computed by min-max normalising each stat across all 30 NBA
    teams and applying the configured weights. Higher is better (defensive
    stats are inverted before normalisation).
    """
    team_scores   = build_team_scores(team_df)
    playoff_abbrs = sorted({abbr for h, a, _ in PLAYOFF_MATCHUPS for abbr in (h, a)})
    return [
        TeamScoreResponse(
            abbreviation=abbr,
            full_name=ABBR_TO_FULL.get(abbr, abbr),
            composite_score=round(team_scores.get(ABBR_TO_FULL.get(abbr, abbr), 0.0), 1),
        )
        for abbr in playoff_abbrs
    ]
