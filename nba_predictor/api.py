"""
api.py
------
FastAPI REST interface for the NBA playoff predictor.
Run with:  uvicorn nba_predictor.api:app --reload
"""

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from nba_predictor.config import ABBR_TO_FULL, PLAYOFF_MATCHUPS, SEASON
from nba_predictor.fetcher import fetch_player_df, fetch_team_df
from nba_predictor.model import build_player_scores, build_team_scores, predict_all

app = FastAPI(
    title="NBA Playoff Predictor API",
    description="Live NBA playoff series predictions powered by the NBA Stats API",
    version="1.0.0",
)


# ── Response models ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    season: str


class PredictionResponse(BaseModel):
    series_label:          str
    home:                  str
    home_team:             str
    away:                  str
    away_team:             str
    home_win_pct:          float
    away_win_pct:          float
    predicted_winner:      str
    predicted_winner_full: str


class TeamScoreResponse(BaseModel):
    abbreviation:    str
    full_name:       str
    composite_score: float


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

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", season=SEASON)


@app.get("/predictions", response_model=list[PredictionResponse])
def get_all_predictions(
    team_df:   pd.DataFrame = Depends(get_team_data),
    player_df: pd.DataFrame = Depends(get_player_data),
):
    return _predictions(team_df, player_df)


@app.get("/predictions/{home}/{away}", response_model=PredictionResponse)
def get_series_prediction(
    home:      str,
    away:      str,
    team_df:   pd.DataFrame = Depends(get_team_data),
    player_df: pd.DataFrame = Depends(get_player_data),
):
    home, away = home.upper(), away.upper()
    if not any(h == home and a == away for h, a, _ in PLAYOFF_MATCHUPS):
        raise HTTPException(status_code=404, detail=f"No active matchup found for {home} vs {away}")
    return next(p for p in _predictions(team_df, player_df) if p.home == home and p.away == away)


@app.get("/teams", response_model=list[TeamScoreResponse])
def get_teams(
    team_df:   pd.DataFrame = Depends(get_team_data),
    player_df: pd.DataFrame = Depends(get_player_data),
):
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
