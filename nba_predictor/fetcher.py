"""
fetcher.py
----------
Fetches NBA stats from the official NBA Stats API via nba_api.
Caches results to disk for 24 hours to avoid repeated API calls.
"""

import time
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats, LeagueDashPlayerStats

SEASON = "2025-26"
CACHE_DIR = Path(__file__).parent.parent / ".data_cache"
CACHE_TTL_HOURS = 24


class FetchError(Exception):
    """Raised when the NBA API request fails."""


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.pkl"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _load_or_build(name: str, force: bool, build_fn) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_path(name)
    if not force and _cache_valid(path):
        return pd.read_pickle(str(path))
    try:
        df = build_fn()
    except Exception as e:
        raise FetchError(f"NBA API error fetching {name}: {e}")
    df.to_pickle(str(path))
    return df


def fetch_team_df(force: bool = False) -> pd.DataFrame:
    """Return a clean team stats DataFrame from the NBA Stats API."""
    def build():
        base = LeagueDashTeamStats(
            season=SEASON,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
            timeout=30,
        ).get_data_frames()[0]

        time.sleep(1)

        adv = LeagueDashTeamStats(
            season=SEASON,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
            timeout=30,
        ).get_data_frames()[0]

        team = base[["TEAM_NAME", "PTS", "FG3M", "AST"]].rename(columns={
            "TEAM_NAME": "team",
            "PTS":       "pts",
            "FG3M":      "3pm",
            "AST":       "ast",
        })
        advanced = adv[["TEAM_NAME", "PACE", "OFF_RATING", "DEF_RATING", "NET_RATING"]].rename(columns={
            "TEAM_NAME":  "team",
            "PACE":       "pace",
            "OFF_RATING": "ortg",
            "DEF_RATING": "drtg",
            "NET_RATING": "net_rtg",
        })
        return team.merge(advanced, on="team")

    return _load_or_build("team_stats", force, build)


def fetch_player_df(force: bool = False) -> pd.DataFrame:
    """Return a clean player stats DataFrame from the NBA Stats API."""
    def build():
        base = LeagueDashPlayerStats(
            season=SEASON,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
            timeout=30,
        ).get_data_frames()[0]

        time.sleep(1)

        adv = LeagueDashPlayerStats(
            season=SEASON,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
            timeout=30,
        ).get_data_frames()[0]

        players = base[["PLAYER_NAME", "TEAM_ABBREVIATION", "PTS", "AST", "REB", "FG3M"]].rename(columns={
            "PLAYER_NAME":       "player",
            "TEAM_ABBREVIATION": "team_id",
            "PTS":               "pts_per_g",
            "AST":               "ast_per_g",
            "REB":               "trb_per_g",
            "FG3M":              "fg3_per_g",
        })

        # PIE (Player Impact Estimate) is the NBA's equivalent of PER; scale to similar range
        pie_df = adv[["PLAYER_NAME", "TEAM_ABBREVIATION", "PIE"]].rename(columns={
            "PLAYER_NAME":       "player",
            "TEAM_ABBREVIATION": "team_id",
            "PIE":               "per",
        }).copy()
        pie_df["per"] = pie_df["per"] * 100

        return players.merge(pie_df, on=["player", "team_id"], how="left")

    return _load_or_build("player_stats", force, build)
