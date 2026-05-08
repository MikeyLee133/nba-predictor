"""
fetcher.py
----------
Fetches NBA stats from the official NBA Stats API via nba_api.
Caches each result to disk for 24 hours.
"""

import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats, LeagueDashPlayerStats

from nba_predictor.config import SEASON

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
        return pd.read_pickle(path)
    try:
        df = build_fn()
    except Exception as e:
        raise FetchError(f"NBA API error fetching {name}: {e}")
    df.to_pickle(path)
    return df


def _fetch_measures(endpoint_cls, last_n: int, base_cols: dict, adv_cols: dict):
    """Fetch Base and Advanced measures from an endpoint, rename columns, and return both."""
    base = endpoint_cls(
        season=SEASON,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        last_n_games=str(last_n),
        timeout=30,
    ).get_data_frames()[0]

    time.sleep(1)

    adv = endpoint_cls(
        season=SEASON,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
        last_n_games=str(last_n),
        timeout=30,
    ).get_data_frames()[0]

    return (
        base[list(base_cols)].rename(columns=base_cols),
        adv[list(adv_cols)].rename(columns=adv_cols),
    )


# ── Team stats ────────────────────────────────────────────────────────────────

def _fetch_raw_team_df(last_n: int = 0) -> pd.DataFrame:
    base_df, adv_df = _fetch_measures(
        LeagueDashTeamStats,
        last_n,
        base_cols={"TEAM_NAME": "team", "PTS": "pts", "FG3M": "3pm", "AST": "ast"},
        adv_cols={"TEAM_NAME": "team", "PACE": "pace", "OFF_RATING": "ortg",
                  "DEF_RATING": "drtg", "NET_RATING": "net_rtg"},
    )
    return base_df.merge(adv_df, on="team")


def fetch_team_df(last_n: int = 0, force: bool = False) -> pd.DataFrame:
    """Return team stats DataFrame. last_n=0 means full season."""
    return _load_or_build(f"team_stats_{SEASON}_{last_n}", force, lambda: _fetch_raw_team_df(last_n))


# ── Player stats ──────────────────────────────────────────────────────────────

def _fetch_raw_player_df(last_n: int = 0) -> pd.DataFrame:
    base_df, adv_df = _fetch_measures(
        LeagueDashPlayerStats,
        last_n,
        base_cols={"PLAYER_NAME": "player", "TEAM_ABBREVIATION": "team_id",
                   "PTS": "pts_per_g", "AST": "ast_per_g", "REB": "trb_per_g", "FG3M": "fg3_per_g"},
        adv_cols={"PLAYER_NAME": "player", "TEAM_ABBREVIATION": "team_id", "PIE": "per"},
    )
    adv_df["per"] = adv_df["per"] * 100  # scale PIE to approximate PER range
    return base_df.merge(adv_df, on=["player", "team_id"], how="left")


def fetch_player_df(last_n: int = 0, force: bool = False) -> pd.DataFrame:
    """Return player stats DataFrame. last_n=0 means full season."""
    return _load_or_build(f"player_stats_{SEASON}_{last_n}", force, lambda: _fetch_raw_player_df(last_n))
