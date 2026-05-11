"""
fetcher.py
----------
Fetches NBA stats from the official NBA Stats API via nba_api.
Caches each result to disk for 24 hours.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats

from nba_predictor.config import API_SLEEP_SECONDS, CACHE_TTL_HOURS, SEASON

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent.parent / ".data_cache"


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
    t0 = time.perf_counter()
    if not force and _cache_valid(path):
        df = pd.read_pickle(path)
        logger.debug("cache hit   %s (%.0fms)", name, (time.perf_counter() - t0) * 1000)
        return df
    try:
        df = build_fn()
    except Exception as e:
        raise FetchError(f"NBA API error fetching {name}: {e}")
    logger.debug("API fetch   %s (%.0fms)", name, (time.perf_counter() - t0) * 1000)
    df.to_pickle(path)
    return df


def _fetch_measures(endpoint_cls, last_n: int, base_cols: dict, adv_cols: dict, season: str = SEASON):
    """Fetch Base and Advanced measures from an endpoint, rename columns, and return both."""
    base = endpoint_cls(
        season=season,
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        last_n_games=str(last_n),
        timeout=30,
    ).get_data_frames()[0]

    time.sleep(API_SLEEP_SECONDS)

    adv = endpoint_cls(
        season=season,
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

def _fetch_raw_team_df(last_n: int = 0, season: str = SEASON) -> pd.DataFrame:
    base_df, adv_df = _fetch_measures(
        LeagueDashTeamStats,
        last_n,
        base_cols={"TEAM_NAME": "team", "PTS": "pts", "FG3M": "3pm", "AST": "ast"},
        adv_cols={"TEAM_NAME": "team", "PACE": "pace", "OFF_RATING": "ortg",
                  "DEF_RATING": "drtg", "NET_RATING": "net_rtg"},
        season=season,
    )
    return base_df.merge(adv_df, on="team")


def fetch_team_df(last_n: int = 0, force: bool = False, season: str = SEASON) -> pd.DataFrame:
    """Return team stats DataFrame. last_n=0 means full season."""
    return _load_or_build(f"team_stats_{season}_{last_n}", force, lambda: _fetch_raw_team_df(last_n, season))


# ── Player stats ──────────────────────────────────────────────────────────────

def _fetch_raw_player_df(last_n: int = 0, season: str = SEASON) -> pd.DataFrame:
    base_df, adv_df = _fetch_measures(
        LeagueDashPlayerStats,
        last_n,
        base_cols={"PLAYER_NAME": "player", "TEAM_ABBREVIATION": "team_id",
                   "PTS": "pts_per_g", "AST": "ast_per_g", "REB": "trb_per_g", "FG3M": "fg3_per_g"},
        adv_cols={"PLAYER_NAME": "player", "TEAM_ABBREVIATION": "team_id", "PIE": "per"},
        season=season,
    )
    adv_df["per"] = adv_df["per"] * 100  # scale PIE to approximate PER range
    return base_df.merge(adv_df, on=["player", "team_id"], how="left")


def fetch_player_df(last_n: int = 0, force: bool = False, season: str = SEASON) -> pd.DataFrame:
    """Return player stats DataFrame. last_n=0 means full season."""
    return _load_or_build(f"player_stats_{season}_{last_n}", force, lambda: _fetch_raw_player_df(last_n, season))


def fetch_seasons_parallel(seasons: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Fetch team and player stats for multiple seasons concurrently.
    Returns {season: {"team": DataFrame, "player": DataFrame}}.
    Failed seasons are omitted and logged as warnings.
    """
    if not seasons:
        return {}

    results: dict[str, dict[str, pd.DataFrame]] = {}

    def _fetch_one(season: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
        return season, fetch_team_df(season=season), fetch_player_df(season=season)

    with ThreadPoolExecutor(max_workers=min(len(seasons), 4)) as executor:
        futures = {executor.submit(_fetch_one, s): s for s in seasons}
        for future in as_completed(futures):
            season = futures[future]
            try:
                s, team_df, player_df = future.result()
                results[s] = {"team": team_df, "player": player_df}
            except FetchError as exc:
                logger.warning("Failed to fetch %s: %s", season, exc)

    return results
