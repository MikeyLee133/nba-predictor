import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from nba_predictor.fetcher import _cache_valid, _load_or_build, FetchError, fetch_seasons_parallel


# ── _cache_valid ──────────────────────────────────────────────────────────────

def test_cache_valid_missing_file(tmp_path):
    assert _cache_valid(tmp_path / "nonexistent.pkl") is False


def test_cache_valid_fresh_file(tmp_path):
    path = tmp_path / "fresh.pkl"
    path.touch()
    assert _cache_valid(path) is True


def test_cache_valid_expired_file(tmp_path):
    path = tmp_path / "old.pkl"
    path.touch()
    expired = (datetime.now() - timedelta(hours=25)).timestamp()
    os.utime(path, (expired, expired))
    assert _cache_valid(path) is False


# ── _load_or_build ────────────────────────────────────────────────────────────

def test_load_or_build_reads_valid_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("nba_predictor.fetcher.CACHE_DIR", tmp_path)
    df = pd.DataFrame({"val": [1, 2, 3]})
    df.to_pickle(tmp_path / "my_data.pkl")

    build_called = []
    result = _load_or_build("my_data", force=False, build_fn=lambda: build_called.append(1) or df)
    assert build_called == []
    assert list(result["val"]) == [1, 2, 3]


def test_load_or_build_calls_build_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("nba_predictor.fetcher.CACHE_DIR", tmp_path)
    df = pd.DataFrame({"val": [42]})
    result = _load_or_build("new_data", force=False, build_fn=lambda: df)
    assert list(result["val"]) == [42]
    assert (tmp_path / "new_data.pkl").exists()


def test_load_or_build_force_bypasses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("nba_predictor.fetcher.CACHE_DIR", tmp_path)
    old_df = pd.DataFrame({"val": [1]})
    old_df.to_pickle(tmp_path / "cached.pkl")

    new_df = pd.DataFrame({"val": [99]})
    result = _load_or_build("cached", force=True, build_fn=lambda: new_df)
    assert list(result["val"]) == [99]


def test_load_or_build_raises_fetch_error_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("nba_predictor.fetcher.CACHE_DIR", tmp_path)

    def bad_build():
        raise RuntimeError("API is down")

    with pytest.raises(FetchError, match="API is down"):
        _load_or_build("fail", force=True, build_fn=bad_build)


def test_load_or_build_creates_cache_dir(tmp_path, monkeypatch):
    cache_dir = tmp_path / "new_cache_dir"
    monkeypatch.setattr("nba_predictor.fetcher.CACHE_DIR", cache_dir)
    assert not cache_dir.exists()
    _load_or_build("data", force=False, build_fn=lambda: pd.DataFrame({"v": [1]}))
    assert cache_dir.exists()


# ── fetch_seasons_parallel ────────────────────────────────────────────────────

def _mock_fetchers(monkeypatch):
    """Replace fetch_team_df and fetch_player_df with fast stubs."""
    def mock_team(last_n=0, force=False, season="2025-26"):
        return pd.DataFrame({"team": [f"Team_{season}"], "season": [season]})

    def mock_player(last_n=0, force=False, season="2025-26"):
        return pd.DataFrame({"player": [f"Player_{season}"], "season": [season]})

    monkeypatch.setattr("nba_predictor.fetcher.fetch_team_df",   mock_team)
    monkeypatch.setattr("nba_predictor.fetcher.fetch_player_df", mock_player)


def test_fetch_seasons_parallel_returns_all_seasons(monkeypatch):
    _mock_fetchers(monkeypatch)
    results = fetch_seasons_parallel(["2022-23", "2023-24"])
    assert set(results.keys()) == {"2022-23", "2023-24"}


def test_fetch_seasons_parallel_each_result_has_team_and_player(monkeypatch):
    _mock_fetchers(monkeypatch)
    results = fetch_seasons_parallel(["2023-24"])
    assert "team"   in results["2023-24"]
    assert "player" in results["2023-24"]


def test_fetch_seasons_parallel_data_matches_season(monkeypatch):
    _mock_fetchers(monkeypatch)
    results = fetch_seasons_parallel(["2022-23", "2023-24"])
    assert results["2022-23"]["team"]["season"].iloc[0] == "2022-23"
    assert results["2023-24"]["team"]["season"].iloc[0] == "2023-24"


def test_fetch_seasons_parallel_omits_failed_seasons(monkeypatch):
    def flaky_team(last_n=0, force=False, season="2025-26"):
        if season == "2022-23":
            raise FetchError("API error")
        return pd.DataFrame({"team": [f"Team_{season}"]})

    def mock_player(last_n=0, force=False, season="2025-26"):
        return pd.DataFrame({"player": [f"Player_{season}"]})

    monkeypatch.setattr("nba_predictor.fetcher.fetch_team_df",   flaky_team)
    monkeypatch.setattr("nba_predictor.fetcher.fetch_player_df", mock_player)

    results = fetch_seasons_parallel(["2022-23", "2023-24"])
    assert "2022-23" not in results
    assert "2023-24" in results


def test_fetch_seasons_parallel_empty_input(monkeypatch):
    _mock_fetchers(monkeypatch)
    assert fetch_seasons_parallel([]) == {}
