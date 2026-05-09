import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from nba_predictor.fetcher import _cache_valid, _load_or_build, FetchError


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
