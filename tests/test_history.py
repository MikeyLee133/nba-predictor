import json
import pytest
from pathlib import Path

from nba_predictor.model import SeriesPrediction
from nba_predictor.history import (
    save_predictions,
    record_outcome,
    load_history,
    accuracy_stats,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _pred(home, away, winner, home_pct=60.0, label=None):
    return SeriesPrediction(
        home=home,
        away=away,
        home_win_pct=home_pct,
        away_win_pct=round(100 - home_pct, 1),
        predicted_winner=winner,
        label=label or f"{home} vs {away}",
    )


# ── save_predictions ──────────────────────────────────────────────────────────

def test_save_creates_file(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC")], "Second Round", path)
    assert path.exists()


def test_save_stores_correct_fields(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC", home_pct=62.0, label="West Semis")], "Second Round", path)
    records = json.loads(path.read_text())
    assert len(records) == 1
    r = records[0]
    assert r["home"] == "OKC"
    assert r["away"] == "LAL"
    assert r["predicted_winner"] == "OKC"
    assert r["home_win_pct"] == 62.0
    assert r["series_label"] == "West Semis"
    assert r["round"] == "Second Round"
    assert r["actual_winner"] is None
    assert r["correct"] is None


def test_save_multiple_predictions(tmp_path):
    path = tmp_path / "history.json"
    preds = [_pred("OKC", "LAL", "OKC"), _pred("NYK", "PHI", "NYK")]
    save_predictions(preds, "Second Round", path)
    records = json.loads(path.read_text())
    assert len(records) == 2


def test_save_does_not_duplicate_same_round(tmp_path):
    path = tmp_path / "history.json"
    preds = [_pred("OKC", "LAL", "OKC", label="West Semis")]
    save_predictions(preds, "Second Round", path)
    save_predictions(preds, "Second Round", path)
    records = json.loads(path.read_text())
    assert len(records) == 1


def test_save_appends_new_round(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC", label="West Semis")], "Second Round", path)
    save_predictions([_pred("OKC", "MIN", "OKC", label="West Finals")], "Conference Finals", path)
    records = json.loads(path.read_text())
    assert len(records) == 2


# ── record_outcome ────────────────────────────────────────────────────────────

def test_record_outcome_sets_actual_winner(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC", label="West Semis")], "Second Round", path)
    record_outcome("West Semis", "OKC", path)
    records = json.loads(path.read_text())
    assert records[0]["actual_winner"] == "OKC"


def test_record_outcome_correct_when_prediction_matches(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC", label="West Semis")], "Second Round", path)
    record_outcome("West Semis", "OKC", path)
    records = json.loads(path.read_text())
    assert records[0]["correct"] is True


def test_record_outcome_incorrect_when_prediction_wrong(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC", label="West Semis")], "Second Round", path)
    record_outcome("West Semis", "LAL", path)
    records = json.loads(path.read_text())
    assert records[0]["correct"] is False


def test_record_outcome_returns_true_when_found(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC", label="West Semis")], "Second Round", path)
    assert record_outcome("West Semis", "OKC", path) is True


def test_record_outcome_returns_false_when_not_found(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC", label="West Semis")], "Second Round", path)
    assert record_outcome("Nonexistent Series", "OKC", path) is False


# ── load_history ──────────────────────────────────────────────────────────────

def test_load_history_returns_empty_list_when_no_file(tmp_path):
    assert load_history(tmp_path / "nonexistent.json") == []


def test_load_history_returns_all_records(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC"), _pred("NYK", "PHI", "NYK")], "Second Round", path)
    assert len(load_history(path)) == 2


# ── accuracy_stats ────────────────────────────────────────────────────────────

def test_accuracy_stats_correct_count(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([
        _pred("OKC", "LAL", "OKC", label="Series A"),
        _pred("NYK", "PHI", "NYK", label="Series B"),
        _pred("DET", "CLE", "DET", label="Series C"),
    ], "Second Round", path)
    record_outcome("Series A", "OKC", path)   # correct
    record_outcome("Series B", "PHI", path)   # wrong
    record_outcome("Series C", "DET", path)   # correct
    stats = accuracy_stats(load_history(path))
    assert stats["correct"] == 2
    assert stats["total"] == 3
    assert stats["pct"] == pytest.approx(66.7, abs=0.1)


def test_accuracy_stats_excludes_unresolved(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([
        _pred("OKC", "LAL", "OKC", label="Series A"),
        _pred("NYK", "PHI", "NYK", label="Series B"),
    ], "Second Round", path)
    record_outcome("Series A", "OKC", path)
    # Series B has no outcome yet
    stats = accuracy_stats(load_history(path))
    assert stats["total"] == 1


def test_accuracy_stats_no_resolved_predictions(tmp_path):
    path = tmp_path / "history.json"
    save_predictions([_pred("OKC", "LAL", "OKC", label="Series A")], "Second Round", path)
    stats = accuracy_stats(load_history(path))
    assert stats["correct"] == 0
    assert stats["total"] == 0
    assert stats["pct"] == 0.0
