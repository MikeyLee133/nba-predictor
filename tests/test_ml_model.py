import numpy as np
import pandas as pd
import pytest

from nba_predictor.ml_model import (
    FEATURE_STATS,
    build_feature_vector,
    build_training_data,
    build_training_records,
    get_team_stats,
    train,
    predict_win_probability,
    feature_importances,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _stats(net_rtg=5.0, drtg=108.0, ortg=113.0, pts=115.0, ast=27.0, three=14.0, pace=99.0):
    return {"net_rtg": net_rtg, "drtg": drtg, "ortg": ortg,
            "pts": pts, "ast": ast, "3pm": three, "pace": pace}


def _records(n=12):
    """Alternating home-win / away-win records with a clear signal in net_rtg."""
    recs = []
    for i in range(n):
        home_won = i % 2 == 0
        recs.append({
            "home_scores": _stats(net_rtg=8.0 if home_won else 0.0),
            "away_scores": _stats(net_rtg=0.0 if home_won else 8.0),
            "home_won": home_won,
        })
    return recs


# ── build_feature_vector ──────────────────────────────────────────────────────

def test_feature_vector_length_matches_feature_stats():
    vec = build_feature_vector(_stats(), _stats())
    assert len(vec) == len(FEATURE_STATS)


def test_feature_vector_computes_home_minus_away():
    home = _stats(net_rtg=6.0)
    away = _stats(net_rtg=2.0)
    vec = build_feature_vector(home, away)
    net_rtg_idx = FEATURE_STATS.index("net_rtg")
    assert vec[net_rtg_idx] == pytest.approx(4.0)


def test_feature_vector_negative_when_away_stronger():
    home = _stats(net_rtg=1.0)
    away = _stats(net_rtg=9.0)
    vec = build_feature_vector(home, away)
    net_rtg_idx = FEATURE_STATS.index("net_rtg")
    assert vec[net_rtg_idx] < 0


def test_feature_vector_zero_for_equal_teams():
    s = _stats()
    vec = build_feature_vector(s, s)
    assert all(v == pytest.approx(0.0) for v in vec)


# ── build_training_data ───────────────────────────────────────────────────────

def test_training_data_shapes():
    X, y = build_training_data(_records(8))
    assert X.shape == (8, len(FEATURE_STATS))
    assert y.shape == (8,)


def test_training_data_labels_correct():
    recs = [
        {"home_scores": _stats(), "away_scores": _stats(), "home_won": True},
        {"home_scores": _stats(), "away_scores": _stats(), "home_won": False},
    ]
    _, y = build_training_data(recs)
    assert y[0] == 1
    assert y[1] == 0


# ── get_team_stats ────────────────────────────────────────────────────────────

def _team_df():
    return pd.DataFrame([{
        "team": "Boston Celtics",
        "net_rtg": 8.5, "drtg": 107.2, "ortg": 115.7,
        "pts": 117.0, "ast": 27.0, "3pm": 15.0, "pace": 99.0,
    }])


def test_get_team_stats_returns_correct_values():
    stats = get_team_stats(_team_df(), "BOS")
    assert stats["net_rtg"] == pytest.approx(8.5)
    assert stats["drtg"] == pytest.approx(107.2)


def test_get_team_stats_returns_empty_for_unknown_abbr():
    assert get_team_stats(_team_df(), "XYZ") == {}


def test_get_team_stats_returns_all_feature_stats():
    stats = get_team_stats(_team_df(), "BOS")
    assert set(FEATURE_STATS).issubset(set(stats.keys()))


# ── build_training_records ────────────────────────────────────────────────────

def test_build_training_records_returns_one_per_resolved_matchup():
    team_df = pd.DataFrame([
        {"team": "Boston Celtics",     "net_rtg": 8.0, "drtg": 107.0, "ortg": 115.0, "pts": 117.0, "ast": 27.0, "3pm": 15.0, "pace": 99.0},
        {"team": "Cleveland Cavaliers","net_rtg": 3.0, "drtg": 112.0, "ortg": 115.0, "pts": 111.0, "ast": 24.0, "3pm": 12.0, "pace": 97.0},
    ])
    historical = {
        "2023-24": {
            "matchups": [("BOS", "CLE", "East Semis")],
            "outcomes": {"East Semis": "BOS"},
        }
    }
    records = build_training_records(historical, {"2023-24": team_df})
    assert len(records) == 1
    assert records[0]["home_won"] is True


def test_build_training_records_skips_missing_team_data():
    team_df = pd.DataFrame([
        {"team": "Boston Celtics", "net_rtg": 8.0, "drtg": 107.0, "ortg": 115.0, "pts": 117.0, "ast": 27.0, "3pm": 15.0, "pace": 99.0},
    ])
    historical = {
        "2023-24": {
            "matchups": [("BOS", "CLE", "East Semis")],  # CLE not in df
            "outcomes": {"East Semis": "BOS"},
        }
    }
    records = build_training_records(historical, {"2023-24": team_df})
    assert len(records) == 0


def test_build_training_records_skips_seasons_without_df():
    historical = {
        "2023-24": {
            "matchups": [("BOS", "CLE", "East Semis")],
            "outcomes": {"East Semis": "BOS"},
        }
    }
    records = build_training_records(historical, {})  # no dfs provided
    assert len(records) == 0


# ── train ─────────────────────────────────────────────────────────────────────

def test_train_returns_model_with_expected_attributes():
    model = train(_records())
    assert model.classifier is not None
    assert model.scaler is not None
    assert model.feature_names == FEATURE_STATS
    assert 0.0 <= model.train_accuracy <= 100.0


def test_train_accuracy_reasonable_with_clear_signal():
    model = train(_records(20))
    assert model.train_accuracy > 50.0


# ── predict_win_probability ───────────────────────────────────────────────────

def test_predict_returns_probability_between_0_and_1():
    model = train(_records())
    prob = predict_win_probability(model, _stats(net_rtg=6.0), _stats(net_rtg=2.0))
    assert 0.0 <= prob <= 1.0


def test_predict_higher_for_stronger_home_team():
    model = train(_records())
    strong = predict_win_probability(model, _stats(net_rtg=9.0), _stats(net_rtg=1.0))
    weak   = predict_win_probability(model, _stats(net_rtg=1.0), _stats(net_rtg=9.0))
    assert strong > weak


# ── feature_importances ───────────────────────────────────────────────────────

def test_feature_importances_has_all_features():
    importances = feature_importances(train(_records()))
    assert set(importances.keys()) == set(FEATURE_STATS)


def test_feature_importances_net_rtg_positive():
    # net_rtg differential is the clearest signal — coefficient should be positive
    model = train(_records(20))
    imp = feature_importances(model)
    assert imp["net_rtg"] > 0
