import numpy as np
import pandas as pd
import pytest

from nba_predictor.ml_model import (
    FEATURE_STATS,
    TEAM_FEATURE_STATS,
    build_feature_vector,
    build_training_data,
    build_training_records,
    cross_validate_loo_season,
    get_team_stats,
    train,
    predict_win_probability,
    feature_importances,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _stats(net_rtg=5.0, drtg=108.0, ortg=113.0, pts=115.0, ast=27.0, three=14.0, pace=99.0, player_score=20.0):
    return {"net_rtg": net_rtg, "drtg": drtg, "ortg": ortg,
            "pts": pts, "ast": ast, "3pm": three, "pace": pace,
            "player_score": player_score}


def _records(n=12, season="2023-24"):
    """Alternating home-win / away-win records with a clear signal in net_rtg."""
    recs = []
    for i in range(n):
        home_won = i % 2 == 0
        recs.append({
            "season":      season,
            "home_scores": _stats(net_rtg=8.0 if home_won else 0.0, player_score=25.0 if home_won else 15.0),
            "away_scores": _stats(net_rtg=0.0 if home_won else 8.0, player_score=15.0 if home_won else 25.0),
            "home_won":    home_won,
        })
    return recs


def _multi_season_records():
    """Records spread across 3 seasons for cross-validation tests."""
    recs = []
    for season in ["2021-22", "2022-23", "2023-24"]:
        recs += _records(n=10, season=season)
    return recs


# ── constants ─────────────────────────────────────────────────────────────────

def test_feature_stats_includes_player_score():
    assert "player_score" in FEATURE_STATS


def test_team_feature_stats_excludes_player_score():
    assert "player_score" not in TEAM_FEATURE_STATS


# ── build_feature_vector ──────────────────────────────────────────────────────

def test_feature_vector_length_matches_feature_stats():
    vec = build_feature_vector(_stats(), _stats())
    assert len(vec) == len(FEATURE_STATS)


def test_feature_vector_computes_home_minus_away():
    home = _stats(net_rtg=6.0)
    away = _stats(net_rtg=2.0)
    vec = build_feature_vector(home, away)
    assert vec[FEATURE_STATS.index("net_rtg")] == pytest.approx(4.0)


def test_feature_vector_player_score_differential():
    home = _stats(player_score=30.0)
    away = _stats(player_score=20.0)
    vec = build_feature_vector(home, away)
    assert vec[FEATURE_STATS.index("player_score")] == pytest.approx(10.0)


def test_feature_vector_negative_when_away_stronger():
    vec = build_feature_vector(_stats(net_rtg=1.0), _stats(net_rtg=9.0))
    assert vec[FEATURE_STATS.index("net_rtg")] < 0


def test_feature_vector_zero_for_equal_teams():
    s = _stats()
    assert all(v == pytest.approx(0.0) for v in build_feature_vector(s, s))


# ── build_training_data ───────────────────────────────────────────────────────

def test_training_data_shapes():
    X, y = build_training_data(_records(8))
    assert X.shape == (8, len(FEATURE_STATS))
    assert y.shape == (8,)


def test_training_data_labels_correct():
    recs = [
        {"season": "2023-24", "home_scores": _stats(), "away_scores": _stats(), "home_won": True},
        {"season": "2023-24", "home_scores": _stats(), "away_scores": _stats(), "home_won": False},
    ]
    _, y = build_training_data(recs)
    assert y[0] == 1 and y[1] == 0


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
    assert stats["drtg"]    == pytest.approx(107.2)


def test_get_team_stats_returns_empty_for_unknown_abbr():
    assert get_team_stats(_team_df(), "XYZ") == {}


def test_get_team_stats_returns_team_feature_stats():
    stats = get_team_stats(_team_df(), "BOS")
    assert set(TEAM_FEATURE_STATS).issubset(set(stats.keys()))


def test_get_team_stats_does_not_include_player_score():
    stats = get_team_stats(_team_df(), "BOS")
    assert "player_score" not in stats


# ── build_training_records ────────────────────────────────────────────────────

def _hist_team_df():
    return pd.DataFrame([
        {"team": "Boston Celtics",     "net_rtg": 8.0, "drtg": 107.0, "ortg": 115.0, "pts": 117.0, "ast": 27.0, "3pm": 15.0, "pace": 99.0},
        {"team": "Cleveland Cavaliers","net_rtg": 3.0, "drtg": 112.0, "ortg": 115.0, "pts": 111.0, "ast": 24.0, "3pm": 12.0, "pace": 97.0},
    ])


def _hist_player_df():
    return pd.DataFrame([
        {"team_id": "BOS", "player": "Tatum",   "per": 28.0, "pts_per_g": 27.0, "ast_per_g": 4.9, "trb_per_g": 8.1, "fg3_per_g": 3.0},
        {"team_id": "CLE", "player": "Mitchell", "per": 22.0, "pts_per_g": 28.0, "ast_per_g": 4.5, "trb_per_g": 4.0, "fg3_per_g": 2.5},
    ])


def _historical():
    return {
        "2023-24": {
            "matchups": [("BOS", "CLE", "East Semis")],
            "outcomes": {"East Semis": "BOS"},
        }
    }


def test_build_training_records_returns_correct_count():
    records = build_training_records(_historical(), {"2023-24": _hist_team_df()})
    assert len(records) == 1


def test_build_training_records_includes_season():
    records = build_training_records(_historical(), {"2023-24": _hist_team_df()})
    assert records[0]["season"] == "2023-24"


def test_build_training_records_home_won_flag():
    records = build_training_records(_historical(), {"2023-24": _hist_team_df()})
    assert records[0]["home_won"] is True


def test_build_training_records_includes_player_score_when_dfs_provided():
    records = build_training_records(
        _historical(),
        {"2023-24": _hist_team_df()},
        season_player_dfs={"2023-24": _hist_player_df()},
    )
    assert "player_score" in records[0]["home_scores"]
    assert "player_score" in records[0]["away_scores"]


def test_build_training_records_player_score_zero_without_player_dfs():
    records = build_training_records(_historical(), {"2023-24": _hist_team_df()})
    assert records[0]["home_scores"].get("player_score", 0.0) == 0.0


def test_build_training_records_skips_missing_team():
    df = pd.DataFrame([{"team": "Boston Celtics", "net_rtg": 8.0, "drtg": 107.0,
                         "ortg": 115.0, "pts": 117.0, "ast": 27.0, "3pm": 15.0, "pace": 99.0}])
    records = build_training_records(_historical(), {"2023-24": df})
    assert len(records) == 0  # CLE not in df


def test_build_training_records_skips_missing_season():
    records = build_training_records(_historical(), {})
    assert len(records) == 0


# ── cross_validate_loo_season ─────────────────────────────────────────────────

def test_cross_validate_returns_required_keys():
    result = cross_validate_loo_season(_multi_season_records())
    assert {"correct", "total", "accuracy", "seasons_tested"}.issubset(result.keys())


def test_cross_validate_accuracy_between_0_and_100():
    result = cross_validate_loo_season(_multi_season_records())
    assert 0.0 <= result["accuracy"] <= 100.0


def test_cross_validate_seasons_tested_matches_unique_seasons():
    recs = _multi_season_records()
    result = cross_validate_loo_season(recs)
    assert result["seasons_tested"] == 3


def test_cross_validate_total_equals_all_test_samples():
    recs = _multi_season_records()  # 30 records, 3 seasons of 10 each
    result = cross_validate_loo_season(recs)
    assert result["total"] == 30


# ── train ─────────────────────────────────────────────────────────────────────

def test_train_returns_model_with_expected_attributes():
    model = train(_records())
    assert model.classifier is not None
    assert model.scaler is not None
    assert model.feature_names == FEATURE_STATS
    assert 0.0 <= model.train_accuracy <= 100.0
    assert model.n_samples == 12


def test_train_accuracy_reasonable_with_clear_signal():
    assert train(_records(20)).train_accuracy > 50.0


# ── predict_win_probability ───────────────────────────────────────────────────

def test_predict_returns_probability_between_0_and_1():
    prob = predict_win_probability(train(_records()), _stats(net_rtg=6.0), _stats(net_rtg=2.0))
    assert 0.0 <= prob <= 1.0


def test_predict_higher_for_stronger_home_team():
    model = train(_records())
    strong = predict_win_probability(model, _stats(net_rtg=9.0), _stats(net_rtg=1.0))
    weak   = predict_win_probability(model, _stats(net_rtg=1.0), _stats(net_rtg=9.0))
    assert strong > weak


# ── feature_importances ───────────────────────────────────────────────────────

def test_feature_importances_has_all_features():
    assert set(feature_importances(train(_records())).keys()) == set(FEATURE_STATS)


def test_feature_importances_net_rtg_positive():
    imp = feature_importances(train(_records(20)))
    assert imp["net_rtg"] > 0
