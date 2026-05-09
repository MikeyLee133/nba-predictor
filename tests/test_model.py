import pandas as pd
import pytest

from nba_predictor.model import (
    _min_max_normalize,
    build_team_scores,
    build_player_scores,
    predict_series,
    predict_all,
)


# ── _min_max_normalize ────────────────────────────────────────────────────────

def test_normalize_basic():
    result = _min_max_normalize(pd.Series([0.0, 50.0, 100.0]))
    assert list(result) == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_invert():
    result = _min_max_normalize(pd.Series([0.0, 50.0, 100.0]), invert=True)
    assert list(result) == pytest.approx([1.0, 0.5, 0.0])


def test_normalize_all_identical_returns_half():
    result = _min_max_normalize(pd.Series([42.0, 42.0, 42.0]))
    assert all(v == pytest.approx(0.5) for v in result)


def test_normalize_two_values():
    result = _min_max_normalize(pd.Series([10.0, 20.0]))
    assert list(result) == pytest.approx([0.0, 1.0])


# ── build_team_scores ─────────────────────────────────────────────────────────

def _team_df(teams: dict) -> pd.DataFrame:
    return pd.DataFrame([{"team": name, **stats} for name, stats in teams.items()])


def test_team_scores_better_team_scores_higher():
    df = _team_df({"Good": {"net_rtg": 10.0}, "Bad": {"net_rtg": -5.0}})
    scores = build_team_scores(df, weights={"net_rtg": 1.0})
    assert scores["Good"] > scores["Bad"]


def test_team_scores_identical_teams_are_equal():
    df = _team_df({"A": {"net_rtg": 5.0}, "B": {"net_rtg": 5.0}})
    scores = build_team_scores(df, weights={"net_rtg": 1.0})
    assert scores["A"] == pytest.approx(scores["B"])


def test_team_scores_missing_stat_skipped_silently():
    df = _team_df({"A": {"net_rtg": 10.0}, "B": {"net_rtg": -5.0}})
    # "pts" is not in df — should not raise
    scores = build_team_scores(df, weights={"net_rtg": 0.8, "pts": 0.2})
    assert "A" in scores and "B" in scores


def test_team_scores_in_range_0_to_100():
    df = _team_df({"A": {"net_rtg": 10.0}, "B": {"net_rtg": 5.0}, "C": {"net_rtg": -5.0}})
    scores = build_team_scores(df, weights={"net_rtg": 1.0})
    for score in scores.values():
        assert 0.0 <= score <= 100.0


def test_team_scores_invert_stat_lower_is_better():
    df = _team_df({"LowDef": {"drtg": 100.0}, "HighDef": {"drtg": 115.0}})
    scores = build_team_scores(df, weights={"drtg": 1.0})
    assert scores["LowDef"] > scores["HighDef"]


# ── build_player_scores ───────────────────────────────────────────────────────

def _player_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_player_scores_better_players_score_higher():
    df = _player_df([
        {"team_id": "GSW", "player": "Star",    "per": 30.0, "pts_per_g": 30.0, "ast_per_g": 8.0, "trb_per_g": 5.0, "fg3_per_g": 3.0},
        {"team_id": "LAL", "player": "Average", "per": 12.0, "pts_per_g":  8.0, "ast_per_g": 2.0, "trb_per_g": 3.0, "fg3_per_g": 0.5},
    ])
    scores = build_player_scores(df)
    assert scores["GSW"] > scores["LAL"]


def test_player_scores_nan_per_dropped():
    df = _player_df([
        {"team_id": "BOS", "player": "A", "per": None, "pts_per_g": 20.0, "ast_per_g": 5.0, "trb_per_g": 4.0, "fg3_per_g": 2.0},
        {"team_id": "BOS", "player": "B", "per": 18.0, "pts_per_g": 15.0, "ast_per_g": 4.0, "trb_per_g": 3.0, "fg3_per_g": 1.0},
    ])
    scores = build_player_scores(df)
    assert scores["BOS"] > 0.0


def test_player_scores_all_nan_returns_zero():
    df = _player_df([
        {"team_id": "MIA", "player": "A", "per": None, "pts_per_g": None, "ast_per_g": None, "trb_per_g": None, "fg3_per_g": None},
    ])
    scores = build_player_scores(df)
    assert scores["MIA"] == 0.0


def test_player_scores_only_top_n_used():
    # Five players; the bottom two should not influence score
    df = _player_df([
        {"team_id": "NYK", "player": f"P{i}", "per": float(30 - i),
         "pts_per_g": float(20 - i), "ast_per_g": 5.0, "trb_per_g": 4.0, "fg3_per_g": 1.0}
        for i in range(5)
    ])
    # Score using top-3 default
    score_5 = build_player_scores(df)["NYK"]
    # Score using only the top 3 players directly
    df_top3 = df.nlargest(3, "per")
    score_top3 = build_player_scores(df_top3)["NYK"]
    assert score_5 == pytest.approx(score_top3)


# ── predict_series ────────────────────────────────────────────────────────────

def test_predict_series_win_pcts_sum_to_100():
    p = predict_series("HOM", "AWY", "Test", {"HOM": 80.0, "AWY": 60.0}, {},
                       team_w=1.0, player_w=0.0, home_mult=1.0)
    assert p.home_win_pct + p.away_win_pct == pytest.approx(100.0)


def test_predict_series_stronger_team_wins():
    p = predict_series("HOM", "AWY", "Test", {"HOM": 80.0, "AWY": 40.0}, {},
                       team_w=1.0, player_w=0.0, home_mult=1.0)
    assert p.predicted_winner == "HOM"
    assert p.home_win_pct > 50.0


def test_predict_series_weaker_home_team_can_lose():
    p = predict_series("HOM", "AWY", "Test", {"HOM": 30.0, "AWY": 80.0}, {},
                       team_w=1.0, player_w=0.0, home_mult=1.0)
    assert p.predicted_winner == "AWY"
    assert p.away_win_pct > 50.0


def test_predict_series_home_court_tips_equal_matchup():
    p = predict_series("HOM", "AWY", "Test", {"HOM": 50.0, "AWY": 50.0}, {},
                       team_w=1.0, player_w=0.0, home_mult=1.04)
    assert p.predicted_winner == "HOM"
    assert p.home_win_pct > 50.0


def test_predict_series_no_home_court_equal_teams_is_fifty_fifty():
    p = predict_series("HOM", "AWY", "Test", {"HOM": 50.0, "AWY": 50.0}, {},
                       team_w=1.0, player_w=0.0, home_mult=1.0)
    assert p.home_win_pct == pytest.approx(50.0)
    assert p.away_win_pct == pytest.approx(50.0)


def test_predict_series_result_fields():
    p = predict_series("HOM", "AWY", "Finals", {"HOM": 70.0, "AWY": 50.0}, {},
                       team_w=1.0, player_w=0.0, home_mult=1.0)
    assert p.home == "HOM"
    assert p.away == "AWY"
    assert p.label == "Finals"


# ── predict_all ───────────────────────────────────────────────────────────────

def test_predict_all_returns_one_result_per_matchup():
    matchups = [("A", "B", "Series 1"), ("C", "D", "Series 2"), ("E", "F", "Series 3")]
    scores = {"A": 60.0, "B": 50.0, "C": 55.0, "D": 45.0, "E": 70.0, "F": 40.0}
    results = predict_all(matchups, scores, {}, team_w=1.0, player_w=0.0, home_mult=1.0)
    assert len(results) == 3


def test_predict_all_preserves_order():
    matchups = [("A", "B", "First"), ("C", "D", "Second")]
    scores = {"A": 60.0, "B": 50.0, "C": 55.0, "D": 45.0}
    results = predict_all(matchups, scores, {}, team_w=1.0, player_w=0.0, home_mult=1.0)
    assert results[0].label == "First"
    assert results[1].label == "Second"
