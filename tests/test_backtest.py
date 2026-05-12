import pandas as pd
import pytest

from nba_predictor.backtest import BacktestResult, run_season_backtest, backtest_accuracy, higher_seed_baseline


# ── helpers ───────────────────────────────────────────────────────────────────

def _team_df(teams: dict) -> pd.DataFrame:
    return pd.DataFrame([{"team": name, **stats} for name, stats in teams.items()])


def _player_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _minimal_player(team, name, per=20.0):
    return {"team_id": team, "player": name, "per": per,
            "pts_per_g": 15.0, "ast_per_g": 4.0, "trb_per_g": 4.0, "fg3_per_g": 1.0}


def _make_dfs():
    team_df = _team_df({
        "Boston Celtics":     {"net_rtg": 8.0, "drtg": 108.0, "ortg": 116.0, "pts": 115.0, "ast": 26.0, "3pm": 14.0, "pace": 98.0},
        "Cleveland Cavaliers":{"net_rtg": 3.0, "drtg": 112.0, "ortg": 115.0, "pts": 110.0, "ast": 24.0, "3pm": 12.0, "pace": 97.0},
    })
    player_df = _player_df([
        _minimal_player("BOS", "Tatum", per=28.0),
        _minimal_player("CLE", "Mitchell", per=22.0),
    ])
    return team_df, player_df


# ── run_season_backtest ───────────────────────────────────────────────────────

def test_returns_one_result_per_matchup_with_known_outcome():
    team_df, player_df = _make_dfs()
    matchups = [("BOS", "CLE", "East Semis: Boston vs Cleveland")]
    outcomes = {"East Semis: Boston vs Cleveland": "BOS"}
    results = run_season_backtest("2023-24", matchups, outcomes, team_df, player_df)
    assert len(results) == 1


def test_skips_matchups_with_no_outcome():
    team_df, player_df = _make_dfs()
    matchups = [
        ("BOS", "CLE", "East Semis: Boston vs Cleveland"),
        ("BOS", "CLE", "Unknown Series"),
    ]
    outcomes = {"East Semis: Boston vs Cleveland": "BOS"}
    results = run_season_backtest("2023-24", matchups, outcomes, team_df, player_df)
    assert len(results) == 1


def test_correct_flag_true_when_prediction_matches_outcome():
    team_df, player_df = _make_dfs()
    matchups = [("BOS", "CLE", "East Semis")]
    # BOS has better stats so model should predict BOS
    team_scores_bos_wins = _team_df({
        "Boston Celtics":     {"net_rtg": 10.0, "drtg": 106.0, "ortg": 116.0, "pts": 118.0, "ast": 27.0, "3pm": 15.0, "pace": 99.0},
        "Cleveland Cavaliers":{"net_rtg": 1.0,  "drtg": 114.0, "ortg": 115.0, "pts": 108.0, "ast": 23.0, "3pm": 11.0, "pace": 96.0},
    })
    results = run_season_backtest("2023-24", matchups, {"East Semis": "BOS"}, team_scores_bos_wins, player_df)
    assert results[0].correct is True


def test_correct_flag_false_when_prediction_wrong():
    team_df, player_df = _make_dfs()
    matchups = [("BOS", "CLE", "East Semis")]
    strong_bos_df = _team_df({
        "Boston Celtics":     {"net_rtg": 10.0, "drtg": 106.0, "ortg": 116.0, "pts": 118.0, "ast": 27.0, "3pm": 15.0, "pace": 99.0},
        "Cleveland Cavaliers":{"net_rtg": 1.0,  "drtg": 114.0, "ortg": 115.0, "pts": 108.0, "ast": 23.0, "3pm": 11.0, "pace": 96.0},
    })
    # Model will predict BOS but actual winner is CLE — should be wrong
    results = run_season_backtest("2023-24", matchups, {"East Semis": "CLE"}, strong_bos_df, player_df)
    assert results[0].correct is False


def test_result_fields_populated():
    team_df, player_df = _make_dfs()
    matchups = [("BOS", "CLE", "East Semis: Boston vs Cleveland")]
    outcomes = {"East Semis: Boston vs Cleveland": "BOS"}
    r = run_season_backtest("2023-24", matchups, outcomes, team_df, player_df)[0]
    assert r.season == "2023-24"
    assert r.series_label == "East Semis: Boston vs Cleveland"
    assert r.home == "BOS"
    assert r.away == "CLE"
    assert r.actual_winner == "BOS"
    assert r.home_win_pct + r.away_win_pct == pytest.approx(100.0)


def test_multiple_matchups_all_returned():
    team_df = _team_df({
        "Boston Celtics":     {"net_rtg": 8.0, "drtg": 108.0, "ortg": 116.0, "pts": 115.0, "ast": 26.0, "3pm": 14.0, "pace": 98.0},
        "Cleveland Cavaliers":{"net_rtg": 3.0, "drtg": 112.0, "ortg": 115.0, "pts": 110.0, "ast": 24.0, "3pm": 12.0, "pace": 97.0},
        "New York Knicks":    {"net_rtg": 5.0, "drtg": 110.0, "ortg": 115.0, "pts": 112.0, "ast": 25.0, "3pm": 13.0, "pace": 97.0},
        "Indiana Pacers":     {"net_rtg": 2.0, "drtg": 113.0, "ortg": 115.0, "pts": 109.0, "ast": 28.0, "3pm": 11.0, "pace": 101.0},
    })
    player_df = _player_df([
        _minimal_player("BOS", "Tatum"), _minimal_player("CLE", "Mitchell"),
        _minimal_player("NYK", "Brunson"), _minimal_player("IND", "Haliburton"),
    ])
    matchups = [
        ("BOS", "CLE", "East Semis 1"),
        ("NYK", "IND", "East Semis 2"),
    ]
    outcomes = {"East Semis 1": "BOS", "East Semis 2": "IND"}
    results = run_season_backtest("2023-24", matchups, outcomes, team_df, player_df)
    assert len(results) == 2


# ── backtest_accuracy ─────────────────────────────────────────────────────────

def _result(correct: bool, season="2023-24") -> BacktestResult:
    return BacktestResult(
        season=season, series_label="Test", home="A", away="B",
        predicted_winner="A" if correct else "B",
        actual_winner="A",
        correct=correct,
        home_win_pct=60.0, away_win_pct=40.0,
    )


def test_accuracy_correct_count():
    results = [_result(True), _result(True), _result(False)]
    stats = backtest_accuracy(results)
    assert stats["correct"] == 2
    assert stats["total"] == 3


def test_accuracy_pct():
    results = [_result(True), _result(True), _result(False), _result(False)]
    stats = backtest_accuracy(results)
    assert stats["pct"] == pytest.approx(50.0)


def test_accuracy_all_correct():
    results = [_result(True)] * 4
    stats = backtest_accuracy(results)
    assert stats["pct"] == pytest.approx(100.0)
    assert stats["correct"] == 4


# ── higher_seed_baseline ──────────────────────────────────────────────────────

def _playoffs(home_wins: list[bool]) -> dict:
    """Build a minimal HISTORICAL_PLAYOFFS dict where home team wins according to home_wins."""
    matchups = [(f"H{i}", f"A{i}", f"Series {i}") for i in range(len(home_wins))]
    outcomes = {
        f"Series {i}": (f"H{i}" if won else f"A{i}")
        for i, won in enumerate(home_wins)
    }
    return {"2023-24": {"matchups": matchups, "outcomes": outcomes}}


def test_higher_seed_baseline_all_home_wins():
    stats = higher_seed_baseline(_playoffs([True, True, True, True]))
    assert stats["correct"] == 4
    assert stats["total"] == 4
    assert stats["pct"] == pytest.approx(100.0)


def test_higher_seed_baseline_mixed():
    stats = higher_seed_baseline(_playoffs([True, True, False, False]))
    assert stats["correct"] == 2
    assert stats["total"] == 4
    assert stats["pct"] == pytest.approx(50.0)


def test_higher_seed_baseline_all_upsets():
    stats = higher_seed_baseline(_playoffs([False, False, False]))
    assert stats["correct"] == 0
    assert stats["pct"] == pytest.approx(0.0)


def test_higher_seed_baseline_multiple_seasons():
    playoffs = {
        "2022-23": {
            "matchups": [("H0", "A0", "S0"), ("H1", "A1", "S1")],
            "outcomes": {"S0": "H0", "S1": "A1"},
        },
        "2023-24": {
            "matchups": [("H2", "A2", "S2")],
            "outcomes": {"S2": "H2"},
        },
    }
    stats = higher_seed_baseline(playoffs)
    assert stats["correct"] == 2
    assert stats["total"] == 3


def test_higher_seed_baseline_skips_missing_outcomes():
    playoffs = {"2023-24": {
        "matchups": [("H0", "A0", "S0"), ("H1", "A1", "S1")],
        "outcomes": {"S0": "H0"},  # S1 has no outcome
    }}
    stats = higher_seed_baseline(playoffs)
    assert stats["total"] == 1


def test_accuracy_empty_results():
    stats = backtest_accuracy([])
    assert stats["correct"] == 0
    assert stats["total"] == 0
    assert stats["pct"] == 0.0


def test_accuracy_grouped_by_season():
    results = [
        _result(True,  season="2023-24"),
        _result(True,  season="2023-24"),
        _result(False, season="2022-23"),
        _result(True,  season="2022-23"),
    ]
    by_season = {}
    for r in results:
        by_season.setdefault(r.season, []).append(r)

    assert backtest_accuracy(by_season["2023-24"])["correct"] == 2
    assert backtest_accuracy(by_season["2022-23"])["correct"] == 1
