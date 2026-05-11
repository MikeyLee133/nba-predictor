import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nba_predictor.api import app, get_team_data, get_player_data
from nba_predictor.config import PLAYOFF_MATCHUPS, SEASON


# ── mock data ─────────────────────────────────────────────────────────────────

def _mock_team_df():
    teams = {
        "DET": ("Detroit Pistons",     -2.0, 114.0, 112.0, 110.0, 24.0, 12.0, 98.0),
        "CLE": ("Cleveland Cavaliers",  5.0, 109.0, 114.0, 113.0, 26.0, 13.0, 99.0),
        "NYK": ("New York Knicks",      3.0, 110.0, 113.0, 112.0, 25.0, 13.0, 97.0),
        "PHI": ("Philadelphia 76ers",   1.0, 112.0, 113.0, 111.0, 24.0, 12.0, 98.0),
        "OKC": ("Oklahoma City Thunder",8.0, 108.0, 116.0, 117.0, 27.0, 14.0, 100.0),
        "LAL": ("Los Angeles Lakers",   2.0, 111.0, 113.0, 111.0, 25.0, 12.0, 98.0),
        "SAS": ("San Antonio Spurs",   -5.0, 116.0, 111.0, 107.0, 23.0, 11.0, 97.0),
        "MIN": ("Minnesota Timberwolves",4.0, 109.0, 113.0, 112.0, 25.0, 12.0, 98.0),
    }
    rows = [{"team": full, "net_rtg": nr, "drtg": dr, "ortg": or_,
             "pts": pts, "ast": ast, "3pm": tpm, "pace": pace}
            for abbr, (full, nr, dr, or_, pts, ast, tpm, pace) in teams.items()]
    return pd.DataFrame(rows)


def _mock_player_df():
    rows = [
        {"team_id": abbr, "player": f"Player {abbr}", "per": 20.0,
         "pts_per_g": 18.0, "ast_per_g": 4.0, "trb_per_g": 4.0, "fg3_per_g": 2.0}
        for abbr in ["DET", "CLE", "NYK", "PHI", "OKC", "LAL", "SAS", "MIN"]
    ]
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def override_deps():
    app.dependency_overrides[get_team_data]   = lambda: _mock_team_df()
    app.dependency_overrides[get_player_data] = lambda: _mock_player_df()
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


# ── GET /health ───────────────────────────────────────────────────────────────

def test_health_returns_200():
    assert client.get("/health").status_code == 200


def test_health_returns_ok_status():
    assert client.get("/health").json()["status"] == "ok"


def test_health_includes_current_season():
    assert client.get("/health").json()["season"] == SEASON


# ── GET /predictions ──────────────────────────────────────────────────────────

def test_predictions_returns_200():
    assert client.get("/predictions").status_code == 200


def test_predictions_returns_list():
    assert isinstance(client.get("/predictions").json(), list)


def test_predictions_count_matches_matchups():
    assert len(client.get("/predictions").json()) == len(PLAYOFF_MATCHUPS)


def test_predictions_response_has_required_fields():
    pred = client.get("/predictions").json()[0]
    for field in ["series_label", "home", "home_team", "away", "away_team",
                  "home_win_pct", "away_win_pct", "predicted_winner", "predicted_winner_full"]:
        assert field in pred, f"Missing field: {field}"


def test_predictions_win_pcts_sum_to_100():
    for pred in client.get("/predictions").json():
        assert abs(pred["home_win_pct"] + pred["away_win_pct"] - 100.0) < 0.2


def test_predictions_winner_is_home_or_away():
    for pred in client.get("/predictions").json():
        assert pred["predicted_winner"] in (pred["home"], pred["away"])


def test_predictions_full_names_populated():
    for pred in client.get("/predictions").json():
        assert len(pred["home_team"]) > 3
        assert len(pred["away_team"]) > 3


# ── GET /predictions/{home}/{away} ────────────────────────────────────────────

def test_single_prediction_returns_200():
    home, away, _ = PLAYOFF_MATCHUPS[0]
    assert client.get(f"/predictions/{home}/{away}").status_code == 200


def test_single_prediction_correct_teams():
    home, away, _ = PLAYOFF_MATCHUPS[0]
    pred = client.get(f"/predictions/{home}/{away}").json()
    assert pred["home"] == home
    assert pred["away"] == away


def test_single_prediction_case_insensitive():
    home, away, _ = PLAYOFF_MATCHUPS[0]
    assert client.get(f"/predictions/{home.lower()}/{away.lower()}").status_code == 200


def test_single_prediction_404_for_unknown_matchup():
    assert client.get("/predictions/XYZ/ABC").status_code == 404


def test_single_prediction_404_detail_mentions_teams():
    resp = client.get("/predictions/XYZ/ABC")
    assert "XYZ" in resp.json()["detail"]


# ── GET /teams ────────────────────────────────────────────────────────────────

def test_teams_returns_200():
    assert client.get("/teams").status_code == 200


def test_teams_returns_list():
    assert isinstance(client.get("/teams").json(), list)


def test_teams_count_matches_playoff_teams():
    expected = len({abbr for h, a, _ in PLAYOFF_MATCHUPS for abbr in (h, a)})
    assert len(client.get("/teams").json()) == expected


def test_teams_have_required_fields():
    team = client.get("/teams").json()[0]
    for field in ["abbreviation", "full_name", "composite_score"]:
        assert field in team


def test_teams_scores_are_numeric():
    for team in client.get("/teams").json():
        assert isinstance(team["composite_score"], (int, float))
