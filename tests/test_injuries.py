import pandas as pd
import pytest

from nba_predictor.model import build_player_scores


def _player_df(rows):
    return pd.DataFrame(rows)


def _star(team, name, per=30.0, pts=28.0):
    return {"team_id": team, "player": name, "per": per,
            "pts_per_g": pts, "ast_per_g": 6.0, "trb_per_g": 5.0, "fg3_per_g": 2.0}


# ── unavailable players are excluded ─────────────────────────────────────────

def test_unavailable_player_excluded_lowers_team_score():
    df = _player_df([
        _star("OKC", "SGA",   per=32.0, pts=30.0),
        _star("OKC", "Holmgren", per=20.0, pts=16.0),
        _star("LAL", "LBJ",   per=24.0, pts=22.0),
    ])
    score_full    = build_player_scores(df)["OKC"]
    score_injured = build_player_scores(df, unavailable={"SGA"})["OKC"]
    assert score_injured < score_full


def test_unavailable_player_not_counted_even_if_highest_per():
    df = _player_df([
        _star("OKC", "SGA",      per=32.0, pts=30.0),
        _star("OKC", "Holmgren", per=20.0, pts=16.0),
    ])
    # SGA has the best PER — marking him out should drop OKC's score
    with_sga    = build_player_scores(df)["OKC"]
    without_sga = build_player_scores(df, unavailable={"SGA"})["OKC"]
    assert without_sga < with_sga


def test_unavailable_does_not_affect_other_teams():
    df = _player_df([
        _star("OKC", "SGA", per=32.0, pts=30.0),
        _star("LAL", "LBJ", per=24.0, pts=22.0),
    ])
    lal_full    = build_player_scores(df)["LAL"]
    lal_injured = build_player_scores(df, unavailable={"SGA"})["LAL"]
    assert lal_full == pytest.approx(lal_injured)


def test_all_players_unavailable_returns_zero():
    df = _player_df([_star("OKC", "SGA")])
    scores = build_player_scores(df, unavailable={"SGA"})
    assert scores.get("OKC", 0.0) == 0.0


def test_empty_unavailable_set_has_no_effect():
    df = _player_df([_star("OKC", "SGA"), _star("LAL", "LBJ")])
    scores_default = build_player_scores(df)
    scores_empty   = build_player_scores(df, unavailable=set())
    assert scores_default["OKC"] == pytest.approx(scores_empty["OKC"])
    assert scores_default["LAL"] == pytest.approx(scores_empty["LAL"])


def test_none_unavailable_matches_default():
    df = _player_df([_star("OKC", "SGA"), _star("LAL", "LBJ")])
    assert build_player_scores(df) == build_player_scores(df, unavailable=None)


def test_unavailable_multiple_players():
    df = _player_df([
        _star("OKC", "SGA",      per=32.0, pts=30.0),
        _star("OKC", "Holmgren", per=20.0, pts=16.0),
        _star("OKC", "Williams", per=15.0, pts=12.0),
    ])
    score_none = build_player_scores(df)["OKC"]
    score_two  = build_player_scores(df, unavailable={"SGA", "Holmgren"})["OKC"]
    assert score_two < score_none
