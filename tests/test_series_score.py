import pytest
from nba_predictor.model import SeriesPrediction, adjust_for_series_score


def _pred(home_pct=55.0):
    return SeriesPrediction(
        home="HOM", away="AWY",
        home_win_pct=home_pct,
        away_win_pct=round(100 - home_pct, 1),
        predicted_winner="HOM" if home_pct >= 50 else "AWY",
        label="Test Series",
    )


# ── 0-0: no adjustment ────────────────────────────────────────────────────────

def test_no_adjustment_at_zero_zero():
    p = _pred(55.0)
    result = adjust_for_series_score(p, home_wins=0, away_wins=0)
    assert result.home_win_pct == pytest.approx(55.0)
    assert result.away_win_pct == pytest.approx(45.0)


# ── win % always sums to 100 ──────────────────────────────────────────────────

def test_win_pcts_sum_to_100_home_leads():
    result = adjust_for_series_score(_pred(), home_wins=2, away_wins=1)
    assert result.home_win_pct + result.away_win_pct == pytest.approx(100.0)


def test_win_pcts_sum_to_100_away_leads():
    result = adjust_for_series_score(_pred(), home_wins=1, away_wins=2)
    assert result.home_win_pct + result.away_win_pct == pytest.approx(100.0)


# ── large lead pushes probability toward leader ───────────────────────────────

def test_home_3_1_lead_strongly_favours_home():
    result = adjust_for_series_score(_pred(50.0), home_wins=3, away_wins=1)
    assert result.home_win_pct > 70.0
    assert result.predicted_winner == "HOM"


def test_away_3_1_lead_strongly_favours_away():
    result = adjust_for_series_score(_pred(50.0), home_wins=1, away_wins=3)
    assert result.away_win_pct > 70.0
    assert result.predicted_winner == "AWY"


def test_home_3_0_lead_near_certain_home_win():
    # blend of model(50%) and historical(99%) = 74.5%
    result = adjust_for_series_score(_pred(50.0), home_wins=3, away_wins=0)
    assert result.home_win_pct > 70.0


def test_away_3_0_lead_near_certain_away_win():
    result = adjust_for_series_score(_pred(50.0), home_wins=0, away_wins=3)
    assert result.away_win_pct > 70.0


# ── series score moves probability in the right direction ────────────────────

def test_home_lead_increases_home_pct_vs_model():
    base = _pred(50.0)
    result = adjust_for_series_score(base, home_wins=2, away_wins=0)
    assert result.home_win_pct > base.home_win_pct


def test_away_lead_decreases_home_pct_vs_model():
    base = _pred(50.0)
    result = adjust_for_series_score(base, home_wins=0, away_wins=2)
    assert result.home_win_pct < base.home_win_pct


def test_close_series_moderate_adjustment():
    result = adjust_for_series_score(_pred(50.0), home_wins=2, away_wins=1)
    # 2-1 lead is meaningful but not decisive — home should be moderately favoured
    assert 55.0 < result.home_win_pct < 80.0


# ── predicted_winner is consistent with win_pct ──────────────────────────────

def test_predicted_winner_matches_higher_pct_home():
    result = adjust_for_series_score(_pred(50.0), home_wins=3, away_wins=1)
    assert result.predicted_winner == ("HOM" if result.home_win_pct >= 50 else "AWY")


def test_predicted_winner_matches_higher_pct_away():
    result = adjust_for_series_score(_pred(50.0), home_wins=1, away_wins=3)
    assert result.predicted_winner == ("HOM" if result.home_win_pct >= 50 else "AWY")


# ── metadata preserved ────────────────────────────────────────────────────────

def test_home_away_and_label_preserved():
    p = _pred()
    result = adjust_for_series_score(p, home_wins=1, away_wins=0)
    assert result.home == p.home
    assert result.away == p.away
    assert result.label == p.label
