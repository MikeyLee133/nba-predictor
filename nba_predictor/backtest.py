"""
backtest.py
-----------
Runs the prediction model against historical seasons and measures accuracy.
All functions are pure — data fetching is the caller's responsibility.
"""

from dataclasses import dataclass

import pandas as pd

from nba_predictor.model import build_player_scores, build_team_scores, predict_all


@dataclass
class BacktestResult:
    season: str
    series_label: str
    home: str
    away: str
    predicted_winner: str
    actual_winner: str
    correct: bool
    home_win_pct: float
    away_win_pct: float


def run_season_backtest(
    season: str,
    matchups: list[tuple[str, str, str]],
    outcomes: dict[str, str],
    team_df: pd.DataFrame,
    player_df: pd.DataFrame,
) -> list[BacktestResult]:
    """
    Predict every series in matchups and compare to known outcomes.
    Series with no entry in outcomes are skipped.
    Returns one BacktestResult per resolved series.
    """
    team_scores   = build_team_scores(team_df)
    player_scores = build_player_scores(player_df)
    predictions   = predict_all(matchups, team_scores, player_scores)

    results = []
    for pred in predictions:
        actual = outcomes.get(pred.label)
        if actual is None:
            continue
        results.append(BacktestResult(
            season=season,
            series_label=pred.label,
            home=pred.home,
            away=pred.away,
            predicted_winner=pred.predicted_winner,
            actual_winner=actual,
            correct=pred.predicted_winner == actual,
            home_win_pct=pred.home_win_pct,
            away_win_pct=pred.away_win_pct,
        ))
    return results


def backtest_accuracy(results: list[BacktestResult]) -> dict:
    """Compute accuracy stats for a list of BacktestResults."""
    correct = sum(1 for r in results if r.correct)
    total   = len(results)
    return {
        "correct": correct,
        "total":   total,
        "pct":     round(correct / total * 100, 1) if total else 0.0,
    }


def higher_seed_baseline(historical_playoffs: dict) -> dict:
    """
    Compute accuracy of always predicting the home team (higher seed) wins.
    Used as a naive baseline to contextualise model accuracy.
    """
    correct = total = 0
    for data in historical_playoffs.values():
        for home, _away, label in data["matchups"]:
            actual = data["outcomes"].get(label)
            if actual is None:
                continue
            correct += int(actual == home)
            total   += 1
    return {
        "correct": correct,
        "total":   total,
        "pct":     round(correct / total * 100, 1) if total else 0.0,
    }
