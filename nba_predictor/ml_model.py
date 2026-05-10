"""
ml_model.py
-----------
Logistic regression model trained on historical playoff series outcomes.
Takes stat differentials (home - away) as features and predicts win probability.

All functions are pure — data fetching is the caller's responsibility.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nba_predictor.config import ABBR_TO_FULL

FEATURE_STATS = ["net_rtg", "drtg", "ortg", "pts", "ast", "3pm", "pace"]


@dataclass
class TrainedModel:
    classifier:     LogisticRegression
    scaler:         StandardScaler
    feature_names:  list[str]
    train_accuracy: float
    n_samples:      int


def build_feature_vector(home_scores: dict, away_scores: dict) -> np.ndarray:
    """Feature vector = home_stat - away_stat for each stat in FEATURE_STATS."""
    return np.array([
        home_scores.get(f, 0.0) - away_scores.get(f, 0.0)
        for f in FEATURE_STATS
    ])


def build_training_data(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of {home_scores, away_scores, home_won} records into
    a feature matrix X and label vector y.
    """
    X = np.array([build_feature_vector(r["home_scores"], r["away_scores"]) for r in records])
    y = np.array([int(r["home_won"]) for r in records])
    return X, y


def get_team_stats(team_df: pd.DataFrame, abbr: str) -> dict:
    """Extract a {stat: value} dict for a team by abbreviation. Returns {} if not found."""
    full_name = ABBR_TO_FULL.get(abbr, abbr)
    row = team_df[team_df["team"] == full_name]
    if row.empty:
        return {}
    return row.iloc[0][FEATURE_STATS].to_dict()


def build_training_records(
    historical_playoffs: dict,
    season_team_dfs: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    Build training records from historical playoff data and pre-fetched DataFrames.
    Skips any matchup where team stats are missing from the DataFrame.

    historical_playoffs: {season: {matchups: [...], outcomes: {...}}}
    season_team_dfs:     {season: team_df}
    """
    records = []
    for season, data in historical_playoffs.items():
        team_df = season_team_dfs.get(season)
        if team_df is None:
            continue
        for home, away, label in data["matchups"]:
            actual = data["outcomes"].get(label)
            if not actual:
                continue
            home_stats = get_team_stats(team_df, home)
            away_stats = get_team_stats(team_df, away)
            if not home_stats or not away_stats:
                continue
            records.append({
                "home_scores": home_stats,
                "away_scores": away_stats,
                "home_won":    actual == home,
            })
    return records


def train(records: list[dict]) -> TrainedModel:
    """Train a logistic regression model on historical series records."""
    X, y = build_training_data(records)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    clf.fit(X_s, y)
    train_acc = round(float((clf.predict(X_s) == y).mean() * 100), 1)
    return TrainedModel(
        classifier=clf,
        scaler=scaler,
        feature_names=FEATURE_STATS,
        train_accuracy=train_acc,
        n_samples=len(records),
    )


def predict_win_probability(model: TrainedModel, home_scores: dict, away_scores: dict) -> float:
    """Return probability (0–1) that the home team wins."""
    x = build_feature_vector(home_scores, away_scores).reshape(1, -1)
    return float(model.classifier.predict_proba(model.scaler.transform(x))[0][1])


def feature_importances(model: TrainedModel) -> dict[str, float]:
    """
    Logistic regression coefficients keyed by feature name.
    Positive = favours home team when home has the higher value.
    """
    return dict(zip(model.feature_names, model.classifier.coef_[0]))
