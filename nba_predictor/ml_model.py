"""
ml_model.py
-----------
Logistic regression model trained on historical playoff series outcomes.

Features: stat differentials (home - away) for 7 team stats + player score.
Calibrated to produce reliable win probabilities on small datasets.
All functions are pure — data fetching is the caller's responsibility.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nba_predictor.config import ABBR_TO_FULL

TEAM_FEATURE_STATS = ["net_rtg", "drtg", "ortg", "pts", "ast", "3pm", "pace"]
FEATURE_STATS      = TEAM_FEATURE_STATS + ["player_score"]


@dataclass
class TrainedModel:
    classifier:     CalibratedClassifierCV
    scaler:         StandardScaler
    feature_names:  list[str]
    train_accuracy: float
    n_samples:      int
    cv_accuracy:    float | None = None


def build_feature_vector(home_scores: dict, away_scores: dict) -> np.ndarray:
    """Feature vector = home_stat - away_stat for each stat in FEATURE_STATS."""
    return np.array([
        home_scores.get(f, 0.0) - away_scores.get(f, 0.0)
        for f in FEATURE_STATS
    ])


def build_training_data(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert records into feature matrix X and label vector y."""
    X = np.array([build_feature_vector(r["home_scores"], r["away_scores"]) for r in records])
    y = np.array([int(r["home_won"]) for r in records])
    return X, y


def get_team_stats(team_df: pd.DataFrame, abbr: str) -> dict:
    """Extract a {stat: value} dict for a team by abbreviation. Returns {} if not found."""
    full_name = ABBR_TO_FULL.get(abbr, abbr)
    row = team_df[team_df["team"] == full_name]
    if row.empty:
        return {}
    return row.iloc[0][TEAM_FEATURE_STATS].to_dict()


def build_training_records(
    historical_playoffs: dict,
    season_team_dfs: dict[str, pd.DataFrame],
    season_player_dfs: dict[str, pd.DataFrame] | None = None,
) -> list[dict]:
    """
    Build training records from historical playoff data and pre-fetched DataFrames.
    Includes player_score differential if season_player_dfs is provided.
    Skips any matchup where team stats are missing.
    """
    from nba_predictor.model import build_player_scores

    records = []
    for season, data in historical_playoffs.items():
        team_df = season_team_dfs.get(season)
        if team_df is None:
            continue

        player_scores: dict[str, float] = {}
        if season_player_dfs and season in season_player_dfs:
            player_scores = build_player_scores(season_player_dfs[season])

        for home, away, label in data["matchups"]:
            actual = data["outcomes"].get(label)
            if not actual:
                continue
            home_stats = get_team_stats(team_df, home)
            away_stats = get_team_stats(team_df, away)
            if not home_stats or not away_stats:
                continue

            home_stats["player_score"] = player_scores.get(home, 0.0)
            away_stats["player_score"] = player_scores.get(away, 0.0)

            records.append({
                "season":      season,
                "home_scores": home_stats,
                "away_scores": away_stats,
                "home_won":    actual == home,
            })
    return records


def _fit_calibrated(X: np.ndarray, y: np.ndarray) -> CalibratedClassifierCV:
    """Fit a calibrated logistic regression. Uses sigmoid calibration for small datasets."""
    base = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    clf  = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    clf.fit(X, y)
    return clf


def train(records: list[dict]) -> TrainedModel:
    """Train a calibrated logistic regression on historical series records."""
    X, y = build_training_data(records)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf  = _fit_calibrated(X_s, y)
    train_acc = round(float(np.mean((clf.predict_proba(X_s)[:, 1] >= 0.5) == y) * 100), 1)
    return TrainedModel(
        classifier=clf,
        scaler=scaler,
        feature_names=FEATURE_STATS,
        train_accuracy=train_acc,
        n_samples=len(records),
    )


def cross_validate_loo_season(records: list[dict]) -> dict:
    """
    Leave-one-season-out cross-validation.
    Trains on all seasons except one, tests on the held-out season, repeats.
    Returns honest held-out accuracy rather than in-sample accuracy.
    """
    seasons = sorted({r["season"] for r in records})
    correct = total = 0

    for test_season in seasons:
        train_recs = [r for r in records if r["season"] != test_season]
        test_recs  = [r for r in records if r["season"] == test_season]

        if len(train_recs) < 6:
            continue
        classes = {r["home_won"] for r in train_recs}
        if len(classes) < 2:
            continue

        model = train(train_recs)
        for r in test_recs:
            prob = predict_win_probability(model, r["home_scores"], r["away_scores"])
            correct += int((prob >= 0.5) == r["home_won"])
            total   += 1

    return {
        "correct":        correct,
        "total":          total,
        "accuracy":       round(correct / total * 100, 1) if total else 0.0,
        "seasons_tested": len(seasons),
    }


def predict_win_probability(model: TrainedModel, home_scores: dict, away_scores: dict) -> float:
    """Return probability (0–1) that the home team wins."""
    x = build_feature_vector(home_scores, away_scores).reshape(1, -1)
    return float(model.classifier.predict_proba(model.scaler.transform(x))[0][1])


def feature_importances(model: TrainedModel) -> dict[str, float]:
    """
    Average logistic regression coefficients across calibrated folds.
    Positive = favours home team when home leads in that stat.
    """
    coefs = np.mean([
        clf.estimator.coef_[0]
        for clf in model.classifier.calibrated_classifiers_
    ], axis=0)
    return dict(zip(model.feature_names, coefs))
