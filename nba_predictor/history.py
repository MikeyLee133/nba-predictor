"""
history.py
----------
Persists predictions and actual outcomes to a JSON file so model
accuracy can be tracked across playoff rounds.
"""

import json
from pathlib import Path

from nba_predictor.model import SeriesPrediction


def load_history(path: Path) -> list[dict]:
    """Load all prediction records. Returns empty list if file doesn't exist."""
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_predictions(predictions: list[SeriesPrediction], round_label: str, path: Path) -> None:
    """Append predictions to history. Skips any series already saved for this round."""
    records = load_history(path)
    existing = {(r["series_label"], r["round"]) for r in records}

    for p in predictions:
        if (p.label, round_label) in existing:
            continue
        records.append({
            "season":           _season_from_path(path),
            "round":            round_label,
            "series_label":     p.label,
            "home":             p.home,
            "away":             p.away,
            "predicted_winner": p.predicted_winner,
            "home_win_pct":     p.home_win_pct,
            "away_win_pct":     p.away_win_pct,
            "actual_winner":    None,
            "correct":          None,
        })

    path.write_text(json.dumps(records, indent=2))


def record_outcome(series_label: str, actual_winner: str, path: Path) -> bool:
    """
    Record the actual winner of a series.
    Returns True if the series was found, False otherwise.
    """
    records = load_history(path)
    for r in records:
        if r["series_label"] == series_label:
            r["actual_winner"] = actual_winner
            r["correct"] = actual_winner == r["predicted_winner"]
            path.write_text(json.dumps(records, indent=2))
            return True
    return False


def accuracy_stats(records: list[dict]) -> dict:
    """Compute accuracy over all resolved predictions (those with actual_winner set)."""
    resolved = [r for r in records if r["actual_winner"] is not None]
    correct = sum(1 for r in resolved if r["correct"])
    total = len(resolved)
    return {
        "correct": correct,
        "total":   total,
        "pct":     round(correct / total * 100, 1) if total else 0.0,
    }


def _season_from_path(path: Path) -> str:
    from nba_predictor.config import SEASON
    return SEASON
