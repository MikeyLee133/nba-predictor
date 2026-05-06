"""
display.py
----------
Responsible for one thing only: formatting and printing output.

No scraping, no parsing, no model logic.
All data arrives as arguments — no global state accessed here.
"""

import pandas as pd
from tabulate import tabulate
from nba_predictor.config import ABBR_TO_FULL, TEAM_STAT_WEIGHTS, PLAYER_STAT_WEIGHTS
from nba_predictor.model import SeriesPrediction


def _full(abbr: str) -> str:
    """Resolve an abbreviation to a full team name, falling back to the abbr."""
    return ABBR_TO_FULL.get(abbr, abbr)


def print_predictions(predictions: list[SeriesPrediction]) -> None:
    """Print a formatted table of series predictions."""
    print("\n" + "=" * 68)
    print("  🏀  NBA PLAYOFF PREDICTOR — 2025-26 Second Round")
    print("=" * 68)

    rows = []
    for p in predictions:
        arrow = "◀" if p.predicted_winner == p.home else "▶"
        rows.append([
            f"{_full(p.home)} ({p.home})",
            f"{p.home_win_pct}%",
            arrow,
            f"{p.away_win_pct}%",
            f"{_full(p.away)} ({p.away})",
            f"→ {_full(p.predicted_winner)}",
        ])

    print(tabulate(
        rows,
        headers=["Home Team", "Win %", "", "Win %", "Away Team", "Predicted Winner"],
        tablefmt="rounded_outline",
    ))
    print()


def print_top_players(player_df: pd.DataFrame, teams: list[str], top_n: int = 3) -> None:
    """Print top N players (by PER) for each team in the given list."""
    print("=" * 68)
    print("  📊  TOP PLAYERS FOR PLAYOFF TEAMS (by PER)")
    print("=" * 68)

    rows = []
    for abbr in teams:
        group = player_df[player_df["team_id"] == abbr].dropna(subset=["per"])
        top = group.nlargest(top_n, "per")
        for _, row in top.iterrows():
            rows.append([
                abbr,
                row.get("player", "?"),
                row.get("pts_per_g", "?"),
                row.get("ast_per_g", "?"),
                row.get("trb_per_g", "?"),
                row.get("per", "?"),
            ])

    print(tabulate(
        rows,
        headers=["Team", "Player", "PPG", "APG", "RPG", "PER"],
        tablefmt="rounded_outline",
        floatfmt=".1f",
    ))
    print()


def print_model_summary() -> None:
    """Print a summary of the weights used in the current model."""
    print("=" * 68)
    print("  ⚙️   MODEL CONFIGURATION")
    print("=" * 68)

    team_rows  = [(s, f"{w*100:.0f}%") for s, w in TEAM_STAT_WEIGHTS.items()]
    player_rows = [(s, f"{w*100:.0f}%") for s, w in PLAYER_STAT_WEIGHTS.items()]

    print("\nTeam stat weights (60% of final score):")
    print(tabulate(team_rows, headers=["Stat", "Weight"], tablefmt="simple"))

    print("\nPlayer stat weights (40% of final score, top-3 by PER):")
    print(tabulate(player_rows, headers=["Stat", "Weight"], tablefmt="simple"))

    print("\nHome-court advantage: +4% multiplier on home team score")
    print("Recent form: last 15 games weighted at 40%, full season at 60%\n")
