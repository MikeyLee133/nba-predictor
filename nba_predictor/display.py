"""
display.py
----------
Responsible for one thing only: formatting and printing output.

No scraping, no parsing, no model logic.
All data arrives as arguments — no global state accessed here.
"""

import pandas as pd
from tabulate import tabulate
from nba_predictor.config import (
    ABBR_TO_FULL, TEAM_STAT_WEIGHTS, PLAYER_STAT_WEIGHTS,
    SEASON, TEAM_SCORE_WEIGHT, PLAYER_SCORE_WEIGHT,
    TOP_PLAYERS_PER_TEAM, RECENT_GAMES,
)
from nba_predictor.model import SeriesPrediction


def _full(abbr: str) -> str:
    """Resolve an abbreviation to a full team name, falling back to the abbr."""
    return ABBR_TO_FULL.get(abbr, abbr)


def print_predictions(predictions: list[SeriesPrediction]) -> None:
    """Print a formatted table of series predictions."""
    print("\n" + "=" * 68)
    print(f"  🏀  NBA PLAYOFF PREDICTOR — {SEASON} Second Round")
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


def print_top_players(player_df: pd.DataFrame, teams: list[str], top_n: int = TOP_PLAYERS_PER_TEAM) -> None:
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

    print(f"\nTeam stat weights ({TEAM_SCORE_WEIGHT*100:.0f}% of final score):")
    print(tabulate(team_rows, headers=["Stat", "Weight"], tablefmt="simple"))

    print(f"\nPlayer stat weights ({PLAYER_SCORE_WEIGHT*100:.0f}% of final score, top-{TOP_PLAYERS_PER_TEAM} by PER):")
    print(tabulate(player_rows, headers=["Stat", "Weight"], tablefmt="simple"))

    print("\nHome-court advantage: +4% multiplier on home team score")
    print(f"Recent form: last {RECENT_GAMES} games\n")
