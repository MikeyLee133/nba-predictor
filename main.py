"""
main.py
-------
Entry point. Orchestrates the pipeline:

    scraper → parser → model → display

This file contains no business logic — it only wires the modules together.
Run with:  python main.py
"""

import sys
from nba_predictor.config import PLAYOFF_MATCHUPS
from nba_predictor.scraper import fetch_team_stats_html, fetch_player_stats_html, ScraperError
from nba_predictor.parser import parse_team_stats, parse_player_stats, ParseError
from nba_predictor.model import build_team_scores, build_player_scores, predict_all
from nba_predictor.display import print_predictions, print_top_players, print_model_summary

def run():
    # ── 1. Fetch ──────────────────────────────────────────────────────────────
    print("\nFetching data from Basketball Reference...")
    try:
        team_html   = fetch_team_stats_html()
        player_html = fetch_player_stats_html()
    except ScraperError as e:
        print(f"\n❌ Network error: {e}")
        print("   Check your internet connection and try again.")
        sys.exit(1)

    # ── 2. Parse ──────────────────────────────────────────────────────────────
    print("Parsing stats...")
    try:
        team_df   = parse_team_stats(team_html)
        player_df = parse_player_stats(player_html)
    except ParseError as e:
        print(f"\n❌ Parse error: {e}")
        sys.exit(1)

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("Running predictions...\n")
    team_scores   = build_team_scores(team_df)
    player_scores = build_player_scores(player_df)
    predictions   = predict_all(PLAYOFF_MATCHUPS, team_scores, player_scores)

    # ── 4. Display ────────────────────────────────────────────────────────────
    print_predictions(predictions)

    playoff_teams = list({abbr for home, away, _ in PLAYOFF_MATCHUPS for abbr in (home, away)})
    print_top_players(player_df, sorted(playoff_teams))
    print_model_summary()


if __name__ == "__main__":
    run()
