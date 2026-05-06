"""
main.py
-------
Entry point. Orchestrates the pipeline:

    fetcher → model → display

Run with:  python main.py
"""

import sys
from nba_predictor.config import PLAYOFF_MATCHUPS
from nba_predictor.fetcher import fetch_team_df, fetch_player_df, FetchError
from nba_predictor.model import build_team_scores, build_player_scores, predict_all
from nba_predictor.display import print_predictions, print_top_players, print_model_summary


def run():
    # ── 1. Fetch ──────────────────────────────────────────────────────────────
    print("\nFetching data from NBA Stats API...")
    try:
        team_df   = fetch_team_df()
        player_df = fetch_player_df()
    except FetchError as e:
        print(f"\n❌ Error: {e}")
        print("   Check your internet connection and try again.")
        sys.exit(1)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    print("Running predictions...\n")
    team_scores   = build_team_scores(team_df)
    player_scores = build_player_scores(player_df)
    predictions   = predict_all(PLAYOFF_MATCHUPS, team_scores, player_scores)

    # ── 3. Display ────────────────────────────────────────────────────────────
    print_predictions(predictions)

    playoff_teams = list({abbr for home, away, _ in PLAYOFF_MATCHUPS for abbr in (home, away)})
    print_top_players(player_df, sorted(playoff_teams))
    print_model_summary()


if __name__ == "__main__":
    run()
