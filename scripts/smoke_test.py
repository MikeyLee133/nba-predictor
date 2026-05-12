"""
scripts/smoke_test.py
---------------------
End-to-end validation against the live NBA Stats API.
Checks that the API is reachable, data schema is intact,
and the full prediction pipeline produces valid output.

Exits with code 1 on any failure — designed for CI.

Usage:
    python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_predictor.config import PLAYOFF_MATCHUPS, PLAYER_STAT_WEIGHTS, TEAM_STAT_WEIGHTS
from nba_predictor.fetcher import fetch_player_df, fetch_team_df
from nba_predictor.model import build_player_scores, build_team_scores, predict_all

PASS = "✓"
FAIL = "✗"
_failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  {PASS}  {message}")
    else:
        print(f"  {FAIL}  {message}")
        _failures.append(message)


def main() -> None:
    print("\n── Fetching team stats ───────────────────────────────")
    team_df = fetch_team_df()
    check(len(team_df) >= 30, f"team_df has {len(team_df)} rows (expected ≥ 30)")
    check("team" in team_df.columns, "team_df has 'team' column")
    for col in TEAM_STAT_WEIGHTS:
        check(col in team_df.columns, f"team_df has stat column '{col}'")

    print("\n── Fetching player stats ─────────────────────────────")
    player_df = fetch_player_df()
    check(len(player_df) > 200, f"player_df has {len(player_df)} rows (expected > 200)")
    for col in ["player", "team_id", "per", "pts_per_g"]:
        check(col in player_df.columns, f"player_df has column '{col}'")
    for col in PLAYER_STAT_WEIGHTS:
        check(col in player_df.columns, f"player_df has stat column '{col}'")

    print("\n── Running prediction pipeline ───────────────────────")
    team_scores   = build_team_scores(team_df)
    player_scores = build_player_scores(player_df)
    predictions   = predict_all(PLAYOFF_MATCHUPS, team_scores, player_scores)

    check(
        len(predictions) == len(PLAYOFF_MATCHUPS),
        f"got {len(predictions)} predictions (expected {len(PLAYOFF_MATCHUPS)})",
    )
    for p in predictions:
        check(abs(p.home_win_pct + p.away_win_pct - 100.0) < 0.2,
              f"{p.label}: win % sums to 100")
        check(0 < p.home_win_pct < 100,
              f"{p.label}: home_win_pct in valid range ({p.home_win_pct})")
        check(p.predicted_winner in (p.home, p.away),
              f"{p.label}: predicted_winner is home or away")

    print("\n── Result ────────────────────────────────────────────")
    if _failures:
        print(f"  {FAIL}  {len(_failures)} check(s) failed:")
        for f in _failures:
            print(f"       • {f}")
        sys.exit(1)
    else:
        print(f"  {PASS}  All checks passed — {len(predictions)} predictions produced\n")


if __name__ == "__main__":
    main()
