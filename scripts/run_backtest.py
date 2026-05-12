"""
scripts/run_backtest.py
-----------------------
Runs the weighted model and ML model against all historical seasons
and prints a comparison table vs naive baselines.

Usage:
    python scripts/run_backtest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nba_predictor.backtest import backtest_accuracy, higher_seed_baseline, run_season_backtest
from nba_predictor.fetcher import fetch_seasons_parallel
from nba_predictor.historical import HISTORICAL_PLAYOFFS
from nba_predictor.ml_model import build_training_records, cross_validate_loo_season


def _bar(pct: float, width: int = 20) -> str:
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    seasons = list(HISTORICAL_PLAYOFFS.keys())
    total_series = sum(len(d["matchups"]) for d in HISTORICAL_PLAYOFFS.values())

    print(f"\nFetching stats for {len(seasons)} seasons ({total_series} series)...")
    season_data = fetch_seasons_parallel(seasons)
    fetched = len(season_data)
    print(f"  Fetched {fetched}/{len(seasons)} seasons successfully\n")

    # ── Weighted model ─────────────────────────────────────────────────────────
    all_results = []
    print("Weighted model — per season:")
    for season, data in HISTORICAL_PLAYOFFS.items():
        if season not in season_data:
            print(f"  {season}  skipped (fetch failed)")
            continue
        results = run_season_backtest(
            season, data["matchups"], data["outcomes"],
            season_data[season]["team"], season_data[season]["player"],
        )
        all_results.extend(results)
        s = backtest_accuracy(results)
        print(f"  {season}  {s['correct']:>2}/{s['total']:>2}  ({s['pct']:>5.1f}%)")

    weighted_stats = backtest_accuracy(all_results)

    # ── ML model (LOSO-CV) ─────────────────────────────────────────────────────
    print("\nML model — leave-one-season-out cross-validation:")
    season_team_dfs   = {s: d["team"]   for s, d in season_data.items()}
    season_player_dfs = {s: d["player"] for s, d in season_data.items()}
    records   = build_training_records(HISTORICAL_PLAYOFFS, season_team_dfs, season_player_dfs)
    cv_stats  = cross_validate_loo_season(records)
    print(f"  {cv_stats['correct']:>2}/{cv_stats['total']:>2}  ({cv_stats['accuracy']:>5.1f}%)"
          f"  across {cv_stats['seasons_tested']} seasons")

    # ── Baselines ──────────────────────────────────────────────────────────────
    baseline = higher_seed_baseline(HISTORICAL_PLAYOFFS)
    coin     = {"correct": baseline["total"] // 2, "total": baseline["total"], "pct": 50.0}

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 52}")
    print(f"{'Model':<28} {'Correct':>7}  {'Accuracy':>8}  {'':20}")
    print(f"{'─' * 52}")

    rows = [
        ("Weighted model",         weighted_stats["correct"], weighted_stats["total"], weighted_stats["pct"]),
        ("ML model (LOSO-CV)",     cv_stats["correct"],       cv_stats["total"],       cv_stats["accuracy"]),
        ("Baseline: higher seed",  baseline["correct"],       baseline["total"],       baseline["pct"]),
        ("Baseline: coin flip",    coin["correct"],           coin["total"],           coin["pct"]),
    ]
    for name, correct, total, pct in rows:
        print(f"  {name:<26} {correct:>2}/{total:<2}  {pct:>6.1f}%  {_bar(pct)}")

    print(f"{'─' * 52}")
    print(f"\nWeighted model beats higher-seed baseline by "
          f"{weighted_stats['pct'] - baseline['pct']:+.1f} percentage points")
    print(f"ML model beats higher-seed baseline by "
          f"{cv_stats['accuracy'] - baseline['pct']:+.1f} percentage points\n")


if __name__ == "__main__":
    main()
