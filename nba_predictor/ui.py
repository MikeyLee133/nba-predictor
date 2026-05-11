"""
ui.py
-----
Streamlit UI components for the NBA playoff predictor.
All rendering logic lives here; app.py only wires data to these functions.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from nba_predictor.backtest import BacktestResult, backtest_accuracy
from nba_predictor.config import ABBR_TO_FULL, FULL_TO_ABBR, TOP_PLAYERS_PER_TEAM
from nba_predictor.history import accuracy_stats, load_history, record_outcome, save_predictions
from nba_predictor.ml_model import TrainedModel, feature_importances
from nba_predictor.model import SeriesPrediction, adjust_for_series_score


def predictions_df(preds: list[SeriesPrediction]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Series":           p.label,
        "Home Team":        ABBR_TO_FULL.get(p.home, p.home),
        "Home Win %":       p.home_win_pct,
        "Away Team":        ABBR_TO_FULL.get(p.away, p.away),
        "Away Win %":       p.away_win_pct,
        "Predicted Winner": ABBR_TO_FULL.get(p.predicted_winner, p.predicted_winner),
    } for p in preds])


def show_predictions(preds: list[SeriesPrediction]) -> None:
    st.dataframe(
        predictions_df(preds),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Home Win %": st.column_config.ProgressColumn(
                "Home Win %", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Away Win %": st.column_config.ProgressColumn(
                "Away Win %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )


def show_players(player_df: pd.DataFrame, playoff_teams: list[str]) -> None:
    rows = []
    for abbr in playoff_teams:
        group = player_df[player_df["team_id"] == abbr].dropna(subset=["per"])
        for _, row in group.nlargest(TOP_PLAYERS_PER_TEAM, "per").iterrows():
            rows.append({
                "Team":   abbr,
                "Player": row.get("player", "?"),
                "PPG":    row.get("pts_per_g"),
                "APG":    row.get("ast_per_g"),
                "RPG":    row.get("trb_per_g"),
                "PER":    row.get("per"),
            })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "PPG": st.column_config.NumberColumn(format="%.1f"),
            "APG": st.column_config.NumberColumn(format="%.1f"),
            "RPG": st.column_config.NumberColumn(format="%.1f"),
            "PER": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def show_tab(label: str, preds: list[SeriesPrediction], player_df: pd.DataFrame, playoff_teams: list[str]) -> None:
    st.subheader(f"Series Predictions — {label}")
    show_predictions(preds)
    st.download_button(
        "⬇ Download CSV",
        data=predictions_df(preds).to_csv(index=False),
        file_name=f"predictions_{label.replace(' ', '_')}.csv",
        mime="text/csv",
        key=f"dl_{label}",
    )
    st.subheader(f"Top Players — {label}")
    show_players(player_df, playoff_teams)


def trend(season_pct: float, recent_pct: float) -> str:
    diff = recent_pct - season_pct
    if diff > 2:
        return "↑"
    if diff < -2:
        return "↓"
    return "→"


def show_comparison(season_preds: list[SeriesPrediction], recent_preds: list[SeriesPrediction]) -> None:
    st.subheader("How Recent Form Changes the Predictions")
    rows = [{
        "Series":            s.label,
        "Home Team":         ABBR_TO_FULL.get(s.home, s.home),
        "Home Season Win %": s.home_win_pct,
        "Home Recent Win %": r.home_win_pct,
        "Home Trend":        trend(s.home_win_pct, r.home_win_pct),
        "Away Team":         ABBR_TO_FULL.get(s.away, s.away),
        "Away Season Win %": s.away_win_pct,
        "Away Recent Win %": r.away_win_pct,
        "Away Trend":        trend(s.away_win_pct, r.away_win_pct),
        "Season Pick":       ABBR_TO_FULL.get(s.predicted_winner, s.predicted_winner),
        "Recent Pick":       ABBR_TO_FULL.get(r.predicted_winner, r.predicted_winner),
    } for s, r in zip(season_preds, recent_preds)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("↑ trending up  ·  ↓ trending down  ·  → roughly same (threshold: 2%)")


def show_live_series(preds: list[SeriesPrediction]) -> None:
    st.subheader("In-Series Adjustment")
    st.caption("Set the current series score to blend model predictions with historical comeback rates.")

    _SCORE_OPTIONS = ["0-0", "1-0", "0-1", "2-0", "0-2", "2-1", "1-2",
                      "3-0", "0-3", "3-1", "1-3", "3-2", "2-3"]

    rows = []
    for p in preds:
        score = st.selectbox(
            f"{ABBR_TO_FULL.get(p.home, p.home)} vs {ABBR_TO_FULL.get(p.away, p.away)}",
            options=_SCORE_OPTIONS,
            key=f"score_{p.label}",
        )
        home_w, away_w = map(int, score.split("-"))
        adjusted = adjust_for_series_score(p, home_w, away_w)
        rows.append({
            "Series":           p.label,
            "Score":            score,
            "Home Win %":       adjusted.home_win_pct,
            "Away Win %":       adjusted.away_win_pct,
            "Predicted Winner": ABBR_TO_FULL.get(adjusted.predicted_winner, adjusted.predicted_winner),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Home Win %": st.column_config.ProgressColumn(
                "Home Win %", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Away Win %": st.column_config.ProgressColumn(
                "Away Win %", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
    )
    st.caption("Blends model prediction 50/50 with historical NBA series comeback rates.")


def show_history(
    current_preds: list[SeriesPrediction],
    round_label: str,
    history_path: Path,
) -> None:
    st.subheader("Prediction History")

    col_save, col_spacer = st.columns([2, 5])
    with col_save:
        if st.button("💾 Save current predictions"):
            save_predictions(current_preds, round_label, history_path)
            st.success(f"Saved {len(current_preds)} predictions for {round_label}")

    records = load_history(history_path)
    if not records:
        st.info("No predictions saved yet. Click 'Save current predictions' to start tracking.")
        return

    # Accuracy summary
    stats = accuracy_stats(records)
    m1, m2, m3 = st.columns(3)
    m1.metric("Correct", stats["correct"])
    m2.metric("Total Resolved", stats["total"])
    m3.metric("Accuracy", f"{stats['pct']}%" if stats["total"] else "—")

    st.divider()

    # Record outcomes for unresolved series
    unresolved = [r for r in records if r["actual_winner"] is None]
    if unresolved:
        st.subheader("Record Outcomes")
        for r in unresolved:
            col_label, col_pick = st.columns([3, 2])
            with col_label:
                st.write(r["series_label"])
            with col_pick:
                winner = st.selectbox(
                    "Actual winner",
                    options=["—", ABBR_TO_FULL.get(r["home"], r["home"]), ABBR_TO_FULL.get(r["away"], r["away"])],
                    key=f"outcome_{r['series_label']}",
                )
                if winner != "—":
                    abbr = FULL_TO_ABBR.get(winner, winner)
                    record_outcome(r["series_label"], abbr, history_path)
                    st.rerun()

    st.divider()

    # Full history table
    st.subheader("All Predictions")
    rows = [{
        "Round":            r["round"],
        "Series":           r["series_label"],
        "Predicted":        ABBR_TO_FULL.get(r["predicted_winner"], r["predicted_winner"]),
        "Actual":           ABBR_TO_FULL.get(r["actual_winner"], "—") if r["actual_winner"] else "—",
        "Result":           "✓" if r["correct"] else ("✗" if r["correct"] is False else "—"),
        "Home Win %":       r["home_win_pct"],
    } for r in records]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_backtest(all_results: list[BacktestResult]) -> None:
    st.subheader("Historical Backtesting")
    st.caption("How well the model predicted past playoff series using that season's stats.")

    if not all_results:
        st.info("No backtest results yet — click 'Run Backtest' in the app.")
        return

    # Overall accuracy
    overall = backtest_accuracy(all_results)
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall Correct", f"{overall['correct']} / {overall['total']}")
    m2.metric("Overall Accuracy", f"{overall['pct']}%")
    m3.metric("Seasons Tested", len({r.season for r in all_results}))

    st.divider()

    # Per-season breakdown
    seasons = sorted({r.season for r in all_results}, reverse=True)
    for season in seasons:
        season_results = [r for r in all_results if r.season == season]
        stats = backtest_accuracy(season_results)
        st.subheader(f"{season} — {stats['correct']}/{stats['total']} ({stats['pct']}%)")
        rows = [{
            "Series":    r.series_label,
            "Predicted": ABBR_TO_FULL.get(r.predicted_winner, r.predicted_winner),
            "Actual":    ABBR_TO_FULL.get(r.actual_winner, r.actual_winner),
            "Result":    "✓" if r.correct else "✗",
            "Home Win %": r.home_win_pct,
        } for r in season_results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_ml_predictions(
    ml_preds: list[SeriesPrediction],
    model: TrainedModel | None,
    weighted_preds: list[SeriesPrediction],
) -> None:
    st.subheader("ML Model — Logistic Regression")
    st.caption("Trained on historical playoff series outcomes. Features are stat differentials (home − away).")

    if not ml_preds or model is None:
        st.info("Click '▶ Train ML Model' to fetch historical stats and train the model.")
        return

    cv_str = f"  ·  Cross-val accuracy: {model.cv_accuracy}%" if model.cv_accuracy is not None else ""
    st.caption(f"Trained on {model.n_samples} historical series · Training accuracy: {model.train_accuracy}%{cv_str}")

    # Predictions comparison
    st.subheader("Predictions vs Weighted Model")
    rows = []
    for ml, wt in zip(ml_preds, weighted_preds):
        rows.append({
            "Series":          ml.label,
            "ML Pick":         ABBR_TO_FULL.get(ml.predicted_winner, ml.predicted_winner),
            "ML Home Win %":   ml.home_win_pct,
            "Weighted Pick":   ABBR_TO_FULL.get(wt.predicted_winner, wt.predicted_winner),
            "Wtd Home Win %":  wt.home_win_pct,
            "Agreement":       "✓" if ml.predicted_winner == wt.predicted_winner else "✗",
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ML Home Win %":  st.column_config.ProgressColumn("ML Home Win %",  min_value=0, max_value=100, format="%.1f%%"),
            "Wtd Home Win %": st.column_config.ProgressColumn("Wtd Home Win %", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    # Feature importances
    st.divider()
    st.subheader("What the Model Learned")
    st.caption("Coefficients show which stats drive predictions. Positive = favours the home team when they lead in that stat.")
    imp = feature_importances(model)
    imp_sorted = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)
    imp_df = pd.DataFrame([{"Stat": k, "Coefficient": round(v, 3)} for k, v in imp_sorted])
    st.bar_chart(imp_df.set_index("Stat")["Coefficient"])
