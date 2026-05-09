"""
ui.py
-----
Streamlit UI components for the NBA playoff predictor.
All rendering logic lives here; app.py only wires data to these functions.
"""

from pathlib import Path

import streamlit as st
import pandas as pd

from nba_predictor.config import ABBR_TO_FULL, TOP_PLAYERS_PER_TEAM
from nba_predictor.model import SeriesPrediction


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
    from nba_predictor.model import adjust_for_series_score

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
    from nba_predictor.history import save_predictions, record_outcome, load_history, accuracy_stats

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
                    from nba_predictor.config import FULL_TO_ABBR
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
