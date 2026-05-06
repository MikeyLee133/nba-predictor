"""
app.py
------
Streamlit web interface for the NBA playoff predictor.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from nba_predictor.config import (
    PLAYOFF_MATCHUPS, ABBR_TO_FULL, TEAM_STAT_WEIGHTS,
    PLAYER_STAT_WEIGHTS, RECENT_GAMES, TOP_PLAYERS_PER_TEAM,
)
from nba_predictor.fetcher import fetch_team_df, fetch_player_df, FetchError
from nba_predictor.model import build_team_scores, build_player_scores, predict_all

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="wide")

st.title("🏀 NBA Playoff Predictor — 2025-26")
st.caption("Stats sourced from the official NBA Stats API")

_, col_btn = st.columns([6, 1])
with col_btn:
    refresh = st.button("🔄 Refresh")

if refresh:
    st.cache_data.clear()


@st.cache_data(ttl=3600, show_spinner=False)
def load_all(force: bool = False):
    season_team_df   = fetch_team_df(last_n=0,          force=force)
    season_player_df = fetch_player_df(last_n=0,         force=force)
    recent_team_df   = fetch_team_df(last_n=RECENT_GAMES, force=force)
    recent_player_df = fetch_player_df(last_n=RECENT_GAMES, force=force)
    return season_team_df, season_player_df, recent_team_df, recent_player_df


with st.spinner("Loading stats..."):
    try:
        season_team_df, season_player_df, recent_team_df, recent_player_df = load_all(force=refresh)
    except FetchError as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

season_preds = predict_all(
    PLAYOFF_MATCHUPS,
    build_team_scores(season_team_df),
    build_player_scores(season_player_df),
)
recent_preds = predict_all(
    PLAYOFF_MATCHUPS,
    build_team_scores(recent_team_df),
    build_player_scores(recent_player_df),
)

playoff_teams = sorted({abbr for home, away, _ in PLAYOFF_MATCHUPS for abbr in (home, away)})


# ── Helper ────────────────────────────────────────────────────────────────────

def predictions_df(preds):
    rows = []
    for p in preds:
        rows.append({
            "Series":           p.label,
            "Home Team":        ABBR_TO_FULL.get(p.home, p.home),
            "Home Win %":       p.home_win_pct,
            "Away Team":        ABBR_TO_FULL.get(p.away, p.away),
            "Away Win %":       p.away_win_pct,
            "Predicted Winner": ABBR_TO_FULL.get(p.predicted_winner, p.predicted_winner),
        })
    return pd.DataFrame(rows)


def show_predictions(preds):
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


def show_players(player_df):
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


def trend(season_pct, recent_pct):
    diff = recent_pct - season_pct
    if diff > 2:
        return "↑"
    elif diff < -2:
        return "↓"
    return "→"


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Full Season", f"Last {RECENT_GAMES} Games", "Comparison"])

with tab1:
    st.subheader("Series Predictions — Full Season")
    show_predictions(season_preds)
    st.subheader("Top Players — Full Season")
    show_players(season_player_df)

with tab2:
    st.subheader(f"Series Predictions — Last {RECENT_GAMES} Games")
    show_predictions(recent_preds)
    st.subheader(f"Top Players — Last {RECENT_GAMES} Games")
    show_players(recent_player_df)

with tab3:
    st.subheader("How Recent Form Changes the Predictions")
    rows = []
    for s, r in zip(season_preds, recent_preds):
        rows.append({
            "Series":           s.label,
            "Home Team":        ABBR_TO_FULL.get(s.home, s.home),
            "Home Season Win %": s.home_win_pct,
            "Home Recent Win %": r.home_win_pct,
            "Home Trend":       trend(s.home_win_pct, r.home_win_pct),
            "Away Team":        ABBR_TO_FULL.get(s.away, s.away),
            "Away Season Win %": s.away_win_pct,
            "Away Recent Win %": r.away_win_pct,
            "Away Trend":       trend(s.away_win_pct, r.away_win_pct),
            "Season Pick":      ABBR_TO_FULL.get(s.predicted_winner, s.predicted_winner),
            "Recent Pick":      ABBR_TO_FULL.get(r.predicted_winner, r.predicted_winner),
        })
    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.caption("↑ trending up  ·  ↓ trending down  ·  → roughly same (threshold: 2%)")

# ── Model Config ──────────────────────────────────────────────────────────────

with st.expander("Model Configuration"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Team Stat Weights (60%)")
        st.dataframe(
            pd.DataFrame([{"Stat": s, "Weight": f"{w*100:.0f}%"} for s, w in TEAM_STAT_WEIGHTS.items()]),
            hide_index=True, use_container_width=True,
        )
    with col2:
        st.subheader("Player Stat Weights (40%)")
        st.dataframe(
            pd.DataFrame([{"Stat": s, "Weight": f"{w*100:.0f}%"} for s, w in PLAYER_STAT_WEIGHTS.items()]),
            hide_index=True, use_container_width=True,
        )
    st.caption("Home-court advantage: +4% multiplier. Top 3 players by PER per team.")
