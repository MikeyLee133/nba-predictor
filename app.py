"""
app.py
------
Streamlit web interface for the NBA playoff predictor.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
from nba_predictor.config import PLAYOFF_MATCHUPS, ABBR_TO_FULL, TEAM_STAT_WEIGHTS, PLAYER_STAT_WEIGHTS
from nba_predictor.fetcher import fetch_team_df, fetch_player_df, FetchError
from nba_predictor.model import build_team_scores, build_player_scores, predict_all

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="wide")

st.title("🏀 NBA Playoff Predictor — 2025-26")
st.caption("Stats sourced from the official NBA Stats API")


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(force: bool = False):
    team_df   = fetch_team_df(force=force)
    player_df = fetch_player_df(force=force)
    return team_df, player_df


_, col_btn = st.columns([6, 1])
with col_btn:
    refresh = st.button("🔄 Refresh")

if refresh:
    st.cache_data.clear()

with st.spinner("Loading stats..."):
    try:
        team_df, player_df = load_data(force=refresh)
    except FetchError as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

team_scores   = build_team_scores(team_df)
player_scores = build_player_scores(player_df)
predictions   = predict_all(PLAYOFF_MATCHUPS, team_scores, player_scores)

# ── Predictions ───────────────────────────────────────────────────────────────

st.header("Series Predictions")

rows = []
for p in predictions:
    rows.append({
        "Series":           p.label,
        "Home Team":        ABBR_TO_FULL.get(p.home, p.home),
        "Home Win %":       p.home_win_pct,
        "Away Team":        ABBR_TO_FULL.get(p.away, p.away),
        "Away Win %":       p.away_win_pct,
        "Predicted Winner": ABBR_TO_FULL.get(p.predicted_winner, p.predicted_winner),
    })

pred_df = pd.DataFrame(rows)

st.dataframe(
    pred_df,
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

# ── Top Players ───────────────────────────────────────────────────────────────

st.header("Top Players by PER")

playoff_teams = sorted({abbr for home, away, _ in PLAYOFF_MATCHUPS for abbr in (home, away)})

player_rows = []
for abbr in playoff_teams:
    group = player_df[player_df["team_id"] == abbr].dropna(subset=["per"])
    for _, row in group.nlargest(3, "per").iterrows():
        player_rows.append({
            "Team":   abbr,
            "Player": row.get("player", "?"),
            "PPG":    row.get("pts_per_g"),
            "APG":    row.get("ast_per_g"),
            "RPG":    row.get("trb_per_g"),
            "PER":    row.get("per"),
        })

st.dataframe(
    pd.DataFrame(player_rows),
    use_container_width=True,
    hide_index=True,
    column_config={
        "PPG": st.column_config.NumberColumn(format="%.1f"),
        "APG": st.column_config.NumberColumn(format="%.1f"),
        "RPG": st.column_config.NumberColumn(format="%.1f"),
        "PER": st.column_config.NumberColumn(format="%.1f"),
    },
)

# ── Model Config ──────────────────────────────────────────────────────────────

with st.expander("Model Configuration"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team Stat Weights (60%)")
        st.dataframe(
            pd.DataFrame(
                [{"Stat": s, "Weight": f"{w*100:.0f}%"} for s, w in TEAM_STAT_WEIGHTS.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with col2:
        st.subheader("Player Stat Weights (40%)")
        st.dataframe(
            pd.DataFrame(
                [{"Stat": s, "Weight": f"{w*100:.0f}%"} for s, w in PLAYER_STAT_WEIGHTS.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )

    st.caption("Home-court advantage: +4% multiplier on home team score. Top 3 players by PER used per team.")
