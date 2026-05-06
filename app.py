"""
app.py
------
Streamlit web interface for the NBA playoff predictor.
Run with:  streamlit run app.py
"""

import streamlit as st
from nba_predictor.config import (
    PLAYOFF_MATCHUPS, TEAM_STAT_WEIGHTS, PLAYER_STAT_WEIGHTS, RECENT_GAMES, SEASON,
)
from nba_predictor.fetcher import fetch_team_df, fetch_player_df, FetchError
from nba_predictor.model import build_team_scores, build_player_scores, predict_all
from nba_predictor.ui import show_tab, show_comparison, show_model_config

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="wide")
st.title(f"🏀 NBA Playoff Predictor — {SEASON}")
st.caption("Stats sourced from the official NBA Stats API")

_, col_btn = st.columns([6, 1])
with col_btn:
    refresh = st.button("🔄 Refresh")

if refresh:
    st.cache_data.clear()


@st.cache_data(ttl=3600, show_spinner=False)
def load_all(force: bool = False):
    return (
        fetch_team_df(last_n=0,           force=force),
        fetch_player_df(last_n=0,          force=force),
        fetch_team_df(last_n=RECENT_GAMES, force=force),
        fetch_player_df(last_n=RECENT_GAMES, force=force),
    )


with st.spinner("Loading stats..."):
    try:
        season_team_df, season_player_df, recent_team_df, recent_player_df = load_all(force=refresh)
    except FetchError as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

season_preds = predict_all(PLAYOFF_MATCHUPS, build_team_scores(season_team_df), build_player_scores(season_player_df))
recent_preds = predict_all(PLAYOFF_MATCHUPS, build_team_scores(recent_team_df),  build_player_scores(recent_player_df))
playoff_teams = sorted({abbr for home, away, _ in PLAYOFF_MATCHUPS for abbr in (home, away)})

tab1, tab2, tab3 = st.tabs(["Full Season", f"Last {RECENT_GAMES} Games", "Comparison"])

with tab1:
    show_tab("Full Season", season_preds, season_player_df, playoff_teams)
with tab2:
    show_tab(f"Last {RECENT_GAMES} Games", recent_preds, recent_player_df, playoff_teams)
with tab3:
    show_comparison(season_preds, recent_preds)

show_model_config(TEAM_STAT_WEIGHTS, PLAYER_STAT_WEIGHTS)
