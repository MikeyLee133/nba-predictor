"""
app.py
------
Streamlit web interface for the NBA playoff predictor.
Run with:  streamlit run app.py
"""

import time
from pathlib import Path
import streamlit as st
from nba_predictor.config import (
    PLAYOFF_MATCHUPS, TEAM_STAT_WEIGHTS, PLAYER_STAT_WEIGHTS,
    TEAM_SCORE_WEIGHT, PLAYER_SCORE_WEIGHT, HOME_COURT_MULTIPLIER,
    RECENT_GAMES, SEASON, PLAYOFF_ROUND, HISTORY_FILE,
)
from nba_predictor.fetcher import fetch_team_df, fetch_player_df, FetchError, CACHE_DIR
from nba_predictor.model import build_team_scores, build_player_scores, predict_all
from nba_predictor.history import save_predictions, record_outcome, load_history, accuracy_stats
from nba_predictor.backtest import run_season_backtest
from nba_predictor.historical import HISTORICAL_PLAYOFFS
from nba_predictor.ui import show_tab, show_comparison, show_live_series, show_history, show_backtest

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="wide")
st.title(f"🏀 NBA Playoff Predictor — {SEASON}")
st.caption("Stats sourced from the official NBA Stats API")

col_fresh, col_btn = st.columns([6, 1])
with col_fresh:
    _cache_file = CACHE_DIR / f"team_stats_{SEASON}_0.pkl"
    if _cache_file.exists():
        _age = time.time() - _cache_file.stat().st_mtime
        _freshness = f"Stats from {int(_age / 60)}m ago" if _age < 3600 else f"Stats from {_age / 3600:.1f}h ago"
    else:
        _freshness = "Stats not yet cached"
    st.caption(_freshness)
with col_btn:
    refresh = st.button("🔄 Refresh")

if refresh:
    st.cache_data.clear()

# ── Sidebar: adjustable model weights ─────────────────────────────────────────

with st.sidebar:
    st.header("Model Weights")

    # Single source of truth for defaults — used by sliders and the reset button
    _DEFAULTS = {
        "blend_team": int(TEAM_SCORE_WEIGHT * 100),
        "hca":        int((HOME_COURT_MULTIPLIER - 1) * 100),
        "t_net_rtg":  int(TEAM_STAT_WEIGHTS["net_rtg"]  * 100),
        "t_drtg":     int(TEAM_STAT_WEIGHTS["drtg"]     * 100),
        "t_ortg":     int(TEAM_STAT_WEIGHTS["ortg"]     * 100),
        "t_pts":      int(TEAM_STAT_WEIGHTS["pts"]      * 100),
        "t_ast":      int(TEAM_STAT_WEIGHTS["ast"]      * 100),
        "t_3pm":      int(TEAM_STAT_WEIGHTS["3pm"]      * 100),
        "t_pace":     int(TEAM_STAT_WEIGHTS["pace"]     * 100),
        "p_pts":      int(PLAYER_STAT_WEIGHTS["pts_per_g"] * 100),
        "p_per":      int(PLAYER_STAT_WEIGHTS["per"]       * 100),
        "p_ast":      int(PLAYER_STAT_WEIGHTS["ast_per_g"] * 100),
        "p_reb":      int(PLAYER_STAT_WEIGHTS["trb_per_g"] * 100),
        "p_3pm":      int(PLAYER_STAT_WEIGHTS["fg3_per_g"] * 100),
    }
    if st.button("↺ Reset to defaults", key="reset_weights"):
        st.session_state.update(_DEFAULTS)
        st.rerun()

    st.subheader("Blend")
    team_blend = st.slider(
        "Team stats weight (%)", 0, 100, _DEFAULTS["blend_team"], key="blend_team",
        help="Percentage of the final score from team-level stats; remainder goes to player star power",
    )
    player_blend = 100 - team_blend
    st.caption(f"Player star power: {player_blend}%")
    team_w  = team_blend  / 100
    player_w = player_blend / 100

    hca = st.slider(
        "Home court advantage (%)", 0, 10, _DEFAULTS["hca"], key="hca",
        help="Bonus multiplier applied to the home team's score",
    )
    home_mult = 1 + hca / 100

    st.divider()
    st.subheader("Team Stat Weights")
    st.caption("Drag to set relative importance — auto-normalized to 100%")
    raw_team = {
        "net_rtg":  st.slider("Net Rating",      0, 100, _DEFAULTS["t_net_rtg"], key="t_net_rtg"),
        "drtg":     st.slider("Def. Rating",     0, 100, _DEFAULTS["t_drtg"],    key="t_drtg"),
        "ortg":     st.slider("Off. Rating",     0, 100, _DEFAULTS["t_ortg"],    key="t_ortg"),
        "pts":      st.slider("Points/G",        0, 100, _DEFAULTS["t_pts"],     key="t_pts"),
        "ast":      st.slider("Assists/G",       0, 100, _DEFAULTS["t_ast"],     key="t_ast"),
        "3pm":      st.slider("3-Pointers/G",    0, 100, _DEFAULTS["t_3pm"],     key="t_3pm"),
        "pace":     st.slider("Pace",            0, 100, _DEFAULTS["t_pace"],    key="t_pace"),
    }
    total_team = sum(raw_team.values()) or 1
    team_stat_weights = {k: v / total_team for k, v in raw_team.items()}

    st.divider()
    st.subheader("Player Stat Weights")
    st.caption("Drag to set relative importance — auto-normalized to 100%")
    raw_player = {
        "pts_per_g": st.slider("Points/G",     0, 100, _DEFAULTS["p_pts"], key="p_pts"),
        "per":       st.slider("PER",          0, 100, _DEFAULTS["p_per"], key="p_per"),
        "ast_per_g": st.slider("Assists/G",    0, 100, _DEFAULTS["p_ast"], key="p_ast"),
        "trb_per_g": st.slider("Rebounds/G",   0, 100, _DEFAULTS["p_reb"], key="p_reb"),
        "fg3_per_g": st.slider("3-Pointers/G", 0, 100, _DEFAULTS["p_3pm"], key="p_3pm"),
    }
    total_player = sum(raw_player.values()) or 1
    player_stat_weights = {k: v / total_player for k, v in raw_player.items()}

    st.divider()
    st.subheader("Injuries / Unavailable")
    st.caption("Excluded from player star-power scoring")


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

# Populate injury multiselect now that player data is loaded
playoff_teams = sorted({abbr for home, away, _ in PLAYOFF_MATCHUPS for abbr in (home, away)})
playoff_players = (
    season_player_df[season_player_df["team_id"].isin(playoff_teams)]
    .dropna(subset=["per"])
    .sort_values("per", ascending=False)["player"]
    .tolist()
)
with st.sidebar:
    unavailable = set(st.multiselect("Players", options=playoff_players, key="unavailable"))

_t0 = time.perf_counter()
season_preds = predict_all(
    PLAYOFF_MATCHUPS,
    build_team_scores(season_team_df, weights=team_stat_weights),
    build_player_scores(season_player_df, weights=player_stat_weights, unavailable=unavailable),
    team_w=team_w, player_w=player_w, home_mult=home_mult,
)
recent_preds = predict_all(
    PLAYOFF_MATCHUPS,
    build_team_scores(recent_team_df, weights=team_stat_weights),
    build_player_scores(recent_player_df, weights=player_stat_weights, unavailable=unavailable),
    team_w=team_w, player_w=player_w, home_mult=home_mult,
)
_model_ms = (time.perf_counter() - _t0) * 1000

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Full Season", f"Last {RECENT_GAMES} Games", "Comparison", "Live Series", "History", "Backtest"])

with tab1:
    show_tab("Full Season", season_preds, season_player_df, playoff_teams)
with tab2:
    show_tab(f"Last {RECENT_GAMES} Games", recent_preds, recent_player_df, playoff_teams)
with tab3:
    show_comparison(season_preds, recent_preds)
with tab4:
    show_live_series(season_preds)
with tab5:
    history_path = Path(HISTORY_FILE)
    show_history(season_preds, PLAYOFF_ROUND, history_path)
with tab6:
    if st.button("▶ Run Backtest"):
        backtest_results = []
        for hist_season, data in HISTORICAL_PLAYOFFS.items():
            with st.spinner(f"Fetching {hist_season} stats..."):
                try:
                    hist_team_df   = fetch_team_df(season=hist_season)
                    hist_player_df = fetch_player_df(season=hist_season)
                    backtest_results += run_season_backtest(
                        hist_season, data["matchups"], data["outcomes"],
                        hist_team_df, hist_player_df,
                    )
                except FetchError as e:
                    st.error(f"Could not fetch {hist_season}: {e}")
        st.session_state["backtest_results"] = backtest_results
    show_backtest(st.session_state.get("backtest_results", []))

st.caption(f"Model computed in {_model_ms:.1f}ms")
