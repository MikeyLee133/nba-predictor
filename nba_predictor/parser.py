"""
parser.py
---------
Responsible for one thing only: turning raw HTML into clean DataFrames.

No HTTP requests, no model logic — just parsing and light cleaning.
"""

import re
import pandas as pd
from bs4 import BeautifulSoup


class ParseError(Exception):
    """Raised when an expected HTML table or column is missing."""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _uncomment(html: str) -> str:
    """Strip HTML comment wrappers so BeautifulSoup can see commented-out tables."""
    return re.sub(r"<!--(.*?)-->", lambda m: m.group(1), html, flags=re.DOTALL)


def _table_to_df(soup: BeautifulSoup, table_id: str) -> pd.DataFrame:
    """
    Locate a <table id=table_id> in the soup and return it as a DataFrame.
    Each cell's `data-stat` attribute becomes the column name.
    Raises ParseError if the table is not found.
    """
    table = soup.find("table", {"id": table_id})
    if table is None:
        raise ParseError(
            f"Table '{table_id}' not found in page. "
            "Basketball Reference may have changed their HTML structure."
        )

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        if "thead" in tr.get("class", []):
            continue
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        rows.append({c.get("data-stat"): c.get_text(strip=True) for c in cells})

    if not rows:
        raise ParseError(f"Table '{table_id}' contained no data rows.")

    return pd.DataFrame(rows)


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce a list of columns to numeric, leaving unparseable values as NaN."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _require_columns(df: pd.DataFrame, required: list[str], context: str):
    """Raise ParseError if any required column is absent after renaming."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ParseError(
            f"{context}: expected columns {missing} are missing. "
            "The site layout may have changed."
        )


# ── Public API ────────────────────────────────────────────────────────────────

def parse_team_stats(html: str) -> pd.DataFrame:
    """
    Parse the NBA season summary page into a single team-stats DataFrame.

    Merges per-game, opponent per-game, and miscellaneous stats tables.
    Returned columns: team, pts, 3pm, ast, pace, ortg, net_rtg, opp_pts, drtg, opp_efg
    """
    soup = BeautifulSoup(_uncomment(html), "html.parser")

    off_raw  = _table_to_df(soup, "per_game-team")
    def_raw  = _table_to_df(soup, "per_game-opponent")
    misc_raw = _table_to_df(soup, "advanced-team")

    off_rename = {
        "team_name": "team",
        "pts_per_g": "pts",
        "fg3_per_g": "3pm",
        "ast_per_g": "ast",
    }
    def_rename = {
        "team_name": "team",
        "pts_per_g": "opp_pts",
        "efg_pct":   "opp_efg",
    }
    misc_rename = {
        "team_name": "team",
        "pace":      "pace",
        "off_rtg":   "ortg",
        "def_rtg":   "drtg",
        "net_rtg":   "net_rtg",
    }

    off_df  = off_raw.rename(columns=off_rename)
    def_df  = def_raw.rename(columns=def_rename)
    misc_df = misc_raw.rename(columns=misc_rename)

    off_df  = off_df[[c for c in off_rename.values()  if c in off_df.columns]]
    def_df  = def_df[[c for c in def_rename.values()  if c in def_df.columns]]
    misc_df = misc_df[[c for c in misc_rename.values() if c in misc_df.columns]]

    _require_columns(off_df,  ["team", "pts"],    "Per-game stats table")
    _require_columns(def_df,  ["team", "opp_pts"], "Opponent stats table")
    _require_columns(misc_df, ["team", "net_rtg", "drtg"], "Miscellaneous stats table")

    merged = pd.merge(off_df, def_df,  on="team")
    merged = pd.merge(merged, misc_df, on="team")

    numeric_cols = [c for c in merged.columns if c != "team"]
    merged = _to_numeric(merged, numeric_cols)

    return merged.dropna(subset=["team"]).reset_index(drop=True)


def parse_player_stats(html: str) -> pd.DataFrame:
    """
    Parse the per-game player stats page into a player DataFrame.

    Returned columns include: player, team_id, pts_per_g, ast_per_g,
    trb_per_g, fg3_per_g, fg_pct, efg_pct, per, ws
    """
    soup = BeautifulSoup(_uncomment(html), "html.parser")
    raw = _table_to_df(soup, "per_game_stats")

    numeric_cols = [
        "pts_per_g", "ast_per_g", "trb_per_g", "fg3_per_g",
        "fg_pct", "efg_pct", "per", "ws",
    ]
    df = _to_numeric(raw, numeric_cols)

    print("  Player table columns:", list(df.columns[:15]))
    _require_columns(df, ["player", "team_id"], "Player stats table")

    # Drop header-repeat rows Basketball Reference sometimes injects
    df = df[df["player"] != "Player"].copy()

    return df.dropna(subset=["team_id"]).reset_index(drop=True)
