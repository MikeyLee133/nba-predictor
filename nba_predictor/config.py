"""
config.py
---------
All constants, weights, and team mappings in one place.
Tweak weights here without touching any other file.
"""

SEASON = "2025-26"

# ── Team mappings ─────────────────────────────────────────────────────────────

ABBR_TO_FULL = {
    # Current playoff teams
    "DET": "Detroit Pistons",
    "CLE": "Cleveland Cavaliers",
    "NYK": "New York Knicks",
    "PHI": "Philadelphia 76ers",
    "OKC": "Oklahoma City Thunder",
    "LAL": "Los Angeles Lakers",
    "SAS": "San Antonio Spurs",
    "MIN": "Minnesota Timberwolves",
    # Historical playoff teams (for backtesting)
    "BOS": "Boston Celtics",
    "IND": "Indiana Pacers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "MIA": "Miami Heat",
    "GSW": "Golden State Warriors",
    "PHX": "Phoenix Suns",
    "MEM": "Memphis Grizzlies",
    "MIL": "Milwaukee Bucks",
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "CHI": "Chicago Bulls",
}

FULL_TO_ABBR = {v: k for k, v in ABBR_TO_FULL.items()}

# ── Current playoff matchups ──────────────────────────────────────────────────
# Each tuple: (home_abbr, away_abbr, series_label)

PLAYOFF_MATCHUPS = [
    ("DET", "CLE", "East Semis: Detroit vs Cleveland"),
    ("NYK", "PHI", "East Semis: NY Knicks vs Philadelphia"),
    ("OKC", "LAL", "West Semis: OKC Thunder vs LA Lakers"),
    ("SAS", "MIN", "West Semis: San Antonio vs Minnesota"),
]

PLAYOFF_ROUND = "Second Round"

HISTORY_FILE = "prediction_history.json"

# ── Model weights ─────────────────────────────────────────────────────────────

TEAM_SCORE_WEIGHT    = 0.60
PLAYER_SCORE_WEIGHT  = 0.40
HOME_COURT_MULTIPLIER = 1.04

# Recent form window (number of games)
RECENT_GAMES = 15

# Disk cache TTL in hours
CACHE_TTL_HOURS = 24

# Individual stat weights for the team composite score (must sum to 1.0)
TEAM_STAT_WEIGHTS = {
    "net_rtg":  0.30,   # best single predictor of team quality
    "drtg":     0.30,   # defensive rating (lower = better)
    "ortg":     0.15,   # offensive rating
    "pts":      0.10,   # points scored per game
    "ast":      0.05,
    "3pm":      0.05,
    "pace":     0.05,
}

# Stats where a LOWER value is better — inverted during normalization
INVERT_STATS = {"drtg"}

# Individual stat weights for the player star-power score (must sum to 1.0)
PLAYER_STAT_WEIGHTS = {
    "pts_per_g": 0.35,
    "per":       0.30,
    "ast_per_g": 0.15,
    "trb_per_g": 0.10,
    "fg3_per_g": 0.10,
}

TOP_PLAYERS_PER_TEAM = 3
PLAYER_SCORE_SCALE   = 2.5

# Historical NBA playoff series win rates for the current leader
# Key: (leader_wins, trailer_wins) → probability the leader wins the series
SERIES_COMEBACK_RATES = {
    (1, 0): 0.60,
    (2, 0): 0.82,
    (2, 1): 0.68,
    (3, 0): 0.99,
    (3, 1): 0.96,
    (3, 2): 0.83,
}

# ── Validation ────────────────────────────────────────────────────────────────

assert abs(sum(TEAM_STAT_WEIGHTS.values())   - 1.0) < 1e-9, f"TEAM_STAT_WEIGHTS must sum to 1.0 (got {sum(TEAM_STAT_WEIGHTS.values()):.4f})"
assert abs(sum(PLAYER_STAT_WEIGHTS.values()) - 1.0) < 1e-9, f"PLAYER_STAT_WEIGHTS must sum to 1.0 (got {sum(PLAYER_STAT_WEIGHTS.values()):.4f})"
assert abs(TEAM_SCORE_WEIGHT + PLAYER_SCORE_WEIGHT - 1.0) < 1e-9, f"TEAM_SCORE_WEIGHT + PLAYER_SCORE_WEIGHT must sum to 1.0 (got {TEAM_SCORE_WEIGHT + PLAYER_SCORE_WEIGHT:.4f})"
