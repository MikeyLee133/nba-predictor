"""
config.py
---------
All constants, weights, and team mappings in one place.
Tweak weights here without touching any other file.
"""

SEASON = "2026"
BASE_URL = "https://www.basketball-reference.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# Polite delay between HTTP requests (seconds)
CRAWL_DELAY = 2

# ── Team mappings ─────────────────────────────────────────────────────────────

# Basketball Reference uses full team names in the stats tables
ABBR_TO_FULL = {
    "DET": "Detroit Pistons",
    "CLE": "Cleveland Cavaliers",
    "NYK": "New York Knicks",
    "PHI": "Philadelphia 76ers",
    "OKC": "Oklahoma City Thunder",
    "LAL": "Los Angeles Lakers",
    "SAS": "San Antonio Spurs",
    "MIN": "Minnesota Timberwolves",
}

# Reverse lookup: full name → abbreviation
FULL_TO_ABBR = {v: k for k, v in ABBR_TO_FULL.items()}

# ── Current playoff matchups ──────────────────────────────────────────────────
# Each tuple: (home_abbr, away_abbr, series_label)

PLAYOFF_MATCHUPS = [
    ("DET", "CLE", "East Semis: Detroit vs Cleveland"),
    ("NYK", "PHI", "East Semis: NY Knicks vs Philadelphia"),
    ("OKC", "LAL", "West Semis: OKC Thunder vs LA Lakers"),
    ("SAS", "MIN", "West Semis: San Antonio vs Minnesota"),
]

# ── Model weights ─────────────────────────────────────────────────────────────

# Blend ratio between team-level and player-level scores
TEAM_SCORE_WEIGHT = 0.60
PLAYER_SCORE_WEIGHT = 0.40

# Home-court advantage multiplier applied to the home team's blended score
HOME_COURT_MULTIPLIER = 1.04

# Individual stat weights for the team composite score (must sum to 1.0)
TEAM_STAT_WEIGHTS = {
    "net_rtg":  0.30,   # best single predictor of team quality
    "drtg":     0.20,   # defensive rating  (lower = better)
    "ortg":     0.15,   # offensive rating
    "pts":      0.10,   # points scored per game
    "opp_pts":  0.10,   # opponent points   (lower = better)
    "ast":      0.05,
    "3pm":      0.05,
    "pace":     0.05,
}

# Stats where a LOWER value is better — these are inverted during normalization
INVERT_STATS = {"drtg", "opp_pts"}

# Individual stat weights for the player star-power score (must sum to 1.0)
PLAYER_STAT_WEIGHTS = {
    "pts_per_g": 0.35,
    "per":       0.30,
    "ast_per_g": 0.15,
    "trb_per_g": 0.10,
    "fg3_per_g": 0.10,
}

# Number of top players (by PER) used to represent each team
TOP_PLAYERS_PER_TEAM = 3

# Scale factor to bring raw player scores (~15–30) into the 0–100 range
# Adjust if scores look unreasonably high or low after changes to PLAYER_STAT_WEIGHTS
PLAYER_SCORE_SCALE = 2.5
