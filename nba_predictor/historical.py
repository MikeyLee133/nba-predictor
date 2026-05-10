"""
historical.py
-------------
Historical NBA playoff matchups and actual outcomes for backtesting.
Each season entry covers the Second Round (Conference Semifinals).

To add a new season after it completes:
  1. Add a new key to HISTORICAL_PLAYOFFS
  2. Fill in matchups (home_abbr, away_abbr, label) and outcomes {label: winner_abbr}
"""

HISTORICAL_PLAYOFFS = {
    "2023-24": {
        "matchups": [
            ("BOS", "CLE", "East Semis: Boston vs Cleveland"),
            ("NYK", "IND", "East Semis: New York vs Indiana"),
            ("OKC", "DAL", "West Semis: OKC vs Dallas"),
            ("DEN", "MIN", "West Semis: Denver vs Minnesota"),
        ],
        "outcomes": {
            "East Semis: Boston vs Cleveland": "BOS",  # Celtics won 4-1
            "East Semis: New York vs Indiana": "IND",  # Pacers won 4-3
            "West Semis: OKC vs Dallas":       "DAL",  # Mavericks won 4-2
            "West Semis: Denver vs Minnesota": "MIN",  # Timberwolves won 4-3
        },
    },
    "2022-23": {
        "matchups": [
            ("BOS", "PHI", "East Semis: Boston vs Philadelphia"),
            ("NYK", "MIA", "East Semis: New York vs Miami"),
            ("DEN", "PHX", "West Semis: Denver vs Phoenix"),
            ("GSW", "LAL", "West Semis: Golden State vs LA Lakers"),
        ],
        "outcomes": {
            "East Semis: Boston vs Philadelphia": "BOS",  # Celtics won 4-3
            "East Semis: New York vs Miami":      "MIA",  # Heat won 4-2
            "West Semis: Denver vs Phoenix":      "DEN",  # Nuggets won 4-2
            "West Semis: Golden State vs LA Lakers": "LAL",  # Lakers won 4-2
        },
    },
}
