"""
historical.py
-------------
Historical NBA playoff matchups and actual outcomes.
Covers all rounds (First Round through Finals) for each season.
Home team is the higher seed in each matchup.

To add a new season after it completes:
  1. Add a new key to HISTORICAL_PLAYOFFS
  2. Fill in all rounds: matchups (home, away, label) and outcomes {label: winner_abbr}
"""

HISTORICAL_PLAYOFFS = {
    "2023-24": {
        "matchups": [
            # First Round — East
            ("BOS", "MIA", "E1: Boston vs Miami"),
            ("NYK", "PHI", "E1: New York vs Philadelphia"),
            ("MIL", "IND", "E1: Milwaukee vs Indiana"),
            ("CLE", "ORL", "E1: Cleveland vs Orlando"),
            # First Round — West
            ("OKC", "NOP", "W1: OKC vs New Orleans"),
            ("DEN", "LAL", "W1: Denver vs LA Lakers"),
            ("MIN", "PHX", "W1: Minnesota vs Phoenix"),
            ("DAL", "LAC", "W1: Dallas vs LA Clippers"),
            # Second Round — East
            ("BOS", "CLE", "E2: Boston vs Cleveland"),
            ("NYK", "IND", "E2: New York vs Indiana"),
            # Second Round — West
            ("OKC", "DAL", "W2: OKC vs Dallas"),
            ("DEN", "MIN", "W2: Denver vs Minnesota"),
            # Conference Finals
            ("BOS", "IND", "ECF: Boston vs Indiana"),
            ("DAL", "MIN", "WCF: Dallas vs Minnesota"),
            # Finals
            ("BOS", "DAL", "Finals: Boston vs Dallas"),
        ],
        "outcomes": {
            "E1: Boston vs Miami":          "BOS",  # 4-1
            "E1: New York vs Philadelphia": "NYK",  # 4-2
            "E1: Milwaukee vs Indiana":     "IND",  # 4-2 upset
            "E1: Cleveland vs Orlando":     "CLE",  # 4-3
            "W1: OKC vs New Orleans":       "OKC",  # 4-0
            "W1: Denver vs LA Lakers":      "DEN",  # 4-1
            "W1: Minnesota vs Phoenix":     "MIN",  # 4-0
            "W1: Dallas vs LA Clippers":    "DAL",  # 4-2
            "E2: Boston vs Cleveland":      "BOS",  # 4-1
            "E2: New York vs Indiana":      "IND",  # 4-3 upset
            "W2: OKC vs Dallas":            "DAL",  # 4-2 upset
            "W2: Denver vs Minnesota":      "MIN",  # 4-3 upset
            "ECF: Boston vs Indiana":       "BOS",  # 4-0
            "WCF: Dallas vs Minnesota":     "DAL",  # 4-1
            "Finals: Boston vs Dallas":     "BOS",  # 4-1
        },
    },
    "2022-23": {
        "matchups": [
            # First Round — East
            ("MIL", "MIA", "E1: Milwaukee vs Miami"),
            ("BOS", "ATL", "E1: Boston vs Atlanta"),
            ("PHI", "BKN", "E1: Philadelphia vs Brooklyn"),
            ("CLE", "NYK", "E1: Cleveland vs New York"),
            # First Round — West
            ("DEN", "MIN", "W1: Denver vs Minnesota"),
            ("MEM", "LAL", "W1: Memphis vs LA Lakers"),
            ("SAC", "GSW", "W1: Sacramento vs Golden State"),
            ("PHX", "LAC", "W1: Phoenix vs LA Clippers"),
            # Second Round — East
            ("BOS", "PHI", "E2: Boston vs Philadelphia"),
            ("NYK", "MIA", "E2: New York vs Miami"),
            # Second Round — West
            ("DEN", "PHX", "W2: Denver vs Phoenix"),
            ("GSW", "LAL", "W2: Golden State vs LA Lakers"),
            # Conference Finals
            ("BOS", "MIA", "ECF: Boston vs Miami"),
            ("DEN", "LAL", "WCF: Denver vs LA Lakers"),
            # Finals
            ("DEN", "MIA", "Finals: Denver vs Miami"),
        ],
        "outcomes": {
            "E1: Milwaukee vs Miami":          "MIA",  # 4-1 upset
            "E1: Boston vs Atlanta":           "BOS",  # 4-2
            "E1: Philadelphia vs Brooklyn":    "PHI",  # 4-0
            "E1: Cleveland vs New York":       "NYK",  # 4-1 upset
            "W1: Denver vs Minnesota":         "DEN",  # 4-1
            "W1: Memphis vs LA Lakers":        "LAL",  # 4-2 upset
            "W1: Sacramento vs Golden State":  "GSW",  # 4-3
            "W1: Phoenix vs LA Clippers":      "PHX",  # 4-1
            "E2: Boston vs Philadelphia":      "BOS",  # 4-3
            "E2: New York vs Miami":           "MIA",  # 4-2
            "W2: Denver vs Phoenix":           "DEN",  # 4-2
            "W2: Golden State vs LA Lakers":   "LAL",  # 4-2
            "ECF: Boston vs Miami":            "MIA",  # 4-3
            "WCF: Denver vs LA Lakers":        "DEN",  # 4-0
            "Finals: Denver vs Miami":         "DEN",  # 4-1
        },
    },
    "2021-22": {
        "matchups": [
            # First Round — East
            ("MIA", "ATL", "E1: Miami vs Atlanta"),
            ("BOS", "BKN", "E1: Boston vs Brooklyn"),
            ("MIL", "CHI", "E1: Milwaukee vs Chicago"),
            ("PHI", "TOR", "E1: Philadelphia vs Toronto"),
            # First Round — West
            ("PHX", "NOP", "W1: Phoenix vs New Orleans"),
            ("MEM", "MIN", "W1: Memphis vs Minnesota"),
            ("GSW", "DEN", "W1: Golden State vs Denver"),
            ("UTA", "DAL", "W1: Utah vs Dallas"),
            # Second Round — East
            ("MIA", "PHI", "E2: Miami vs Philadelphia"),
            ("BOS", "MIL", "E2: Boston vs Milwaukee"),
            # Second Round — West
            ("PHX", "DAL", "W2: Phoenix vs Dallas"),
            ("MEM", "GSW", "W2: Memphis vs Golden State"),
            # Conference Finals
            ("MIA", "BOS", "ECF: Miami vs Boston"),
            ("GSW", "DAL", "WCF: Golden State vs Dallas"),
            # Finals
            ("GSW", "BOS", "Finals: Golden State vs Boston"),
        ],
        "outcomes": {
            "E1: Miami vs Atlanta":          "MIA",  # 4-1
            "E1: Boston vs Brooklyn":        "BOS",  # 4-0
            "E1: Milwaukee vs Chicago":      "MIL",  # 4-1
            "E1: Philadelphia vs Toronto":   "PHI",  # 4-2
            "W1: Phoenix vs New Orleans":    "PHX",  # 4-2
            "W1: Memphis vs Minnesota":      "MEM",  # 4-2
            "W1: Golden State vs Denver":    "GSW",  # 4-1
            "W1: Utah vs Dallas":            "DAL",  # 4-2 upset
            "E2: Miami vs Philadelphia":     "MIA",  # 4-2
            "E2: Boston vs Milwaukee":       "BOS",  # 4-3
            "W2: Phoenix vs Dallas":         "DAL",  # 4-3 upset
            "W2: Memphis vs Golden State":   "GSW",  # 4-2
            "ECF: Miami vs Boston":          "BOS",  # 4-3
            "WCF: Golden State vs Dallas":   "GSW",  # 4-1
            "Finals: Golden State vs Boston":"GSW",  # 4-2
        },
    },
}
