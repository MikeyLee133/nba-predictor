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
            "E1: Miami vs Atlanta":           "MIA",  # 4-1
            "E1: Boston vs Brooklyn":         "BOS",  # 4-0
            "E1: Milwaukee vs Chicago":       "MIL",  # 4-1
            "E1: Philadelphia vs Toronto":    "PHI",  # 4-2
            "W1: Phoenix vs New Orleans":     "PHX",  # 4-2
            "W1: Memphis vs Minnesota":       "MEM",  # 4-2
            "W1: Golden State vs Denver":     "GSW",  # 4-1
            "W1: Utah vs Dallas":             "DAL",  # 4-2 upset
            "E2: Miami vs Philadelphia":      "MIA",  # 4-2
            "E2: Boston vs Milwaukee":        "BOS",  # 4-3
            "W2: Phoenix vs Dallas":          "DAL",  # 4-3 upset
            "W2: Memphis vs Golden State":    "GSW",  # 4-2
            "ECF: Miami vs Boston":           "BOS",  # 4-3
            "WCF: Golden State vs Dallas":    "GSW",  # 4-1
            "Finals: Golden State vs Boston": "GSW",  # 4-2
        },
    },
    "2020-21": {
        "matchups": [
            # First Round — East
            ("PHI", "WAS", "E1: Philadelphia vs Washington"),
            ("BKN", "BOS", "E1: Brooklyn vs Boston"),
            ("MIL", "MIA", "E1: Milwaukee vs Miami"),
            ("NYK", "ATL", "E1: New York vs Atlanta"),
            # First Round — West
            ("UTA", "MEM", "W1: Utah vs Memphis"),
            ("PHX", "LAL", "W1: Phoenix vs LA Lakers"),
            ("DEN", "POR", "W1: Denver vs Portland"),
            ("LAC", "DAL", "W1: LA Clippers vs Dallas"),
            # Second Round — East
            ("PHI", "ATL", "E2: Philadelphia vs Atlanta"),
            ("MIL", "BKN", "E2: Milwaukee vs Brooklyn"),
            # Second Round — West
            ("UTA", "LAC", "W2: Utah vs LA Clippers"),
            ("PHX", "DEN", "W2: Phoenix vs Denver"),
            # Conference Finals
            ("MIL", "ATL", "ECF: Milwaukee vs Atlanta"),
            ("PHX", "LAC", "WCF: Phoenix vs LA Clippers"),
            # Finals
            ("MIL", "PHX", "Finals: Milwaukee vs Phoenix"),
        ],
        "outcomes": {
            "E1: Philadelphia vs Washington": "PHI",  # 4-1
            "E1: Brooklyn vs Boston":         "BKN",  # 4-1
            "E1: Milwaukee vs Miami":         "MIL",  # 4-0
            "E1: New York vs Atlanta":        "ATL",  # 4-1 upset
            "W1: Utah vs Memphis":            "UTA",  # 4-1
            "W1: Phoenix vs LA Lakers":       "PHX",  # 4-2 upset (defending champs out)
            "W1: Denver vs Portland":         "DEN",  # 4-0
            "W1: LA Clippers vs Dallas":      "LAC",  # 4-3
            "E2: Philadelphia vs Atlanta":    "ATL",  # 4-3 upset
            "E2: Milwaukee vs Brooklyn":      "MIL",  # 4-3
            "W2: Utah vs LA Clippers":        "LAC",  # 4-2
            "W2: Phoenix vs Denver":          "PHX",  # 4-0
            "ECF: Milwaukee vs Atlanta":      "MIL",  # 4-2
            "WCF: Phoenix vs LA Clippers":    "PHX",  # 4-2
            "Finals: Milwaukee vs Phoenix":   "MIL",  # 4-2 (Giannis 50 pts game 6)
        },
    },
    "2019-20": {
        "matchups": [
            # First Round — East (Bubble — no true home court; higher seed listed first)
            ("MIL", "ORL", "E1: Milwaukee vs Orlando"),
            ("TOR", "BKN", "E1: Toronto vs Brooklyn"),
            ("BOS", "PHI", "E1: Boston vs Philadelphia"),
            ("MIA", "IND", "E1: Miami vs Indiana"),
            # First Round — West
            ("LAL", "POR", "W1: LA Lakers vs Portland"),
            ("LAC", "DAL", "W1: LA Clippers vs Dallas"),
            ("DEN", "UTA", "W1: Denver vs Utah"),
            ("HOU", "OKC", "W1: Houston vs OKC"),
            # Second Round — East
            ("MIL", "MIA", "E2: Milwaukee vs Miami"),
            ("BOS", "TOR", "E2: Boston vs Toronto"),
            # Second Round — West
            ("LAL", "HOU", "W2: LA Lakers vs Houston"),
            ("DEN", "LAC", "W2: Denver vs LA Clippers"),
            # Conference Finals
            ("MIA", "BOS", "ECF: Miami vs Boston"),
            ("LAL", "DEN", "WCF: LA Lakers vs Denver"),
            # Finals
            ("LAL", "MIA", "Finals: LA Lakers vs Miami"),
        ],
        "outcomes": {
            "E1: Milwaukee vs Orlando":   "MIL",  # 4-1
            "E1: Toronto vs Brooklyn":    "TOR",  # 4-0
            "E1: Boston vs Philadelphia": "BOS",  # 4-0
            "E1: Miami vs Indiana":       "MIA",  # 4-0
            "W1: LA Lakers vs Portland":  "LAL",  # 4-1
            "W1: LA Clippers vs Dallas":  "LAC",  # 4-2
            "W1: Denver vs Utah":         "DEN",  # 4-3 (Murray iconic)
            "W1: Houston vs OKC":         "HOU",  # 4-1
            "E2: Milwaukee vs Miami":     "MIA",  # 4-1 upset (Bubble Heat)
            "E2: Boston vs Toronto":      "BOS",  # 4-3
            "W2: LA Lakers vs Houston":   "LAL",  # 4-1
            "W2: Denver vs LA Clippers":  "DEN",  # 4-3 (blew 3-1 lead)
            "ECF: Miami vs Boston":       "MIA",  # 4-2
            "WCF: LA Lakers vs Denver":   "LAL",  # 4-1
            "Finals: LA Lakers vs Miami": "LAL",  # 4-2
        },
    },
    "2018-19": {
        "matchups": [
            # First Round — East
            ("MIL", "DET", "E1: Milwaukee vs Detroit"),
            ("TOR", "ORL", "E1: Toronto vs Orlando"),
            ("PHI", "BKN", "E1: Philadelphia vs Brooklyn"),
            ("BOS", "IND", "E1: Boston vs Indiana"),
            # First Round — West
            ("GSW", "LAC", "W1: Golden State vs LA Clippers"),
            ("DEN", "SAS", "W1: Denver vs San Antonio"),
            ("POR", "OKC", "W1: Portland vs OKC"),
            ("HOU", "UTA", "W1: Houston vs Utah"),
            # Second Round — East
            ("MIL", "BOS", "E2: Milwaukee vs Boston"),
            ("TOR", "PHI", "E2: Toronto vs Philadelphia"),
            # Second Round — West
            ("GSW", "HOU", "W2: Golden State vs Houston"),
            ("POR", "DEN", "W2: Portland vs Denver"),
            # Conference Finals
            ("MIL", "TOR", "ECF: Milwaukee vs Toronto"),
            ("GSW", "POR", "WCF: Golden State vs Portland"),
            # Finals
            ("GSW", "TOR", "Finals: Golden State vs Toronto"),
        ],
        "outcomes": {
            "E1: Milwaukee vs Detroit":      "MIL",  # 4-0
            "E1: Toronto vs Orlando":        "TOR",  # 4-1
            "E1: Philadelphia vs Brooklyn":  "PHI",  # 4-1
            "E1: Boston vs Indiana":         "BOS",  # 4-0
            "W1: Golden State vs LA Clippers":"GSW",  # 4-2
            "W1: Denver vs San Antonio":     "DEN",  # 4-3
            "W1: Portland vs OKC":           "POR",  # 4-1
            "W1: Houston vs Utah":           "HOU",  # 4-1
            "E2: Milwaukee vs Boston":       "MIL",  # 4-1
            "E2: Toronto vs Philadelphia":   "TOR",  # 4-3 (Kawhi's shot)
            "W2: Golden State vs Houston":   "GSW",  # 4-2
            "W2: Portland vs Denver":        "POR",  # 4-3
            "ECF: Milwaukee vs Toronto":     "TOR",  # 4-2
            "WCF: Golden State vs Portland": "GSW",  # 4-0
            "Finals: Golden State vs Toronto":"TOR",  # 4-2
        },
    },
}
