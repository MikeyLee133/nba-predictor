# NBA Playoff Predictor

Predicts NBA playoff series winners using live team and player stats from the official NBA Stats API.

## Project structure

```
nba_predictor/
├── main.py                   # CLI entry point
├── app.py                    # Streamlit web app
└── nba_predictor/
    ├── config.py             # All constants, weights, and team mappings
    ├── fetcher.py            # Fetches stats from NBA Stats API (with disk cache)
    ├── model.py              # Prediction logic (no I/O)
    └── display.py            # Formats and prints CLI output
```

## Install

```bash
pip install requests pandas tabulate nba_api streamlit
```

## Run

**Web app (recommended):**
```bash
streamlit run app.py
```

**Command line:**
```bash
python main.py
```

## Features

- **Full season vs recent form** — the web app has three tabs: Full Season, Last 15 Games, and a Comparison tab that shows win percentages side by side with trend arrows (↑↓→)
- **Live data** — stats are fetched from the official NBA Stats API and cached to disk for 24 hours
- **Refresh button** — force a fresh fetch at any time from the web app

## How the model works

### Data
- **Team stats** — net rating, offensive/defensive ratings, pace, points, assists, 3PM
- **Player stats** — PIE (Player Impact Estimate), points, assists, rebounds, 3PM for top 3 players per team

### Scoring

1. **Team composite score (60%)** — each of 8 stats is min-max normalized across all 30 teams, then multiplied by its weight (see `config.TEAM_STAT_WEIGHTS`)

2. **Player star-power score (40%)** — top 3 players by PIE per team are averaged across 5 stats (see `config.PLAYER_STAT_WEIGHTS`)

3. **Blend + home-court** — the two scores are combined and the home team receives a +4% multiplier. Win probability is each team's share of the combined score.

### Tuning

All weights live in `config.py`. Change `TEAM_STAT_WEIGHTS`, `PLAYER_STAT_WEIGHTS`, `TEAM_SCORE_WEIGHT`, `PLAYER_SCORE_WEIGHT`, `HOME_COURT_MULTIPLIER`, or `RECENT_GAMES` without touching any other file.

To update matchups for a new round, edit `PLAYOFF_MATCHUPS` in `config.py`.
