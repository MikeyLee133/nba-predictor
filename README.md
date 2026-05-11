# NBA Playoff Predictor

Predicts NBA playoff series winners using live team and player stats from the official NBA Stats API.

## Project structure

```
nba_predictor/
├── main.py                   # CLI entry point
├── app.py                    # Streamlit web app
├── tests/                    # pytest test suite (64 tests)
└── nba_predictor/
    ├── config.py             # All constants, weights, and team mappings
    ├── fetcher.py            # Fetches stats from NBA Stats API (with disk cache)
    ├── model.py              # Prediction logic (no I/O)
    ├── history.py            # Prediction persistence and accuracy tracking
    ├── ui.py                 # Streamlit UI components
    └── display.py            # Formats and prints CLI output
```

## Install

```bash
pip install -r requirements.txt
```

## Run

**Web app (recommended):**
```bash
streamlit run app.py
```

**REST API:**
```bash
uvicorn nba_predictor.api:app --reload
```
Interactive docs available at `http://localhost:8000/docs`.

**Docker (runs both Streamlit and API together):**
```bash
docker compose up --build
```

**Command line:**
```bash
python main.py
```

**Tests:**
```bash
pytest
```

## Features

- **Adjustable model weights** — sidebar sliders tune team/player blend, home-court advantage, and all individual stat weights; predictions update instantly. Reset to defaults with one button.
- **Injury / unavailable flags** — mark players as out from the sidebar; they are excluded from star-power scoring immediately
- **Live series adjustment** — set the current series score (e.g. 3-1) to blend model predictions with historical NBA comeback rates
- **Prediction history** — save predictions each round, record actual outcomes, and track model accuracy over time
- **Full season vs recent form** — five tabs: Full Season, Last 15 Games, Comparison (with trend arrows ↑↓→), Live Series, and History
- **Live data** — stats fetched from the official NBA Stats API, cached to disk for 24 hours with a freshness indicator showing how old the data is
- **Download CSV** — export any predictions table with one click

## How the model works

### Data
- **Team stats** — net rating, offensive/defensive ratings, pace, points, assists, 3PM (7 stats total)
- **Player stats** — PIE (Player Impact Estimate), points, assists, rebounds, 3PM for top 3 players per team

### Scoring

1. **Team composite score (60% default)** — each stat is min-max normalized across all 30 teams, then multiplied by its weight (see `config.TEAM_STAT_WEIGHTS`)

2. **Player star-power score (40% default)** — top 3 players by PIE per team are averaged across 5 stats (see `config.PLAYER_STAT_WEIGHTS`). Players marked unavailable are excluded before scoring.

3. **Blend + home-court** — the two scores are combined and the home team receives a +4% multiplier by default. Win probability is each team's share of the combined score.

4. **Live series adjustment** — when a series is in progress, win probability is blended 50/50 with historical NBA series survival rates (e.g. a 3-1 leader has won ~96% of series historically).

### Tuning

All default weights live in `config.py`. The web app sidebar lets you adjust them live without editing any files. To change the defaults, edit `TEAM_STAT_WEIGHTS`, `PLAYER_STAT_WEIGHTS`, `TEAM_SCORE_WEIGHT`, `PLAYER_SCORE_WEIGHT`, `HOME_COURT_MULTIPLIER`, or `RECENT_GAMES` in `config.py`.

To update matchups for a new round, edit `PLAYOFF_MATCHUPS`, `ABBR_TO_FULL`, and `PLAYOFF_ROUND` in `config.py`.
