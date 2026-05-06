# NBA Playoff Predictor

Predicts NBA playoff series winners using live team and player stats
scraped from Basketball Reference.

## Project structure

```
nba_predictor/
├── main.py                   # Entry point — orchestrates the pipeline
└── nba_predictor/
    ├── config.py             # All constants, weights, team mappings
    ├── scraper.py            # Fetches raw HTML (no parsing)
    ├── parser.py             # Turns HTML into DataFrames (no HTTP)
    ├── model.py              # Prediction logic (no I/O)
    └── display.py            # Formats and prints output (no logic)
```

Each module has exactly one responsibility. To change the model, edit
`config.py` or `model.py`. To add a new data source, edit `scraper.py`
and `parser.py` only.

## Install

```bash
pip install requests beautifulsoup4 pandas tabulate
```

## Run

```bash
python main.py
```

## How it works

### Data sources
- **Team stats** — offensive/defensive ratings, net rating, pace, points, assists, 3PM
- **Player stats** — PER, points, assists, rebounds, 3PM for every player

### Scoring model

1. **Team composite score (60%)** — each of 8 stats is min-max normalized
   across all 30 teams, then multiplied by its weight (see `config.TEAM_STAT_WEIGHTS`).

2. **Player star-power score (40%)** — top 3 players by PER per team are
   averaged across 5 stats (see `config.PLAYER_STAT_WEIGHTS`).

3. **Blend + home-court** — the two scores are combined and the home team
   receives a +4% multiplier. Win probability is each team's share of the
   combined blended score.

### Tuning

All weights live in `config.py`. Change `TEAM_STAT_WEIGHTS`,
`PLAYER_STAT_WEIGHTS`, `TEAM_SCORE_WEIGHT`, `PLAYER_SCORE_WEIGHT`, or
`HOME_COURT_MULTIPLIER` without touching any other file.

To update matchups for a new round, edit `PLAYOFF_MATCHUPS` in `config.py`.
