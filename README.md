# NBA Playoff Predictor

Predicts NBA playoff series winners using live team and player stats from the official NBA Stats API.

![NBA Playoff Predictor](assets/demo.gif)

## Results

Backtested across **90 playoff series (2018–19 through 2023–24), all rounds**:

| Model | Correct | Accuracy |
|---|---|---|
| Weighted model | 65 / 90 | **72.2%** |
| ML model (LOSO-CV) | 62 / 82 | **75.6%** |
| Baseline: always pick higher seed | 69 / 90 | 76.7% |
| Baseline: coin flip | 45 / 90 | 50.0% |

The ML model accuracy is reported using **leave-one-season-out cross-validation** (trains on 5 seasons, tests on the held-out season) — no data leakage. Both models are competitive with the higher-seed naive baseline, which captures most of the predictive signal in playoff seeding. Run `python scripts/run_backtest.py` to reproduce.

## Project structure

```
nba_predictor/
├── main.py                   # CLI entry point
├── app.py                    # Streamlit web app
├── scripts/run_backtest.py   # Backtest script with baseline comparison
├── tests/                    # pytest test suite (139 tests)
└── nba_predictor/
    ├── config.py             # All constants, weights, and team mappings
    ├── fetcher.py            # Fetches stats from NBA Stats API (with disk cache)
    ├── model.py              # Prediction logic (no I/O)
    ├── ml_model.py           # Logistic regression model
    ├── backtest.py           # Historical accuracy testing
    ├── history.py            # Prediction persistence and accuracy tracking
    ├── historical.py         # 6 seasons of past playoff matchups and outcomes
    ├── api.py                # FastAPI REST interface
    ├── ui.py                 # Streamlit UI components
    └── display.py            # Formats and prints CLI output
```

## Install

```bash
pip install .          # production dependencies
pip install ".[dev]"   # include pytest, ruff, mypy
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

**CLI:**
```bash
python main.py
```

**Backtest:**
```bash
python scripts/run_backtest.py
```

**Tests:**
```bash
pytest
```

## Features

- **Adjustable model weights** — sidebar sliders tune team/player blend, home-court advantage, and all individual stat weights; predictions update instantly
- **ML model tab** — logistic regression trained on 90 historical series, compared against the weighted model with feature importance chart
- **Injury / unavailable flags** — mark players as out; excluded from star-power scoring immediately
- **Live series adjustment** — set the current series score to blend predictions with historical NBA comeback rates
- **Prediction history** — save predictions, record actual outcomes, track accuracy over time
- **Full season vs recent form** — Full Season, Last 15 Games, Comparison, Live Series, History, Backtest, and ML Model tabs
- **REST API** — 4 endpoints with OpenAPI docs at `/docs`
- **Live data** — stats from the NBA Stats API, cached to disk for 24 hours

## How the model works

### Data
- **Team stats** — net rating, offensive/defensive ratings, pace, points, assists, 3PM (7 stats)
- **Player stats** — PIE (Player Impact Estimate), points, assists, rebounds, 3PM for top 3 players per team

### Scoring

1. **Team composite score (60% default)** — each stat is min-max normalized across all 30 teams, then multiplied by its weight (see `config.TEAM_STAT_WEIGHTS`)
2. **Player star-power score (40% default)** — top 3 players by PIE per team averaged across 5 stats. Unavailable players are excluded before selection.
3. **Blend + home-court** — scores combined with a +4% home-court multiplier. Win probability is each team's share of the combined score.
4. **Live series adjustment** — blends model prediction 50/50 with historical NBA series survival rates.

### ML model

Logistic regression trained on ~90 historical playoff series. Features are stat differentials (home − away) for 7 team stats plus player score. `StandardScaler` normalises features; `CalibratedClassifierCV` ensures reliable probability outputs. Evaluated with leave-one-season-out CV to avoid data leakage.

### Tuning

All weights live in `config.py`. The sidebar lets you adjust them live. To update for a new round, edit `PLAYOFF_MATCHUPS`, `ABBR_TO_FULL`, and `PLAYOFF_ROUND` in `config.py`.
