# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Web app (primary interface)
streamlit run app.py

# CLI
python main.py

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test file
pytest tests/test_model.py

# Run a single test
pytest tests/test_model.py::test_team_scores_better_team_scores_higher
```

## Architecture

Data flows in one direction: **fetcher → model → ui/display**. Nothing upstream imports from downstream.

```
fetcher.py      pulls from NBA Stats API, writes .pkl disk cache
    ↓
model.py        pure functions, no I/O — takes DataFrames, returns SeriesPrediction objects
    ↓
history.py      persists predictions and outcomes to prediction_history.json
ui.py           Streamlit components (web app)
display.py      tabulate-based printing (CLI only)
```

**`config.py` is the single source of truth** for all weights, multipliers, matchups, and team mappings. All other files import from it; none duplicate those values — except sidebar slider defaults in `app.py`, which intentionally read from config at startup.

## Key design decisions

**Two rendering paths share the same model.** `app.py` + `ui.py` serve the Streamlit app; `main.py` + `display.py` serve the CLI. Both call the same `model.py` functions. `model.py` accepts optional weight overrides (`weights=`, `team_w=`, `player_w=`, `home_mult=`, `unavailable=`) that default to config values, keeping the CLI working without changes.

**PIE, not PER.** The player metric column is named `per` throughout the codebase, but the data source is the NBA API's `PIE` (Player Impact Estimate) field, multiplied by 100 to approximate PER's numeric range. This happens in `fetcher.py:_fetch_raw_player_df`.

**Disk cache lives in `.data_cache/`** as `.pkl` files, keyed by season, stat type, and `last_n` window (e.g. `team_stats_2025-26_0.pkl`, `player_stats_2025-26_15.pkl`). TTL is 24 hours. Including SEASON in the key means changing the season in config automatically bypasses stale cache. The Streamlit in-memory cache TTL is 1 hour (separate from disk).

**Sidebar weights are live but not persisted.** Sliders rerender predictions instantly but reset on page reload. The "Reset to defaults" button clears all 13 slider session state keys at once. To change defaults permanently, edit `config.py`.

**Prediction history is stored in `prediction_history.json`** (gitignored). `history.py` handles saving, outcome recording, and accuracy stats. The History tab in the web app surfaces this data.

**Series score adjustment blends two signals.** `adjust_for_series_score()` in `model.py` takes the model's win % and blends it 50/50 with historical NBA series comeback rates from `config.SERIES_COMEBACK_RATES`. At 0-0 the prediction is unchanged.

## Test suite

64 tests across four files — all pure unit tests, no API calls:
- `tests/test_model.py` — normalization, scoring, prediction logic
- `tests/test_fetcher.py` — cache validity and `_load_or_build` behaviour
- `tests/test_history.py` — save/load/outcome recording/accuracy stats
- `tests/test_injuries.py` — unavailable player exclusion from scoring
- `tests/test_series_score.py` — series score adjustment blending

## Adding a new playoff round

Edit `PLAYOFF_MATCHUPS`, `ABBR_TO_FULL` / `FULL_TO_ABBR`, and `PLAYOFF_ROUND` in `config.py`. No other file needs to change.
