# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Web app (primary interface)
streamlit run app.py

# CLI
python main.py

# Install dependencies
pip install requests pandas tabulate nba_api streamlit
```

No test suite or linter is configured.

## Architecture

Data flows in one direction: **fetcher → model → ui/display**. Nothing upstream imports from downstream.

```
fetcher.py      pulls from NBA Stats API, writes .pkl disk cache
    ↓
model.py        pure functions, no I/O — takes DataFrames, returns SeriesPrediction objects
    ↓
ui.py           Streamlit components (web app)
display.py      tabulate-based printing (CLI only)
```

**`config.py` is the single source of truth** for all weights, multipliers, matchups, and team mappings. All other files import from it; none duplicate those values — except sidebar slider defaults in `app.py`, which intentionally read from config at startup.

## Key design decisions

**Two rendering paths share the same model.** `app.py` + `ui.py` serve the Streamlit app; `main.py` + `display.py` serve the CLI. Both call the same `model.py` functions. `model.py` accepts optional weight overrides (`weights=`, `team_w=`, `player_w=`, `home_mult=`) that default to config values, keeping the CLI working without changes.

**PIE, not PER.** The player metric column is named `per` throughout the codebase, but the data source is the NBA API's `PIE` (Player Impact Estimate) field, multiplied by 100 to approximate PER's numeric range. This happens in `fetcher.py:_fetch_raw_player_df`.

**Disk cache lives in `.data_cache/`** as `.pkl` files, keyed by stat type and `last_n` window (e.g. `team_stats_0.pkl`, `player_stats_15.pkl`). TTL is 24 hours. The Streamlit refresh button calls `st.cache_data.clear()` then re-fetches with `force=True`. The Streamlit in-memory cache TTL is 1 hour (separate from the disk cache).

**Sidebar weights are live but not persisted.** Adjusting sliders in the web app rerenders predictions instantly but resets on page reload. To change defaults, edit `config.py`.

## Adding a new playoff round

Edit `PLAYOFF_MATCHUPS` and `ABBR_TO_FULL` / `FULL_TO_ABBR` in `config.py`. No other file needs to change.
