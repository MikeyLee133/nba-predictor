# Changelog

All notable changes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.2.0] - 2026-05-10 — Production Ready

### Added
- **FastAPI REST API** with 4 endpoints: `GET /health`, `GET /predictions`, `GET /predictions/{home}/{away}`, `GET /teams`
- **GitHub Actions CI** — pytest runs automatically on every push and pull request
- **Docker** — `docker compose up --build` runs Streamlit (:8501) and FastAPI (:8000) together with a shared named volume for the stats cache
- **In-memory TTL cache** on the API — first request reads from disk, subsequent requests serve from memory (~0ms)
- **Parallel historical fetching** — `fetch_seasons_parallel()` fetches all 6 seasons concurrently; cold run drops from ~12s to ~1s
- **Enriched OpenAPI docs** — field descriptions, example values, tag groups, and route docstrings at `/docs`

### Changed
- Replaced `print()` with Python `logging` module in `fetcher.py` and `main.py`; cache timing is now `DEBUG` level (silent by default)
- `API_SLEEP_SECONDS` moved to `config.py` — consistent with every other tunable constant
- Removed duplicate `NOP`/`TOR`/`UTA` keys from `ABBR_TO_FULL` (silent data integrity bug)
- Removed unused `player_df` parameter from `GET /teams` — was triggering a needless fetch
- Simplified `train_accuracy` computation in `ml_model.py`

---

## [1.1.0] - 2026-05-09 — ML & Advanced Features

### Added
- **Logistic regression ML model** trained on ~90 historical playoff series (6 seasons, all rounds)
- **Leave-one-season-out cross-validation** — reports honest held-out accuracy, not inflated training accuracy
- **Probability calibration** (`CalibratedClassifierCV`) for reliable win probability outputs on small datasets
- **Player star-power feature** — 8th feature in the ML model using PIE-based score differentials
- **Historical backtesting** across 2018-19 through 2023-24 (all rounds) — shows per-season and overall accuracy
- **Prediction history** — save predictions each round, record actual outcomes, track accuracy over time (History tab)
- **Injury/unavailable player flags** — sidebar multiselect excludes players from star-power scoring instantly
- **In-series score adjustment** — blends model predictions 50/50 with historical NBA series comeback rates (Live Series tab)
- `historical.py` — 6 seasons of playoff matchups and outcomes (~90 series)
- `backtest.py`, `history.py`, `ml_model.py` modules, each with full test coverage

### Fixed
- Reset weights button now uses `st.session_state.update()` instead of key deletion — sliders correctly snap back to defaults

---

## [1.0.0] - 2026-05-08 — Core Predictor

### Added
- **Streamlit web app** with Full Season, Last 15 Games, Comparison, Live Series, and History tabs
- **Sidebar weight sliders** — 15 live-adjustable parameters (team/player blend, home-court advantage, 7 team stat weights, 5 player stat weights); auto-normalised to 100%
- **Data freshness indicator** — shows how old the cached stats are next to the Refresh button
- **CSV export** — download any predictions table with one click
- **Disk-based caching** — 24-hour TTL, season-keyed filenames (`team_stats_2025-26_0.pkl`) for automatic invalidation on season change
- **Weight validation** — `AssertionError` with actual sum if weights in `config.py` don't add to 1.0
- **Test suite** — 64 tests covering model logic, caching, history, injury flags, and series score adjustment (zero API calls)
- `CLAUDE.md` with architecture notes, design decisions, and development commands
- `requirements.txt`

### Architecture
- `fetcher → model → ui/display` pipeline — nothing upstream imports downstream
- `config.py` as single source of truth for all weights, multipliers, matchups, and constants
- Two rendering paths (Streamlit + CLI) share the same pure `model.py` functions
