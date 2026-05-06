"""
scraper.py
----------
Responsible for one thing only: fetching raw HTML from Basketball Reference.

Returns raw response text — no parsing, no business logic.
HTML is cached to disk for 24 hours to avoid hitting rate limits.
"""

import time
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import requests
from nba_predictor.config import BASE_URL, HEADERS, CRAWL_DELAY, SEASON

CACHE_DIR = Path(__file__).parent.parent / ".html_cache"
CACHE_TTL_HOURS = 24


class ScraperError(Exception):
    """Raised when an HTTP request fails or returns unexpected content."""


def _cache_path(url: str) -> Path:
    key = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{key}.html"


def _cache_valid(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_TTL_HOURS)


def _get(url: str, force: bool = False, retries: int = 3) -> str:
    """
    Fetch a URL and return the response text.
    Serves from disk cache if fresh (< 24 h). Pass force=True to bypass cache.
    Retries up to `retries` times on 429/503 with exponential backoff.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cached = _cache_path(url)

    if not force and _cache_valid(cached):
        return cached.read_text(encoding="utf-8")

    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code in (429, 503) and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            html = resp.text
            cached.write_text(html, encoding="utf-8")
            return html
        except requests.exceptions.Timeout:
            raise ScraperError(f"Request timed out: {url}")
        except requests.exceptions.HTTPError as e:
            raise ScraperError(f"HTTP {e.response.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            raise ScraperError(f"Request failed: {e}")

    raise ScraperError(f"Failed to fetch after {retries} attempts: {url}")


def fetch_team_stats_html(force: bool = False) -> str:
    """Return HTML of the NBA season summary page (contains team stat tables)."""
    url = f"{BASE_URL}/leagues/NBA_{SEASON}.html"
    print(f"  GET {url}")
    html = _get(url, force=force)
    if not _cache_valid(_cache_path(url)):
        time.sleep(CRAWL_DELAY)
    return html


def fetch_player_stats_html(force: bool = False) -> str:
    """Return HTML of the per-game player stats page."""
    url = f"{BASE_URL}/leagues/NBA_{SEASON}_per_game.html"
    print(f"  GET {url}")
    html = _get(url, force=force)
    if not _cache_valid(_cache_path(url)):
        time.sleep(CRAWL_DELAY)
    return html


def fetch_advanced_player_stats_html(force: bool = False) -> str:
    """Return HTML of the advanced player stats page (contains PER)."""
    url = f"{BASE_URL}/leagues/NBA_{SEASON}_advanced.html"
    print(f"  GET {url}")
    html = _get(url, force=force)
    if not _cache_valid(_cache_path(url)):
        time.sleep(CRAWL_DELAY)
    return html
