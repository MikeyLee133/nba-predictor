"""
scraper.py
----------
Responsible for one thing only: fetching raw HTML from Basketball Reference.

Returns raw response text — no parsing, no business logic.
"""

import time
import requests
from nba_predictor.config import BASE_URL, HEADERS, CRAWL_DELAY, SEASON


class ScraperError(Exception):
    """Raised when an HTTP request fails or returns unexpected content."""


def _get(url: str) -> str:
    """
    Fetch a URL and return the response text.
    Raises ScraperError on HTTP errors or timeouts.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.exceptions.Timeout:
        raise ScraperError(f"Request timed out: {url}")
    except requests.exceptions.HTTPError as e:
        raise ScraperError(f"HTTP {e.response.status_code} for {url}")
    except requests.exceptions.RequestException as e:
        raise ScraperError(f"Request failed: {e}")


def fetch_team_stats_html() -> str:
    """Return raw HTML of the NBA season summary page (contains team stat tables)."""
    url = f"{BASE_URL}/leagues/NBA_{SEASON}.html"
    print(f"  GET {url}")
    html = _get(url)
    time.sleep(CRAWL_DELAY)
    return html


def fetch_player_stats_html() -> str:
    """Return raw HTML of the per-game player stats page."""
    url = f"{BASE_URL}/leagues/NBA_{SEASON}_per_game.html"
    print(f"  GET {url}")
    html = _get(url)
    time.sleep(CRAWL_DELAY)
    return html
