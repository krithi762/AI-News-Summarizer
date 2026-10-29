"""
utils.py
--------
Helper utilities and styling functions for the Streamlit UI/UX.
Provides centralized CSS injection, responsive styling tokens, safe date formatting,
text truncators, image fallback handlers, and modular UI components.
"""

import re
from datetime import datetime
from typing import Any
import streamlit as st
import news_fetcher
import database


import theme


# ---------------------------------------------------------------------------
# Centralized Custom CSS (Delegates to theme.py)
# ---------------------------------------------------------------------------

def apply_custom_css() -> None:
    """
    Injects centralized, professional CSS styling into the Streamlit app.
    Supports dynamic Light Mode and Dark Mode switching via theme.py.
    """
    theme.apply_theme()


# ---------------------------------------------------------------------------
# Formatting & Text Utilities
# ---------------------------------------------------------------------------

def format_date_human(raw_date: str | None) -> str:
    """
    Converts various date string formats (ISO, RFC 2822) into a concise readable string.
    Falls back gracefully to the original string or 'Recent'.
    """
    if not raw_date or not isinstance(raw_date, str):
        return "Recently"

    clean_str = raw_date.strip()

    # Try ISO 8601 (e.g. 2026-08-28T13:45:00Z)
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S GMT",
    ):
        try:
            dt = datetime.strptime(clean_str[:25].strip(), fmt)
            return dt.strftime("%b %d, %Y")
        except Exception:
            continue

    # Fallback to simple extraction of YYYY-MM-DD
    match = re.search(r"(\d{4}-\d{2}-\d{2})", clean_str)
    if match:
        return match.group(1)

    return clean_str[:30]


def truncate_description(desc: str | None, max_chars: int = 190) -> str:
    """
    Truncates an article description cleanly to max_chars without cutting words in half.
    """
    if not desc or not isinstance(desc, str):
        return "No summary description available for this article."

    cleaned = " ".join(desc.split())
    if len(cleaned) <= max_chars:
        return cleaned

    truncated = cleaned[:max_chars].rsplit(" ", 1)[0]
    return f"{truncated}..."


# ---------------------------------------------------------------------------
# Data Fetcher Wrapper with Source Selection
# ---------------------------------------------------------------------------

def fetch_news_with_source(
    category: str,
    limit: int = 10,
    source_mode: str = "Automatic",
) -> list[dict[str, Any]]:
    """
    Fetches news honoring the user's category selection and source selection mode
    ('Automatic', 'NewsAPI', 'RSS').
    """
    # Handle 'All' category
    target_category = category
    if category == "All":
        target_category = "Technology"

    if source_mode == "NewsAPI":
        articles = news_fetcher.fetch_news_from_api(category=target_category, limit=limit)
        # If NewsAPI returned nothing, fall back to RSS safely
        if not articles:
            articles = news_fetcher.fetch_news_from_rss(category=target_category, limit=limit)
    elif source_mode == "RSS":
        articles = news_fetcher.fetch_news_from_rss(category=target_category, limit=limit)
    else:  # Automatic
        articles = news_fetcher.fetch_news(category=target_category, limit=limit)

    # Persist in SQLite
    if articles:
        database.save_articles_batch(articles)

    return articles
