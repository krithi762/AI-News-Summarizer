"""
news_fetcher.py
---------------
Module responsible for collecting news articles from external sources.
Supports NewsAPI as the primary news provider and RSS feeds as an automatic fallback.
"""

import os
import re
import html
import logging
from typing import Any
import requests
import feedparser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Category Mappings & Centralized Configurations
# ---------------------------------------------------------------------------

SUPPORTED_CATEGORIES: list[str] = [
    "Technology",
    "Business",
    "Sports",
    "Science",
    "Health",
    "World",
    "Entertainment",
]

# Mapping to NewsAPI categories
NEWS_API_CATEGORY_MAP: dict[str, str] = {
    "Technology": "technology",
    "Business": "business",
    "Sports": "sports",
    "Science": "science",
    "Health": "health",
    "World": "general",
    "Entertainment": "entertainment",
}

# Centralized RSS Feed sources for each category
RSS_FEED_MAP: dict[str, list[dict[str, str]]] = {
    "Technology": [
        {"name": "BBC Tech", "url": "http://feeds.bbci.co.uk/news/technology/rss.xml"},
        {"name": "NYT Tech", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"},
        {"name": "The Verge", "url": "https://www.theverge.com/rss/index.xml"},
    ],
    "Business": [
        {"name": "BBC Business", "url": "http://feeds.bbci.co.uk/news/business/rss.xml"},
        {"name": "NYT Business", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"},
        {"name": "MarketWatch", "url": "https://feeds.content.dowjones.io/public/rss/mw_topstories"},
    ],
    "Sports": [
        {"name": "BBC Sport", "url": "http://feeds.bbci.co.uk/sport/rss.xml"},
        {"name": "NYT Sports", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml"},
        {"name": "ESPN", "url": "https://www.espn.com/espn/rss/news"},
    ],
    "Science": [
        {"name": "BBC Science", "url": "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml"},
        {"name": "NYT Science", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml"},
        {"name": "Phys.org", "url": "https://phys.org/rss-feed/"},
    ],
    "Health": [
        {"name": "BBC Health", "url": "http://feeds.bbci.co.uk/news/health/rss.xml"},
        {"name": "NYT Health", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml"},
        {"name": "Medical News Today", "url": "https://www.medicalnewstoday.com/feed"},
    ],
    "World": [
        {"name": "BBC World", "url": "http://feeds.bbci.co.uk/news/world/rss.xml"},
        {"name": "NYT World", "url": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"},
        {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    ],
    "Entertainment": [
        {"name": "BBC Entertainment", "url": "http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml"},
        {"name": "NYT Arts", "url": "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml"},
        {"name": "Variety", "url": "https://variety.com/feed/"},
    ],
}

# Standard HTTP headers for network requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ---------------------------------------------------------------------------
# Helper Utility Functions
# ---------------------------------------------------------------------------

def _clean_html_text(raw_html: str | None) -> str | None:
    """Removes HTML tags and decodes HTML entities from text."""
    if not raw_html or not isinstance(raw_html, str):
        return None
    # Remove HTML tags
    clean_text = re.sub(r"<[^>]+>", "", raw_html)
    # Decode HTML entities (e.g. &amp;, &quot;)
    clean_text = html.unescape(clean_text)
    # Strip unnecessary whitespaces
    clean_text = " ".join(clean_text.split())
    return clean_text if clean_text else None


def _extract_rss_image(entry: Any) -> str | None:
    """Safely extracts image thumbnail URL from RSS feed entry metadata."""
    # 1. Check media_content
    if hasattr(entry, "media_content") and entry.media_content:
        for media in entry.media_content:
            if isinstance(media, dict) and "url" in media:
                return media["url"]

    # 2. Check media_thumbnail
    if hasattr(entry, "media_thumbnail") and entry.media_thumbnail:
        for thumb in entry.media_thumbnail:
            if isinstance(thumb, dict) and "url" in thumb:
                return thumb["url"]

    # 3. Check enclosures
    if hasattr(entry, "enclosures") and entry.enclosures:
        for enc in entry.enclosures:
            if isinstance(enc, dict) and enc.get("type", "").startswith("image/") and "href" in enc:
                return enc["href"]

    # 4. Check embedded img tag in summary / description
    summary = entry.get("summary", "") or entry.get("description", "")
    if summary and isinstance(summary, str):
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', summary, re.IGNORECASE)
        if img_match:
            return img_match.group(1)

    return None


def _standardize_article(
    title: str | None,
    description: str | None,
    source: str | None,
    url: str | None,
    image_url: str | None,
    author: str | None,
    published_at: str | None,
    category: str,
    provider: str,
) -> dict[str, Any] | None:
    """
    Validates and standardizes article data into a unified dictionary structure.
    Returns None if mandatory fields (title, url) are missing.
    """
    if not title or not isinstance(title, str) or not title.strip():
        return None
    if not url or not isinstance(url, str) or not url.strip():
        return None

    return {
        "title": title.strip(),
        "description": _clean_html_text(description),
        "source": source.strip() if source and isinstance(source, str) else "Unknown Source",
        "url": url.strip(),
        "image_url": image_url.strip() if image_url and isinstance(image_url, str) else None,
        "author": author.strip() if author and isinstance(author, str) else None,
        "published_at": published_at.strip() if published_at and isinstance(published_at, str) else None,
        "category": category,
        "provider": provider,
    }


# ---------------------------------------------------------------------------
# NewsAPI Fetcher
# ---------------------------------------------------------------------------

def fetch_news_from_api(category: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Fetches articles from NewsAPI for a given category.

    Parameters:
    - category (str): The news category to query.
    - limit (int): Maximum number of articles to return.

    Returns:
    - list[dict]: List of standardized article dictionaries.
    """
    api_key = os.getenv("NEWS_API_KEY", "").strip()
    if not api_key or api_key == "your_news_api_key_here":
        logger.info("NewsAPI key is not configured. Skipping NewsAPI.")
        return []

    api_category = NEWS_API_CATEGORY_MAP.get(category, "general")
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": api_category,
        "language": "en",
        "pageSize": min(limit * 2, 50),  # Request extra to account for filtering
    }
    headers = {
        "X-Api-Key": api_key,
        "User-Agent": DEFAULT_HEADERS["User-Agent"],
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 401:
            logger.warning("NewsAPI error: Invalid or expired API key.")
            return []
        if response.status_code == 429:
            logger.warning("NewsAPI error: Rate limit exceeded.")
            return []
        response.raise_for_status()

        data = response.json()
        if data.get("status") != "ok":
            logger.warning("NewsAPI response status not OK: %s", data.get("message", "Unknown error"))
            return []

        raw_articles = data.get("articles", [])
        standardized_articles: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for art in raw_articles:
            if not isinstance(art, dict):
                continue

            art_url = art.get("url")
            if not art_url or art_url in seen_urls:
                continue

            # Exclude removed articles placeholder titles
            art_title = art.get("title")
            if art_title and "[Removed]" in art_title:
                continue

            source_name = None
            if isinstance(art.get("source"), dict):
                source_name = art["source"].get("name")

            standardized = _standardize_article(
                title=art_title,
                description=art.get("description"),
                source=source_name,
                url=art_url,
                image_url=art.get("urlToImage"),
                author=art.get("author"),
                published_at=art.get("publishedAt"),
                category=category,
                provider="NewsAPI",
            )

            if standardized:
                seen_urls.add(art_url)
                standardized_articles.append(standardized)
                if len(standardized_articles) >= limit:
                    break

        return standardized_articles

    except requests.exceptions.RequestException as e:
        logger.warning("Network or request error while fetching from NewsAPI: %s", type(e).__name__)
        return []
    except Exception as e:
        logger.warning("Unexpected error during NewsAPI parsing: %s", type(e).__name__)
        return []


# ---------------------------------------------------------------------------
# RSS Feeds Fetcher
# ---------------------------------------------------------------------------

def fetch_news_from_rss(category: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Fetches articles from predefined RSS feeds for a given category.

    Parameters:
    - category (str): The news category to query.
    - limit (int): Maximum number of articles to return.

    Returns:
    - list[dict]: List of standardized article dictionaries.
    """
    feeds = RSS_FEED_MAP.get(category, RSS_FEED_MAP.get("Technology", []))
    standardized_articles: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for feed_info in feeds:
        feed_name = feed_info.get("name", "RSS Feed")
        feed_url = feed_info.get("url", "")
        if not feed_url:
            continue

        try:
            # Fetch with custom headers to prevent 403 Forbidden on restrictive RSS feeds
            resp = requests.get(feed_url, headers=DEFAULT_HEADERS, timeout=8)
            if resp.status_code != 200:
                logger.warning("RSS Feed %s returned status %d", feed_name, resp.status_code)
                continue

            parsed_feed = feedparser.parse(resp.content)
            entries = getattr(parsed_feed, "entries", [])

            for entry in entries:
                art_url = getattr(entry, "link", None)
                if not art_url or art_url in seen_urls:
                    continue

                art_title = getattr(entry, "title", None)
                art_desc = getattr(entry, "summary", None) or getattr(entry, "description", None)
                art_published = getattr(entry, "published", None) or getattr(entry, "updated", None)
                art_author = getattr(entry, "author", None)
                art_image = _extract_rss_image(entry)

                standardized = _standardize_article(
                    title=art_title,
                    description=art_desc,
                    source=feed_name,
                    url=art_url,
                    image_url=art_image,
                    author=art_author,
                    published_at=art_published,
                    category=category,
                    provider="RSS",
                )

                if standardized:
                    seen_urls.add(art_url)
                    standardized_articles.append(standardized)
                    if len(standardized_articles) >= limit:
                        return standardized_articles

        except Exception as e:
            logger.warning("Failed to fetch/parse RSS feed %s: %s", feed_name, type(e).__name__)
            continue

    return standardized_articles[:limit]


# ---------------------------------------------------------------------------
# Main Unified News Fetcher
# ---------------------------------------------------------------------------

def fetch_news(category: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Main entry point for fetching news articles.
    Attempts NewsAPI first if configured; falls back automatically to RSS feeds.

    Parameters:
    - category (str): Selected category (e.g. Technology, Business, Sports).
    - limit (int): Desired number of articles (default 10).

    Returns:
    - list[dict]: List of standardized article dictionaries.
    """
    # Sanitize category
    if category not in SUPPORTED_CATEGORIES:
        category = "Technology"

    articles: list[dict[str, Any]] = []

    # 1. Try NewsAPI
    api_key = os.getenv("NEWS_API_KEY", "").strip()
    if api_key and api_key != "your_news_api_key_here":
        logger.info("Attempting to fetch news from NewsAPI for category: %s", category)
        articles = fetch_news_from_api(category=category, limit=limit)

    # 2. Fallback to RSS if NewsAPI returned nothing or wasn't configured
    if not articles:
        logger.info("Fetching news from RSS fallback for category: %s", category)
        articles = fetch_news_from_rss(category=category, limit=limit)

    return articles
