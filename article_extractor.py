"""
article_extractor.py
--------------------
Module responsible for extracting full article body text, author, publication date,
and metadata from web URLs.
Uses newspaper4k as the primary extractor with an automated BeautifulSoup4 fallback.
"""

import logging
import re
import warnings
from typing import Any
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# Suppress harmless optional NLP warnings from newspaper
warnings.filterwarnings("ignore", category=UserWarning, module="newspaper")

from newspaper import Article, Config

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default browser request headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Minimum words required for an article extraction to be considered valid
MIN_ARTICLE_WORD_COUNT = 25


# ---------------------------------------------------------------------------
# URL Validation
# ---------------------------------------------------------------------------

def validate_url(url: str | None) -> bool:
    """
    Validates whether a given URL is a properly formed HTTP/HTTPS web address.

    Parameters:
    - url (str | None): URL string to validate.

    Returns:
    - bool: True if URL is valid, False otherwise.
    """
    if not url or not isinstance(url, str):
        return False
    
    clean_url = url.strip()
    if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        return False

    try:
        parsed = urlparse(clean_url)
        return bool(parsed.scheme in ("http", "https") and parsed.netloc)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_article_text(text: str | None) -> str:
    """
    Cleans raw extracted article text for readability and downstream AI processing.
    Preserves meaningful paragraphs, sentence structure, and punctuation.

    Parameters:
    - text (str | None): Raw extracted text.

    Returns:
    - str: Cleaned, structured article text.
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace non-breaking spaces and irregular whitespace characters
    cleaned = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")

    # Split into lines and trim whitespace per line
    lines = [line.strip() for line in cleaned.split("\n")]

    # Filter out empty lines while preserving structural paragraph breaks
    cleaned_paragraphs: list[str] = []
    for line in lines:
        if not line:
            continue

        # Normalize multiple spaces inside each line
        normalized_line = re.sub(r"[ \t]+", " ", line)

        # Skip common boilerplate text / cookie notices
        lower_line = normalized_line.lower()
        if any(
            boilerplate in lower_line
            for boilerplate in [
                "all rights reserved",
                "subscribe to our newsletter",
                "cookie policy",
                "terms of service",
                "sign up for free",
                "follow us on twitter",
                "share this article",
                "advertisement",
            ]
        ) and len(normalized_line.split()) < 10:
            continue

        cleaned_paragraphs.append(normalized_line)

    # Join meaningful paragraphs with double newlines
    return "\n\n".join(cleaned_paragraphs).strip()


# ---------------------------------------------------------------------------
# Primary Extractor (newspaper4k)
# ---------------------------------------------------------------------------

def _extract_with_newspaper(url: str) -> dict[str, Any] | None:
    """
    Attempts article extraction using newspaper4k.

    Parameters:
    - url (str): Article URL.

    Returns:
    - dict or None: Extracted metadata and text if successful, None otherwise.
    """
    try:
        config = Config()
        config.browser_user_agent = DEFAULT_HEADERS["User-Agent"]
        config.request_timeout = 10
        config.fetch_images = False
        config.memoize_articles = False

        article = Article(url, config=config)
        article.download()
        article.parse()

        title = article.title.strip() if article.title else None
        text = clean_article_text(article.text)
        authors = ", ".join(article.authors) if article.authors else None
        published_at = str(article.publish_date) if article.publish_date else None

        word_count = len(text.split()) if text else 0

        # Check if meaningful text was extracted
        if text and word_count >= MIN_ARTICLE_WORD_COUNT:
            return {
                "title": title,
                "author": authors,
                "published_at": published_at,
                "text": text,
                "word_count": word_count,
            }

        logger.info("Newspaper4k extracted insufficient text (%d words). Trying fallback.", word_count)
        return None

    except Exception as e:
        logger.warning("Newspaper4k extraction failed for %s: %s", url, type(e).__name__)
        return None


# ---------------------------------------------------------------------------
# Fallback Extractor (requests + BeautifulSoup4)
# ---------------------------------------------------------------------------

def _extract_with_bs4(url: str) -> dict[str, Any] | None:
    """
    Fallback article extraction using requests and BeautifulSoup4.

    Parameters:
    - url (str): Article URL.

    Returns:
    - dict or None: Extracted metadata and text if successful, None otherwise.
    """
    try:
        response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
        if response.status_code != 200:
            logger.warning("BS4 fallback request returned status %d for %s", response.status_code, url)
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        # 1. Extract Title
        title = None
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        elif soup.title:
            title = soup.title.get_text().strip()

        # 2. Extract Author
        author = None
        author_tag = (
            soup.find("meta", attrs={"name": "author"})
            or soup.find("meta", property="article:author")
            or soup.find("meta", attrs={"name": "twitter:creator"})
        )
        if author_tag and author_tag.get("content"):
            author = author_tag["content"].strip()

        # 3. Extract Published Date
        published_at = None
        pub_tag = (
            soup.find("meta", property="article:published_time")
            or soup.find("meta", attrs={"name": "pubdate"})
            or soup.find("meta", attrs={"name": "date"})
            or soup.find("time")
        )
        if pub_tag:
            published_at = pub_tag.get("content") or pub_tag.get("datetime") or pub_tag.get_text().strip()

        # 4. Remove unwanted HTML elements (scripts, styles, nav, headers, footers, ads)
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            element.decompose()

        # 5. Locate article container
        article_body = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"(article-body|story-body|entry-content|post-content|content__article-body)", re.I))
            or soup.find("body")
        )

        paragraphs: list[str] = []
        if article_body:
            for p in article_body.find_all("p"):
                p_text = p.get_text().strip()
                if len(p_text.split()) > 4:  # Ignore trivial 1-3 word snippets
                    paragraphs.append(p_text)

        raw_text = "\n\n".join(paragraphs)
        cleaned_text = clean_article_text(raw_text)
        word_count = len(cleaned_text.split()) if cleaned_text else 0

        if cleaned_text and word_count >= MIN_ARTICLE_WORD_COUNT:
            return {
                "title": title,
                "author": author,
                "published_at": published_at,
                "text": cleaned_text,
                "word_count": word_count,
            }

        logger.info("BS4 fallback extracted insufficient text (%d words).", word_count)
        return None

    except Exception as e:
        logger.warning("BS4 fallback extraction failed for %s: %s", url, type(e).__name__)
        return None


# ---------------------------------------------------------------------------
# Main Unified Article Extractor
# ---------------------------------------------------------------------------

def extract_article(url: str, source_hint: str | None = None) -> dict[str, Any]:
    """
    Main extraction interface. Validates URL, extracts body text using newspaper4k,
    and falls back to BeautifulSoup4 if needed.

    Parameters:
    - url (str): The web URL of the article.
    - source_hint (str | None): Optional source name hint (e.g. 'BBC News').

    Returns:
    - dict[str, Any]: Standardized article dictionary containing:
      {
          "success": bool,
          "title": str | None,
          "author": str | None,
          "published_at": str | None,
          "text": str | None,
          "word_count": int,
          "url": str,
          "source": str | None,
          "error": str | None,
      }
    """
    # 1. Validate URL
    if not validate_url(url):
        return {
            "success": False,
            "title": None,
            "author": None,
            "published_at": None,
            "text": None,
            "word_count": 0,
            "url": str(url) if url else "",
            "source": source_hint,
            "error": "Invalid or unsupported article URL format.",
        }

    clean_url = url.strip()

    # 2. Try Primary Extractor (newspaper4k)
    extracted = _extract_with_newspaper(clean_url)

    # 3. Try Fallback Extractor (requests + BS4) if primary yielded no sufficient text
    if not extracted:
        logger.info("Attempting BeautifulSoup4 fallback extraction for: %s", clean_url)
        extracted = _extract_with_bs4(clean_url)

    # 4. Handle Failure / Success
    if not extracted or not extracted.get("text"):
        return {
            "success": False,
            "title": extracted.get("title") if extracted else None,
            "author": extracted.get("author") if extracted else None,
            "published_at": extracted.get("published_at") if extracted else None,
            "text": None,
            "word_count": 0,
            "url": clean_url,
            "source": source_hint,
            "error": "Unable to extract readable article content from this URL.",
        }

    return {
        "success": True,
        "title": extracted.get("title"),
        "author": extracted.get("author"),
        "published_at": extracted.get("published_at"),
        "text": extracted.get("text"),
        "word_count": extracted.get("word_count", 0),
        "url": clean_url,
        "source": source_hint,
        "error": None,
    }
