"""
database.py
-----------
Module responsible for local SQLite database management, schema creation,
article persistence, JSON serialization for multi-item fields (key points, keywords),
and retrieving previously stored AI summaries, key points, categories, and tags.

Database Path: data/news.db

Features:
- Thread-safe local SQLite database connection manager.
- Automatic table creation and indexing on application startup.
- Unique URL constraint preventing duplicate article entries.
- Non-destructive updates: preserves previously extracted content and AI results.
- Robust JSON serialization/deserialization for list fields (key points, keywords).
- Parameterized SQL queries preventing SQL injection vulnerabilities.
- Comprehensive exception handling and detailed logging.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from typing import Any, Generator

# Setup module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Database File & Directory Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "news.db")


def get_db_path() -> str:
    """
    Returns the absolute path to the SQLite database file.
    """
    return DB_PATH


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a configured SQLite connection with row factory
    set to sqlite3.Row, ensuring proper commit, rollback, and connection cleanup.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Database operation failed, rolled back transaction: %s", e)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Database Initialization & Schema Definition
# ---------------------------------------------------------------------------

def init_database() -> bool:
    """
    Initializes the SQLite database, creating the data/ directory, the articles
    table, and required performance indexes if they do not already exist.

    Returns:
    - bool: True if initialization was successful, False otherwise.
    """
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create main articles table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    description TEXT,
                    source TEXT,
                    url TEXT NOT NULL UNIQUE,
                    image_url TEXT,
                    author TEXT,
                    published_at TEXT,
                    category TEXT,
                    article_text TEXT,
                    word_count INTEGER,
                    summary TEXT,
                    summary_length TEXT,
                    key_points TEXT,
                    keywords TEXT,
                    ai_category TEXT,
                    category_confidence REAL,
                    category_method TEXT,
                    sentiment TEXT,
                    sentiment_confidence REAL,
                    sentiment_explanation TEXT,
                    sentiment_method TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );
                """
            )

            # Safe column migration for existing databases
            cursor.execute("PRAGMA table_info(articles);")
            existing_cols = {row["name"] for row in cursor.fetchall()}
            if "sentiment" not in existing_cols:
                cursor.execute("ALTER TABLE articles ADD COLUMN sentiment TEXT;")
            if "sentiment_confidence" not in existing_cols:
                cursor.execute("ALTER TABLE articles ADD COLUMN sentiment_confidence REAL;")
            if "sentiment_explanation" not in existing_cols:
                cursor.execute("ALTER TABLE articles ADD COLUMN sentiment_explanation TEXT;")
            if "sentiment_method" not in existing_cols:
                cursor.execute("ALTER TABLE articles ADD COLUMN sentiment_method TEXT;")

            # Create indexing for frequent lookup fields
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_ai_category ON articles(ai_category);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_sentiment ON articles(sentiment);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at);")

        logger.info("Database initialized successfully at: %s", DB_PATH)
        return True
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)
        return False


# ---------------------------------------------------------------------------
# Deserialization & Helper Functions
# ---------------------------------------------------------------------------

def _current_timestamp_iso() -> str:
    """
    Returns current UTC timestamp in ISO 8601 string format.
    """
    return datetime.now(timezone.utc).isoformat()


def _row_to_article_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """
    Converts a sqlite3.Row object into a Python dictionary, safely decoding
    JSON-encoded list fields ('key_points', 'keywords').
    """
    if row is None:
        return None

    data = dict(row)

    # Safely deserialize key_points JSON string
    if data.get("key_points"):
        try:
            parsed_kp = json.loads(data["key_points"])
            data["key_points"] = parsed_kp if isinstance(parsed_kp, list) else []
        except (json.JSONDecodeError, TypeError):
            data["key_points"] = []
    else:
        data["key_points"] = []

    # Safely deserialize keywords JSON string
    if data.get("keywords"):
        try:
            parsed_kw = json.loads(data["keywords"])
            data["keywords"] = parsed_kw if isinstance(parsed_kw, list) else []
        except (json.JSONDecodeError, TypeError):
            data["keywords"] = []
    else:
        data["keywords"] = []

    return data


# ---------------------------------------------------------------------------
# Article CRUD & Insertion Operations
# ---------------------------------------------------------------------------

def save_article(article_data: dict[str, Any]) -> int | None:
    """
    Inserts a newly fetched article or updates metadata if URL exists,
    preserving any previously extracted text, summaries, and AI results.

    Parameters:
    - article_data (dict[str, Any]): Article dictionary containing title, url, source, etc.

    Returns:
    - int | None: Article database ID if successful, None otherwise.
    """
    url = (article_data.get("url") or "").strip()
    title = (article_data.get("title") or "").strip()

    if not url or not title:
        logger.warning("Cannot save article with empty title or URL.")
        return None

    now_iso = _current_timestamp_iso()

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO articles (
                    title, description, source, url, image_url,
                    author, published_at, category, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    description = COALESCE(excluded.description, articles.description),
                    source = COALESCE(excluded.source, articles.source),
                    image_url = COALESCE(excluded.image_url, articles.image_url),
                    author = COALESCE(excluded.author, articles.author),
                    published_at = COALESCE(excluded.published_at, articles.published_at),
                    category = COALESCE(excluded.category, articles.category),
                    updated_at = excluded.updated_at;
                """,
                (
                    title,
                    article_data.get("description"),
                    article_data.get("source"),
                    url,
                    article_data.get("image_url"),
                    article_data.get("author"),
                    article_data.get("published_at"),
                    article_data.get("category"),
                    now_iso,
                    now_iso,
                ),
            )

            # Retrieve article ID
            cursor.execute("SELECT id FROM articles WHERE url = ?", (url,))
            row = cursor.fetchone()
            return row["id"] if row else None
    except Exception as e:
        logger.error("Failed to save article with URL [%s]: %s", url, e)
        return None


def save_articles_batch(articles: list[dict[str, Any]]) -> int:
    """
    Inserts or updates a batch of fetched articles within a single transaction.

    Parameters:
    - articles (list[dict[str, Any]]): List of article dictionaries.

    Returns:
    - int: Count of articles successfully processed.
    """
    if not articles:
        return 0

    now_iso = _current_timestamp_iso()
    saved_count = 0

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            for article in articles:
                url = (article.get("url") or "").strip()
                title = (article.get("title") or "").strip()

                if not url or not title:
                    continue

                cursor.execute(
                    """
                    INSERT INTO articles (
                        title, description, source, url, image_url,
                        author, published_at, category, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        title = excluded.title,
                        description = COALESCE(excluded.description, articles.description),
                        source = COALESCE(excluded.source, articles.source),
                        image_url = COALESCE(excluded.image_url, articles.image_url),
                        author = COALESCE(excluded.author, articles.author),
                        published_at = COALESCE(excluded.published_at, articles.published_at),
                        category = COALESCE(excluded.category, articles.category),
                        updated_at = excluded.updated_at;
                    """,
                    (
                        title,
                        article.get("description"),
                        article.get("source"),
                        url,
                        article.get("image_url"),
                        article.get("author"),
                        article.get("published_at"),
                        article.get("category"),
                        now_iso,
                        now_iso,
                    ),
                )
                saved_count += 1
        logger.info("Batch saved %d articles to database.", saved_count)
        return saved_count
    except Exception as e:
        logger.error("Failed to batch save articles: %s", e)
        return 0


def get_article_by_url(url: str) -> dict[str, Any] | None:
    """
    Retrieves an article record by URL, deserializing JSON fields into lists.

    Parameters:
    - url (str): Article unique URL.

    Returns:
    - dict[str, Any] | None: Article data dictionary if found, None otherwise.
    """
    if not url:
        return None

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM articles WHERE url = ?", (url.strip(),))
            row = cursor.fetchone()
            return _row_to_article_dict(row)
    except Exception as e:
        logger.error("Error retrieving article by URL [%s]: %s", url, e)
        return None


def get_article_by_id(article_id: int) -> dict[str, Any] | None:
    """
    Retrieves an article record by its integer primary key ID.

    Parameters:
    - article_id (int): Database row ID.

    Returns:
    - dict[str, Any] | None: Article data dictionary if found, None otherwise.
    """
    if not article_id:
        return None

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
            row = cursor.fetchone()
            return _row_to_article_dict(row)
    except Exception as e:
        logger.error("Error retrieving article by ID [%d]: %s", article_id, e)
        return None


def get_recent_articles(limit: int = 20) -> list[dict[str, Any]]:
    """
    Returns the most recently stored or updated articles from the database.

    Parameters:
    - limit (int): Maximum number of articles to return.

    Returns:
    - list[dict[str, Any]]: List of article dictionaries.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM articles
                ORDER BY updated_at DESC, id DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [_row_to_article_dict(row) for row in rows if row]
    except Exception as e:
        logger.error("Error fetching recent articles: %s", e)
        return []


def get_article_count() -> int:
    """
    Returns the total count of articles stored in the database.

    Returns:
    - int: Number of article records.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as total FROM articles;")
            row = cursor.fetchone()
            return int(row["total"]) if row else 0
    except Exception as e:
        logger.error("Error counting articles in database: %s", e)
        return 0


def get_ai_processed_count() -> int:
    """
    Returns count of articles that have at least one AI-generated field
    (summary, key points, category, or keywords).

    Returns:
    - int: Number of AI-processed articles.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as total FROM articles
                WHERE summary IS NOT NULL
                   OR key_points IS NOT NULL
                   OR ai_category IS NOT NULL
                   OR keywords IS NOT NULL
                   OR sentiment IS NOT NULL;
                """
            )
            row = cursor.fetchone()
            return int(row["total"]) if row else 0
    except Exception as e:
        logger.error("Error counting AI processed articles in database: %s", e)
        return 0


def get_distinct_sources() -> list[str]:
    """
    Retrieves a sorted list of unique news publisher/source names stored in the database.

    Returns:
    - list[str]: Distinct non-empty source names.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT source FROM articles
                WHERE source IS NOT NULL AND TRIM(source) != ''
                ORDER BY source ASC;
                """
            )
            rows = cursor.fetchall()
            return [row["source"].strip() for row in rows if row["source"] and row["source"].strip()]
    except Exception as e:
        logger.error("Error fetching distinct sources from database: %s", e)
        return []


def search_articles(
    query: str = "",
    category: str | None = None,
    source: str | None = None,
    sentiment: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
    sort_by: str = "newest",
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Searches and filters stored news articles using safe, parameterized SQL queries.

    Parameters:
    - query (str): Search keywords across title, description, article_text, keywords, and metadata.
    - category (str | None): Category filter ('All', 'Technology', 'Business', etc.).
    - source (str | None): Source filter ('All Sources', 'BBC Tech', etc.).
    - sentiment (str | None): Sentiment filter ('All', 'Positive', 'Neutral', 'Negative').
    - date_filter (str): Quick date filter ('All Time', 'Today', 'Last 7 Days', 'Last 30 Days', 'Custom Range').
    - start_date (str | None): Custom range start date (YYYY-MM-DD).
    - end_date (str | None): Custom range end date (YYYY-MM-DD).
    - sort_by (str): Sorting option ('newest', 'oldest', 'relevance').
    - limit (int): Maximum number of results to return.

    Returns:
    - list[dict[str, Any]]: List of matching article dictionaries.
    """
    clauses: list[str] = []
    params: list[Any] = []

    # 1. Multi-word Search Query Filter (Parameterized LIKE across fields)
    cleaned_query = (query or "").strip()
    query_terms = [t.strip() for t in cleaned_query.split() if t.strip()]

    if query_terms:
        for term in query_terms:
            term_like = f"%{term}%"
            clauses.append(
                """
                (
                    title LIKE ? OR
                    description LIKE ? OR
                    article_text LIKE ? OR
                    keywords LIKE ? OR
                    category LIKE ? OR
                    ai_category LIKE ? OR
                    source LIKE ?
                )
                """
            )
            params.extend([term_like] * 7)

    # 2. Category Filter (Checks ai_category first, falls back to category)
    if category and category != "All":
        clauses.append("(LOWER(COALESCE(ai_category, category)) = LOWER(?) OR LOWER(category) = LOWER(?))")
        params.extend([category, category])

    # 3. Source Filter
    if source and source != "All Sources":
        clauses.append("LOWER(source) = LOWER(?)")
        params.append(source)

    # 4. Sentiment Filter (Feature 13)
    if sentiment and sentiment != "All":
        clauses.append("LOWER(sentiment) = LOWER(?)")
        params.append(sentiment)

    # 5. Date Filtering
    now_utc = datetime.now(timezone.utc)
    if date_filter == "Today":
        today_cutoff = now_utc.strftime("%Y-%m-%d")
        clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
        params.extend([today_cutoff, today_cutoff])
    elif date_filter == "Last 7 Days":
        cutoff_7d = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
        clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
        params.extend([cutoff_7d, cutoff_7d])
    elif date_filter == "Last 30 Days":
        cutoff_30d = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")
        clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
        params.extend([cutoff_30d, cutoff_30d])
    elif date_filter == "Custom Range" or (start_date or end_date):
        if start_date:
            clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
            params.extend([str(start_date)[:10], str(start_date)[:10]])
        if end_date:
            end_date_str = str(end_date)[:10] + "T23:59:59"
            clauses.append("(published_at <= ? OR (published_at IS NULL AND created_at <= ?))")
            params.extend([end_date_str, end_date_str])

    # Construct WHERE clause
    where_sql = " AND ".join(clauses) if clauses else "1=1"

    # 5. Base Order By SQL
    norm_sort = (sort_by or "newest").lower()
    if norm_sort == "oldest":
        order_sql = "ORDER BY COALESCE(published_at, created_at) ASC, id ASC"
    else:
        # Default or relevance base ordering
        order_sql = "ORDER BY COALESCE(published_at, created_at) DESC, id DESC"

    # 6. SQL Limit
    safe_limit = max(1, min(limit, 200))
    fetch_limit = safe_limit * 3 if norm_sort == "relevance" and cleaned_query else safe_limit

    sql = f"""
        SELECT * FROM articles
        WHERE {where_sql}
        {order_sql}
        LIMIT ?;
    """
    params.append(fetch_limit)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            articles = [_row_to_article_dict(row) for row in rows if row]

        # 7. Local Relevance Ranking (Term weighting: title > keywords > description > text)
        if norm_sort == "relevance" and cleaned_query and articles:
            q_lower = cleaned_query.lower()
            q_words = [w.lower() for w in query_terms]

            def compute_relevance(art: dict[str, Any]) -> float:
                score = 0.0
                art_title = (art.get("title") or "").lower()
                art_desc = (art.get("description") or "").lower()
                art_text = (art.get("article_text") or "").lower()
                art_kws = " ".join([str(k).lower() for k in art.get("keywords", [])])

                # Exact phrase match in title
                if q_lower in art_title:
                    score += 50.0

                # Term matches
                for word in q_words:
                    if word in art_title:
                        score += 15.0
                    if word in art_kws:
                        score += 10.0
                    if word in art_desc:
                        score += 5.0
                    if word in art_text:
                        score += 1.0

                return score

            articles.sort(key=compute_relevance, reverse=True)
            return articles[:safe_limit]

        return articles[:safe_limit]

    except Exception as e:
        logger.error("Error executing article search: %s", e)
        return []


# ---------------------------------------------------------------------------
# Specific Field Update Operations (Extraction & AI Results)
# ---------------------------------------------------------------------------

def update_article_content(
    url: str,
    article_text: str,
    word_count: int,
    title: str | None = None,
    author: str | None = None,
    published_at: str | None = None,
) -> bool:
    """
    Updates the extracted body text, word count, and optional metadata for an article.

    Parameters:
    - url (str): Article unique URL.
    - article_text (str): Cleaned article body text.
    - word_count (int): Extracted text word count.
    - title (str | None): Updated article title if discovered during extraction.
    - author (str | None): Author string if discovered.
    - published_at (str | None): Publication date if discovered.

    Returns:
    - bool: True if updated successfully, False otherwise.
    """
    if not url or not article_text:
        return False

    now_iso = _current_timestamp_iso()

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET
                    article_text = ?,
                    word_count = ?,
                    title = COALESCE(?, title),
                    author = COALESCE(?, author),
                    published_at = COALESCE(?, published_at),
                    updated_at = ?
                WHERE url = ?;
                """,
                (
                    article_text.strip(),
                    word_count,
                    title.strip() if title else None,
                    author.strip() if author else None,
                    published_at.strip() if published_at else None,
                    now_iso,
                    url.strip(),
                ),
            )
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Failed to update article content for URL [%s]: %s", url, e)
        return False


def save_summary(url: str, summary: str, summary_length: str) -> bool:
    """
    Persists an AI-generated summary and its length category for an article.

    Parameters:
    - url (str): Article unique URL.
    - summary (str): Generated summary text.
    - summary_length (str): Summary length preset ('short', 'medium', 'detailed').

    Returns:
    - bool: True if saved successfully, False otherwise.
    """
    if not url or not summary:
        return False

    now_iso = _current_timestamp_iso()

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET
                    summary = ?,
                    summary_length = ?,
                    updated_at = ?
                WHERE url = ?;
                """,
                (summary.strip(), summary_length.strip(), now_iso, url.strip()),
            )
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Failed to save summary for URL [%s]: %s", url, e)
        return False


def save_key_points(url: str, key_points: list[str]) -> bool:
    """
    Persists AI-extracted key points as a JSON string for an article.

    Parameters:
    - url (str): Article unique URL.
    - key_points (list[str]): List of extracted key points.

    Returns:
    - bool: True if saved successfully, False otherwise.
    """
    if not url or not isinstance(key_points, list):
        return False

    now_iso = _current_timestamp_iso()
    json_points = json.dumps(key_points)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET
                    key_points = ?,
                    updated_at = ?
                WHERE url = ?;
                """,
                (json_points, now_iso, url.strip()),
            )
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Failed to save key points for URL [%s]: %s", url, e)
        return False


def save_category(
    url: str,
    category: str,
    confidence: float | None = None,
    method: str | None = None,
) -> bool:
    """
    Persists AI-classified category, confidence score, and classification method.

    Parameters:
    - url (str): Article unique URL.
    - category (str): Valid category name.
    - confidence (float | None): AI model confidence score (0.0 to 1.0).
    - method (str | None): Classification method ('gemini' or 'keyword_fallback').

    Returns:
    - bool: True if saved successfully, False otherwise.
    """
    if not url or not category:
        return False

    now_iso = _current_timestamp_iso()

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET
                    ai_category = ?,
                    category_confidence = ?,
                    category_method = ?,
                    updated_at = ?
                WHERE url = ?;
                """,
                (
                    category.strip(),
                    confidence if confidence is not None else None,
                    method.strip() if method else None,
                    now_iso,
                    url.strip(),
                ),
            )
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Failed to save category for URL [%s]: %s", url, e)
        return False


def save_keywords(url: str, keywords: list[str], method: str | None = None) -> bool:
    """
    Persists AI-extracted keywords as a JSON string for an article.

    Parameters:
    - url (str): Article unique URL.
    - keywords (list[str]): List of keyword strings.
    - method (str | None): Extraction method ('gemini' or 'local_fallback').

    Returns:
    - bool: True if saved successfully, False otherwise.
    """
    if not url or not isinstance(keywords, list):
        return False

    now_iso = _current_timestamp_iso()
    json_keywords = json.dumps(keywords)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET
                    keywords = ?,
                    updated_at = ?
                WHERE url = ?;
                """,
                (json_keywords, now_iso, url.strip()),
            )
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Failed to save keywords for URL [%s]: %s", url, e)
        return False


def save_sentiment(
    url: str,
    sentiment: str,
    confidence: float | None = None,
    explanation: str | None = None,
    method: str | None = None,
) -> bool:
    """
    Persists AI-analyzed sentiment, confidence score, explanation, and method.

    Parameters:
    - url (str): Article unique URL.
    - sentiment (str): Sentiment classification ('Positive', 'Neutral', 'Negative').
    - confidence (float | None): Confidence estimate (0.0 to 1.0).
    - explanation (str | None): 1-2 sentence explanation of the sentiment.
    - method (str | None): Analysis method ('gemini' or 'local_fallback').

    Returns:
    - bool: True if saved successfully, False otherwise.
    """
    if not url or not sentiment:
        return False

    now_iso = _current_timestamp_iso()

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE articles
                SET
                    sentiment = ?,
                    sentiment_confidence = ?,
                    sentiment_explanation = ?,
                    sentiment_method = ?,
                    updated_at = ?
                WHERE url = ?;
                """,
                (
                    sentiment.strip(),
                    confidence if confidence is not None else None,
                    explanation.strip() if explanation else None,
                    method.strip() if method else None,
                    now_iso,
                    url.strip(),
                ),
            )
            return cursor.rowcount > 0
    except Exception as e:
        logger.error("Failed to save sentiment for URL [%s]: %s", url, e)
        return False


def get_sentiment_by_url(url: str) -> dict[str, Any] | None:
    """
    Retrieves stored sentiment details for an article by its URL.

    Parameters:
    - url (str): Article unique URL.

    Returns:
    - dict[str, Any] | None: Sentiment fields dictionary if found, None otherwise.
    """
    if not url:
        return None

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sentiment, sentiment_confidence, sentiment_explanation, sentiment_method
                FROM articles WHERE url = ?;
                """,
                (url.strip(),),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error("Error retrieving sentiment for URL [%s]: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# Feature 12: News Analytics Dashboard Queries
# ---------------------------------------------------------------------------

from collections import Counter


def _build_filter_clause(
    category: str | None = None,
    sentiment: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[str, list[Any]]:
    """
    Constructs parameterized WHERE clause and parameter list for dashboard filtering.

    Parameters:
    - category (str | None): Category filter ('All', 'Technology', etc.).
    - sentiment (str | None): Sentiment filter ('All', 'Positive', etc.).
    - date_filter (str): Quick date filter ('All Time', 'Today', 'Last 7 Days', 'Last 30 Days', 'Custom Range').
    - start_date (str | None): Custom range starting date (YYYY-MM-DD).
    - end_date (str | None): Custom range ending date (YYYY-MM-DD).

    Returns:
    - tuple[str, list[Any]]: (where_sql_string, parameter_values_list)
    """
    clauses: list[str] = []
    params: list[Any] = []

    # Category Filter (Checks ai_category first, falls back to category)
    if category and category != "All":
        clauses.append("(LOWER(COALESCE(ai_category, category)) = LOWER(?) OR LOWER(category) = LOWER(?))")
        params.extend([category, category])

    # Sentiment Filter
    if sentiment and sentiment != "All":
        clauses.append("LOWER(sentiment) = LOWER(?)")
        params.append(sentiment)

    # Date Filter
    now_utc = datetime.now(timezone.utc)
    if date_filter == "Today":
        today_cutoff = now_utc.strftime("%Y-%m-%d")
        clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
        params.extend([today_cutoff, today_cutoff])
    elif date_filter == "Last 7 Days":
        cutoff_7d = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")
        clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
        params.extend([cutoff_7d, cutoff_7d])
    elif date_filter == "Last 30 Days":
        cutoff_30d = (now_utc - timedelta(days=30)).strftime("%Y-%m-%d")
        clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
        params.extend([cutoff_30d, cutoff_30d])
    elif date_filter == "Custom Range" or (start_date or end_date):
        if start_date:
            clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
            params.extend([str(start_date)[:10], str(start_date)[:10]])
        if end_date:
            end_date_str = str(end_date)[:10] + "T23:59:59"
            clauses.append("(published_at <= ? OR (published_at IS NULL AND created_at <= ?))")
            params.extend([end_date_str, end_date_str])

    where_sql = " AND ".join(clauses) if clauses else "1=1"
    return where_sql, params


def get_dashboard_summary_metrics(
    category: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Computes aggregated top-level dashboard metrics using SQLite SQL aggregation functions.

    Returns:
    - dict[str, Any]: Aggregated metric counts and averages.
    """
    where_sql, params = _build_filter_clause(category=category, sentiment=None, date_filter=date_filter, start_date=start_date, end_date=end_date)
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    week_str = (now_utc - timedelta(days=7)).strftime("%Y-%m-%d")

    sql = f"""
        SELECT
            COUNT(*) as total_articles,
            COUNT(DISTINCT COALESCE(ai_category, category)) as total_categories,
            COUNT(DISTINCT source) as total_sources,
            SUM(CASE WHEN summary IS NOT NULL OR key_points IS NOT NULL OR ai_category IS NOT NULL OR keywords IS NOT NULL THEN 1 ELSE 0 END) as ai_processed_count,
            SUM(CASE WHEN (published_at >= '{today_str}' OR (published_at IS NULL AND created_at >= '{today_str}')) THEN 1 ELSE 0 END) as today_articles,
            SUM(CASE WHEN (published_at >= '{week_str}' OR (published_at IS NULL AND created_at >= '{week_str}')) THEN 1 ELSE 0 END) as week_articles,
            AVG(CASE WHEN word_count > 0 THEN word_count ELSE NULL END) as avg_word_count
        FROM articles
        WHERE {where_sql};
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            if not row:
                return {
                    "total_articles": 0,
                    "total_categories": 0,
                    "total_sources": 0,
                    "ai_processed_count": 0,
                    "today_articles": 0,
                    "week_articles": 0,
                    "avg_word_count": 0,
                }
            return {
                "total_articles": int(row["total_articles"] or 0),
                "total_categories": int(row["total_categories"] or 0),
                "total_sources": int(row["total_sources"] or 0),
                "ai_processed_count": int(row["ai_processed_count"] or 0),
                "today_articles": int(row["today_articles"] or 0),
                "week_articles": int(row["week_articles"] or 0),
                "avg_word_count": int(round(row["avg_word_count"])) if row["avg_word_count"] else 0,
            }
    except Exception as e:
        logger.error("Error computing dashboard summary metrics: %s", e)
        return {
            "total_articles": 0,
            "total_categories": 0,
            "total_sources": 0,
            "ai_processed_count": 0,
            "today_articles": 0,
            "week_articles": 0,
            "avg_word_count": 0,
        }


def get_articles_over_time(
    category: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Returns chronological daily article publication volumes for trend charts.

    Returns:
    - list[dict[str, Any]]: List of {'date': 'YYYY-MM-DD', 'count': N}
    """
    where_sql, params = _build_filter_clause(category=category, sentiment=None, date_filter=date_filter, start_date=start_date, end_date=end_date)
    sql = f"""
        SELECT
            substr(COALESCE(published_at, created_at), 1, 10) as article_date,
            COUNT(*) as article_count
        FROM articles
        WHERE {where_sql}
          AND (published_at IS NOT NULL OR created_at IS NOT NULL)
        GROUP BY article_date
        ORDER BY article_date ASC;
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [{"date": row["article_date"], "count": int(row["article_count"])} for row in rows if row["article_date"]]
    except Exception as e:
        logger.error("Error fetching articles over time: %s", e)
        return []


def get_category_distribution(
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Returns article counts grouped by category.

    Returns:
    - list[dict[str, Any]]: List of {'category': name, 'count': N}
    """
    where_sql, params = _build_filter_clause(category=None, sentiment=None, date_filter=date_filter, start_date=start_date, end_date=end_date)
    sql = f"""
        SELECT
            COALESCE(ai_category, category, 'General') as cat_name,
            COUNT(*) as article_count
        FROM articles
        WHERE {where_sql}
        GROUP BY cat_name
        ORDER BY article_count DESC;
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [{"category": row["cat_name"], "count": int(row["article_count"])} for row in rows if row["cat_name"]]
    except Exception as e:
        logger.error("Error fetching category distribution: %s", e)
        return []


def get_source_distribution(
    category: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Returns top news sources by article volume.

    Returns:
    - list[dict[str, Any]]: List of {'source': name, 'count': N}
    """
    where_sql, params = _build_filter_clause(category=category, sentiment=None, date_filter=date_filter, start_date=start_date, end_date=end_date)
    sql = f"""
        SELECT
            COALESCE(source, 'Unknown') as source_name,
            COUNT(*) as article_count
        FROM articles
        WHERE {where_sql}
          AND source IS NOT NULL AND TRIM(source) != ''
        GROUP BY source_name
        ORDER BY article_count DESC
        LIMIT ?;
    """
    params.append(limit)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [{"source": row["source_name"], "count": int(row["article_count"])} for row in rows if row["source_name"]]
    except Exception as e:
        logger.error("Error fetching source distribution: %s", e)
        return []


def get_top_keywords(
    category: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 15,
) -> list[dict[str, Any]]:
    """
    Extracts and aggregates frequencies of stored JSON keywords across articles.

    Returns:
    - list[dict[str, Any]]: List of {'keyword': word, 'count': N}
    """
    where_sql, params = _build_filter_clause(category=category, sentiment=None, date_filter=date_filter, start_date=start_date, end_date=end_date)
    sql = f"""
        SELECT keywords FROM articles
        WHERE {where_sql}
          AND keywords IS NOT NULL AND TRIM(keywords) != '';
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()

        counter: Counter[str] = Counter()
        for row in rows:
            raw = row["keywords"]
            if not raw:
                continue
            try:
                kws = json.loads(raw)
                if isinstance(kws, list):
                    for kw in kws:
                        if isinstance(kw, str) and kw.strip():
                            clean_kw = kw.strip().title()
                            counter[clean_kw] += 1
            except Exception:
                continue

        return [{"keyword": kw, "count": count} for kw, count in counter.most_common(top_n)]
    except Exception as e:
        logger.error("Error aggregating top keywords: %s", e)
        return []


def get_sentiment_distribution(
    category: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Returns article counts grouped by sentiment (Positive, Neutral, Negative).

    Returns:
    - list[dict[str, Any]]: List of {'sentiment': name, 'count': N}
    """
    where_sql, params = _build_filter_clause(category, None, date_filter, start_date, end_date)
    sql = f"""
        SELECT
            COALESCE(sentiment, 'Unclassified') as sentiment_name,
            COUNT(*) as article_count
        FROM articles
        WHERE {where_sql}
          AND sentiment IS NOT NULL AND TRIM(sentiment) != ''
        GROUP BY sentiment_name
        ORDER BY article_count DESC;
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [{"sentiment": row["sentiment_name"], "count": int(row["article_count"])} for row in rows if row["sentiment_name"]]
    except Exception as e:
        logger.error("Error fetching sentiment distribution: %s", e)
        return []


def get_ai_processing_breakdown(
    category: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """
    Calculates completion counts and percentages for each AI pipeline feature.

    Returns:
    - dict[str, Any]: AI completion breakdown.
    """
    where_sql, params = _build_filter_clause(category, None, date_filter, start_date, end_date)
    sql = f"""
        SELECT
            COUNT(*) as total_articles,
            SUM(CASE WHEN summary IS NOT NULL AND TRIM(summary) != '' THEN 1 ELSE 0 END) as summary_count,
            SUM(CASE WHEN key_points IS NOT NULL AND TRIM(key_points) != '' THEN 1 ELSE 0 END) as key_points_count,
            SUM(CASE WHEN ai_category IS NOT NULL AND TRIM(ai_category) != '' THEN 1 ELSE 0 END) as category_count,
            SUM(CASE WHEN keywords IS NOT NULL AND TRIM(keywords) != '' THEN 1 ELSE 0 END) as keywords_count,
            SUM(CASE WHEN sentiment IS NOT NULL AND TRIM(sentiment) != '' THEN 1 ELSE 0 END) as sentiment_count
        FROM articles
        WHERE {where_sql};
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            row = cursor.fetchone()
            if not row:
                return {
                    "total": 0,
                    "summary_count": 0, "summary_pct": 0.0,
                    "key_points_count": 0, "key_points_pct": 0.0,
                    "category_count": 0, "category_pct": 0.0,
                    "keywords_count": 0, "keywords_pct": 0.0,
                    "sentiment_count": 0, "sentiment_pct": 0.0,
                }
            total = int(row["total_articles"] or 0)
            s_cnt = int(row["summary_count"] or 0)
            kp_cnt = int(row["key_points_count"] or 0)
            cat_cnt = int(row["category_count"] or 0)
            kw_cnt = int(row["keywords_count"] or 0)
            sent_cnt = int(row["sentiment_count"] or 0)

            return {
                "total": total,
                "summary_count": s_cnt,
                "summary_pct": round((s_cnt / total * 100), 1) if total > 0 else 0.0,
                "key_points_count": kp_cnt,
                "key_points_pct": round((kp_cnt / total * 100), 1) if total > 0 else 0.0,
                "category_count": cat_cnt,
                "category_pct": round((cat_cnt / total * 100), 1) if total > 0 else 0.0,
                "keywords_count": kw_cnt,
                "keywords_pct": round((kw_cnt / total * 100), 1) if total > 0 else 0.0,
                "sentiment_count": sent_cnt,
                "sentiment_pct": round((sent_cnt / total * 100), 1) if total > 0 else 0.0,
            }
    except Exception as e:
        logger.error("Error computing AI processing breakdown: %s", e)
        return {
            "total": 0,
            "summary_count": 0, "summary_pct": 0.0,
            "key_points_count": 0, "key_points_pct": 0.0,
            "category_count": 0, "category_pct": 0.0,
            "keywords_count": 0, "keywords_pct": 0.0,
            "sentiment_count": 0, "sentiment_pct": 0.0,
        }


def get_summary_type_distribution(
    category: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Returns breakdown of summaries by length preset (Short, Medium, Detailed).

    Returns:
    - list[dict[str, Any]]: List of {'type': length_type, 'count': N}
    """
    where_sql, params = _build_filter_clause(category, None, date_filter, start_date, end_date)
    sql = f"""
        SELECT
            COALESCE(summary_length, 'Standard') as length_type,
            COUNT(*) as summary_count
        FROM articles
        WHERE {where_sql}
          AND summary IS NOT NULL AND TRIM(summary) != ''
        GROUP BY length_type
        ORDER BY summary_count DESC;
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [{"type": row["length_type"].capitalize(), "count": int(row["summary_count"])} for row in rows]
    except Exception as e:
        logger.error("Error fetching summary type distribution: %s", e)
        return []


def get_dashboard_recent_articles(
    category: str | None = None,
    date_filter: str = "All Time",
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Returns recent articles with AI completion indicators for the dashboard table.

    Returns:
    - list[dict[str, Any]]: List of article overview dictionaries.
    """
    where_sql, params = _build_filter_clause(category, None, date_filter, start_date, end_date)
    sql = f"""
        SELECT
            id, title, source, url,
            COALESCE(ai_category, category, 'General') as category,
            published_at, created_at,
            sentiment,
            (summary IS NOT NULL AND TRIM(summary) != '') as has_summary,
            (key_points IS NOT NULL AND TRIM(key_points) != '') as has_key_points,
            (ai_category IS NOT NULL AND TRIM(ai_category) != '') as has_ai_category,
            (keywords IS NOT NULL AND TRIM(keywords) != '') as has_keywords,
            (sentiment IS NOT NULL AND TRIM(sentiment) != '') as has_sentiment
        FROM articles
        WHERE {where_sql}
        ORDER BY COALESCE(published_at, created_at) DESC, id DESC
        LIMIT ?;
    """
    params.append(limit)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [dict(row) for row in rows if row]
    except Exception as e:
        logger.error("Error fetching dashboard recent articles: %s", e)
        return []


def get_articles_for_trends(
    start_date_iso: str | None = None,
    end_date_iso: str | None = None,
    category: str | None = None,
    source: str | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """
    Fetches article metadata, keywords, titles, categories, and sentiments
    for dynamic trending topic analysis using safe parameterized queries.

    Parameters:
    - start_date_iso (str | None): Inclusive start date cutoff in ISO format.
    - end_date_iso (str | None): Exclusive end date cutoff in ISO format.
    - category (str | None): Category filter ('All', 'Technology', etc.).
    - source (str | None): Source publisher filter ('All Sources', etc.).
    - limit (int): Maximum records to retrieve.

    Returns:
    - list[dict[str, Any]]: List of article dictionaries with metadata and keywords.
    """
    clauses: list[str] = []
    params: list[Any] = []

    if start_date_iso:
        clauses.append("(published_at >= ? OR (published_at IS NULL AND created_at >= ?))")
        params.extend([start_date_iso, start_date_iso])

    if end_date_iso:
        clauses.append("(published_at < ? OR (published_at IS NULL AND created_at < ?))")
        params.extend([end_date_iso, end_date_iso])

    if category and category != "All":
        clauses.append("(LOWER(COALESCE(ai_category, category)) = LOWER(?) OR LOWER(category) = LOWER(?))")
        params.extend([category, category])

    if source and source != "All Sources":
        clauses.append("LOWER(source) = LOWER(?)")
        params.append(source)

    where_sql = " AND ".join(clauses) if clauses else "1=1"

    sql = f"""
        SELECT
            id, title, description, url, source,
            COALESCE(ai_category, category, 'General') as category,
            keywords, sentiment,
            COALESCE(published_at, created_at) as article_date,
            published_at, created_at
        FROM articles
        WHERE {where_sql}
        ORDER BY COALESCE(published_at, created_at) DESC
        LIMIT ?;
    """
    params.append(limit)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [dict(row) for row in rows if row]
    except Exception as e:
        logger.error("Error fetching articles for trends: %s", e)
        return []



