"""
trending_topics.py
------------------
Module responsible for detecting the most frequently discussed and currently
relevant trending topics across news articles stored locally in the SQLite database.

Core Features:
- Transparent, explainable frequency + recency weighting algorithm.
- Centralized topic normalization and alias mapping (e.g., 'AI' -> 'Artificial Intelligence').
- Stop-topic filtering to eliminate generic headline noise.
- Multi-occurrence deduplication: each topic counts at most once per article.
- Historical period comparison for trend movement classification:
  * 'Rising' (🔥)
  * 'Stable' (→)
  * 'Declining' (↓)
  * 'Current' (⚡, when historical comparison data is insufficient)
- Dominant sentiment aggregation for each trending topic.
- Offline-first execution: 100% SQLite based, zero external API or Gemini calls.
"""

import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any

import database

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Centralized Topic Normalization & Aliases
# ---------------------------------------------------------------------------

TOPIC_ALIASES: dict[str, str] = {
    # Artificial Intelligence & Computing
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "genai": "Generative AI",
    "generative ai": "Generative AI",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "large language model": "Large Language Models",
    "large language models": "Large Language Models",
    "machine learning": "Machine Learning",
    "ml": "Machine Learning",
    "deep learning": "Machine Learning",
    "openai": "OpenAI",
    "chatgpt": "ChatGPT",
    "gpt": "ChatGPT",
    "gemini": "Google Gemini",
    "google gemini": "Google Gemini",
    "anthropic": "Anthropic",
    "claude": "Anthropic Claude",
    "nvidia": "Nvidia",
    "semiconductor": "Semiconductors",
    "semiconductors": "Semiconductors",
    "chips": "Semiconductors",
    "cybersecurity": "Cybersecurity",
    "cyber security": "Cybersecurity",
    "robotics": "Robotics",
    "robots": "Robotics",

    # Business & Economy
    "stock market": "Stock Markets",
    "stock markets": "Stock Markets",
    "stocks": "Stock Markets",
    "wall street": "Stock Markets",
    "inflation": "Inflation & Economy",
    "economy": "Global Economy",
    "interest rates": "Federal Reserve & Rates",
    "fed": "Federal Reserve & Rates",
    "crypto": "Cryptocurrency",
    "cryptocurrency": "Cryptocurrency",
    "cryptocurrencies": "Cryptocurrency",
    "bitcoin": "Bitcoin",
    "ethereum": "Ethereum",

    # Science & Environment
    "climate change": "Climate Change",
    "global warming": "Climate Change",
    "renewable energy": "Renewable Energy",
    "solar energy": "Renewable Energy",
    "ev": "Electric Vehicles",
    "electric vehicles": "Electric Vehicles",
    "electric cars": "Electric Vehicles",
    "nasa": "NASA Space Missions",
    "space exploration": "Space Exploration",
    "spacex": "SpaceX",
    "astronomy": "Space Exploration",

    # Sports
    "cricket": "Cricket",
    "cricket world cup": "Cricket World Cup",
    "icc": "Cricket (ICC)",
    "ipl": "IPL Cricket",
    "premier league": "Premier League Football",
    "champions league": "UEFA Champions League",
    "football": "Football / Soccer",
    "soccer": "Football / Soccer",
    "formula 1": "Formula 1",
    "f1": "Formula 1",
    "olympics": "Olympic Games",
    "tennis": "Tennis Grand Slam",

    # Health
    "covid": "COVID-19",
    "coronavirus": "COVID-19",
    "vaccine": "Vaccines & Immunization",
    "vaccines": "Vaccines & Immunization",
    "cancer research": "Cancer Research",
    "mental health": "Mental Health",
}

# Stop-topic words / generic terms to eliminate
STOP_TOPICS: set[str] = {
    "news", "article", "report", "reports", "reporting", "reported",
    "today", "yesterday", "tomorrow", "people", "world", "information",
    "update", "updates", "year", "years", "day", "days", "time", "times",
    "week", "weeks", "month", "months", "new", "latest", "breaking",
    "overview", "story", "stories", "event", "events", "thing", "things",
    "way", "ways", "say", "says", "said", "post", "video", "photo",
    "image", "source", "online", "percent", "first", "last", "good",
    "great", "big", "major", "daily", "summary", "press", "media",
    "country", "countries", "state", "states", "government", "official",
    "officials", "spokesperson", "statement", "case", "cases", "part",
}

ENGLISH_STOPWORDS: set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


# ---------------------------------------------------------------------------
# Topic Normalization Helper
# ---------------------------------------------------------------------------

def normalize_topic(raw_topic: str | None) -> str | None:
    """
    Cleans and canonicalizes a raw keyword or topic phrase.
    Applies alias mappings and rejects generic stop-topics.

    Parameters:
    - raw_topic (str | None): Raw topic or keyword string.

    Returns:
    - str | None: Canonical normalized topic or None if filtered out.
    """
    if not raw_topic or not isinstance(raw_topic, str):
        return None

    # Strip quotes, punctuation, brackets, numbers at edges
    cleaned = raw_topic.strip()
    cleaned = re.sub(r"^[\d\W_]+|[\d\W_]+$", "", cleaned).strip()

    if not cleaned or len(cleaned) < 2:
        return None

    lower_topic = cleaned.lower()

    # Direct alias lookup
    if lower_topic in TOPIC_ALIASES:
        return TOPIC_ALIASES[lower_topic]

    # Stop-topic check
    if lower_topic in STOP_TOPICS:
        return None

    # Reject topics made entirely of single stopwords
    words = [w for w in re.split(r"\s+", lower_topic) if w]
    if all(w in ENGLISH_STOPWORDS or w in STOP_TOPICS for w in words):
        return None

    # Standard Title Case for general multi-word or single-word topics
    # Keep uppercase acronyms (e.g., NASA, UK, USA, AI, BBC) intact
    if cleaned.isupper() and len(cleaned) <= 5:
        return cleaned

    # Format into Title Case
    return " ".join(word.capitalize() if word.lower() not in {"and", "of", "the", "in", "for", "on"} else word.lower() for word in cleaned.split())


# ---------------------------------------------------------------------------
# Fallback Extraction Helper (for articles without AI keywords)
# ---------------------------------------------------------------------------

def _extract_fallback_topics_from_article(title: str | None, description: str | None) -> list[str]:
    """
    Extracts high-signal topic candidates from article headline and description
    when no AI keywords have been generated yet.
    """
    text = f"{title or ''} {description or ''}"
    if not text.strip():
        return []

    # Match capitalized multi-word phrases (proper nouns) or key terms
    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    extracted: list[str] = []

    for phrase in phrases:
        norm = normalize_topic(phrase)
        if norm and norm not in extracted:
            extracted.append(norm)

    return extracted[:4]


# ---------------------------------------------------------------------------
# Recency Weighting Calculation
# ---------------------------------------------------------------------------

def _calculate_recency_weight(article_date_str: str | None, now_utc: datetime, window_days: int) -> float:
    """
    Computes a recency decay weight between 0.25 and 1.0 based on days elapsed.

    Formula:
    recency_weight = max(0.25, 1.0 - 0.10 * days_ago)
    """
    if not article_date_str:
        return 0.5  # Neutral default when timestamp missing

    try:
        clean_date_str = article_date_str.replace("Z", "+00:00")
        if len(clean_date_str) == 10:
            clean_date_str += "T00:00:00+00:00"

        art_dt = datetime.fromisoformat(clean_date_str)
        if art_dt.tzinfo is None:
            art_dt = art_dt.replace(tzinfo=timezone.utc)

        days_ago = max(0.0, (now_utc - art_dt).total_seconds() / 86400.0)

        # Decay smoothly based on time elapsed
        decay_rate = 0.75 / max(1, window_days)
        weight = max(0.25, 1.0 - (decay_rate * days_ago))
        return round(weight, 3)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main Trending Topics Algorithm (Feature 14)
# ---------------------------------------------------------------------------

def get_trending_topics(
    days: int = 7,
    limit: int = 10,
    category: str | None = None,
    source: str | None = None,
    min_articles: int = 1,
) -> list[dict[str, Any]]:
    """
    Identifies and ranks the most frequently and recently discussed topics
    from news articles stored in the SQLite database.

    Parameters:
    - days (int): Number of recent days to evaluate (e.g., 1, 3, 7, 30).
    - limit (int): Maximum number of top trending topics to return (5, 10, 15).
    - category (str | None): Optional category filter ('All', 'Technology', etc.).
    - source (str | None): Optional publisher/source filter.
    - min_articles (int): Minimum distinct article threshold for a topic to qualify.

    Returns:
    - list[dict[str, Any]]: List of trending topic dictionaries with structure:
      [
          {
              "topic": str,
              "article_count": int,
              "score": float,
              "trend": str ("Rising" | "Stable" | "Declining" | "Current"),
              "dominant_sentiment": str ("Positive" | "Neutral" | "Negative" | None),
              "recent_count": int,
              "previous_count": int,
              "sample_titles": list[str],
          }
      ]
    """
    now_utc = datetime.now(timezone.utc)
    current_cutoff_iso = (now_utc - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    prev_cutoff_iso = (now_utc - timedelta(days=days * 2)).strftime("%Y-%m-%dT%H:%M:%S")

    # 1. Fetch current period articles from SQLite
    current_articles = database.get_articles_for_trends(
        start_date_iso=current_cutoff_iso,
        end_date_iso=None,
        category=category,
        source=source,
        limit=1000,
    )

    # 2. Fetch previous comparable period articles for trend direction
    previous_articles = database.get_articles_for_trends(
        start_date_iso=prev_cutoff_iso,
        end_date_iso=current_cutoff_iso,
        category=category,
        source=source,
        limit=1000,
    )

    if not current_articles and not previous_articles:
        logger.info("No stored articles found in date window (%d days).", days)
        return []

    # If current period has no articles (e.g., database has older articles),
    # fallback to all available articles so user gets meaningful insights
    if not current_articles:
        current_articles = database.get_articles_for_trends(
            start_date_iso=None,
            end_date_iso=None,
            category=category,
            source=source,
            limit=500,
        )

    # 3. Process current period topics
    topic_scores: dict[str, float] = defaultdict(float)
    topic_articles_count: dict[str, int] = defaultdict(int)
    topic_sentiments: dict[str, Counter[str]] = defaultdict(Counter)
    topic_sample_titles: dict[str, list[str]] = defaultdict(list)

    for art in current_articles:
        art_title = art.get("title") or "Untitled"
        art_desc = art.get("description") or ""
        art_date = art.get("article_date")
        art_sentiment = art.get("sentiment")
        raw_keywords_json = art.get("keywords")

        recency_w = _calculate_recency_weight(art_date, now_utc, days)

        # Extract topics for this article
        article_topics: set[str] = set()

        if raw_keywords_json:
            try:
                kw_list = json.loads(raw_keywords_json)
                if isinstance(kw_list, list):
                    for kw in kw_list:
                        norm = normalize_topic(kw)
                        if norm:
                            article_topics.add(norm)
            except Exception:
                pass

        # If no keywords stored, use fallback topic extractor
        if not article_topics:
            fallback_topics = _extract_fallback_topics_from_article(art_title, art_desc)
            for fb_t in fallback_topics:
                article_topics.add(fb_t)

        # Attribute each unique topic from this article (counted at most once per article)
        for t in article_topics:
            topic_articles_count[t] += 1
            topic_scores[t] += recency_w
            if len(topic_sample_titles[t]) < 3:
                topic_sample_titles[t].append(art_title)
            if art_sentiment and art_sentiment.strip():
                topic_sentiments[t][art_sentiment.strip().capitalize()] += 1

    # 4. Process previous period topics for movement comparison
    previous_topic_counts: dict[str, int] = defaultdict(int)
    has_prev_data = len(previous_articles) >= 2

    for prev_art in previous_articles:
        prev_topics: set[str] = set()
        raw_kw = prev_art.get("keywords")
        if raw_kw:
            try:
                kw_list = json.loads(raw_kw)
                if isinstance(kw_list, list):
                    for kw in kw_list:
                        norm = normalize_topic(kw)
                        if norm:
                            prev_topics.add(norm)
            except Exception:
                pass

        if not prev_topics:
            fb = _extract_fallback_topics_from_article(prev_art.get("title"), prev_art.get("description"))
            for t in fb:
                prev_topics.add(t)

        for t in prev_topics:
            previous_topic_counts[t] += 1

    # 5. Build and rank trending topics list
    ranked_topics: list[dict[str, Any]] = []

    for topic, count in topic_articles_count.items():
        if count < min_articles and len(current_articles) > 5:
            continue

        score = round(topic_scores[topic], 1)
        prev_cnt = previous_topic_counts.get(topic, 0)

        # Determine trend direction
        if not has_prev_data:
            trend_direction = "Current"
        else:
            diff = count - prev_cnt
            # Apply tolerance so 1 minor change isn't overstated on large volume
            tolerance = 1 if max(count, prev_cnt) > 4 else 0
            if diff > tolerance:
                trend_direction = "Rising"
            elif diff < -tolerance:
                trend_direction = "Declining"
            else:
                trend_direction = "Stable"

        # Determine dominant sentiment
        dominant_sentiment: str | None = None
        s_counter = topic_sentiments[topic]
        if s_counter:
            dominant_sentiment = s_counter.most_common(1)[0][0]

        ranked_topics.append(
            {
                "topic": topic,
                "article_count": count,
                "score": score,
                "trend": trend_direction,
                "dominant_sentiment": dominant_sentiment,
                "recent_count": count,
                "previous_count": prev_cnt,
                "sample_titles": topic_sample_titles[topic],
            }
        )

    # 6. Sort by Trending Score (primary) and Article Count (secondary)
    ranked_topics.sort(key=lambda item: (item["score"], item["article_count"]), reverse=True)

    return ranked_topics[:limit]
