"""
duplicate_detector.py
---------------------
Module responsible for detecting duplicate, syndicated, or highly related news articles
across different publishers and URLs using TF-IDF (Term Frequency-Inverse Document Frequency)
vectorization and Cosine Similarity from scikit-learn.

Key Features:
- 100% local, lightweight, and fast NLP similarity computation (Zero external API / Gemini calls).
- Cross-publisher and cross-URL comparison: detects identical events written with varying wording.
- Immediate shortcut evaluation for identical URLs (100% duplicate).
- Title-weighted representation & dual title/content scoring: emphasizes headline terminology over body filler.
- Configurable heuristic thresholds for 'Duplicate / Same Story', 'Related', and 'Different'.
- Safe fallback handling if scikit-learn is uninstalled or candidate list is empty.
- Non-destructive: performs comparisons dynamically without modifying or deleting database records.
"""

import re
import logging
from typing import Any

# Setup module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configurable Similarity Thresholds (Heuristic Constants)
# ---------------------------------------------------------------------------

# Similarity >= 0.80: Articles describe essentially the same news event / story
DUPLICATE_THRESHOLD: float = 0.80

# 0.55 <= Similarity < 0.80: Articles discuss closely related topics or follow-up developments
RELATED_THRESHOLD: float = 0.55

# Maximum number of recent database articles to fetch as candidates
DEFAULT_CANDIDATE_LIMIT: int = 100

# Maximum words of body text to include in similarity calculation (avoids dilution)
MAX_BODY_WORDS_FOR_SIMILARITY: int = 300

# Title repetition multiplier for TF-IDF weighting
DEFAULT_TITLE_WEIGHT: int = 2


# ---------------------------------------------------------------------------
# Dependency Check
# ---------------------------------------------------------------------------

def is_sklearn_available() -> bool:
    """
    Checks if scikit-learn is installed and importable.
    """
    try:
        import sklearn  # noqa: F401
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Text Preprocessing & Preparation
# ---------------------------------------------------------------------------

def preprocess_text(raw_text: str | None) -> str:
    """
    Normalizes and cleans raw text for TF-IDF feature extraction.
    Converts to lowercase, strips punctuation, collapses whitespaces, and removes HTML remnants.

    Parameters:
    - raw_text (str | None): Raw input string.

    Returns:
    - str: Cleaned string ready for vectorization.
    """
    if not raw_text or not isinstance(raw_text, str):
        return ""

    # Convert to lowercase
    cleaned = raw_text.lower()

    # Remove HTML tags if any remain
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)

    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Normalize punctuation to spaces (keeping alphanumeric characters)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)

    # Normalize multiple whitespaces into a single space
    cleaned = " ".join(cleaned.split())

    return cleaned.strip()


def prepare_article_text(
    article: dict[str, Any],
    title_weight: int = DEFAULT_TITLE_WEIGHT,
    max_body_words: int = MAX_BODY_WORDS_FOR_SIMILARITY,
) -> str:
    """
    Constructs a composite, weighted textual representation of an article for similarity comparison.
    Headlines receive higher weighting because news titles contain core subject entities and actions.

    Parameters:
    - article (dict[str, Any]): Article dictionary containing title, description, and article_text.
    - title_weight (int): Multiplier for title text presence.
    - max_body_words (int): Maximum words of body text to append.

    Returns:
    - str: Preprocessed composite string.
    """
    if not article or not isinstance(article, dict):
        return ""

    title = preprocess_text(article.get("title", ""))
    description = preprocess_text(article.get("description", ""))
    
    # Extract body text or fallback to empty
    raw_body = article.get("article_text") or ""
    clean_body = preprocess_text(raw_body)
    body_words = clean_body.split()[:max_body_words]
    limited_body = " ".join(body_words)

    # Weighted composition: repeat title `title_weight` times, then description, then body snippet
    weighted_parts: list[str] = []
    
    if title:
        weighted_parts.extend([title] * max(1, title_weight))
    if description:
        weighted_parts.append(description)
    if limited_body:
        weighted_parts.append(limited_body)

    return " ".join(weighted_parts).strip()


# ---------------------------------------------------------------------------
# Similarity Classification & Scoring
# ---------------------------------------------------------------------------

def classify_relationship(similarity_score: float) -> str:
    """
    Classifies the relationship between two articles based on textual cosine similarity score.

    Note: These classifications represent textual similarity heuristics, not absolute factual identity.

    Parameters:
    - similarity_score (float): Cosine similarity value between 0.0 and 1.0.

    Returns:
    - str: Relationship label ('Duplicate / Same Story', 'Related', or 'Different').
    """
    if similarity_score >= DUPLICATE_THRESHOLD:
        return "Duplicate / Same Story"
    elif similarity_score >= RELATED_THRESHOLD:
        return "Related"
    else:
        return "Different"


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculates the cosine similarity score between two individual text strings using TF-IDF.

    Parameters:
    - text1 (str): First text string.
    - text2 (str): Second text string.

    Returns:
    - float: Cosine similarity score between 0.0 and 1.0.
    """
    clean_t1 = preprocess_text(text1)
    clean_t2 = preprocess_text(text2)

    if not clean_t1 or not clean_t2:
        return 0.0

    if clean_t1 == clean_t2:
        return 1.0

    if not is_sklearn_available():
        logger.warning("scikit-learn is not installed. Unable to compute TF-IDF cosine similarity.")
        return 0.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 1),
            use_idf=False,
            norm="l2",
        )
        tfidf_matrix = vectorizer.fit_transform([clean_t1, clean_t2])
        score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        return max(0.0, min(1.0, round(score, 4)))
    except Exception as e:
        logger.error("Error calculating pairwise text similarity: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Core Duplicate & Related News Search
# ---------------------------------------------------------------------------

def find_similar_articles(
    target_article: dict[str, Any] | None,
    candidate_articles: list[dict[str, Any]] | None,
    threshold: float = RELATED_THRESHOLD,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Finds and ranks articles similar to the target article from a candidate set.
    Excludes the target article itself, handles exact URL matches immediately,
    and uses TF-IDF + Cosine Similarity over titles, descriptions, and body texts.

    Parameters:
    - target_article (dict[str, Any] | None): The reference article to compare against.
    - candidate_articles (list[dict[str, Any]] | None): Candidate articles pool (e.g. from SQLite).
    - threshold (float): Minimum similarity score threshold (default: RELATED_THRESHOLD = 0.55).
    - max_results (int): Maximum number of similar articles to return.

    Returns:
    - list[dict[str, Any]]: List of match dictionaries sorted by similarity descending:
      [
          {
              "article": dict,
              "similarity": float (0.0 to 1.0),
              "similarity_percent": float (0.0 to 100.0),
              "relationship": str ("Duplicate / Same Story" | "Related"),
          },
          ...
      ]
    """
    if not target_article or not isinstance(target_article, dict):
        return []

    if not candidate_articles or not isinstance(candidate_articles, list):
        return []

    target_url = (target_article.get("url") or "").strip()
    target_title = preprocess_text(target_article.get("title", ""))
    target_text = prepare_article_text(target_article)

    if not target_text and not target_url:
        return []

    # 1. Filter out the target article itself from candidate list
    filtered_candidates: list[dict[str, Any]] = []
    for cand in candidate_articles:
        if not isinstance(cand, dict):
            continue
        cand_url = (cand.get("url") or "").strip()
        # Skip identical database ID or identical non-empty URL if exact same instance
        if target_article.get("id") and cand.get("id") and target_article.get("id") == cand.get("id"):
            continue
        if target_url and cand_url and target_url == cand_url and not cand.get("id"):
            continue
        filtered_candidates.append(cand)

    if not filtered_candidates:
        return []

    # 2. Check scikit-learn availability
    if not is_sklearn_available():
        logger.warning("scikit-learn is not available. Skipping batch TF-IDF similarity calculation.")
        return []

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # 3. Prepare candidate representations and identify URL matches
    candidate_titles: list[str] = []
    candidate_texts: list[str] = []
    valid_candidates: list[dict[str, Any]] = []
    immediate_url_matches: list[dict[str, Any]] = []

    for cand in filtered_candidates:
        cand_url = (cand.get("url") or "").strip()
        # Fast path: exact URL duplicate
        if target_url and cand_url and target_url == cand_url:
            immediate_url_matches.append({
                "article": cand,
                "similarity": 1.0,
                "similarity_percent": 100.0,
                "relationship": "Duplicate / Same Story",
            })
            continue

        c_text = prepare_article_text(cand)
        c_title = preprocess_text(cand.get("title", ""))
        if c_text or c_title:
            candidate_titles.append(c_title)
            candidate_texts.append(c_text)
            valid_candidates.append(cand)

    results: list[dict[str, Any]] = list(immediate_url_matches)

    # 4. Compute Title & Composite Content TF-IDF Similarity
    if valid_candidates:
        try:
            # 4a. Title Similarity (Term Frequency normalized)
            title_scores = [0.0] * len(valid_candidates)
            if target_title and any(candidate_titles):
                title_corpus = [target_title] + candidate_titles
                v_title = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), use_idf=False, norm="l2")
                m_title = v_title.fit_transform(title_corpus)
                title_scores = cosine_similarity(m_title[0:1], m_title[1:])[0]

            # 4b. Content Similarity (Full composite text with Term Frequency)
            content_scores = [0.0] * len(valid_candidates)
            if target_text and any(candidate_texts):
                content_corpus = [target_text] + candidate_texts
                v_content = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), use_idf=False, norm="l2")
                m_content = v_content.fit_transform(content_corpus)
                content_scores = cosine_similarity(m_content[0:1], m_content[1:])[0]

            # 4c. Blend scores: headline similarity is highly influential in news
            for cand, t_score, c_score in zip(valid_candidates, title_scores, content_scores):
                # Combined score takes the maximum of title alignment or weighted blend
                combined_score = max(float(t_score), float(c_score), float(t_score) * 0.6 + float(c_score) * 0.4)
                score = max(0.0, min(1.0, round(combined_score, 4)))

                if score >= threshold:
                    rel_label = classify_relationship(score)
                    results.append({
                        "article": cand,
                        "similarity": score,
                        "similarity_percent": round(score * 100.0, 1),
                        "relationship": rel_label,
                    })

        except Exception as e:
            logger.error("Error computing batch TF-IDF similarity: %s", e)

    # 5. Sort by similarity score descending
    results.sort(key=lambda item: item["similarity"], reverse=True)

    # 6. Deduplicate results by URL if duplicate records exist
    seen_urls: set[str] = set()
    deduped_results: list[dict[str, Any]] = []
    for item in results:
        art_url = (item["article"].get("url") or "").strip()
        if art_url and art_url in seen_urls:
            continue
        if art_url:
            seen_urls.add(art_url)
        deduped_results.append(item)

    return deduped_results[:max_results]


# ---------------------------------------------------------------------------
# High-Level Collection Duplicate Detector & Grouping (Optional)
# ---------------------------------------------------------------------------

def group_related_articles(
    articles: list[dict[str, Any]],
    threshold: float = RELATED_THRESHOLD,
) -> list[list[dict[str, Any]]]:
    """
    Groups a list of articles into connected clusters of related/duplicate stories
    using pairwise TF-IDF cosine similarity.

    Parameters:
    - articles (list[dict[str, Any]]): List of article dictionaries.
    - threshold (float): Minimum similarity threshold to form a link.

    Returns:
    - list[list[dict[str, Any]]]: List of story clusters (each cluster is a list of articles).
    """
    if not articles or len(articles) <= 1:
        return [[art] for art in articles] if articles else []

    if not is_sklearn_available():
        return [[art] for art in articles]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        prepared_texts = [prepare_article_text(art) for art in articles]
        # Filter articles that have valid text
        non_empty_indices = [i for i, t in enumerate(prepared_texts) if t.strip()]
        
        if not non_empty_indices:
            return [[art] for art in articles]

        sub_corpus = [prepared_texts[i] for i in non_empty_indices]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
        tfidf_matrix = vectorizer.fit_transform(sub_corpus)
        sim_matrix = cosine_similarity(tfidf_matrix)

        n = len(sub_corpus)
        adj: dict[int, list[int]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        # Connected components traversal
        visited: set[int] = set()
        clusters: list[list[dict[str, Any]]] = []

        for i in range(n):
            if i not in visited:
                component: list[int] = []
                queue = [i]
                visited.add(i)
                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                clusters.append([articles[non_empty_indices[idx]] for idx in component])

        return clusters

    except Exception as e:
        logger.error("Error clustering related articles: %s", e)
        return [[art] for art in articles]
