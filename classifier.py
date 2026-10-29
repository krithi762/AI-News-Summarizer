"""
classifier.py
-------------
Module responsible for classifying extracted news articles into standardized
predefined categories using the Google Gemini API, with a robust keyword-based
fallback mechanism.

Supported Categories:
- Technology
- Business
- Sports
- Science
- Health
- World
- Entertainment

Features:
- Reuses the existing Gemini configuration and client execution from summarizer.py.
- Validates article title and text with graceful handling for long articles.
- Strict anti-hallucination prompt instructing Gemini to identify the single PRIMARY category.
- Safe response parser extracting category and model confidence (0.0 - 1.0).
- Automatic category normalization for common variations (e.g., 'Tech' -> 'Technology').
- Weighted keyword fallback classifier when Gemini API is unavailable or unconfigured.
- Consistent, structured dictionary return format with type hints and docstrings.
"""

import os
import re
import logging
from typing import Any
from dotenv import load_dotenv

import summarizer

# Ensure environment variables are loaded
load_dotenv()

# Setup module logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Category Definitions & Standard Constants
# ---------------------------------------------------------------------------

VALID_CATEGORIES: list[str] = [
    "Technology",
    "Business",
    "Sports",
    "Science",
    "Health",
    "World",
    "Entertainment",
]

CATEGORY_DEFINITIONS: dict[str, str] = {
    "Technology": (
        "Artificial intelligence, software, hardware, computers, smartphones, "
        "cybersecurity, internet technology, programming, robotics, technology companies, "
        "and emerging technologies."
    ),
    "Business": (
        "Companies, finance, markets, stocks, investments, banking, economy, "
        "trade, startups, corporate announcements, business strategy, and employment markets."
    ),
    "Sports": (
        "Cricket, football, soccer, tennis, basketball, athletics, Olympics, "
        "tournaments, players, teams, coaches, and sporting events."
    ),
    "Science": (
        "Scientific research, discoveries, astronomy, physics, chemistry, biology, "
        "climate research, space science, experiments, and scientific institutions."
    ),
    "Health": (
        "Medicine, diseases, healthcare, hospitals, treatments, medical research, "
        "public health, nutrition, mental wellness, and health policy."
    ),
    "World": (
        "International events, countries, diplomacy, wars/conflicts, international politics, "
        "disasters, global events, and major events that do not fit another category."
    ),
    "Entertainment": (
        "Movies, television, music, celebrities, actors, artists, awards, "
        "streaming, cultural events, and entertainment industry news."
    ),
}

# Aliases and variations for safe category normalization
CATEGORY_NORMALIZATION_MAP: dict[str, str] = {
    # Technology
    "technology": "Technology",
    "tech": "Technology",
    "infotech": "Technology",
    "information technology": "Technology",
    "software": "Technology",
    "ai": "Technology",
    "hardware": "Technology",
    "cybersecurity": "Technology",
    "robotics": "Technology",
    "computing": "Technology",
    # Business
    "business": "Business",
    "biz": "Business",
    "finance": "Business",
    "financial": "Business",
    "economy": "Business",
    "economics": "Business",
    "markets": "Business",
    "market": "Business",
    "stocks": "Business",
    "stock": "Business",
    "trade": "Business",
    "corporate": "Business",
    # Sports
    "sports": "Sports",
    "sport": "Sports",
    "athletics": "Sports",
    "football": "Sports",
    "soccer": "Sports",
    "cricket": "Sports",
    "tennis": "Sports",
    "basketball": "Sports",
    # Science
    "science": "Science",
    "scientific": "Science",
    "space": "Science",
    "astronomy": "Science",
    "physics": "Science",
    "chemistry": "Science",
    "biology": "Science",
    # Health
    "health": "Health",
    "healthcare": "Health",
    "medical": "Health",
    "medicine": "Health",
    "wellness": "Health",
    "fitness": "Health",
    # World
    "world": "World",
    "global": "World",
    "international": "World",
    "general": "World",
    "geopolitics": "World",
    "politics": "World",
    "diplomacy": "World",
    # Entertainment
    "entertainment": "Entertainment",
    "entertain": "Entertainment",
    "movies": "Entertainment",
    "movie": "Entertainment",
    "film": "Entertainment",
    "cinema": "Entertainment",
    "music": "Entertainment",
    "celebrity": "Entertainment",
    "celebrities": "Entertainment",
    "hollywood": "Entertainment",
    "tv": "Entertainment",
    "television": "Entertainment",
    "arts": "Entertainment",
    "art": "Entertainment",
}

# Keywords used for fallback rule-based classification
FALLBACK_KEYWORDS: dict[str, list[str]] = {
    "Technology": [
        "ai", "artificial intelligence", "software", "hardware", "apple", "google", "microsoft",
        "nvidia", "meta", "openai", "semiconductor", "chip", "chipset", "app", "apps", "smartphone",
        "phone", "cybersecurity", "hack", "hacker", "cloud", "server", "algorithm", "robot",
        "robotics", "gadget", "crypto", "blockchain", "tech", "laptop", "processor", "gpu", "cpu"
    ],
    "Business": [
        "market", "markets", "stock", "stocks", "shares", "wall street", "investor", "investors",
        "investment", "revenue", "profit", "earnings", "quarterly", "bank", "banking", "inflation",
        "economy", "economic", "gdp", "trade", "deal", "acquisition", "merger", "startup", "ceo",
        "financial", "finance", "treasury", "interest rate", "nasdaq", "dow jones", "valuation"
    ],
    "Sports": [
        "match", "game", "tournament", "championship", "cup", "league", "score", "scores",
        "goal", "goals", "player", "players", "coach", "team", "teams", "win", "won", "victory",
        "defeat", "cricket", "football", "soccer", "tennis", "basketball", "olympics", "fifa",
        "nba", "nfl", "ipl", "premier league", "grand slam", "stadium", "athlete", "athletes"
    ],
    "Science": [
        "scientist", "scientists", "research", "researchers", "study", "studies", "discovery",
        "discoveries", "space", "nasa", "esa", "isro", "telescope", "planet", "planetarium",
        "galaxy", "astronomy", "physics", "quantum", "chemistry", "biology", "climate", "fossil",
        "species", "mars", "moon", "asteroid", "laboratory", "experiment", "experiments"
    ],
    "Health": [
        "health", "healthcare", "hospital", "hospitals", "doctor", "doctors", "patient", "patients",
        "disease", "diseases", "virus", "infection", "vaccine", "vaccines", "treatment", "therapy",
        "clinical", "drug", "drugs", "fda", "who", "cancer", "diabetes", "mental health", "diet",
        "nutrition", "medicine", "medical", "surgery", "outbreak", "wellness", "symptom", "symptoms"
    ],
    "World": [
        "president", "prime minister", "government", "parliament", "congress", "diplomat", "diplomacy",
        "summit", "treaty", "united nations", "un", "nato", "war", "conflict", "military", "troops",
        "border", "refugee", "refugees", "foreign", "embassy", "election", "elections", "sanctions",
        "crisis", "protest", "protests", "treaty", "nation", "nations", "geopolitical"
    ],
    "Entertainment": [
        "movie", "movies", "film", "films", "actor", "actors", "actress", "actresses", "director",
        "cinema", "box office", "trailer", "hollywood", "bollywood", "music", "song", "songs",
        "album", "singer", "singers", "concert", "tour", "grammy", "oscar", "oscars", "emmy",
        "streaming", "netflix", "disney", "celebrity", "celebrities", "drama", "series", "season"
    ],
}

# Maximum words of article text to send for classification
MAX_CLASSIFY_WORDS = 1500


# ---------------------------------------------------------------------------
# Normalization & Keyword Fallback Helpers
# ---------------------------------------------------------------------------

def normalize_category(raw_category: str | None) -> str | None:
    """
    Normalizes a category string to one of the 7 valid predefined categories.
    Handles capitalization, whitespace, and common synonym variations.
    """
    if not raw_category or not isinstance(raw_category, str):
        return None

    cleaned = raw_category.strip()
    # Direct exact match check (case-insensitive)
    for valid_cat in VALID_CATEGORIES:
        if cleaned.lower() == valid_cat.lower():
            return valid_cat

    # Clean punctuation and check dictionary mapping
    sanitized = re.sub(r"[^\w\s]", "", cleaned).strip().lower()
    if sanitized in CATEGORY_NORMALIZATION_MAP:
        return CATEGORY_NORMALIZATION_MAP[sanitized]

    # Partial / substring match check
    for key, mapped_cat in CATEGORY_NORMALIZATION_MAP.items():
        if key in sanitized:
            return mapped_cat

    return None


def _classify_with_keywords(title: str | None, article_text: str | None) -> dict[str, Any]:
    """
    Lightweight rule-based fallback classifier that scores categories
    based on keyword occurrences in the title and article text.
    Title matches are weighted 3x higher than body text matches.
    """
    safe_title = (title or "").lower()
    safe_text = (article_text or "").lower()

    # If neither title nor text is provided
    if not safe_title.strip() and not safe_text.strip():
        return {
            "success": False,
            "category": None,
            "confidence": None,
            "method": "keyword_fallback",
            "error": "No title or article content provided for classification.",
        }

    # Limit body text length for scoring
    limited_body_words = safe_text.split()[:MAX_CLASSIFY_WORDS]
    limited_body = " ".join(limited_body_words)

    scores: dict[str, int] = {cat: 0 for cat in VALID_CATEGORIES}

    for cat, keywords in FALLBACK_KEYWORDS.items():
        for kw in keywords:
            # Word boundary regex pattern for accurate matching
            kw_pattern = r"\b" + re.escape(kw) + r"\b"
            
            # Title matches (3 points per match)
            title_matches = len(re.findall(kw_pattern, safe_title))
            scores[cat] += title_matches * 3

            # Body matches (1 point per match)
            body_matches = len(re.findall(kw_pattern, limited_body))
            scores[cat] += body_matches

    # Find category with highest positive score
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_cat, best_score = sorted_scores[0]

    if best_score > 0:
        logger.info("Fallback keyword classifier assigned category: %s (score: %d)", best_cat, best_score)
        return {
            "success": True,
            "category": best_cat,
            "confidence": None,
            "method": "keyword_fallback",
            "error": None,
        }

    # If no keyword matched, default to World as general category
    logger.info("Fallback keyword classifier found no strong matches. Defaulting to 'World'.")
    return {
        "success": True,
        "category": "World",
        "confidence": None,
        "method": "keyword_fallback",
        "error": None,
    }


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def _build_classification_system_instruction() -> str:
    """
    Constructs strict anti-hallucination system instructions for article categorization.
    """
    cat_bullets = "\n".join([f"- **{cat}**: {desc}" for cat, desc in CATEGORY_DEFINITIONS.items()])
    
    return (
        "You are an expert, objective news classification system.\n"
        "Your task is to analyze the provided news article and classify it into EXACTLY ONE "
        "primary category from the allowed list.\n\n"
        "ALLOWED CATEGORIES & DEFINITIONS:\n"
        f"{cat_bullets}\n\n"
        "CRITICAL CLASSIFICATION RULES:\n"
        "1. Select the single category that best represents the article's PRIMARY subject.\n"
        "2. Do NOT select a category simply because a related keyword appears in passing (e.g. an article about a tech company's stock earnings is 'Business', while an article about an athlete launching a film is 'Entertainment').\n"
        "3. You must choose ONLY from the 7 allowed categories: Technology, Business, Sports, Science, Health, World, Entertainment.\n"
        "4. Base your judgment strictly and only on the supplied title and article text.\n"
        "5. Estimate your model confidence score as a decimal value between 0.0 and 1.0 (e.g., 0.95 for very clear primary subjects, 0.70 for mixed subjects).\n"
        "6. Output EXACTLY two lines in the following structured format:\n"
        "CATEGORY: <AllowedCategoryName>\n"
        "CONFIDENCE: <DecimalValueBetween0.0And1.0>\n\n"
        "Do NOT output conversational greetings, preambles, explanations, markdown quotes, or notes."
    )


def _build_classification_user_prompt(title: str | None, article_text: str | None) -> str:
    """
    Constructs the user prompt for news classification.
    """
    safe_title = (title or "").strip()
    safe_text = (article_text or "").strip()

    # Limit article body words to avoid excessive token usage
    words = safe_text.split()
    if len(words) > MAX_CLASSIFY_WORDS:
        safe_text = " ".join(words[:MAX_CLASSIFY_WORDS]) + "\n[... article text truncated for classification ...]"

    return (
        f"INSTRUCTIONS:\n"
        f"- Read the article title and text below.\n"
        f"- Determine the single primary category and output in the requested format:\n"
        f"CATEGORY: <CategoryName>\n"
        f"CONFIDENCE: <0.0-1.0>\n\n"
        f"--- START ARTICLE METADATA & CONTENT ---\n"
        f"TITLE: {safe_title if safe_title else 'N/A'}\n\n"
        f"TEXT:\n{safe_text if safe_text else 'N/A'}\n"
        f"--- END ARTICLE METADATA & CONTENT ---"
    )


# ---------------------------------------------------------------------------
# Response Parsing Helper
# ---------------------------------------------------------------------------

def _parse_gemini_classification(raw_text: str | None) -> tuple[str | None, float | None]:
    """
    Safely parses the raw Gemini model response to extract the category and confidence.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None, None

    category_match = re.search(r"CATEGORY\s*:\s*([A-Za-z\s]+)", raw_text, flags=re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", raw_text, flags=re.IGNORECASE)

    raw_cat = category_match.group(1).strip() if category_match else None
    normalized_cat = normalize_category(raw_cat) if raw_cat else None

    confidence: float | None = None
    if confidence_match:
        try:
            val = float(confidence_match.group(1).strip())
            # Clamping value between 0.0 and 1.0
            confidence = max(0.0, min(1.0, round(val, 2)))
        except (ValueError, TypeError):
            confidence = None

    # If direct regex didn't find CATEGORY line, attempt fallback scan over raw text lines
    if not normalized_cat:
        for line in raw_text.splitlines():
            line_clean = line.strip()
            candidate = normalize_category(line_clean)
            if candidate:
                normalized_cat = candidate
                break

    return normalized_cat, confidence


# ---------------------------------------------------------------------------
# Main Classification Interface (Feature 6)
# ---------------------------------------------------------------------------

def classify_article(
    title: str | None,
    article_text: str | None,
) -> dict[str, Any]:
    """
    Classifies a news article into one of 7 predefined categories based on its
    title and extracted text using Google Gemini AI, with automatic keyword fallback.

    Parameters:
    - title (str | None): Article headline/title.
    - article_text (str | None): Cleaned article body text.

    Returns:
    - dict[str, Any]: Standardized classification result dictionary:
      {
          "success": bool,
          "category": str | None,
          "confidence": float | None,
          "method": str ("gemini" | "keyword_fallback"),
          "error": str | None,
      }
    """
    # 1. Input Validation
    clean_title = (title or "").strip()
    clean_text = (article_text or "").strip()

    if not clean_title and not clean_text:
        return {
            "success": False,
            "category": None,
            "confidence": None,
            "method": None,
            "error": "No title or article text provided for classification.",
        }

    # 2. Check Gemini Configuration
    if not summarizer.is_gemini_configured():
        logger.info("Gemini API is not configured. Using keyword fallback classifier.")
        return _classify_with_keywords(clean_title, clean_text)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = summarizer.get_configured_model()
    system_instruction = _build_classification_system_instruction()
    user_prompt = _build_classification_user_prompt(clean_title, clean_text)

    # 3. Execute Gemini API Call with Error Handling & Fallback
    try:
        raw_response = summarizer._execute_gemini_call(
            api_key=api_key,
            model_name=model_name,
            system_instruction=system_instruction,
            prompt=user_prompt,
        )

        parsed_category, parsed_confidence = _parse_gemini_classification(raw_response)

        if parsed_category and parsed_category in VALID_CATEGORIES:
            return {
                "success": True,
                "category": parsed_category,
                "confidence": parsed_confidence,
                "method": "gemini",
                "error": None,
            }

        # If Gemini returned an unmappable category, fallback to keyword matching
        logger.warning(
            "Gemini response could not be mapped to a valid category. Raw response: %s. Falling back to keywords.",
            raw_response
        )
        return _classify_with_keywords(clean_title, clean_text)

    except Exception as e:
        logger.error("Gemini classification failed with error: %s. Initiating keyword fallback.", e)
        # On Gemini error, fallback gracefully to keyword matching
        fallback_res = _classify_with_keywords(clean_title, clean_text)
        if fallback_res.get("success"):
            return fallback_res

        friendly_err = summarizer._map_gemini_error(e)
        return {
            "success": False,
            "category": None,
            "confidence": None,
            "method": "keyword_fallback",
            "error": friendly_err,
        }
