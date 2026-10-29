"""
sentiment_analyzer.py
---------------------
Module responsible for analyzing the overall sentiment and tone of extracted news
articles using the Google Gemini API, with a robust rule-based local fallback lexicon.

Supported Sentiments:
- Positive: Favorable developments, achievements, growth, success, recoveries, strong performance.
- Neutral: Factual reports, balanced developments, routine announcements, mixed non-dominant coverage.
- Negative: Losses, disasters, conflicts, risks, decline, criticism, casualties, serious concerns.

Features:
- Reuses the existing Gemini configuration and client execution from summarizer.py.
- Validates article title and text with sensible text chunking for long articles.
- Robust anti-hallucination prompt architecture distinguishing news event tone from subjective emotion.
- Safe response parser extracting sentiment, confidence (0.0 to 1.0), and 1-2 sentence explanation.
- Automatic sentiment normalization (e.g. 'POSITIVE' -> 'Positive').
- Fast, rule-based local fallback lexicon when Gemini API is unavailable or unconfigured.
- Non-destructive SQLite persistence and retrieval integration.
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
# Sentiment Definitions & Constants
# ---------------------------------------------------------------------------

VALID_SENTIMENTS: list[str] = [
    "Positive",
    "Neutral",
    "Negative",
]

# Normalization dictionary for common model variations and aliases
SENTIMENT_NORMALIZATION_MAP: dict[str, str] = {
    "positive": "Positive",
    "pos": "Positive",
    "favorable": "Positive",
    "optimistic": "Positive",
    "bullish": "Positive",
    "growth": "Positive",
    "success": "Positive",
    "neutral": "Neutral",
    "neu": "Neutral",
    "balanced": "Neutral",
    "mixed": "Neutral",
    "factual": "Neutral",
    "objective": "Neutral",
    "negative": "Negative",
    "neg": "Negative",
    "unfavorable": "Negative",
    "pessimistic": "Negative",
    "bearish": "Negative",
    "adverse": "Negative",
    "critical": "Negative",
}

# Maximum words of article text to send to Gemini for sentiment evaluation
MAX_SENTIMENT_WORDS = 1500

# Minimum word count required for reliable sentiment analysis
MIN_SENTIMENT_WORD_COUNT = 15

# ---------------------------------------------------------------------------
# Fallback Lexicon for Offline Rule-based Sentiment Analysis
# ---------------------------------------------------------------------------

POSITIVE_LEXICON: set[str] = {
    "growth", "grow", "grows", "growing", "grew", "gain", "gains", "gained",
    "success", "successful", "successfully", "succeed", "succeeds",
    "improve", "improves", "improved", "improvement", "improvements",
    "win", "wins", "won", "winning", "winner", "winners", "victory", "victories",
    "profit", "profits", "profitable", "profitability", "record", "surge", "surges", "surged",
    "benefit", "benefits", "benefited", "beneficial", "advantage", "advantages",
    "achievement", "achievements", "achieve", "achieved", "breakthrough", "breakthroughs",
    "recovery", "recoveries", "recover", "recovers", "recovered", "boost", "boosts", "boosted",
    "innovation", "innovations", "innovative", "expansion", "expands", "expanded",
    "milestone", "milestones", "optimism", "optimistic", "favorable", "triumph", "triumphs",
    "strength", "strengthen", "strengthens", "strengthened", "strong", "positive",
    "soar", "soars", "soared", "rally", "rallies", "rallied", "celebrate", "celebrates", "celebrated",
}

NEGATIVE_LEXICON: set[str] = {
    "loss", "losses", "lost", "lose", "loses", "losing",
    "decline", "declines", "declined", "declining", "drop", "drops", "dropped", "dropping",
    "fall", "falls", "fell", "falling", "slump", "slumps", "slumped", "crash", "crashed",
    "death", "deaths", "dead", "die", "dies", "died", "fatal", "fatalities", "fatality",
    "casualty", "casualties", "kill", "kills", "killed", "killing", "murder", "murdered",
    "crisis", "crises", "disaster", "disasters", "disastrous", "tragedy", "tragic",
    "failure", "failures", "fail", "fails", "failed", "failing", "bankrupt", "bankruptcy",
    "damage", "damages", "damaged", "destroy", "destroys", "destroyed", "destruction",
    "attack", "attacks", "attacked", "war", "wars", "conflict", "conflicts", "strike", "strikes",
    "risk", "risks", "risky", "threat", "threats", "threaten", "threatens", "threatened",
    "criticism", "criticisms", "criticize", "criticizes", "criticized", "critic", "critics",
    "warning", "warnings", "warn", "warns", "warned", "concern", "concerns", "concerned",
    "struggle", "struggles", "struggled", "struggling", "scandal", "scandals", "fraud",
    "deficit", "lawsuit", "lawsuits", "sue", "sued", "violation", "violations", "negative",
}


# ---------------------------------------------------------------------------
# Normalization & Rule-Based Fallback
# ---------------------------------------------------------------------------

def normalize_sentiment(raw_sentiment: str | None) -> str | None:
    """
    Normalizes an incoming sentiment label to one of the 3 supported standard sentiments:
    'Positive', 'Neutral', or 'Negative'.
    """
    if not raw_sentiment or not isinstance(raw_sentiment, str):
        return None

    cleaned = raw_sentiment.strip()

    # Exact match check
    for valid_s in VALID_SENTIMENTS:
        if cleaned.lower() == valid_s.lower():
            return valid_s

    # Clean punctuation and dictionary lookup
    sanitized = re.sub(r"[^\w\s]", "", cleaned).strip().lower()
    if sanitized in SENTIMENT_NORMALIZATION_MAP:
        return SENTIMENT_NORMALIZATION_MAP[sanitized]

    # Partial / substring check
    for key, mapped_s in SENTIMENT_NORMALIZATION_MAP.items():
        if key in sanitized:
            return mapped_s

    return None


def _analyze_with_local_fallback(title: str | None, article_text: str | None) -> dict[str, Any]:
    """
    Lightweight rule-based fallback sentiment analyzer using positive/negative lexicons.
    Title matches are weighted 2.5x higher than body text matches.
    """
    safe_title = (title or "").lower()
    safe_text = (article_text or "").lower()

    if not safe_title.strip() and not safe_text.strip():
        return {
            "success": False,
            "sentiment": None,
            "confidence": None,
            "explanation": "No title or article content provided for sentiment analysis.",
            "method": "local_fallback",
            "error": "No title or article content provided for sentiment analysis.",
        }

    # Limit words for scoring
    limited_body_words = safe_text.split()[:MAX_SENTIMENT_WORDS]
    limited_body = " ".join(limited_body_words)

    pos_score = 0.0
    neg_score = 0.0

    # 1. Score title keywords
    for word in re.findall(r"\b[a-z]{3,}\b", safe_title):
        if word in POSITIVE_LEXICON:
            pos_score += 2.5
        elif word in NEGATIVE_LEXICON:
            neg_score += 2.5

    # 2. Score body keywords
    for word in re.findall(r"\b[a-z]{3,}\b", limited_body):
        if word in POSITIVE_LEXICON:
            pos_score += 1.0
        elif word in NEGATIVE_LEXICON:
            neg_score += 1.0

    diff = pos_score - neg_score

    if diff >= 2.0:
        sentiment = "Positive"
        explanation = "The article contains predominantly positive developments, gains, or successes."
    elif diff <= -2.0:
        sentiment = "Negative"
        explanation = "The article contains predominantly negative events, risks, losses, or concerns."
    else:
        sentiment = "Neutral"
        explanation = "The article presents balanced, routine, or factual information without a dominant positive or negative tone."

    logger.info("Local fallback sentiment assigned '%s' (pos=%.1f, neg=%.1f)", sentiment, pos_score, neg_score)

    return {
        "success": True,
        "sentiment": sentiment,
        "confidence": None,
        "explanation": explanation,
        "method": "local_fallback",
        "error": None,
    }


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def _build_sentiment_system_instruction() -> str:
    """
    Constructs strict anti-hallucination system instructions for news sentiment analysis.
    """
    return (
        "You are an expert, objective news sentiment analysis system.\n"
        "Your task is to analyze the provided news article and classify the overall sentiment/tone "
        "of the reported content into EXACTLY ONE category: Positive, Neutral, or Negative.\n\n"
        "SENTIMENT DEFINITIONS:\n"
        "- Positive: The article mainly describes favorable outcomes, growth, success, achievements, breakthroughs, economic recoveries, or beneficial announcements.\n"
        "- Neutral: The article primarily reports factual information, balanced coverage, routine events, or mixed developments without a single dominant tone.\n"
        "- Negative: The article mainly describes losses, failures, natural disasters, casualties, conflicts, risks, economic declines, criticism, or serious concerns.\n\n"
        "CRITICAL RULES:\n"
        "1. Classify the overall sentiment of the REPORTED REAL-WORLD EVENTS and outcomes, not merely whether the journalist used polite language.\n"
        "2. Base your judgment strictly and solely on the provided article text and title.\n"
        "3. Do NOT make outside assumptions, political judgments, or speculate.\n"
        "4. If positive and negative points are equally balanced, choose Neutral.\n"
        "5. Provide a model confidence estimate as a decimal between 0.0 and 1.0 (e.g. 0.90 for unequivocal tone, 0.70 for nuanced/mixed coverage).\n"
        "6. Provide a concise 1 to 2 sentence explanation summarizing why this sentiment was chosen.\n"
        "7. Output EXACTLY three lines in the following structured format:\n"
        "SENTIMENT: <Positive|Neutral|Negative>\n"
        "CONFIDENCE: <DecimalBetween0.0And1.0>\n"
        "EXPLANATION: <Short 1-2 sentence explanation>\n\n"
        "Do NOT include conversational preambles, greetings, or markdown code blocks."
    )


def _build_sentiment_user_prompt(title: str | None, article_text: str | None) -> str:
    """
    Constructs the user prompt for news sentiment analysis.
    """
    safe_title = (title or "").strip()
    safe_text = (article_text or "").strip()

    words = safe_text.split()
    if len(words) > MAX_SENTIMENT_WORDS:
        safe_text = " ".join(words[:MAX_SENTIMENT_WORDS]) + "\n[... article text truncated for sentiment evaluation ...]"

    return (
        f"INSTRUCTIONS:\n"
        f"- Read the news title and text below.\n"
        f"- Determine the dominant overall sentiment and output in the requested 3-line format:\n"
        f"SENTIMENT: <Positive|Neutral|Negative>\n"
        f"CONFIDENCE: <0.0-1.0>\n"
        f"EXPLANATION: <1-2 sentences>\n\n"
        f"--- START ARTICLE METADATA & CONTENT ---\n"
        f"TITLE: {safe_title if safe_title else 'N/A'}\n\n"
        f"TEXT:\n{safe_text if safe_text else 'N/A'}\n"
        f"--- END ARTICLE METADATA & CONTENT ---"
    )


# ---------------------------------------------------------------------------
# Response Parsing Helper
# ---------------------------------------------------------------------------

def _parse_gemini_sentiment(raw_text: str | None) -> tuple[str | None, float | None, str | None]:
    """
    Safely parses the raw Gemini model response to extract sentiment, confidence, and explanation.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None, None, None

    sentiment_match = re.search(r"SENTIMENT\s*:\s*([A-Za-z\s]+)", raw_text, flags=re.IGNORECASE)
    confidence_match = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", raw_text, flags=re.IGNORECASE)
    explanation_match = re.search(r"EXPLANATION\s*:\s*(.+)", raw_text, flags=re.IGNORECASE | re.DOTALL)

    raw_sentiment = sentiment_match.group(1).strip() if sentiment_match else None
    normalized_sentiment = normalize_sentiment(raw_sentiment) if raw_sentiment else None

    confidence: float | None = None
    if confidence_match:
        try:
            val = float(confidence_match.group(1).strip())
            confidence = max(0.0, min(1.0, round(val, 2)))
        except (ValueError, TypeError):
            confidence = None

    explanation: str | None = None
    if explanation_match:
        raw_exp = explanation_match.group(1).strip()
        # Clean extra newlines or subsequent lines
        explanation = " ".join(raw_exp.split())
        # Truncate if model provided unnecessarily long text
        if len(explanation) > 280:
            explanation = explanation[:277] + "..."

    # Fallback line scan if regex did not match SENTIMENT: line
    if not normalized_sentiment:
        for line in raw_text.splitlines():
            line_clean = line.strip()
            candidate = normalize_sentiment(line_clean)
            if candidate:
                normalized_sentiment = candidate
                break

    return normalized_sentiment, confidence, explanation


# ---------------------------------------------------------------------------
# Main Sentiment Analysis Interface (Feature 13)
# ---------------------------------------------------------------------------

def analyze_sentiment(
    title: str | None,
    article_text: str | None,
) -> dict[str, Any]:
    """
    Analyzes the overall sentiment and tone of a news article based on its title
    and body content using Google Gemini AI, with automatic rule-based local fallback.

    Parameters:
    - title (str | None): Article headline or title.
    - article_text (str | None): Cleaned article body text.

    Returns:
    - dict[str, Any]: Standardized sentiment result dictionary:
      {
          "success": bool,
          "sentiment": str | None ("Positive" | "Neutral" | "Negative"),
          "confidence": float | None (0.0 to 1.0),
          "explanation": str | None,
          "method": str ("gemini" | "local_fallback"),
          "error": str | None,
      }
    """
    clean_title = (title or "").strip()
    clean_text = (article_text or "").strip()

    # 1. Input Validation
    if not clean_title and not clean_text:
        return {
            "success": False,
            "sentiment": None,
            "confidence": None,
            "explanation": None,
            "method": None,
            "error": "No title or article text provided for sentiment analysis.",
        }

    total_words = len(clean_title.split()) + len(clean_text.split())
    if total_words < MIN_SENTIMENT_WORD_COUNT:
        logger.warning("Article text too short for reliable sentiment analysis (%d words).", total_words)
        return {
            "success": True,
            "sentiment": "Neutral",
            "confidence": None,
            "explanation": "The article does not contain enough information for reliable sentiment analysis.",
            "method": "local_fallback",
            "error": None,
        }

    # 2. Check Gemini Configuration
    if not summarizer.is_gemini_configured():
        logger.info("Gemini API is not configured. Using local fallback sentiment analyzer.")
        return _analyze_with_local_fallback(clean_title, clean_text)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = summarizer.get_configured_model()
    system_instruction = _build_sentiment_system_instruction()
    user_prompt = _build_sentiment_user_prompt(clean_title, clean_text)

    # 3. Execute Gemini API Call with Error Handling & Fallback
    try:
        raw_response = summarizer._execute_gemini_call(
            api_key=api_key,
            model_name=model_name,
            prompt=user_prompt,
            system_instruction=system_instruction,
        )

        sentiment, confidence, explanation = _parse_gemini_sentiment(raw_response)

        if sentiment and sentiment in VALID_SENTIMENTS:
            if not explanation:
                if sentiment == "Positive":
                    explanation = "The article focuses primarily on positive developments and favorable outcomes."
                elif sentiment == "Negative":
                    explanation = "The article focuses primarily on losses, risks, challenges, or critical concerns."
                else:
                    explanation = "The article presents factual or balanced reporting without a single dominant tone."

            logger.info("Gemini classified article sentiment as '%s' (confidence: %s)", sentiment, confidence)
            return {
                "success": True,
                "sentiment": sentiment,
                "confidence": confidence,
                "explanation": explanation,
                "method": "gemini",
                "error": None,
            }
        else:
            logger.warning("Gemini returned unrecognized sentiment. Triggering local fallback.")
            fallback_res = _analyze_with_local_fallback(clean_title, clean_text)
            return fallback_res

    except Exception as e:
        err_msg = summarizer._map_gemini_error(e)
        logger.warning("Gemini sentiment call encountered error (%s). Using local fallback.", err_msg)
        fallback_res = _analyze_with_local_fallback(clean_title, clean_text)
        fallback_res["error"] = err_msg
        return fallback_res
