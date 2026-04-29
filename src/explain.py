"""
explain.py
----------
Generates short, human-readable "why recommended" blurbs for each result.

This is a simple rule-based approach — no LLM required.
The explanation is constructed by comparing keywords in the user query to
structured fields in the restaurant row.

Example output:
  "Matched your interest in Italian food. Highly rated (4.5 ★) with 312 reviews.
   Located in Philadelphia. Moderate price range."
"""

import re
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# Keywords that signal certain user preferences
PRICE_KEYWORDS    = {"cheap", "inexpensive", "budget", "affordable", "low cost"}
QUIET_KEYWORDS    = {"quiet", "calm", "peaceful", "study", "work"}
ROMANTIC_KEYWORDS = {"romantic", "date", "date night", "intimate", "cozy"}
LATE_KEYWORDS     = {"late", "late night", "midnight", "after midnight"}

PRICE_LABEL = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}


def _extract_query_keywords(query: str) -> set[str]:
    """Lowercase and tokenize the query into a set of words."""
    return set(re.findall(r"\b\w+\b", query.lower()))


def _price_label(price_range) -> str:
    # Yelp data stores price_range as floats (1.0, 2.5, 1.667, ...) so we
    # round to the nearest integer tier before looking up the symbol.
    try:
        key = int(round(float(price_range)))
    except (TypeError, ValueError):
        return "unknown price range"
    return PRICE_LABEL.get(key, "unknown price range")


def generate_explanation(row: pd.Series, query: str) -> str:
    """
    Build a short explanation sentence for one restaurant result.

    Args:
        row:   A single row from the reranked results DataFrame.
        query: The original user query string.

    Returns:
        A human-readable string explaining why this restaurant was recommended.
    """
    parts = []
    query_words = _extract_query_keywords(query)

    # --- Cuisine / category match ---
    categories = str(row.get("categories", "") or "")
    if categories:
        # Show first two category tags
        top_cats = [c.strip() for c in categories.split(",")][:2]
        parts.append(f"Serves {', '.join(top_cats)}.")

    # --- Location ---
    neighborhood = str(row.get("neighborhood", "") or "")
    city = str(row.get("city", "") or "")
    location = neighborhood if neighborhood and neighborhood != "nan" else city
    if location and location != "nan":
        parts.append(f"Located in {location}.")

    # --- Rating ---
    stars = row.get("stars")
    review_count = row.get("review_count")
    if stars is not None:
        review_str = f" ({int(review_count)} reviews)" if review_count else ""
        parts.append(f"Rated {stars} ★{review_str}.")

    # --- Price ---
    price_range = row.get("price_range")
    if price_range and str(price_range) not in ("nan", "None"):
        parts.append(f"Price: {_price_label(price_range)}.")

    # --- Contextual matches based on query keywords ---
    if query_words & QUIET_KEYWORDS:
        noise = str(row.get("attributes_noise_level", "") or "")
        if "quiet" in noise.lower():
            parts.append("Known to be quiet — great for studying or working.")

    if query_words & ROMANTIC_KEYWORDS:
        ambience = str(row.get("attributes_ambience", "") or "")
        if "romantic" in ambience.lower():
            parts.append("Has a romantic ambience.")

    if query_words & PRICE_KEYWORDS:
        if str(price_range) in ("1", "2"):
            parts.append("Fits your budget.")

    if not parts:
        parts.append("Closely matched your query based on reviews and profile.")

    return " ".join(parts)


def add_explanations(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Add an 'explanation' column to the results DataFrame.

    Args:
        df:    Reranked results DataFrame (one row per restaurant).
        query: The original user query.

    Returns:
        The same DataFrame with an added 'explanation' column.
    """
    df = df.copy()
    df["explanation"] = df.apply(lambda row: generate_explanation(row, query), axis=1)
    return df


# TODO: Improve explanation by extracting a relevant sentence from the actual review text
# TODO: Consider using a small LLM for richer, query-grounded explanations in a future version
