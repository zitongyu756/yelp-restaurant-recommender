"""
rerank.py
---------
Reranks the candidate restaurants returned by retrieve.py using structured
signals: star rating, review count, and (optionally) price-range intent
extracted from the user's query.

Why rerank?
  Semantic similarity alone can surface relevant restaurants that have low
  quality or very few reviews.  The reranker blends the similarity score with
  quality signals derived from structured metadata, giving better overall
  results.

Scoring:
  final_score = W_SIMILARITY * normalized_similarity
              + W_STARS      * normalized_stars
              + W_POPULARITY * normalized_log_review_count
              + price_match_bonus     (only if the query expresses price intent)

The three main weights are applied in one place here (instead of split
across helpers) to make the formula easy to read and tune.
"""

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Core reranking weights — the three components sum to 1.0
# ---------------------------------------------------------------------------
W_SIMILARITY = 0.60   # semantic similarity to the query
W_STARS      = 0.25   # normalized star rating
W_POPULARITY = 0.15   # log-normalized review count (proxy for popularity)

# ---------------------------------------------------------------------------
# Price preference matching — applied as an additive bonus only when the
# query contains explicit price-intent words. Zero otherwise, so the base
# formula above is unchanged for neutral queries.
# ---------------------------------------------------------------------------
PRICE_LOW_KEYWORDS = {
    "cheap", "inexpensive", "budget", "affordable",
    "low cost", "low-cost",
}
PRICE_HIGH_KEYWORDS = {
    "fancy", "upscale", "fine dining", "expensive",
    "high-end", "high end", "luxury", "luxurious",
}

PRICE_MATCH_BONUS      = 0.10   # added when price_range aligns with query intent
PRICE_MISMATCH_PENALTY = 0.05   # subtracted when price_range opposes query intent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_min_max(series: pd.Series) -> pd.Series:
    """Scale a Series to [0, 1] using min-max normalization."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


def _detect_price_intent(query: str | None) -> str | None:
    """
    Return 'low' if the query suggests a budget preference, 'high' if it
    suggests an upscale preference, otherwise None.
    """
    if not query:
        return None
    q = query.lower()
    if any(kw in q for kw in PRICE_LOW_KEYWORDS):
        return "low"
    if any(kw in q for kw in PRICE_HIGH_KEYWORDS):
        return "high"
    return None


def _price_match_bonus(df: pd.DataFrame, intent: str | None) -> np.ndarray:
    """
    Per-row additive bonus based on how well each row's price_range matches
    the detected intent. Returns a zero vector when no intent is given.

    Yelp's ``RestaurantsPriceRange2`` maps to 1 ($), 2 ($$), 3 ($$$), 4 ($$$$).
    We treat 1-2 as "low" and 3-4 as "high".
    """
    if intent is None:
        return np.zeros(len(df), dtype=np.float32)

    prices = pd.to_numeric(df["price_range"], errors="coerce")
    is_low = prices.isin([1, 2]).to_numpy()
    is_high = prices.isin([3, 4]).to_numpy()

    bonus = np.zeros(len(df), dtype=np.float32)
    if intent == "low":
        bonus[is_low]  = PRICE_MATCH_BONUS
        bonus[is_high] = -PRICE_MISMATCH_PENALTY
    else:  # "high"
        bonus[is_high] = PRICE_MATCH_BONUS
        bonus[is_low]  = -PRICE_MISMATCH_PENALTY
    return bonus


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def rerank(candidates: pd.DataFrame, query: str | None = None) -> pd.DataFrame:
    """
    Rerank candidate restaurants by combining semantic similarity with
    structured quality signals (and optional price-intent matching).

    Args:
        candidates: DataFrame returned by retrieve.retrieve(). Must contain
                    columns: similarity_score, stars, review_count, price_range.
        query:      Original user query. Only used to detect price intent
                    ("cheap" / "upscale" etc). Safe to omit.

    Returns:
        DataFrame sorted by final_score descending, with a new 'final_score'
        column added.
    """
    if candidates.empty:
        return candidates

    # Normalize each signal to [0, 1] within the candidate pool
    norm_similarity = normalize_min_max(candidates["similarity_score"])
    stars = pd.to_numeric(candidates["stars"], errors="coerce").fillna(0)
    reviews = pd.to_numeric(candidates["review_count"], errors="coerce").fillna(0)
    norm_stars = normalize_min_max(stars)
    norm_reviews = normalize_min_max(np.log1p(reviews))  # log dampens extreme counts

    # Price intent — zero contribution when the query is price-neutral
    price_intent = _detect_price_intent(query)
    price_bonus = _price_match_bonus(candidates, price_intent)

    candidates = candidates.copy()
    candidates["final_score"] = (
        W_SIMILARITY * norm_similarity
        + W_STARS      * norm_stars
        + W_POPULARITY * norm_reviews
        + price_bonus
    )

    reranked = candidates.sort_values("final_score", ascending=False).reset_index(drop=True)
    logger.info(
        "Reranked %d candidates (price_intent=%s)", len(reranked), price_intent,
    )
    return reranked
