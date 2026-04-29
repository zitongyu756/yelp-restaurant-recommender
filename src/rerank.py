"""
rerank.py
---------
Reranks the candidate restaurants returned by retrieve.py using structured
signals: star rating, review count, and price range.

Why rerank?
  Semantic similarity alone can surface relevant restaurants that have low
  quality or very few reviews.  The reranker blends the similarity score with
  a quality score derived from structured metadata, giving better overall
  results.

Strategy (simple weighted sum):
  final_score = w_sim   * similarity_score
              + w_stars * normalized_stars
              + w_pop   * normalized_review_count
              + w_price * price_match_score

All weights are configurable at the top of this file.
"""

import regex as re
import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

CHEAP_KEYWORDS = {"cheap", "affordable", "budget"}
MODERATE_KEYWORDS = {"moderate", "reasonable", "fair price"}
EXPENSIVE_KEYWORDS = {"expensive", "pricey", "costly"}
VERY_EXPENSIVE_KEYWORDS = {"very expensive", "overpriced", "luxury"}

# ---------------------------------------------------------------------------
# Reranking weights — must sum to 1.0 for interpretability (not required)
# ---------------------------------------------------------------------------
W_SIMILARITY   = 0.48   # semantic similarity to the query
W_STARS        = 0.20   # normalized star rating
W_POPULARITY   = 0.12   # log-normalized review count (proxy for popularity)
W_PRICE_MATCH = 0.20          # price range match

def _extract_query_keywords(query: str) -> set[str]:
    """Lowercase and tokenize the query into a set of words."""
    return set(re.findall(r"\b\w+\b", query.lower()))

def normalize_min_max(series: pd.Series) -> pd.Series:
    """Scale a Series to [0, 1] using min-max normalization."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_quality_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute a quality score in [0, 1] from structured metadata.
    Uses star rating and log review count.
    """
    stars = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
    review_count = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)

    norm_stars = normalize_min_max(stars)
    norm_reviews = normalize_min_max(np.log1p(review_count))  # log dampens extreme counts

    quality = W_STARS * norm_stars + W_POPULARITY * norm_reviews
    return quality

def compute_price_score(df: pd.DataFrame, query: str) -> pd.Series:
    """
    Compute a price score in [0, 1] from user query and price range.
    """
    if not query:
        return pd.Series(np.ones(len(df)), index=df.index)

    query_words = _extract_query_keywords(query)
    wants_cheap = bool(query_words & CHEAP_KEYWORDS)
    wants_expensive = bool(query_words & EXPENSIVE_KEYWORDS)

    if not wants_cheap and not wants_expensive:
        return pd.Series(np.ones(len(df)), index=df.index)

    price = pd.to_numeric(df["price_range"], errors="coerce").fillna(2.0)
    if wants_cheap:
        score = 1.0 - ((price - 1.0) / 3.0)
    else:
        score = (price - 1.0) / 3.0
    return score.clip(0, 1)

def rerank(candidates: pd.DataFrame, query: str = "") -> pd.DataFrame:
    """
    Rerank a DataFrame of candidate restaurants using a weighted combination
    of semantic similarity and structured quality signals.

    Args:
        candidates: DataFrame returned by retrieve.retrieve().
                    Must contain columns: similarity_score, stars, review_count.
        query: The original user query, used for price preference matching.

    Returns:
        DataFrame sorted by final_score descending, with a new 'final_score' column.
    """
    if candidates.empty:
        return candidates

    norm_similarity = normalize_min_max(candidates["similarity_score"])
    quality_score = compute_quality_score(candidates)
    price_score = compute_price_score(candidates, query)
    
    candidates = candidates.copy()
    candidates["final_score"] = (
        W_SIMILARITY * norm_similarity
        + quality_score  # already includes W_STARS and W_POPULARITY weights
        + W_PRICE_MATCH * price_score
    )

    reranked = candidates.sort_values("final_score", ascending=False).reset_index(drop=True)
    logger.info("Reranked %d candidates", len(reranked))
    return reranked


# TODO: Add price preference matching (e.g., if query contains "cheap", boost low price_range)
# TODO: Experiment with different weight combinations
