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
W_SIMILARITY   = 0.40   # semantic similarity to the query
W_STARS        = 0.15   # normalized star rating
W_POPULARITY   = 0.10   # log-normalized review count (proxy for popularity)
W_PRICE_MATCH  = 0.15   # price range match
W_LOCATION     = 0.20   # exact neighborhood match

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
    Uses a Bayesian Average for star rating to prevent places with a single 
    5-star review from unfairly outranking well-established places.
    """
    stars = pd.to_numeric(df["stars"], errors="coerce").fillna(0)
    review_count = pd.to_numeric(df["review_count"], errors="coerce").fillna(0)

    # Bayesian Average for stars
    # C = assumed global average rating (3.5 is typical for Yelp)
    # m = confidence threshold (e.g., 50 reviews)
    C = 3.5
    m = 50.0
    bayesian_stars = (review_count * stars + m * C) / (review_count + m)

    norm_stars = normalize_min_max(bayesian_stars)
    norm_reviews = normalize_min_max(np.log1p(review_count))  # log dampens extreme counts

    quality = W_STARS * norm_stars + W_POPULARITY * norm_reviews
    return quality

def compute_price_score(df: pd.DataFrame, query: str) -> pd.Series:
    """
    Compute a price score in [0, 1] from user query and price range (1 to 4).
    """
    if not query:
        return pd.Series(np.ones(len(df)), index=df.index)

    query_lower = query.lower()
    
    # Use regex to properly match multi-word phrases like "very expensive"
    wants_cheap = any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in CHEAP_KEYWORDS)
    wants_moderate = any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in MODERATE_KEYWORDS)
    wants_expensive = any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in EXPENSIVE_KEYWORDS)
    wants_very_expensive = any(re.search(rf"\b{re.escape(kw)}\b", query_lower) for kw in VERY_EXPENSIVE_KEYWORDS)

    ideal_prices = []
    if wants_cheap:
        ideal_prices.append(1.0)
    if wants_moderate:
        ideal_prices.append(2.0)
    if wants_expensive:
        ideal_prices.append(3.0)
    if wants_very_expensive:
        ideal_prices.append(4.0)

    if not ideal_prices:
        return pd.Series(np.ones(len(df)), index=df.index)

    price = pd.to_numeric(df["price_range"], errors="coerce").fillna(2.0)
    
    scores = []
    for ideal in ideal_prices:
        # Score is 1.0 at ideal price, decreasing by 0.33 for each tier away
        score = 1.0 - (np.abs(price - ideal) / 3.0)
        scores.append(score)
        
    final_score = pd.concat(scores, axis=1).max(axis=1)
    return final_score.clip(0, 1)

def compute_location_score(df: pd.DataFrame, query: str) -> pd.Series:
    """
    Compute a location score (0.0 or 1.0). If the query explicitly mentions
    the restaurant's neighborhood or city, it gets a massive relevance boost.
    """
    if not query:
        return pd.Series(np.zeros(len(df)), index=df.index)
        
    query_lower = query.lower()
    scores = np.zeros(len(df))
    
    # Safely get columns (they might be missing depending on how profiles were built)
    hoods = df.get("neighborhood", pd.Series([""] * len(df)))
    cities = df.get("city", pd.Series([""] * len(df)))
    
    for i, (hood, city) in enumerate(zip(hoods, cities)):
        hood_str = str(hood).lower() if pd.notna(hood) else ""
        city_str = str(city).lower() if pd.notna(city) else ""
        
        if hood_str and len(hood_str) > 3 and hood_str in query_lower:
            scores[i] = 1.0
        elif city_str and len(city_str) > 3 and city_str in query_lower:
            scores[i] = 1.0
            
    return pd.Series(scores, index=df.index)

def rerank(candidates: pd.DataFrame, query: str = "") -> pd.DataFrame:
    """
    Rerank a DataFrame of candidate restaurants using a weighted combination
    of semantic similarity and structured quality signals.

    Args:
        candidates: DataFrame returned by retrieve.retrieve().
                    Must contain columns: similarity_score, stars, review_count.
        query: The original user query, used for price preference matching and dynamic weighting.

    Returns:
        DataFrame sorted by final_score descending, with a new 'final_score' column.
    """
    if candidates.empty:
        return candidates

    # Dynamic Weighting based on query intent
    query_lower = query.lower()
    
    # Default weights
    w_sim = W_SIMILARITY
    w_stars = W_STARS
    w_pop = W_POPULARITY
    w_price = W_PRICE_MATCH
    w_loc = W_LOCATION

    # Check for popularity/review count intent
    if any(kw in query_lower for kw in ["most reviews", "popular", "many reviews", "most reviewed"]):
        w_pop = 0.50
        w_sim = 0.15
        w_stars = 0.10
        w_price = 0.10
        w_loc = 0.15

    # Check for rating intent
    elif any(kw in query_lower for kw in ["highest rated", "best rated", "top rated"]):
        w_stars = 0.50
        w_sim = 0.15
        w_pop = 0.10
        w_price = 0.10
        w_loc = 0.15

    norm_similarity = normalize_min_max(candidates["similarity_score"])
    
    # Recompute quality score using dynamic weights instead of globals
    stars = pd.to_numeric(candidates["stars"], errors="coerce").fillna(0)
    review_count = pd.to_numeric(candidates["review_count"], errors="coerce").fillna(0)
    C = 3.5
    m = 50.0
    bayesian_stars = (review_count * stars + m * C) / (review_count + m)
    
    norm_stars = normalize_min_max(bayesian_stars)
    norm_reviews = normalize_min_max(np.log1p(review_count))
    
    quality_score = w_stars * norm_stars + w_pop * norm_reviews
    price_score = compute_price_score(candidates, query)
    loc_score = compute_location_score(candidates, query)
    
    # Re-allocate weights if specific constraints aren't used in the query
    # so we don't artificially lower the overall Match Score.
    if loc_score.max() == 0.0:
        w_sim += w_loc
        w_loc = 0.0
        
    if (price_score == 1.0).all():
        w_sim += w_price
        w_price = 0.0
    
    candidates = candidates.copy()
    candidates["final_score"] = (
        w_sim * norm_similarity
        + quality_score
        + w_price * price_score
        + w_loc * loc_score
    )

    reranked = candidates.sort_values("final_score", ascending=False).reset_index(drop=True)

    # Apply cluster-based diversity penalty
    # This ensures that we don't just show 5 restaurants with the exact same "vibe" (cluster).
    # We penalize restaurants if their cluster has already appeared higher in the ranking.
    if "cluster_id" in reranked.columns:
        seen_clusters = set()
        penalized_scores = []
        for _, row in reranked.iterrows():
            cluster = row.get("cluster_id")
            score = row["final_score"]
            if pd.notna(cluster) and cluster in seen_clusters:
                score -= 0.10  # 10% penalty for redundant cluster
            elif pd.notna(cluster):
                seen_clusters.add(cluster)
            penalized_scores.append(score)
        
        reranked["final_score"] = penalized_scores
        reranked = reranked.sort_values("final_score", ascending=False).reset_index(drop=True)

    logger.info("Reranked %d candidates with diversity penalty applied", len(reranked))
    return reranked


# TODO: Add price preference matching (e.g., if query contains "cheap", boost low price_range)
# TODO: Experiment with different weight combinations
