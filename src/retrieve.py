"""
retrieve.py
-----------
Takes a natural-language query, embeds it, and retrieves the most
semantically similar restaurants from the precomputed embedding matrix.

Steps:
  1. Load the SentenceTransformer model (cached after first call).
  2. Embed the user query into an L2-normalized vector.
  3. Compute cosine similarity (as a dot product, since both sides are
     pre-normalized) against all restaurant embeddings.
  4. Optionally apply a soft penalty to rows whose ``categories`` don't
     contain the requested cuisine keyword.
  5. Return the top-k most similar rows as a DataFrame.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    PROFILES_CSV,
    EMBEDDINGS_NPY,
    EMBEDDING_MODEL_NAME,
    TOP_K_RETRIEVE,
)
from src.similarity import dot_product_one_to_many, top_k_indices
from src.utils import get_logger, load_csv, load_embeddings

logger = get_logger(__name__)

# Amount to subtract from a row's similarity score when the cuisine soft
# filter is active and the row's categories don't contain the keyword.
# Applied *before* top-k selection so filter intent influences the ranking
# without hard-excluding otherwise strong matches.
SOFT_FILTER_PENALTY = 0.15

# Module-level cache so the model and data are loaded only once per session.
_model: SentenceTransformer | None = None
_profiles_df: pd.DataFrame | None = None
_embeddings: np.ndarray | None = None


def _load_resources() -> None:
    """Load model, profiles, and embeddings into module-level cache."""
    global _model, _profiles_df, _embeddings

    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if _profiles_df is None:
        logger.info("Loading restaurant profiles from %s", PROFILES_CSV)
        _profiles_df = load_csv(PROFILES_CSV)

    if _embeddings is None:
        logger.info("Loading embeddings from %s", EMBEDDINGS_NPY)
        _embeddings = load_embeddings(EMBEDDINGS_NPY)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single query string into an L2-normalized 1-D embedding vector.

    Args:
        query: Natural language query, e.g. "quiet cafe for studying".

    Returns:
        1-D float32 numpy array of shape (D,), unit norm.
    """
    _load_resources()
    vector = _model.encode(query, convert_to_numpy=True).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


def _soft_filter_penalty(df: pd.DataFrame, cuisine: str | None) -> np.ndarray:
    """
    Per-row penalty array for rows whose ``categories`` don't contain the
    requested cuisine keyword (case-insensitive substring match). Returns a
    zero vector if ``cuisine`` is None.

    Note: neighborhood filtering is intentionally not supported — the
    Philadelphia Yelp data has no populated neighborhood column and the
    ``address`` column stores only street addresses, so neighborhood
    information must be expressed inside the natural-language query
    instead (it's picked up semantically via profile_text).
    """
    if not cuisine:
        return np.zeros(len(df), dtype=np.float32)

    cats = df["categories"].fillna("").astype(str).str.lower()
    mismatch = ~cats.str.contains(cuisine.lower(), regex=False)
    return mismatch.to_numpy(dtype=np.float32) * SOFT_FILTER_PENALTY


def retrieve(
    query: str,
    top_k: int = TOP_K_RETRIEVE,
    cuisine: str | None = None,
) -> pd.DataFrame:
    """
    Retrieve the top-k most relevant restaurants for a natural-language query.

    Args:
        query:   User's free-text query.
        top_k:   Number of candidates to return before reranking.
        cuisine: Optional cuisine keyword (e.g. "italian"). Rows whose
                 ``categories`` field doesn't contain this term get a
                 similarity penalty — they can still surface if the semantic
                 match is strong enough.

    Returns:
        DataFrame with the top-k most similar restaurants, sorted by score
        descending. Includes a 'similarity_score' column (post-penalty).
    """
    _load_resources()

    query_vec = embed_query(query)
    scores = dot_product_one_to_many(query_vec, _embeddings)

    if cuisine:
        scores = scores - _soft_filter_penalty(_profiles_df, cuisine)

    indices = top_k_indices(scores, top_k)

    results = _profiles_df.iloc[indices].copy()
    results["similarity_score"] = scores[indices]
    results = results.reset_index(drop=True)

    logger.info(
        "Retrieved %d candidates for query='%s' (cuisine=%s)",
        len(results), query, cuisine,
    )
    return results
