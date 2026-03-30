"""
retrieve.py
-----------
Takes a natural-language query, embeds it, and retrieves the most
semantically similar restaurants from the precomputed embedding matrix.

Steps:
  1. Load the SentenceTransformer model (cached after first call).
  2. Embed the user query into a vector.
  3. Compute cosine similarity against all restaurant embeddings.
  4. Return the top-k most similar rows as a DataFrame.
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
from src.similarity import cosine_similarity_one_to_many, top_k_indices
from src.utils import get_logger, load_csv, load_embeddings

logger = get_logger(__name__)

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
    Encode a single query string into a 1-D embedding vector.

    Args:
        query: Natural language query, e.g. "quiet cafe near NYU".

    Returns:
        1-D float32 numpy array of shape (D,).
    """
    _load_resources()
    vector = _model.encode(query, convert_to_numpy=True)
    return vector.astype(np.float32)


def retrieve(query: str, top_k: int = TOP_K_RETRIEVE) -> pd.DataFrame:
    """
    Retrieve the top-k most relevant restaurants for a natural-language query.

    Args:
        query:  User's free-text query.
        top_k:  Number of candidates to return before reranking.

    Returns:
        DataFrame with the top-k most similar restaurants, sorted by
        cosine similarity descending.  Includes a 'similarity_score' column.
    """
    _load_resources()

    query_vec = embed_query(query)
    similarities = cosine_similarity_one_to_many(query_vec, _embeddings)
    indices = top_k_indices(similarities, top_k)

    results = _profiles_df.iloc[indices].copy()
    results["similarity_score"] = similarities[indices]
    results = results.reset_index(drop=True)

    logger.info("Retrieved %d candidates for query: '%s'", len(results), query)
    return results


# TODO: Add support for filtering by neighborhood or cuisine before retrieval
# TODO: Consider normalizing embeddings at build time to avoid repeated norm computation
