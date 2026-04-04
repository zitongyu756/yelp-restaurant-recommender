"""
similarity.py
-------------
Manual cosine similarity implementation using only NumPy.

Cosine similarity between two vectors a and b is:
    cos(a, b) = (a · b) / (||a|| * ||b||)

A value of 1.0 means identical direction (most similar).
A value of 0.0 means orthogonal (no similarity).

We implement a vectorized version that computes the similarity between
one query vector and an entire matrix of candidate vectors at once.
"""

import numpy as np


def cosine_similarity_one_to_many(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and every row
    in a 2-D matrix.

    Args:
        query:  1-D array of shape (D,) — the embedded user query.
        matrix: 2-D array of shape (N, D) — the embedded restaurant profiles.

    Returns:
        similarities: 1-D array of shape (N,), values in [-1, 1].
                      Higher means more similar.

    Raises:
        ValueError: if query or matrix have unexpected shapes.
    """
    if query.ndim != 1:
        raise ValueError(f"query must be 1-D, got shape {query.shape}")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {matrix.shape}")
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Dimension mismatch: query has {query.shape[0]} dims, "
            f"matrix has {matrix.shape[1]} dims"
        )

    # Compute dot products between query and every row
    dot_products = matrix @ query                      # shape (N,)

    # Compute L2 norms
    query_norm  = np.linalg.norm(query)                # scalar
    matrix_norms = np.linalg.norm(matrix, axis=1)     # shape (N,)

    # Avoid division by zero for zero-norm vectors
    denominator = matrix_norms * query_norm
    denominator = np.where(denominator == 0, 1e-10, denominator)

    similarities = dot_products / denominator
    return similarities


def top_k_indices(similarities: np.ndarray, k: int) -> np.ndarray:
    """
    Return the indices of the top-k highest similarity scores.

    Args:
        similarities: 1-D array of similarity scores.
        k:            Number of top results to return.

    Returns:
        indices: 1-D array of length min(k, len(similarities)),
                 sorted by descending similarity.
    """
    k = min(k, len(similarities))
    # argpartition is faster than full sort for large arrays,
    # but we sort the top-k afterwards to get a proper ranking.
    top_indices = np.argpartition(similarities, -k)[-k:]
    # Sort these k indices by their similarity value (descending)
    sorted_top = top_indices[np.argsort(similarities[top_indices])[::-1]]
    return sorted_top
