"""
test_similarity.py
------------------
Unit tests for src/similarity.py.

Run with:
    pytest tests/test_similarity.py -v

These tests verify the manual cosine similarity implementation against
known values and edge cases.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.similarity import cosine_similarity_one_to_many, top_k_indices


class TestCosineSimilarityOneToMany:
    """Tests for cosine_similarity_one_to_many."""

    def test_identical_vectors_return_one(self):
        """The similarity between a vector and itself should be 1.0."""
        query = np.array([1.0, 2.0, 3.0])
        matrix = np.array([[1.0, 2.0, 3.0]])
        scores = cosine_similarity_one_to_many(query, matrix)
        assert scores.shape == (1,)
        assert pytest.approx(scores[0], abs=1e-6) == 1.0

    def test_orthogonal_vectors_return_zero(self):
        """Orthogonal vectors have no shared direction — similarity should be 0."""
        query = np.array([1.0, 0.0])
        matrix = np.array([[0.0, 1.0]])
        scores = cosine_similarity_one_to_many(query, matrix)
        assert pytest.approx(scores[0], abs=1e-6) == 0.0

    def test_opposite_vectors_return_negative_one(self):
        """Vectors pointing in opposite directions should return -1.0."""
        query = np.array([1.0, 0.0])
        matrix = np.array([[-1.0, 0.0]])
        scores = cosine_similarity_one_to_many(query, matrix)
        assert pytest.approx(scores[0], abs=1e-6) == -1.0

    def test_scale_invariance(self):
        """Cosine similarity is scale-invariant: scaling a vector doesn't change the result."""
        query = np.array([1.0, 1.0])
        matrix = np.array([[2.0, 2.0], [3.0, 3.0]])
        scores = cosine_similarity_one_to_many(query, matrix)
        # All three vectors point in the same direction
        assert pytest.approx(scores[0], abs=1e-6) == 1.0
        assert pytest.approx(scores[1], abs=1e-6) == 1.0

    def test_multiple_rows_returns_correct_shape(self):
        """Output shape should match the number of rows in matrix."""
        query = np.random.rand(16)
        matrix = np.random.rand(50, 16)
        scores = cosine_similarity_one_to_many(query, matrix)
        assert scores.shape == (50,)

    def test_scores_are_in_valid_range(self):
        """Cosine similarity values should always be in [-1, 1]."""
        rng = np.random.default_rng(42)
        query = rng.standard_normal(64)
        matrix = rng.standard_normal((200, 64))
        scores = cosine_similarity_one_to_many(query, matrix)
        assert np.all(scores >= -1.0 - 1e-6)
        assert np.all(scores <= 1.0 + 1e-6)

    def test_known_value(self):
        """Verify against a hand-computed value."""
        # cos([1,1], [1,0]) = (1*1 + 1*0) / (sqrt(2) * 1) = 1/sqrt(2)
        query = np.array([1.0, 1.0])
        matrix = np.array([[1.0, 0.0]])
        scores = cosine_similarity_one_to_many(query, matrix)
        expected = 1.0 / np.sqrt(2)
        assert pytest.approx(scores[0], abs=1e-6) == expected

    def test_dimension_mismatch_raises(self):
        """Mismatched dimensions should raise a ValueError."""
        query = np.array([1.0, 2.0])
        matrix = np.array([[1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cosine_similarity_one_to_many(query, matrix)

    def test_wrong_query_shape_raises(self):
        """A 2-D query array should raise a ValueError."""
        query = np.array([[1.0, 2.0]])  # 2-D instead of 1-D
        matrix = np.array([[1.0, 2.0]])
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity_one_to_many(query, matrix)


class TestTopKIndices:
    """Tests for top_k_indices."""

    def test_returns_correct_top_k(self):
        """top_k_indices should return the k highest-scoring indices."""
        scores = np.array([0.1, 0.9, 0.4, 0.7, 0.3])
        indices = top_k_indices(scores, k=2)
        assert set(indices) == {1, 3}

    def test_sorted_descending(self):
        """Results should be sorted highest-to-lowest similarity."""
        scores = np.array([0.1, 0.9, 0.4, 0.7, 0.3])
        indices = top_k_indices(scores, k=3)
        # Indices in order: 1 (0.9), 3 (0.7), 2 (0.4)
        assert list(indices) == [1, 3, 2]

    def test_k_larger_than_array_returns_all(self):
        """Asking for more results than available should not raise an error."""
        scores = np.array([0.5, 0.8, 0.2])
        indices = top_k_indices(scores, k=10)
        assert len(indices) == 3
