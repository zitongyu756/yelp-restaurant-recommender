"""
embed.py
--------
Encodes restaurant profile texts into dense embedding vectors using a
pretrained sentence-transformer model.

Input:  data/processed/restaurant_profiles.csv  (profile_text column)
Output: data/processed/embeddings.npy           (float32 array, shape [N, D])

The row order in embeddings.npy matches the row order in restaurant_profiles.csv.
Always load both files together so indices stay aligned.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import (
    PROFILES_CSV,
    EMBEDDINGS_NPY,
    EMBEDDING_MODEL_NAME,
)
from src.utils import get_logger, load_csv, save_embeddings

logger = get_logger(__name__)


def load_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformer model (downloads on first call)."""
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Encode a list of strings into a 2-D float32 NumPy array of L2-normalized
    row vectors.

    Args:
        model:  A loaded SentenceTransformer model.
        texts:  List of strings to embed.

    Returns:
        embeddings: shape (len(texts), embedding_dim), dtype float32, rows
        have unit L2 norm so cosine similarity reduces to a dot product at
        query time.
    """
    logger.info("Embedding %d texts...", len(texts))
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=64,
    ).astype(np.float32)

    # L2-normalize rows so downstream retrieval can skip the per-query norm.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    return embeddings / norms


def run(
    profiles_csv: Path = PROFILES_CSV,
    output_npy: Path = EMBEDDINGS_NPY,
) -> np.ndarray:
    """
    Full embedding pipeline.
    Loads profiles, embeds them, and saves the result.
    Returns the embedding matrix.
    """
    profiles_csv = Path(profiles_csv)
    output_npy = Path(output_npy)

    df = load_csv(profiles_csv)
    texts = df["profile_text"].fillna("").tolist()

    model = load_model()
    embeddings = embed_texts(model, texts)

    save_embeddings(embeddings, output_npy)
    logger.info(
        "Saved embeddings %s to %s", str(embeddings.shape), output_npy
    )

    return embeddings


# TODO: If the dataset is very large, consider batching into chunks and saving
#       partial .npy files to avoid running out of memory.
