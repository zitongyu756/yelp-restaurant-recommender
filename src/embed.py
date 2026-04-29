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

    partial_arrays = []
    for idx, chunk in enumerate(chunks):
        logger.info("  chunk %d/%d (%d texts)...", idx + 1, len(chunks), len(chunk))
        chunk_emb = model.encode(
            chunk,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64,
        ).astype(np.float32)

        # save partial so a crash mid-run doesn't lose all progress
        if output_npy is not None:
            partial_path = Path(output_npy).with_name(
                Path(output_npy).stem + f"_chunk_{idx}.npy"
            )
            np.save(str(partial_path), chunk_emb)
            logger.info("  saved partial: %s", partial_path)

        partial_arrays.append(chunk_emb)

    embeddings = np.concatenate(partial_arrays, axis=0)

    # clean up partial files now that the full array is assembled
    if output_npy is not None:
        for idx in range(len(chunks)):
            partial_path = Path(output_npy).with_name(
                Path(output_npy).stem + f"_chunk_{idx}.npy"
            )
            if partial_path.exists():
                partial_path.unlink()
        logger.info("Cleaned up %d partial chunk files", len(chunks))

    return embeddings

def run(
    profiles_csv: Path = PROFILES_CSV,
    output_npy: Path = EMBEDDINGS_NPY,
    chunk_size: int = 5000,
) -> np.ndarray:
    """
    Full embedding pipeline.
    Loads profiles, embeds them, and saves the result.
    Returns the embedding matrix.

    Uses chunked embedding when the dataset exceeds chunk_size rows, so a
    very large dataset won't OOM and partial progress is saved to disk.
    Set chunk_size smaller on machines with limited RAM.
    """
    profiles_csv = Path(profiles_csv)
    output_npy = Path(output_npy)

    df = load_csv(profiles_csv)
    texts = df["profile_text"].fillna("").tolist()

    model = load_model()
    embeddings = embed_texts(model, texts)

     # use chunked path for large datasets, simple path for small ones
    if len(texts) > chunk_size:
        embeddings = embed_texts_chunked(
            model, texts, chunk_size=chunk_size, output_npy=output_npy
        )
    else:
        embeddings = embed_texts(model, texts)
    
    save_embeddings(embeddings, output_npy)
    logger.info(
        "Saved embeddings %s to %s", str(embeddings.shape), output_npy
    )

    return embeddings

