"""
utils.py
--------
Shared helpers used across multiple modules:
  - logging setup
  - loading / saving CSVs and NumPy arrays
  - small general-purpose utilities
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame.  Raises FileNotFoundError with a helpful message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n"
            "Run the preprocessing scripts first (see README)."
        )
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: Path | str) -> None:
    """Save a DataFrame to CSV, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_embeddings(path: Path | str) -> np.ndarray:
    """Load a NumPy .npy file containing embedding vectors."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {path}\n"
            "Run scripts/run_embed.py first."
        )
    return np.load(str(path))


def save_embeddings(embeddings: np.ndarray, path: Path | str) -> None:
    """Save a NumPy array to a .npy file, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), embeddings)
