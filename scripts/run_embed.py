"""
run_embed.py
------------
Offline script: Embed all restaurant profiles using sentence-transformers.

Must be run AFTER run_build_profiles.py.

Usage:
    python scripts/run_embed.py

Input:
    data/processed/restaurant_profiles.csv

Output:
    data/processed/embeddings.npy   (float32, shape [N, embedding_dim])

Note:
    The first run downloads the model (~80 MB) to the local cache.
    Subsequent runs reuse the cached model.
    On CPU this may take several minutes for large datasets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embed
from src.config import PROFILES_CSV, EMBEDDINGS_NPY
from src.utils import get_logger

logger = get_logger("run_embed")


def main():
    if not PROFILES_CSV.exists():
        logger.error(
            "Profiles file not found: %s\n"
            "Run scripts/run_build_profiles.py first.",
            PROFILES_CSV,
        )
        sys.exit(1)

    logger.info("=== Step 3: Embedding restaurant profiles ===")
    embeddings = embed.run(
        profiles_csv=PROFILES_CSV,
        output_npy=EMBEDDINGS_NPY,
    )
    logger.info(
        "Done. Embeddings shape: %s saved to %s",
        str(embeddings.shape),
        EMBEDDINGS_NPY,
    )


if __name__ == "__main__":
    main()
