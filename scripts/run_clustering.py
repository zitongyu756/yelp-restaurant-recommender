"""
run_clustering.py
-----------------
Clusters the restaurant embeddings using our custom K-means implementation
from scratch. The resulting cluster assignments are added to the profiles CSV.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROFILES_CSV, EMBEDDINGS_NPY
from src.kmeans import kmeans
from src.utils import get_logger, load_csv, load_embeddings, save_csv

logger = get_logger(__name__)

NUM_CLUSTERS = 15

def run():
    logger.info("Loading profiles and embeddings...")
    df = load_csv(PROFILES_CSV)
    embeddings = load_embeddings(EMBEDDINGS_NPY)

    if len(df) != embeddings.shape[0]:
        logger.error("Mismatch between profiles and embeddings count.")
        return

    logger.info(f"Running K-means with k={NUM_CLUSTERS}...")
    centroids, labels = kmeans(embeddings, k=NUM_CLUSTERS, max_iters=200)
    
    logger.info("Clustering completed. Assigning labels...")
    df["cluster_id"] = labels
    
    save_csv(df, PROFILES_CSV)
    logger.info("Saved updated profiles with cluster IDs.")

if __name__ == "__main__":
    run()
