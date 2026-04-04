"""
config.py
---------
Single source of truth for all paths and project-wide constants.

Edit this file to change data locations, the embedding model, or NYC filter rules.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root directory (the repo root, one level above src/)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR       = ROOT_DIR / "data"
RAW_DIR        = DATA_DIR / "raw"
INTERIM_DIR    = DATA_DIR / "interim"
PROCESSED_DIR  = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Raw Yelp input files (downloaded manually, never committed)
# ---------------------------------------------------------------------------
BUSINESS_JSON  = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_JSON    = RAW_DIR / "yelp_academic_dataset_review.json"

# ---------------------------------------------------------------------------
# Intermediate / processed output files
# ---------------------------------------------------------------------------
PHILLY_RESTAURANTS_CSV = PROCESSED_DIR / "philly_restaurants.csv"
PHILLY_REVIEWS_CSV     = PROCESSED_DIR / "philly_reviews.csv"
PROFILES_CSV           = PROCESSED_DIR / "restaurant_profiles.csv"
EMBEDDINGS_NPY         = PROCESSED_DIR / "embeddings.npy"

# ---------------------------------------------------------------------------
# Target filtering settings (Philadelphia, PA)
# ---------------------------------------------------------------------------
# Yelp stores city as a free-text string. We accept the values below.
TARGET_CITY_VALUES = {
    "Philadelphia",
}
TARGET_STATE_VALUE = "PA"

# Yelp categories to treat as "restaurant / food" businesses
RESTAURANT_CATEGORY_KEYWORDS = [
    "Restaurants",
    "Food",
    "Bars",
    "Nightlife",
    "Coffee",
    "Cafes",
    "Bakery",
    "Diners",
]

# Minimum number of reviews a restaurant must have to be included
MIN_REVIEW_COUNT = 20 

# Must be currently open
REQUIRE_IS_OPEN = True

# Review recency filter
START_YEAR = 2019

# ---------------------------------------------------------------------------
# Profile builder settings
# ---------------------------------------------------------------------------
# How many review texts to sample per restaurant when building the profile
MAX_REVIEWS_PER_RESTAURANT = 5

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2 is fast, small, and works well for sentence similarity.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
TOP_K_RETRIEVE = 20   # how many candidates to fetch before reranking
TOP_K_DISPLAY  = 5    # how many results to show in the UI
