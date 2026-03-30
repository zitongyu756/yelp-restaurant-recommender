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
NYC_RESTAURANTS_CSV    = INTERIM_DIR / "nyc_restaurants.csv"
PROFILES_CSV           = PROCESSED_DIR / "restaurant_profiles.csv"
EMBEDDINGS_NPY         = PROCESSED_DIR / "embeddings.npy"

# ---------------------------------------------------------------------------
# NYC filter settings
# ---------------------------------------------------------------------------
# Yelp stores city as a free-text string.  We accept any of the values below.
NYC_CITY_VALUES = {
    "New York",
    "New York City",
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
    "The Bronx",
}

# Yelp categories to treat as "restaurant / food" businesses
RESTAURANT_CATEGORY_KEYWORDS = [
    "Restaurants",
    "Food",
    "Bars",
    "Nightlife",
    "Coffee",
    "Cafes",
]

# Minimum number of reviews a restaurant must have to be included
MIN_REVIEW_COUNT = 10

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
