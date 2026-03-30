"""
run_build_profiles.py
---------------------
Offline script: Build one searchable text profile per NYC restaurant.

Must be run AFTER run_preprocess.py.

Usage:
    python scripts/run_build_profiles.py

Input:
    data/interim/nyc_restaurants.csv
    data/raw/yelp_academic_dataset_review.json  (optional but recommended)

Output:
    data/processed/restaurant_profiles.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_profiles
from src.config import NYC_RESTAURANTS_CSV, REVIEW_JSON, PROFILES_CSV
from src.utils import get_logger

logger = get_logger("run_build_profiles")


def main():
    if not NYC_RESTAURANTS_CSV.exists():
        logger.error(
            "Interim file not found: %s\n"
            "Run scripts/run_preprocess.py first.",
            NYC_RESTAURANTS_CSV,
        )
        sys.exit(1)

    if not REVIEW_JSON.exists():
        logger.warning(
            "Review file not found at %s. Profiles will be built from metadata only.\n"
            "This is fine for testing, but results will be less rich.",
            REVIEW_JSON,
        )

    logger.info("=== Step 2: Building restaurant profiles ===")
    df = build_profiles.run(
        restaurants_csv=NYC_RESTAURANTS_CSV,
        review_json=REVIEW_JSON,
        output_csv=PROFILES_CSV,
    )
    logger.info("Done. %d profiles saved to %s", len(df), PROFILES_CSV)


if __name__ == "__main__":
    main()
