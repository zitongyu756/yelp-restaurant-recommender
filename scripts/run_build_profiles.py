"""
run_build_profiles.py
---------------------
Offline script: build one searchable text profile per Philadelphia restaurant.

Must be run AFTER run_preprocess.py (which produces philly_restaurants.csv
and philly_reviews.csv).

Usage:
    python scripts/run_build_profiles.py

Input:
    data/processed/philly_restaurants.csv
    data/processed/philly_reviews.csv

Output:
    data/processed/restaurant_profiles.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import build_profiles
from src.config import PHILLY_RESTAURANTS_CSV, PHILLY_REVIEWS_CSV, PROFILES_CSV
from src.utils import get_logger

logger = get_logger("run_build_profiles")


def main():
    if not PHILLY_RESTAURANTS_CSV.exists():
        logger.error(
            "Restaurants file not found: %s\n"
            "Run scripts/run_preprocess.py first.",
            PHILLY_RESTAURANTS_CSV,
        )
        sys.exit(1)

    if not PHILLY_REVIEWS_CSV.exists():
        logger.warning(
            "Reviews file not found at %s.\n"
            "Profiles will be built from metadata only (less rich results).",
            PHILLY_REVIEWS_CSV,
        )

    logger.info("=== Step 2: Building restaurant profiles ===")
    df = build_profiles.run(
        restaurants_csv=PHILLY_RESTAURANTS_CSV,
        reviews_csv=PHILLY_REVIEWS_CSV,
        output_csv=PROFILES_CSV,
    )
    logger.info("Done. %d profiles saved to %s", len(df), PROFILES_CSV)


if __name__ == "__main__":
    main()
