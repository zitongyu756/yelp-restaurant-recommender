"""
run_preprocess.py
-----------------
Offline script: Filter the raw Yelp business JSON down to Philadelphia restaurants.

Usage:
    python scripts/run_preprocess.py

Input:
    data/raw/yelp_academic_dataset_business.json

Output:
    data/interim/philly_restaurants.csv
"""

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocess
from src.config import BUSINESS_JSON, PHILLY_RESTAURANTS_CSV
from src.utils import get_logger

logger = get_logger("run_preprocess")


def main():
    if not BUSINESS_JSON.exists():
        logger.error(
            "Raw business file not found: %s\n"
            "Download the Yelp Open Dataset and place the JSON files in data/raw/",
            BUSINESS_JSON,
        )
        sys.exit(1)

    logger.info("=== Step 1: Preprocessing ===")
    df = preprocess.run(
        business_json=BUSINESS_JSON,
        output_csv=PHILLY_RESTAURANTS_CSV,
    )
    logger.info("Done. %d Philadelphia restaurants saved to %s", len(df), PHILLY_RESTAURANTS_CSV)


if __name__ == "__main__":
    main()
