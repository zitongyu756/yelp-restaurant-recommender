"""
join_reviews.py
---------------
Streaming script to join Philadelphia restaurant data with the huge review dataset.

Input:
    data/interim/philly_restaurants.csv
    data/raw/yelp_academic_dataset_review.json (5.3GB)

Output:
    data/interim/philly_reviews.csv
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Ensure the repo root is on sys.path so src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BUSINESS_CSV, PHILLY_REVIEWS_CSV
from src.utils import get_logger

logger = get_logger("join_reviews")

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
# BUSINESS_CSV is handled in main below, but I'll use THE ONE FROM CONFIG
from src.config import PHILLY_RESTAURANTS_CSV as BIZ_CSV
REVIEW_JSON = ROOT_DIR / "data" / "raw" / "yelp_academic_dataset_review.json"
OUTPUT_CSV = PHILLY_REVIEWS_CSV

def main():
    if not BUSINESS_CSV.exists():
        logger.error("Business CSV not found: %s", BUSINESS_CSV)
        sys.exit(1)
    if not REVIEW_JSON.exists():
        logger.error("Review JSON not found: %s", REVIEW_JSON)
        sys.exit(1)

    logger.info("Loading filtered business IDs...")
    biz_df = pd.read_csv(BUSINESS_CSV)
    # Keep only relevant columns for joining
    biz_subset = biz_df[['business_id', 'name', 'city', 'state']]
    valid_biz_ids = set(biz_subset['business_id'].unique())
    logger.info("Total Philadelphia businesses to match: %d", len(valid_biz_ids))

    from src.config import START_YEAR
    logger.info("Starting year filter: %d", START_YEAR)

    logger.info("Streaming review dataset and filtering (this may take a while)...")
    matched_reviews = []
    count_total = 0
    count_matched = 0

    with open(REVIEW_JSON, "r", encoding="utf-8") as f:
        for line in f:
            count_total += 1
            if count_total % 500000 == 0:
                logger.info("Processed %d reviews... Matched %d so far.", count_total, count_matched)
            
            review = json.loads(line)
            # Filter by business ID and recency
            review_date = review.get('date', '')
            if review['business_id'] in valid_biz_ids and review_date >= f"{START_YEAR}-01-01":
                matched_reviews.append(review)
                count_matched += 1

    logger.info("Finished filtering. Total matched: %d / %d", count_matched, count_total)

    logger.info("Joining with business metadata...")
    reviews_df = pd.DataFrame(matched_reviews)
    final_df = reviews_df.merge(biz_subset, on='business_id', how='left')

    logger.info("Saving to %s...", OUTPUT_CSV)
    final_df.to_csv(OUTPUT_CSV, index=False)
    logger.info("Done! Joined dataset saved successfully.")

if __name__ == "__main__":
    main()
