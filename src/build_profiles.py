"""
build_profiles.py
-----------------
Builds a single searchable text "profile" for each NYC restaurant by combining:
  - Structured metadata: name, cuisine, city, neighborhood, price, rating
  - Selected attributes: WiFi, noise level, ambience
  - Sampled review text from real customers

Input:
  data/interim/nyc_restaurants.csv
  data/raw/yelp_academic_dataset_review.json (streamed line by line)

Output:
  data/processed/restaurant_profiles.csv
  Columns: business_id, name, stars, review_count, price_range, profile_text
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.config import (
    NYC_RESTAURANTS_CSV,
    REVIEW_JSON,
    PROFILES_CSV,
    MAX_REVIEWS_PER_RESTAURANT,
)
from src.utils import get_logger, load_csv, save_csv

logger = get_logger(__name__)


def load_reviews_for_businesses(
    review_json: Path, business_ids: set, max_per_business: int
) -> dict[str, list[str]]:
    """
    Stream through the Yelp review JSON and collect up to `max_per_business`
    review texts for each business_id in `business_ids`.

    Streaming avoids loading the full ~5 GB review file into memory at once.

    Returns a dict: { business_id: [review_text, ...] }
    """
    reviews: dict[str, list[str]] = defaultdict(list)

    logger.info("Streaming reviews from %s ...", review_json)
    with open(review_json, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            bid = record.get("business_id")
            text = record.get("text", "").strip()

            if bid in business_ids and len(reviews[bid]) < max_per_business:
                reviews[bid].append(text)

            if i % 500_000 == 0 and i > 0:
                logger.info("  Processed %d review lines so far...", i)

    logger.info("Collected reviews for %d businesses", len(reviews))
    return dict(reviews)


def format_price(price_range) -> str:
    """Convert Yelp price range (1-4) to a readable label."""
    mapping = {"1": "inexpensive", "2": "moderate", "3": "pricey", "4": "very expensive"}
    return mapping.get(str(price_range), "unknown price")


def build_profile_text(row: pd.Series, review_texts: list[str]) -> str:
    """
    Construct a single text block for one restaurant.
    This text is what gets embedded — make it rich but concise.

    Example output:
      "Joe's Pizza. Cuisine: Pizza, Italian. Located in Manhattan, New York.
       Rating: 4.5 stars. Price: inexpensive. Noise: average. WiFi: free.
       Reviews: Best slice in the city. Classic NY pizza, always fresh..."
    """
    parts = []

    # Name and cuisine
    name = str(row.get("name", ""))
    categories = str(row.get("categories", ""))
    parts.append(f"{name}.")
    if categories:
        parts.append(f"Cuisine: {categories}.")

    # Location
    neighborhood = str(row.get("neighborhood", "") or "")
    city = str(row.get("city", "") or "")
    location_parts = [p for p in [neighborhood, city] if p]
    if location_parts:
        parts.append(f"Located in {', '.join(location_parts)}.")

    # Rating and price
    stars = row.get("stars")
    if stars is not None:
        parts.append(f"Rating: {stars} stars.")

    price_label = format_price(row.get("price_range"))
    parts.append(f"Price: {price_label}.")

    # Attributes
    noise = row.get("attributes_noise_level")
    if noise and str(noise) not in ("None", "nan"):
        parts.append(f"Noise level: {noise}.")

    wifi = row.get("attributes_wifi")
    if wifi and str(wifi) not in ("None", "nan", "'no'", "no"):
        parts.append(f"WiFi: {wifi}.")

    ambience = row.get("attributes_ambience")
    if ambience and str(ambience) not in ("None", "nan"):
        # Ambience is a stringified dict; include it as-is for now
        parts.append(f"Ambience: {ambience}.")

    # Sampled review text
    if review_texts:
        combined_reviews = " ".join(review_texts)
        parts.append(f"Reviews: {combined_reviews}")

    return " ".join(parts)


def run(
    restaurants_csv: Path = NYC_RESTAURANTS_CSV,
    review_json: Path = REVIEW_JSON,
    output_csv: Path = PROFILES_CSV,
) -> pd.DataFrame:
    """
    Full profile-building pipeline.
    Reads the NYC restaurant CSV + Yelp reviews, builds profiles, writes output CSV.
    """
    logger.info("Loading NYC restaurants from %s", restaurants_csv)
    df = load_csv(restaurants_csv)
    logger.info("Building profiles for %d restaurants", len(df))

    business_ids = set(df["business_id"].tolist())

    # Load reviews only if the review file exists (skip gracefully in dev)
    reviews_by_id: dict[str, list[str]] = {}
    if review_json.exists():
        reviews_by_id = load_reviews_for_businesses(
            review_json, business_ids, MAX_REVIEWS_PER_RESTAURANT
        )
    else:
        logger.warning(
            "Review file not found at %s. Profiles will be built from metadata only.", review_json
        )

    # Build a profile text for each restaurant
    profile_rows = []
    for _, row in df.iterrows():
        bid = row["business_id"]
        review_texts = reviews_by_id.get(bid, [])
        profile_text = build_profile_text(row, review_texts)
        profile_rows.append(
            {
                "business_id": bid,
                "name": row.get("name"),
                "stars": row.get("stars"),
                "review_count": row.get("review_count"),
                "price_range": row.get("price_range"),
                "city": row.get("city"),
                "neighborhood": row.get("neighborhood"),
                "categories": row.get("categories"),
                "profile_text": profile_text,
            }
        )

    df_profiles = pd.DataFrame(profile_rows)
    save_csv(df_profiles, output_csv)
    logger.info("Saved %d restaurant profiles to %s", len(df_profiles), output_csv)

    return df_profiles


# TODO: Experiment with different amounts of review text to include
# TODO: Consider deduplicating near-identical review snippets
