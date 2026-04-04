"""
preprocess.py
-------------
Loads the raw Yelp business JSON and filters it down to NYC restaurants.

Input:  data/raw/yelp_academic_dataset_business.json
Output: data/interim/nyc_restaurants.csv

Columns kept in the output:
  business_id, name, city, state, neighborhood, address,
  stars, review_count, categories, price_range,
  attributes_wifi, attributes_noise_level, attributes_ambience
"""

import json
import pandas as pd
from pathlib import Path

from src.config import (
    BUSINESS_JSON,
    PHILLY_RESTAURANTS_CSV,
    TARGET_CITY_VALUES,
    TARGET_STATE_VALUE,
    RESTAURANT_CATEGORY_KEYWORDS,
    MIN_REVIEW_COUNT,
    REQUIRE_IS_OPEN,
)
from src.utils import get_logger, save_csv

logger = get_logger(__name__)


def load_business_json(path: Path) -> pd.DataFrame:
    """
    Read the Yelp business JSON file line by line.
    Each line is a separate JSON object (newline-delimited JSON).
    Returns a DataFrame with one row per business.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d total businesses from %s", len(records), path)
    return pd.DataFrame(records)


def is_target_area(city: str, state: str) -> bool:
    """Return True if the city and state match Philadelphia, PA."""
    return str(city).strip() in TARGET_CITY_VALUES and str(state).strip() == TARGET_STATE_VALUE


def is_restaurant(categories: str) -> bool:
    """Return True if any restaurant-related keyword appears in the categories string."""
    if not isinstance(categories, str):
        return False
    # Use case-insensitive matching for robustness
    cats_lower = categories.lower()
    return any(kw.lower() in cats_lower for kw in RESTAURANT_CATEGORY_KEYWORDS)


def extract_attributes(attributes: dict | None) -> dict:
    """
    Pull a small set of useful attributes from the nested Yelp attributes dict.
    Returns safe defaults (None) if an attribute is missing.
    """
    if not isinstance(attributes, dict):
        return {
            "attributes_wifi": None,
            "attributes_noise_level": None,
            "attributes_ambience": None,
        }

    # WiFi: values like 'free', 'paid', 'no', or None
    wifi = attributes.get("WiFi")

    # NoiseLevel: values like 'quiet', 'average', 'loud', 'very_loud', or None
    noise = attributes.get("NoiseLevel")

    # Ambience: stored as a stringified dict, e.g. "{'romantic': False, 'casual': True}"
    ambience = attributes.get("Ambience")

    return {
        "attributes_wifi": wifi,
        "attributes_noise_level": noise,
        "attributes_ambience": ambience,
    }


def filter_philly_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all filters and return a clean DataFrame of Philadelphia restaurants.

    Filters applied:
      1. City/State must match Philadelphia, PA
      2. Categories must contain a restaurant keyword
      3. review_count >= MIN_REVIEW_COUNT
      4. is_open == 1 (if REQUIRE_IS_OPEN is True)
      5. stars must be non-null
    """
    # Filter by city and state
    area_mask = df.apply(lambda x: is_target_area(x["city"], x["state"]), axis=1)
    df_philly = df[area_mask].copy()
    logger.info("After Philadelphia city/state filter: %d businesses", len(df_philly))

    # Filter by is_open
    if REQUIRE_IS_OPEN:
        df_philly = df_philly[df_philly["is_open"] == 1].copy()
        logger.info("After is_open filter: %d businesses", len(df_philly))

    # Filter by category
    rest_mask = df_philly["categories"].apply(is_restaurant)
    df_rest = df_philly[rest_mask].copy()
    logger.info("After restaurant category filter: %d businesses", len(df_rest))

    # Filter by minimum review count
    df_rest = df_rest[df_rest["review_count"] >= MIN_REVIEW_COUNT].copy()
    logger.info("After min review count filter (%d): %d businesses", MIN_REVIEW_COUNT, len(df_rest))

    # Drop rows with missing stars
    df_rest = df_rest.dropna(subset=["stars"]).copy()

    return df_rest


def build_output_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename columns; extract nested attributes into flat columns.
    Returns the final interim DataFrame ready to be saved.
    """
    # Extract attributes from the nested dict column
    attr_records = df["attributes"].apply(extract_attributes).tolist()
    attr_df = pd.DataFrame(attr_records, index=df.index)

    # Yelp stores price range under "RestaurantsPriceRange2" inside attributes
    def get_price(attributes):
        if isinstance(attributes, dict):
            return attributes.get("RestaurantsPriceRange2")
        return None

    df = df.copy()
    df["price_range"] = df["attributes"].apply(get_price)

    # Combine with extracted attribute columns
    df = pd.concat([df, attr_df], axis=1)

    # Keep only the columns we need downstream
    columns = [
        "business_id",
        "name",
        "city",
        "state",
        "address",
        "stars",
        "review_count",
        "is_open",
        "categories",
        "price_range",
        "attributes_wifi",
        "attributes_noise_level",
        "attributes_ambience",
    ]

    # "neighborhood" is not always present; add it safely
    if "neighborhood" in df.columns:
        columns.insert(4, "neighborhood")
    else:
        df["neighborhood"] = None
        columns.insert(4, "neighborhood")

    return df[columns].reset_index(drop=True)


def run(business_json: Path = BUSINESS_JSON, output_csv: Path = PHILLY_RESTAURANTS_CSV) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Reads raw Yelp business data and writes the Philadelphia restaurant CSV.
    Returns the resulting DataFrame.
    """
    logger.info("Starting preprocessing...")

    df_raw = load_business_json(business_json)
    df_filtered = filter_philly_restaurants(df_raw)
    df_output = build_output_dataframe(df_filtered)

    save_csv(df_output, output_csv)
    logger.info("Saved %d Philadelphia restaurants to %s", len(df_output), output_csv)

    return df_output


# TODO: Add additional filters if needed (e.g., only open businesses)
# TODO: Consider normalizing the 'categories' field into a list
