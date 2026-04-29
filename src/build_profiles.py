"""
build_profiles.py
-----------------
Builds a single searchable text "profile" for each Philadelphia restaurant
by combining structured metadata with sampled customer review text.

What goes into a profile:
  - Name and cuisine categories
  - Address and city (neighborhood is blank for most Philly entries)
  - Star rating and price tier
  - Selected attributes: WiFi, noise level, ambience (True keys only)
  - Up to MAX_REVIEWS_PER_RESTAURANT review snippets (first N in the CSV)

Input:
  data/processed/philly_restaurants.csv   (from run_preprocess.py)
  data/processed/philly_reviews.csv       (from run_preprocess.py)

Output:
  data/processed/restaurant_profiles.csv
  Columns: business_id, name, stars, review_count, price_range,
           city, address, categories, profile_text
"""

import ast
import regex as re
from pathlib import Path

import pandas as pd

from src.config import (
    PHILLY_RESTAURANTS_CSV,
    PHILLY_REVIEWS_CSV,
    PROFILES_CSV,
    MAX_REVIEWS_PER_RESTAURANT,
)
from src.utils import get_logger, load_csv, save_csv

logger = get_logger(__name__)

# Maximum characters to keep from each individual review snippet.
# Sentence-transformers truncate at ~256 tokens anyway; keeping snippets
# short prevents one verbose review from drowning out the others.
MAX_REVIEW_CHARS = 300


# ---------------------------------------------------------------------------
# Attribute cleaning helpers
# ---------------------------------------------------------------------------

def clean_attr_value(val) -> str:
    """
    Strip Python 2-style u'...' quoting that Yelp data uses for string attrs.

    Examples:
      u'free'   → 'free'
      u'quiet'  → 'quiet'
      'average' → 'average'
      None / nan → ''
    """
    if not val or str(val) in ("None", "nan"):
        return ""
    s = str(val).strip()
    # Remove leading u (Python 2 unicode marker) then strip quotes
    if s.startswith("u'") or s.startswith('u"'):
        s = s[1:]
    s = s.strip("'\"")
    return s.strip()


def parse_ambience(ambience_str) -> str:
    """
    Parse the Yelp ambience attribute (a stringified Python dict) and return
    a comma-separated string of the True keys.

    Example input:  "{'romantic': False, 'classy': True, 'casual': True}"
    Example output: "classy, casual"

    Returns an empty string if the input is missing, None, or unparseable.
    """
    if not ambience_str or str(ambience_str) in ("None", "nan"):
        return ""
    try:
        # Replace Python 2 u'' prefixes so ast.literal_eval can parse the dict
        cleaned = str(ambience_str).replace("u'", "'").replace('u"', '"')
        d = ast.literal_eval(cleaned)
        true_keys = [k for k, v in d.items() if v is True]
        return ", ".join(true_keys)
    except Exception:
        return ""


def format_price(price_range) -> str:
    """Convert Yelp price range (1–4) to a human-readable label."""
    mapping = {
        "1": "inexpensive",
        "2": "moderate",
        "3": "pricey",
        "4": "very expensive",
    }
    return mapping.get(str(price_range), "")


def extract_price_from_reviews(reviews: list[str]) -> float:
    """
    Analyze review text to determine an approximate price level, excluding negators.

    Args:
        reviews: List of review texts for a restaurant.

    Returns:
        A scalar representing the costliness (1.0 = inexpensive, 4.0 = very expensive),
        or 0.0 if no price information is found.
    """
    price_keywords = {
        1.0: ["cheap", "affordable", "budget"],
        2.0: ["moderate", "reasonable", "fair price"],
        3.0: ["expensive", "pricey", "costly"],
        4.0: ["very expensive", "overpriced", "luxury"]
    }

    negators = ["not", "n't", "no"]

    # Combine all reviews into one text block for analysis
    combined_reviews = " ".join(reviews).lower()

    category_counts = {level: 0 for level in price_keywords}

    for price_level, keywords in price_keywords.items():
        for keyword in keywords:
            # Match keywords but exclude those preceded by negators
            matches = re.findall(rf"(?<!\b(?:{'|'.join(negators)})\s)\b{keyword}\b", combined_reviews)
            category_counts[price_level] += len(matches)

    # Calculate weighted average of costliness based on counts
    total_mentions = sum(category_counts.values())
    if total_mentions == 0:
        return 0.0

    weighted_costliness = sum(level * count for level, count in category_counts.items()) / total_mentions
    return weighted_costliness


# ---------------------------------------------------------------------------
# Review loading
# ---------------------------------------------------------------------------

def load_reviews_for_businesses(
    reviews_csv: Path,
    business_ids: set,
    max_per_business: int,
) -> dict[str, list[str]]:
    """
    Load pre-filtered reviews from philly_reviews.csv and group them by
    business_id, keeping the first `max_per_business` entries per restaurant.

    We only read the two columns we need (business_id, text) to stay memory
    efficient — philly_reviews.csv has ~377k rows.

    Returns:
        { business_id: [review_text, review_text, ...] }
    """
    logger.info("Loading reviews from %s ...", reviews_csv)
    df = pd.read_csv(reviews_csv, usecols=["business_id", "text"])

    # Keep only reviews that belong to our set of restaurants
    df = df[df["business_id"].isin(business_ids)]
    df = df.dropna(subset=["text"])

    reviews: dict[str, list[str]] = {}
    for bid, group in df.groupby("business_id"):
        # Take the first max_per_business rows (CSV order = chronological from
        # run_preprocess's JOIN — deterministic and easy to debug)
        texts = group["text"].tolist()[:max_per_business]
        reviews[bid] = texts

    logger.info("Collected reviews for %d / %d businesses", len(reviews), len(business_ids))
    return reviews


# ---------------------------------------------------------------------------
# Profile text builder
# ---------------------------------------------------------------------------

def build_profile_text(row: pd.Series, review_texts: list[str]) -> str:
    """
    Construct a single rich text block for one restaurant.
    This text is what gets fed directly to the embedding model.

    Example output:
      "Joe's Steaks + Soda Shop. Cuisine: Cheesesteaks, American. Located at
       6030 Torresdale Ave, Philadelphia. Rating: 4.5 stars. Price: inexpensive.
       Noise level: average. WiFi: free. Ambience: casual, casual.
       Reviews: Best cheesesteak in the city hands down..."
    """
    parts = []

    # --- Name and cuisine ---
    name = str(row.get("name", "") or "")
    categories = str(row.get("categories", "") or "")
    if name:
        parts.append(f"{name}.")
    if categories and categories != "nan":
        parts.append(f"Cuisine: {categories}.")

    # --- Location: use address + city since neighborhood is mostly blank ---
    address = str(row.get("address", "") or "")
    city = str(row.get("city", "") or "")
    location_parts = [p for p in [address, city] if p and p != "nan"]
    if location_parts:
        parts.append(f"Located at {', '.join(location_parts)}.")

    # --- Rating ---
    stars = row.get("stars")
    if stars is not None:
        parts.append(f"Rating: {stars} stars.")

    # --- Price ---
    price_label = format_price(row.get("price_range"))
    if price_label:
        parts.append(f"Price: {price_label}.")

    # --- Attributes (cleaned) ---
    noise = clean_attr_value(row.get("attributes_noise_level"))
    if noise and noise not in ("no", "none"):
        parts.append(f"Noise level: {noise}.")

    wifi = clean_attr_value(row.get("attributes_wifi"))
    if wifi and wifi not in ("no", "none"):
        parts.append(f"WiFi: {wifi}.")

    ambience = parse_ambience(row.get("attributes_ambience"))
    if ambience:
        parts.append(f"Ambience: {ambience}.")

    # --- Sampled review snippets ---
    if review_texts:
        snippets = [t[:MAX_REVIEW_CHARS] for t in review_texts]
        combined = " ".join(snippets)
        parts.append(f"Reviews: {combined}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run(
    restaurants_csv: Path = PHILLY_RESTAURANTS_CSV,
    reviews_csv: Path = PHILLY_REVIEWS_CSV,
    output_csv: Path = PROFILES_CSV,
) -> pd.DataFrame:
    """
    Full profile-building pipeline.

    1. Load philly_restaurants.csv
    2. Load and group philly_reviews.csv by business_id
    3. For each restaurant, build a profile_text string
    4. Save restaurant_profiles.csv

    Returns the resulting DataFrame.
    """
    logger.info("Loading restaurants from %s", restaurants_csv)
    df = load_csv(restaurants_csv)
    logger.info("Building profiles for %d restaurants", len(df))

    business_ids = set(df["business_id"].tolist())

    # Load reviews (skip gracefully if file is missing — metadata-only mode)
    reviews_by_id: dict[str, list[str]] = {}
    if Path(reviews_csv).exists():
        reviews_by_id = load_reviews_for_businesses(
            reviews_csv, business_ids, MAX_REVIEWS_PER_RESTAURANT
        )
    else:
        logger.warning(
            "Reviews CSV not found at %s. Profiles will use metadata only.", reviews_csv
        )

    # Build one profile row per restaurant
    profile_rows = []
    for _, row in df.iterrows():
        bid = row["business_id"]
        review_texts = reviews_by_id.get(bid, [])
        profile_text = build_profile_text(row, review_texts)

        # Determine approximate price from reviews
        approx_price = extract_price_from_reviews(review_texts)
        price_range = approx_price if approx_price > 0 else row.get("price_range")

        profile_rows.append(
            {
                "business_id": bid,
                "name": row.get("name"),
                "stars": row.get("stars"),
                "review_count": row.get("review_count"),
                "price_range": price_range,
                "city": row.get("city"),
                "address": row.get("address"),
                "categories": row.get("categories"),
                "profile_text": profile_text,
            }
        )

    df_profiles = pd.DataFrame(profile_rows)
    save_csv(df_profiles, output_csv)
    logger.info("Saved %d restaurant profiles to %s", len(df_profiles), output_csv)

    return df_profiles
