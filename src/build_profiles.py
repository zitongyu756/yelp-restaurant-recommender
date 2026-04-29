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
from collections import Counter

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
    Estimate price tier (1–4) from review text using a broad natural-language vocabulary.

    The original keyword list was too narrow (people rarely write "fair price" or
    "moderate" verbatim). This version matches the phrases reviewers actually use,
    e.g. "great value", "steep", "wallet-friendly", "splurge", "worth every penny".

    Negation is handled by checking the 3 words preceding each match.
    Requires at least 3 signal hits for a confident estimate.

    Args:
        reviews: List of review texts for a restaurant.

    Returns:
        Estimated price tier as a float (1.0–4.0), or 0.0 if signal is too weak.
    """
    # Phrases people actually write in Yelp reviews, grouped by price tier
    PRICE_SIGNALS: dict[float, list[str]] = {
        1.0: [
            "cheap", "cheapest", "so cheap", "super cheap", "dirt cheap",
            "very cheap", "inexpensive", "affordable", "budget",
            "wallet.friendly", "pocket.friendly", "easy on the wallet",
            "easy on the pocket", "great value", "good value", "excellent value",
            "amazing value", "best value", "value for money", "worth every penny",
            "won't break the bank", "doesn't break the bank", "bargain",
            "low price", "low prices", "low cost", "price is right",
            "prices are right", "price can't be beat", "can't beat the price",
        ],
        2.0: [
            "moderate", "moderately priced", "mid.range", "midrange",
            "middle of the road", "fair price", "fair prices",
            "decent price", "decent prices", "reasonable", "reasonably priced",
            "average price", "average prices", "normal price", "standard price",
            "not too expensive", "not too pricey", "not too cheap",
        ],
        3.0: [
            "pricey", "pricy", "expensive", "costly", "steep", "steeply priced",
            "a bit pricey", "a bit expensive", "a bit steep", "kind of pricey",
            "on the pricier side", "on the expensive side", "runs a bit high",
            "pay a premium", "premium price", "premium prices",
            "high price", "high prices", "high end", "not cheap",
            "not inexpensive", "overpriced", "overprice",
        ],
        4.0: [
            "very expensive", "super expensive", "extremely expensive",
            "very pricey", "very steep", "way too expensive", "way overpriced",
            "luxury", "luxurious", "upscale", "fine dining", "splurge",
            "special occasion", "white tablecloth", "michelin", "tasting menu",
            "once in a lifetime", "celebratory meal", "high end restaurant",
        ],
    }

    NEGATORS = {"not", "no", "never", "without", "n't", "hardly", "barely"}

    combined = " ".join(reviews).lower()
    category_counts: dict[float, int] = {level: 0 for level in PRICE_SIGNALS}

    for price_level, phrases in PRICE_SIGNALS.items():
        for phrase in phrases:
            # Convert '.' wildcard shorthand (e.g. wallet.friendly) to \w* regex
            wildcard = phrase.replace(".", r"\w*")
            pattern = rf"\b{wildcard}\b"
            for match in re.finditer(pattern, combined):
                # Check the 4 words immediately before the match for negators
                preceding_text = combined[max(0, match.start() - 35) : match.start()]
                preceding_words = preceding_text.split()
                is_negated = bool(NEGATORS.intersection(preceding_words[-4:]))
                if not is_negated:
                    category_counts[price_level] += 1

    total_mentions = sum(category_counts.values())
    # Require at least 2 signal hits to avoid noise from a single accidental word
    if total_mentions < 2:
        return 0.0

    weighted_costliness = (
        sum(level * count for level, count in category_counts.items()) / total_mentions
    )
    return round(weighted_costliness, 2)


def extract_review_keywords(reviews: list[str], top_n: int = 15) -> str:
    """
    Extract the most frequent meaningful words from a large pool of reviews.
    This concentrates the overall sentiment into a dense list of keywords
    that the embedding model can easily pick up on.
    """
    if not reviews:
        return ""
        
    combined = " ".join(reviews).lower()
    # Extract words between 4 and 12 characters (filters out a/the/and automatically)
    words = re.findall(r"\b[a-z]{4,12}\b", combined)
    
    # Common words that don't help with recommendation semantics
    stopwords = {
        "this", "that", "with", "from", "they", "have", "were", "there", 
        "which", "their", "what", "when", "about", "would", "could", 
        "should", "been", "some", "very", "just", "like", "good", "great", 
        "food", "place", "restaurant", "really", "back", "also", "only", 
        "even", "well", "much", "time", "here", "because", "other", "them", 
        "than", "more", "then", "make", "made", "best", "always", "will",
        "never", "ever", "came", "went", "come", "go", "going", "got",
        "get", "getting", "one", "two", "three", "first", "next", "last",
        "definitely", "highly", "recommend", "recommended", "ordered",
        "order", "menu", "service", "staff", "friendly", "nice", "love",
        "loved", "amazing", "awesome", "excellent", "delicious", "tasty",
        "super", "pretty", "little", "right", "sure", "thing", "think",
        "thought", "know", "knew", "say", "said", "tell", "told", "people",
        "us", "we", "our", "you", "your", "my", "me", "i", "am", "is",
        "are", "was", "be", "do", "did", "done", "can", "cannot", "can't",
        "won't", "don't", "didn't", "has", "had"
    }
    
    filtered = [w for w in words if w not in stopwords]
    most_common = [word for word, count in Counter(filtered).most_common(top_n)]
    
    return ", ".join(most_common)


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

    # --- Concentrated Review Keywords ---
    keywords = extract_review_keywords(review_texts)
    if keywords:
        parts.append(f"Popular Keywords: {keywords}.")

    # --- Sampled review snippets ---
    if review_texts:
        # Only take the first 3 reviews for full snippets to avoid token limits
        snippets = [t[:MAX_REVIEW_CHARS] for t in review_texts[:3]]
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

        # Fill in missing Yelp price using review-text estimation.
        # Only apply when Yelp has no price data — never override a valid Yelp value,
        # since Yelp's own data is more reliable than keyword counting.
        yelp_price = row.get("price_range")
        if pd.isna(yelp_price):
            approx_price = extract_price_from_reviews(review_texts)
            price_range = approx_price if approx_price > 0 else yelp_price
        else:
            price_range = yelp_price

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
