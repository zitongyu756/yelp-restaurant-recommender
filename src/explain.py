"""
explain.py
----------
Generates short, human-readable "why recommended" blurbs for each result.

This is a simple rule-based approach — no LLM required.
The explanation is constructed by comparing keywords in the user query to
structured fields in the restaurant row, plus an extractive summary of the
review snippets already stored in profile_text.

Example output:
  "Serves Pizza, Italian. Located in Philadelphia. Rated 4.5 ★ (312 reviews).
   Price: $$.  Reviewers say: 'Best cheesesteak in the city — crispy roll,
   perfectly seasoned meat. A Philly institution.'"
"""

import re
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

# Keywords that signal certain user preferences
PRICE_KEYWORDS    = {"cheap", "inexpensive", "budget", "affordable", "low cost"}
QUIET_KEYWORDS    = {"quiet", "calm", "peaceful", "study", "work"}
ROMANTIC_KEYWORDS = {"romantic", "date", "date night", "intimate", "cozy"}
LATE_KEYWORDS     = {"late", "late night", "midnight", "after midnight"}

PRICE_LABEL = {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}

# Human-readable names for the K-Means clusters (0-14)
CLUSTER_NAMES = {
    0: "Asian Fusion & Chinese",
    1: "Classic American & Pubs",
    2: "Fast Casual & Grocery",
    3: "Bakeries & Desserts",
    4: "Cafes & Coffee",
    5: "Sandwiches & Cheesesteaks",
    6: "Indian & Middle Eastern",
    7: "Mexican & Latin",
    8: "Japanese & Sushi",
    9: "Pizza & Casual Italian",
    10: "Italian & Specialty Food",
    11: "Delis & Brunch",
    12: "Lounges & Entertainment",
    13: "Seafood & Soul Food",
    14: "Brunch & Modern American"
}

# Generic filler phrases that add no information value — sentences containing
# these are penalised during extractive scoring.
_BOILERPLATE = re.compile(
    r"\b(great place|good food|great food|good service|great service|"
    r"highly recommend|will (definitely |)be back|can'?t go wrong|"
    r"must (try|visit)|worth (a |the )?visit|love this place|"
    r"amazing place|wonderful place|fantastic place|nice place|"
    r"best (place|restaurant) (ever|in the city)|always (a |)good)\b",
    re.IGNORECASE,
)


def _extract_query_keywords(query: str) -> set[str]:
    """Lowercase and tokenize the query into a set of words."""
    return set(re.findall(r"\b\w+\b", query.lower()))


def _price_label(price_range) -> str:
    """Convert a price_range value (int, float, or str) to a $ label.

    Yelp stores price as 1–4. The CSV may hold floats (e.g. 2.0, 2.5)
    because of averaged aggregations, so we round to the nearest integer.
    Returns an empty string for missing / un-parseable values.
    """
    try:
        tier = str(round(float(price_range)))
        return PRICE_LABEL.get(tier, "")
    except (ValueError, TypeError):
        return ""


def _score_sentence(sentence: str, query_words: set[str]) -> float:
    """
    Score a single sentence for how useful it is as a review highlight.

    Higher is better. Scoring factors:
      + Length in an ideal range (30–120 chars): specific but not rambling
      + High unique-word ratio: diverse vocabulary → more informative
      + Contains query-related words: directly relevant to what user asked
      - Starts with "I " (often too personal / anecdotal)
      - Matches boilerplate patterns (generic praise)
    """
    s = sentence.strip()
    length = len(s)
    if length < 20:
        return -1.0  # Too short to be useful

    score = 0.0

    # Ideal length bonus
    if 30 <= length <= 120:
        score += 2.0
    elif length <= 200:
        score += 1.0

    # Unique word ratio — proxy for information density
    words = re.findall(r"\b\w+\b", s.lower())
    if words:
        unique_ratio = len(set(words)) / len(words)
        score += unique_ratio * 2.0

    # Query relevance bonus
    sentence_words = set(re.findall(r"\b\w+\b", s.lower()))
    overlap = sentence_words & query_words
    score += len(overlap) * 0.5

    # Penalties
    if s.lower().startswith("i "):
        score -= 0.5
    if _BOILERPLATE.search(s):
        score -= 2.0

    return score


def _extract_review_highlight(profile_text: str, query: str, max_chars: int = 160) -> str:
    """
    Extract the 1–2 best sentences from the review portion of profile_text
    as a short, representative highlight.

    Args:
        profile_text: The full profile_text string from the profiles CSV.
                      Review snippets begin after the "Reviews: " marker.
        query:        The user's search query (used to boost relevant sentences).
        max_chars:    Maximum total length of the returned highlight string.

    Returns:
        A quoted highlight string like:
          '"Crispy roll, perfectly seasoned meat. Best cheesesteak in Philly."'
        or an empty string if no usable review text is found.
    """
    if not profile_text or str(profile_text) in ("nan", "None", ""):
        return ""

    # Extract the review portion (everything after "Reviews: ")
    match = re.search(r"Reviews:\s*(.+)$", str(profile_text), re.DOTALL)
    if not match:
        return ""

    review_block = match.group(1).strip()

    # Split into sentences on . ! ? boundaries
    raw_sentences = re.split(r"(?<=[.!?])\s+", review_block)

    query_words = _extract_query_keywords(query)

    # Score all sentences and pick the best ones that fit within max_chars
    scored = [
        (sent.strip(), _score_sentence(sent, query_words))
        for sent in raw_sentences
        if len(sent.strip()) >= 20
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected: list[str] = []
    total_len = 0
    for sent, score in scored:
        if score < 0:
            break
        if total_len + len(sent) + 1 > max_chars:
            break
        selected.append(sent)
        total_len += len(sent) + 1
        if len(selected) == 2:
            break

    if not selected:
        return ""

    highlight = " ".join(selected)
    
    # Bold the user's query keywords in the highlight
    for word in query_words:
        if len(word) > 3:  # Only bold meaningful words
            pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
            highlight = pattern.sub(r"**\1**", highlight)
            
    # Ensure it ends with punctuation
    if highlight and highlight[-1] not in ".!?*\"":
        highlight += "."
    return f'"{highlight}"'


def generate_explanation(row: pd.Series, query: str) -> str:
    """
    Build a short explanation sentence for one restaurant result.

    Args:
        row:   A single row from the reranked results DataFrame.
        query: The original user query string.

    Returns:
        A human-readable string explaining why this restaurant was recommended.
    """
    parts = []
    query_words = _extract_query_keywords(query)

    # --- Cuisine / category match ---
    categories = str(row.get("categories", "") or "")
    if categories:
        # Show first two category tags
        top_cats = [c.strip() for c in categories.split(",")][:2]
        parts.append(f"Serves {', '.join(top_cats)}.")

    # --- Location ---
    neighborhood = str(row.get("neighborhood", "") or "")
    address = str(row.get("address", "") or "")
    city = str(row.get("city", "") or "")
    
    if address and address != "nan":
        if neighborhood and neighborhood != "nan":
            parts.append(f"Located at {address} ({neighborhood}).")
        else:
            parts.append(f"Located at {address}.")
    elif neighborhood and neighborhood != "nan":
        parts.append(f"Located in {neighborhood}.")
    elif city and city != "nan":
        parts.append(f"Located in {city}.")
        
    # Check for exact neighborhood match
    if neighborhood and neighborhood != "nan" and neighborhood.lower() in query.lower():
        parts.append("Exact neighborhood match.")

    # --- Rating ---
    stars = row.get("stars")
    review_count = row.get("review_count")
    if stars is not None:
        review_str = f" ({int(review_count)} reviews)" if review_count else ""
        parts.append(f"Rated {stars} ★{review_str}.")

    # --- Price ---
    price_range = row.get("price_range")
    price_str = _price_label(price_range)
    if price_str:
        parts.append(f"Price: {price_str}.")
    else:
        parts.append("Price: N/A.")
        
    # --- Vibe Cluster ---
    cluster_id = row.get("cluster_id")
    if pd.notna(cluster_id):
        vibe_name = CLUSTER_NAMES.get(int(cluster_id), "General")
        parts.append(f"Vibe Category: {vibe_name}.")

    # --- Contextual matches based on query keywords ---
    if query_words & QUIET_KEYWORDS:
        noise = str(row.get("attributes_noise_level", "") or "")
        if "quiet" in noise.lower():
            parts.append("Known to be quiet — great for studying or working.")

    if query_words & ROMANTIC_KEYWORDS:
        ambience = str(row.get("attributes_ambience", "") or "")
        if "romantic" in ambience.lower():
            parts.append("Has a romantic ambience.")

    if query_words & PRICE_KEYWORDS:
        if str(price_range) in ("1", "2"):
            parts.append("Fits your budget.")

    # --- Review highlight ---
    profile_text = str(row.get("profile_text", "") or "")
    highlight = _extract_review_highlight(profile_text, query)
    if highlight:
        parts.append(f"Reviewers say: {highlight}")

    if not parts:
        parts.append("Closely matched your query based on reviews and profile.")

    # Two trailing spaces before \n = markdown hard line break, so each segment
    # renders on its own line inside st.info() / st.markdown().
    return "  \n".join(parts)


def add_explanations(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """
    Add an 'explanation' column to the results DataFrame.

    Args:
        df:    Reranked results DataFrame (one row per restaurant).
        query: The original user query.

    Returns:
        The same DataFrame with an added 'explanation' column.
    """
    df = df.copy()
    # Use a list comprehension instead of df.apply() to avoid a pandas edge case
    # where apply() on an empty DataFrame returns a DataFrame instead of a Series,
    # which would crash with "Cannot set a DataFrame with multiple columns to the
    # single column explanation".
    df["explanation"] = [
        generate_explanation(row, query) for _, row in df.iterrows()
    ]
    return df
