"""
streamlit_app.py
----------------
The main Streamlit web application for the Yelp Restaurant Recommender.

Run with:
    streamlit run app/streamlit_app.py

Flow:
  1. User types a natural language query.
  2. App calls retrieve() to find the most semantically similar restaurants.
  3. App calls rerank() to blend similarity with quality signals.
  4. App calls add_explanations() to generate a short blurb per result.
  5. App displays the top results in a clean card layout.

If the processed data files are not found (i.e., the preprocessing pipeline
hasn't been run yet), the app displays mock data so the UI can be developed
and tested independently.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make sure src/ is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import TOP_K_DISPLAY, TOP_K_RETRIEVE, PROFILES_CSV, EMBEDDINGS_NPY
from src.explain import add_explanations
from src.rerank import rerank
from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Philadelphia Restaurant Recommender",
    page_icon="🍽️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Helper: detect whether processed data is available
# ---------------------------------------------------------------------------

def processed_data_exists() -> bool:
    return PROFILES_CSV.exists() and EMBEDDINGS_NPY.exists()


# ---------------------------------------------------------------------------
# Helper: mock data for UI development before the pipeline is run
# ---------------------------------------------------------------------------

def get_mock_results(query: str) -> pd.DataFrame:
    """Return a small hardcoded DataFrame so the UI is testable without real data."""
    mock = pd.DataFrame(
        [
            {
                "name": "Reading Terminal Market",
                "categories": "Public Markets, Food Court, Bakeries, Seafood",
                "city": "Philadelphia",
                "address": "51 N 12th St",
                "stars": 4.5,
                "review_count": 5721,
                "price_range": "2",
                "similarity_score": 0.88,
                "final_score": 0.85,
                "explanation": "Serves Public Markets, Food Court. Located in Philadelphia. Rated 4.5 ★ (5721 reviews). Price: $$.",
            },
            {
                "name": "Zahav",
                "categories": "Middle Eastern, Israeli, Bars",
                "city": "Philadelphia",
                "address": "237 St James Pl",
                "stars": 4.5,
                "review_count": 3065,
                "price_range": "4",
                "similarity_score": 0.85,
                "final_score": 0.82,
                "explanation": "Serves Middle Eastern, Israeli. Located in Philadelphia. Rated 4.5 ★ (3065 reviews). Price: $$$$.",
            },
            {
                "name": "Parc",
                "categories": "French, Wine Bars, American (New), Breakfast & Brunch",
                "city": "Philadelphia",
                "address": "227 S 18th St",
                "stars": 4.0,
                "review_count": 2761,
                "price_range": "3",
                "similarity_score": 0.82,
                "final_score": 0.79,
                "explanation": "Serves French, Wine Bars. Located in Philadelphia. Rated 4.0 ★ (2761 reviews). Price: $$$.",
            },
            {
                "name": "Pizzeria Beddia",
                "categories": "Italian, Pizza",
                "city": "Philadelphia",
                "address": "1313 North Lee St",
                "stars": 4.0,
                "review_count": 599,
                "price_range": "2",
                "similarity_score": 0.79,
                "final_score": 0.76,
                "explanation": "Serves Italian, Pizza. Located in Philadelphia. Rated 4.0 ★ (599 reviews). Price: $$.",
            },
            {
                "name": "Joe's Steaks + Soda Shop",
                "categories": "Cheesesteaks, Sandwiches, American (Traditional)",
                "city": "Philadelphia",
                "address": "1 W Girard Ave",
                "stars": 4.0,
                "review_count": 392,
                "price_range": "2",
                "similarity_score": 0.77,
                "final_score": 0.74,
                "explanation": "Serves Cheesesteaks, Sandwiches. Located in Philadelphia. Rated 4.0 ★ (392 reviews). Price: $$.",
            },
        ]
    )
    return mock


# ---------------------------------------------------------------------------
# Main search function — delegates to real pipeline or mock fallback
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading restaurant data...")
def load_retrieve_module():
    """
    Import and return the retrieve module only when real data is available.
    Wrapped in cache_resource so the model is loaded only once per session.
    """
    from src import retrieve
    retrieve._load_resources()
    return retrieve


def run_search(query: str) -> pd.DataFrame:
    """
    Run the full retrieve → rerank → explain pipeline for a given query.
    Falls back to mock data if processed files don't exist yet.
    """
    if not processed_data_exists():
        st.warning(
            "Processed data not found. Showing mock results.\n\n"
            "Run the preprocessing pipeline (see README) to use real data.",
            icon="⚠️",
        )
        mock = get_mock_results(query)
        return add_explanations(mock, query)

    retrieve_module = load_retrieve_module()
    candidates = retrieve_module.retrieve(query, top_k=TOP_K_RETRIEVE)
    reranked = rerank(candidates, query=query)
    top = reranked.head(TOP_K_DISPLAY)
    results = add_explanations(top, query)
    return results


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.title("🍽️ Philadelphia Restaurant Recommender")
st.caption("Powered by Yelp data · Semantic search · Philadelphia only")

st.markdown(
    "Type what you're looking for in plain English and we'll find the best match."
)

# Example queries shown as clickable buttons
EXAMPLE_QUERIES = [
    "quiet cafe with good coffee to get work done",
    "romantic italian restaurant for date night",
    "cheap cheesesteak sandwich shop",
    "casual brunch spot with a great breakfast menu",
    "upscale seafood restaurant with cocktails",
]

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(cols, EXAMPLE_QUERIES):
    if col.button(example, use_container_width=True):
        st.session_state["query_input"] = example

# Text input — pre-filled from example button clicks via session state
query = st.text_input(
    label="Your query",
    placeholder="e.g. cheap spicy noodles near NYU",
    key="query_input",
    label_visibility="collapsed",
)

search_clicked = st.button("Search", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if search_clicked and query.strip():
    with st.spinner("Finding restaurants..."):
        results = run_search(query.strip())

    st.markdown(f"### Top {len(results)} results for: *{query}*")
    st.divider()

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        price_display = {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}.get(
            str(row.get("price_range", "")), ""
        )
        stars_display = f"{row.get('stars', '?')} ★"
        review_display = f"{int(row.get('review_count', 0)):,} reviews"

        with st.container():
            col_rank, col_info = st.columns([1, 9])
            with col_rank:
                st.markdown(f"### {rank}")
            with col_info:
                st.markdown(f"#### {row.get('name', 'Unknown')}")
                meta_parts = [stars_display, review_display]
                if price_display:
                    meta_parts.append(price_display)
                neighborhood = str(row.get("neighborhood", "") or "")
                city = str(row.get("city", "") or "")
                location = neighborhood if neighborhood and neighborhood != "nan" else city
                if location and location != "nan":
                    meta_parts.append(f"📍 {location}")
                st.caption("  ·  ".join(meta_parts))
                st.markdown(f"*{row.get('categories', '')}*")
                st.info(row.get("explanation", ""), icon="💡")

        st.divider()

elif search_clicked and not query.strip():
    st.error("Please enter a query before searching.")

# ---------------------------------------------------------------------------
# Sidebar: data status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data Status")
    if processed_data_exists():
        st.success("Processed data found ✓", icon="✅")
    else:
        st.error("Processed data not found", icon="❌")
        st.markdown(
            "Run the pipeline first:\n"
            "```bash\n"
            "python scripts/run_preprocess.py\n"
            "python scripts/run_build_profiles.py\n"
            "python scripts/run_embed.py\n"
            "```"
        )
    st.markdown("---")
    st.markdown("**Model:** `all-MiniLM-L6-v2`")
    st.markdown(f"**Results shown:** {TOP_K_DISPLAY}")
    st.markdown(f"**Candidates retrieved:** {TOP_K_RETRIEVE}")
