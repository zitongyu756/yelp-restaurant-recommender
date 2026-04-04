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
    page_title="NYC Restaurant Recommender",
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
                "name": "Joe's Pizza",
                "categories": "Pizza, Italian",
                "city": "Manhattan",
                "neighborhood": "Greenwich Village",
                "stars": 4.5,
                "review_count": 1203,
                "price_range": "1",
                "similarity_score": 0.91,
                "final_score": 0.87,
                "explanation": "Serves Pizza, Italian. Located in Greenwich Village. Rated 4.5 ★ (1203 reviews). Price: $.",
            },
            {
                "name": "Xi'an Famous Foods",
                "categories": "Chinese, Noodles",
                "city": "Manhattan",
                "neighborhood": "East Village",
                "stars": 4.3,
                "review_count": 876,
                "price_range": "1",
                "similarity_score": 0.85,
                "final_score": 0.81,
                "explanation": "Serves Chinese, Noodles. Located in East Village. Rated 4.3 ★ (876 reviews). Price: $.",
            },
            {
                "name": "Ippudo NY",
                "categories": "Ramen, Japanese",
                "city": "Manhattan",
                "neighborhood": "East Village",
                "stars": 4.2,
                "review_count": 2340,
                "price_range": "2",
                "similarity_score": 0.82,
                "final_score": 0.79,
                "explanation": "Serves Ramen, Japanese. Located in East Village. Rated 4.2 ★ (2340 reviews). Price: $$.",
            },
            {
                "name": "Dirt Candy",
                "categories": "Vegetarian, American",
                "city": "Manhattan",
                "neighborhood": "Lower East Side",
                "stars": 4.6,
                "review_count": 512,
                "price_range": "3",
                "similarity_score": 0.79,
                "final_score": 0.76,
                "explanation": "Serves Vegetarian, American. Located in Lower East Side. Rated 4.6 ★ (512 reviews). Price: $$$.",
            },
            {
                "name": "Roberta's",
                "categories": "Pizza, Italian, Bar",
                "city": "Brooklyn",
                "neighborhood": "Bushwick",
                "stars": 4.4,
                "review_count": 1890,
                "price_range": "2",
                "similarity_score": 0.77,
                "final_score": 0.74,
                "explanation": "Serves Pizza, Italian. Located in Bushwick, Brooklyn. Rated 4.4 ★ (1890 reviews). Price: $$.",
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
    reranked = rerank(candidates)
    top = reranked.head(TOP_K_DISPLAY)
    results = add_explanations(top, query)
    return results


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.title("🍽️ NYC Restaurant Recommender")
st.caption("Powered by Yelp data · Semantic search · NYC only")

st.markdown(
    "Type what you're looking for in plain English and we'll find the best match."
)

# Example queries shown as clickable buttons
EXAMPLE_QUERIES = [
    "quiet cafe to study near NYU",
    "date night italian in SoHo",
    "cheap spicy ramen in East Village",
    "brunch with bottomless mimosas in Williamsburg",
    "rooftop bar with great views in Manhattan",
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
