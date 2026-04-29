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
    page_title="Philly Restaurant Recommender",
    page_icon="🍽️",
    layout="wide", # Wider layout for premium feel
    initial_sidebar_state="expanded",
)

# Premium CSS Styling - Tailored for our App (Dark Mode Compatible)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Search Bar */
    .stTextInput input {
        border-radius: 12px !important;
        padding: 14px 20px !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        font-size: 1.05rem !important;
        transition: all 0.2s ease;
        background-color: transparent !important;
    }
    .stTextInput input:focus {
        border-color: #FF4B4B !important;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2) !important;
    }
    
    /* Elegant Result Cards (targets only bordered containers) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* Stylize the metrics */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #FF4B4B;
    }
    
    /* Badge styling */
    .feature-badge {
        display: inline-block;
        padding: 4px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        border-radius: 20px;
        background: rgba(255, 75, 75, 0.1);
        color: #FF4B4B;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(255, 75, 75, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

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


def run_search(query: str, min_stars: float = 0.0) -> pd.DataFrame:
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
        if min_stars > 0:
            mock = mock[mock["stars"] >= min_stars]
        return add_explanations(mock, query)

    retrieve_module = load_retrieve_module()
    candidates = retrieve_module.retrieve(query, top_k=TOP_K_RETRIEVE)
    
    if min_stars > 0:
        candidates = candidates[candidates["stars"] >= min_stars].copy()

    reranked = rerank(candidates, query)
    top = reranked.head(TOP_K_DISPLAY)
    results = add_explanations(top, query)
    return results


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

st.title("🍽️ AI-Powered Philly Restaurant Recommender")
st.caption("Not just keywords. We read the reviews so you don't have to.")

with st.expander("Why is this better than regular Yelp?"):
    st.markdown("""
        **Regular Yelp** forces you to use rigid filters (e.g., checking the "Italian" box and the "$$" box).
        **This app** uses **Semantic AI (Sentence Transformers)** to actually *understand* the meaning of your search.
        
        It mathematically reads thousands of customer reviews to find the exact "vibe" you are looking for. 
        Try asking it complex questions like:
        - *"Where can I take my parents for a quiet anniversary dinner?"*
        - *"A loud sports bar with cheap wings and good beer"*
    """)

st.markdown("---")

# Example queries shown as clickable buttons
EXAMPLE_QUERIES = [
    "quiet cafe to study near UPenn",
    "date night italian in Rittenhouse",
    "cheap spicy noodles in Chinatown",
    "brunch spot in Old City",
    "craft beer bar in Fishtown",
]

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLE_QUERIES))
for col, example in zip(cols, EXAMPLE_QUERIES):
    if col.button(example, use_container_width=True):
        st.session_state["query_input"] = example

# Text input — pre-filled from example button clicks via session state
query = st.text_input(
    label="Your query",
    placeholder="e.g. cheap cheesesteak in South Philly",
    key="query_input",
    label_visibility="collapsed",
)

search_clicked = st.button("Search", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

if search_clicked and query.strip():
    with st.spinner("Finding the best restaurants..."):
        # We'll pull the min_stars from session state if it exists, default to 0.0
        min_stars = st.session_state.get("min_stars", 0.0)
        results = run_search(query.strip(), min_stars)

    if results.empty:
        st.warning(
            "No matching restaurants found in the dataset. "
            "Try lowering the minimum star rating or changing your keywords.",
            icon="⚠️",
        )
    else:
        st.markdown(f"### Top {len(results)} results for: *{query}*")
        st.divider()

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            # price_range is stored as float string ("1.0", "2.0" etc.) in the CSV,
            # so round to nearest int before looking up the symbol.
            _pr = row.get("price_range", "")
            try:
                _pr_key = str(round(float(_pr)))
            except (ValueError, TypeError):
                _pr_key = ""
            price_display = {"1": "$", "2": "$$", "3": "$$$", "4": "$$$$"}.get(
                _pr_key, "Price: Unknown"
            )
            stars_display = f"{row.get('stars', '?')} ★"
            review_display = f"{int(row.get('review_count', 0)):,} reviews"
            
            # Final score converted to a percentage
            match_pct = int(row.get("final_score", 0) * 100)
            
            # Semantic Similarity Score (the pure AI match)
            semantic_score = int(row.get("similarity_score", 0) * 100)
            
            # Format rank to be 01, 02, etc.
            rank_str = f"{rank:02d}"
            
            with st.container(border=True):
                col_rank, col_info, col_metrics = st.columns([1, 6, 2])
                with col_rank:
                    st.markdown(f"<h1 style='color:#FF4B4B;'>#{rank}</h1>", unsafe_allow_html=True)
                with col_info:
                    st.markdown(f"### {row.get('name', 'Unknown')}")
                    meta_parts = [stars_display, review_display]
                    
                    if price_display:
                        meta_parts.append(price_display)
                        
                    neighborhood = str(row.get("neighborhood", "") or "")
                    address = str(row.get("address", "") or "")
                    city = str(row.get("city", "") or "")
                    
                    if address and address != "nan":
                        meta_parts.append(f"📍 {address}")
                    elif neighborhood and neighborhood != "nan":
                        meta_parts.append(f"📍 {neighborhood}")
                    elif city and city != "nan":
                        meta_parts.append(f"📍 {city}")
                        
                    st.caption("  ·  ".join(meta_parts))
                    st.markdown(f"*{row.get('categories', '')}*")
                    
                    # Badges for vibe elements
                    badges_html = ""
                    if "quiet" in str(row.get("explanation", "")).lower():
                        badges_html += "<span class='feature-badge'>🤫 Quiet Setting</span>"
                    if "romantic" in str(row.get("explanation", "")).lower():
                        badges_html += "<span class='feature-badge'>❤️ Romantic</span>"
                    if "budget" in str(row.get("explanation", "")).lower():
                        badges_html += "<span class='feature-badge'>💰 Budget Friendly</span>"
                        
                    if badges_html:
                        st.markdown(badges_html, unsafe_allow_html=True)
                        
                    st.info(row.get("explanation", ""), icon="💡")
                with col_metrics:
                    st.metric("Overall Match", f"{match_pct}%")
                    st.progress(match_pct / 100.0)
                    st.caption(f"🤖 Semantic AI Score: **{semantic_score}%**")

elif search_clicked and not query.strip():
    st.error("Please enter a query before searching.")

# ---------------------------------------------------------------------------
# Sidebar: data status & filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Search Filters")
    st.slider(
        "Minimum Star Rating", 
        min_value=0.0, max_value=5.0, step=0.5, value=0.0,
        key="min_stars"
    )
    
    st.divider()
    
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
    st.markdown("**Price guide**")
    st.markdown(
        "- `$` — Inexpensive (under \\$10)\n"
        "- `$$` — Moderate (\\$11–\\$30)\n"
        "- `$$$` — Pricey (\\$31–\\$60)\n"
        "- `$$$$` — Very expensive (\\$61+)"
    )
    st.caption("Yelp price tiers per person, excluding tax and tip.")
    st.markdown("---")
    st.markdown("**Model:** `all-MiniLM-L6-v2`")
    st.markdown(f"**Results shown:** {TOP_K_DISPLAY}")
    st.markdown(f"**Candidates retrieved:** {TOP_K_RETRIEVE}")
