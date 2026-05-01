# 🍽️ AI-Powered Yelp Restaurant Recommender (Philadelphia)

This project is an advanced, **Semantic AI-driven** restaurant recommendation system specifically built for Philadelphia, PA, using the Yelp Academic Dataset. 

Unlike traditional filter-based search engines (like the generic Yelp app), this system mathematically "reads" thousands of customer reviews and understands the **semantic meaning** of natural language queries (e.g., *"quiet date night italian spot"* or *"cheap late night pizza in center city"*).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Semantic_Search-brightgreen)

## ✨ Key Features

1. **Semantic Search via NLP embeddings:**
   Instead of just keyword matching, the app distills the sentiment of up to 40 customer reviews per restaurant into dense keywords and embeddings (using `all-MiniLM-L6-v2`).
2. **Intent Detection:**
   Dynamically adjusts weight distributions by detecting "quality" (e.g., "best") or "popularity" search intents within the user's query, seamlessly supplementing semantic search.
3. **Dynamic Weight Reallocation & Quality Floor Penalty:**
   The system calculates an "Overall Match Score" by blending Semantic Similarity, Bayesian Average Star Rating, Popularity, Price, and Exact Location match. Low-rated establishments receive a quality floor penalty. If you omit price or location from your query, the algorithm dynamically and proportionally re-allocates those weights back to the semantic score to prevent score dilution!
4. **Custom K-Means Clustering (Algorithm from Scratch):**
   We implemented a vectorized K-Means clustering algorithm from scratch using NumPy to group restaurants into 15 distinct "Vibe Categories" based purely on review semantics. 
5. **Diversity Penalty:**
   The reranker applies a penalty if it tries to recommend too many restaurants from the same "Vibe Cluster" sequentially, ensuring a varied set of results.
6. **Cuisine Soft Filter:**
   A user-selectable cuisine filter applies a soft penalty rather than a rigid exclusion, ensuring that strong semantic intent is never overridden.
7. **Extractive Summarization:**
   The UI generates a "Why this matches" explanation blurb by algorithmically finding the highest-value sentences in the reviews and bolding the exact words that overlap with your query.

---

## 📁 Project Structure

- `app/`
  - `streamlit_app.py`: The beautiful, premium web interface.
  - `api.py`: FastAPI backend endpoint.
  - `static/`: HTML/JS/CSS static assets and the interactive project presentation (`presentation.html`).
- `data/`
  - `raw/`: Original Yelp JSON files (excluded from Git).
  - `processed/`: Filtered datasets, restaurant profiles, and `.npy` embedding matrices.
- `notebooks/`: Exploratory Data Analysis & visual profiling.
- `scripts/`: Execution scripts to build the data pipeline step-by-step.
- `src/`: Core library logic:
  - `build_profiles.py`: Distills reviews into dense keyword highlights.
  - `embed.py`: Generates the sentence transformer embeddings.
  - `retrieve.py`: High-speed vector dot-product similarity search.
  - `rerank.py`: Bayesian averages, location boosting, and dynamic weighting.
  - `explain.py`: Rule-based explanation generation and review extraction.
  - `kmeans.py`: The custom clustering algorithm.

---

## 🚀 Getting Started

### 1. Prerequisites & Setup
Make sure you have Python 3.10+ installed.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Data Pipeline
*(Note: You must have the raw Yelp datasets in `data/raw/` to run the early preprocessing steps).*

```bash
# 1. Clean data and filter for Philly
python scripts/run_preprocess.py
python scripts/join_reviews.py

# 2. Build NLP profiles (extracting keywords from 40 reviews per restaurant)
python scripts/run_build_profiles.py

# 3. Generate Semantic Embeddings (~3,000 vectors)
python scripts/run_embed.py

# 4. Assign Vibe Categories using Custom K-Means
python scripts/run_clustering.py
```

### 3. Launch the Web Interface
Once the data pipeline is fully run, start the FastAPI server (the current
demo, with the cuisine filter, dynamic reranker weights, and the slide deck):

```bash
uvicorn app.api:app --reload
```

Open `http://localhost:8000` for the demo and `http://localhost:8000/presentation.html` for the slides.

> Legacy: an older Streamlit UI is still available via
> `streamlit run app/streamlit_app.py` at `http://localhost:8501`, but it
> doesn't include the recent reranker, cuisine filter, or vibe labels.

---

## 📊 Filtering Criteria (The Dataset)

This system focuses on high-quality, relevant data:
- **City**: Philadelphia, PA
- **Minimum Reviews**: 20
- **Status**: Currently Open (filters out permanently closed businesses)
- **Review Recency**: 2019 onwards (pre-COVID reviews are largely excluded to preserve relevance)
