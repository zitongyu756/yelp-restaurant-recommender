# Yelp Restaurant Recommender (New York City)

A semantic search restaurant recommendation app built on the [Yelp Open Dataset](https://www.yelp.com/dataset), scoped exclusively to **New York City**.

Users type a natural-language query such as:

- `"quiet cafe to study near NYU"`
- `"date night italian restaurant in SoHo"`
- `"cheap spicy noodles open late in Brooklyn"`

The app embeds the query with the same sentence-transformer model used to build restaurant profiles, computes cosine similarity, reranks by structured signals (rating, review count, price), and displays the top results with a short explanation.

---

## Project Scope — NYC Only

We filter the Yelp dataset to:

- **City**: New York (covers Manhattan, Brooklyn, Queens, Bronx, Staten Island)
- **Category**: Restaurants / Food businesses
- **Minimum reviews**: configurable (default 10) to keep quality signal

Neighborhoods such as SoHo, East Village, Astoria, and Williamsburg are preserved in the structured restaurant profile and used as part of the searchable text.

---

## Folder Structure

```
yelp-restaurant-recommender/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/              # Original Yelp JSON files (never committed, too large)
│   ├── interim/          # Filtered NYC restaurant subset (CSV)
│   └── processed/        # Final profiles CSV + embeddings .npy file
│
├── notebooks/            # Exploratory notebooks (EDA, experiments)
│
├── src/                  # All reusable Python modules
│   ├── __init__.py
│   ├── config.py         # Paths, constants, model name, NYC filter settings
│   ├── preprocess.py     # Load + filter Yelp data → data/interim/
│   ├── build_profiles.py # Combine structured + review text → restaurant profiles
│   ├── embed.py          # Embed profiles with sentence-transformers → .npy
│   ├── similarity.py     # Manual cosine similarity (no sklearn)
│   ├── retrieve.py       # Embed query, find top-k similar restaurants
│   ├── rerank.py         # Boost results using rating / review_count / price
│   ├── explain.py        # Generate short "why recommended" text per result
│   └── utils.py          # Shared helpers (logging, loading, saving)
│
├── app/
│   └── streamlit_app.py  # Full Streamlit UI (loads processed data, runs search)
│
├── scripts/              # One-time offline processing scripts
│   ├── run_preprocess.py
│   ├── run_build_profiles.py
│   └── run_embed.py
│
└── tests/
    ├── __init__.py
    └── test_similarity.py
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/zitongyu756/yelp-restaurant-recommender.git
cd yelp-restaurant-recommender
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Yelp Open Dataset

Visit https://www.yelp.com/dataset and download the dataset.  
Place the following files inside `data/raw/`:

```
data/raw/yelp_academic_dataset_business.json
data/raw/yelp_academic_dataset_review.json
```

> These files are large (~1 GB and ~5 GB respectively).  
> They are listed in `.gitignore` and must **never** be committed to the repo.

---

## Data Pipeline (run once, offline)

Run the three preprocessing scripts **in order** before launching the app.

### Step 1 — Filter to NYC restaurants

```bash
python scripts/run_preprocess.py
```

Reads `data/raw/yelp_academic_dataset_business.json`, filters to NYC restaurants,
and writes `data/interim/nyc_restaurants.csv`.

### Step 2 — Build restaurant profiles

```bash
python scripts/run_build_profiles.py
```

Reads `data/interim/nyc_restaurants.csv` + `data/raw/yelp_academic_dataset_review.json`,
combines structured metadata with sampled review text per restaurant,
and writes `data/processed/restaurant_profiles.csv`.

### Step 3 — Embed profiles

```bash
python scripts/run_embed.py
```

Loads `data/processed/restaurant_profiles.csv`, encodes each profile text with
`sentence-transformers`, and writes `data/processed/embeddings.npy`.  
This step is slow the first time (model download + GPU/CPU inference).

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

> On first launch without processed data, the app shows mock results so the UI
> can be developed independently of the data pipeline.

---

## Module Guide

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth for all paths and constants |
| `preprocess.py` | Filter Yelp JSON → clean NYC restaurant CSV |
| `build_profiles.py` | Build one searchable text block per restaurant |
| `embed.py` | Encode profile text → float32 numpy matrix |
| `similarity.py` | Cosine similarity (implemented from scratch with NumPy) |
| `retrieve.py` | Query → top-k restaurant indices |
| `rerank.py` | Reorder by rating, review count, price tier |
| `explain.py` | Generate short human-readable "why recommended" blurb |
| `utils.py` | Load/save helpers, logging setup |
| `streamlit_app.py` | Interactive UI — calls retrieve + rerank + explain |

---

## Example Queries

Try these in the running app:

- `quiet cafe to study near NYU`
- `date night italian restaurant in SoHo`
- `cheap spicy ramen in East Village`
- `rooftop bar with great views in Manhattan`
- `halal cart street food late night`
- `brunch with bottomless mimosas in Williamsburg`
- `kid-friendly pizza in Brooklyn`

---

## Team Division of Work (Suggested)

| Person | Modules |
|---|---|
| A | `preprocess.py` + `run_preprocess.py` |
| B | `build_profiles.py` + `run_build_profiles.py` |
| C | `embed.py` + `similarity.py` + `run_embed.py` |
| D | `retrieve.py` + `rerank.py` + `explain.py` |
| E | `streamlit_app.py` + `tests/` + README |

---

## Technical Stack

- **Python 3.10+**
- **pandas** — tabular data processing
- **numpy** — array math and embedding storage
- **sentence-transformers** — pretrained text embedding model (`all-MiniLM-L6-v2`)
- **Streamlit** — web UI
- No database, no Docker, no external APIs, no authentication
