# 🍜 Yelp Restaurant Recommender

A natural language restaurant recommendation system built on the [Yelp Open Dataset](https://www.yelp.com/dataset). Instead of rigid filters like cuisine type or price range, users can describe what they're looking for in plain English — *"quiet cafe to study"* or *"romantic Italian for date night"* — and receive personalized, explainable recommendations.

---

## 🎯 Project Overview

Existing restaurant platforms rely on predefined categories that fail to capture nuanced user intent. This project bridges that gap by combining:

- **Semantic search** via pretrained sentence embeddings (`sentence-transformers`)
- **Structured data** (ratings, price, location, cuisine)
- **Unstructured review text** to build rich restaurant profiles
- **Reranking** that blends similarity scores with structured signals
- **Clustering** (k-means) to diversify and group recommendations

The result is a web application where users type a natural language query and receive ranked restaurant recommendations with explanations.

---

## 📁 Repo Structure

```
yelp-restaurant-recommender/
│
├── README.md                  # Project overview and setup instructions
├── requirements.txt           # Python dependencies
│
├── data/
│   └── preprocess.py          # Load and clean Yelp dataset; filter to single city; merge reviews with restaurant metadata
│
├── embeddings/
│   └── embed.py               # Generate sentence embeddings for restaurant profiles and user queries using sentence-transformers
│
├── retrieval/
│   └── search.py              # Compute cosine similarity between query and restaurant embeddings; return top-k candidates
│
├── reranking/
│   └── rerank.py              # Rerank retrieved candidates using structured features (rating, price, review sentiment)
│
├── clustering/
│   └── cluster.py             # Apply k-means clustering to group restaurants into latent segments for diversity
│
├── app/
│   └── app.py                 # Streamlit/Flask web application; handles user input and displays recommendations
│
└── utils/
    └── helpers.py             # Shared utility functions (text cleaning, score normalization, etc.)
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/zitongyu756/yelp-restaurant-recommender.git
cd yelp-restaurant-recommender
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the Yelp Dataset
Download from [https://www.yelp.com/dataset](https://www.yelp.com/dataset) and place the JSON files in a `data/raw/` directory.

### 4. Run the app
```bash
streamlit run app/app.py
```

---

## 🛠️ Tech Stack

- **Embeddings:** `sentence-transformers`
- **Similarity Search:** `scikit-learn` (cosine similarity)
- **Clustering:** `scikit-learn` (k-means)
- **Sentiment Analysis:** `transformers` or `VADER`
- **Web App:** `Streamlit`
- **Data Processing:** `pandas`, `numpy`

---

## 👥 Team

| Member | Module |
|--------|--------|
| Andy Xu | `data/preprocess.py` |
| Junting Wu | `embeddings/embed.py` |
| Akash Datla | `retrieval/search.py` + `reranking/rerank.py` |
| Aurora Zhang | `app/app.py` |
| Olivia Yu | `clustering/cluster.py` |

---

## 📚 Course

This project is a final project for a Machine Learning course. Methods applied include semantic embeddings, cosine similarity retrieval, structured feature reranking, and unsupervised clustering.
