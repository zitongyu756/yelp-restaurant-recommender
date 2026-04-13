# Yelp Restaurant Recommender - Philadelphia

This project builds a restaurant recommendation system for Philadelphia, PA, using the Yelp Academic Dataset.

## Project Structure

- `data/raw/`: Original Yelp JSON files (excluded from Git).
- `data/processed/`: Filtered Philadelphia datasets.
  - `philly_restaurants.csv`: 2,963 open restaurants with 20+ reviews.
  - `philly_reviews.csv`: 137,755 reviews from 2019 onwards.
- `notebooks/`: Exploratory Data Analysis.
- `scripts/`: Data processing pipelines.
- `src/`: Core library logic and configurations.

## Getting Started

### Prerequisites
Make sure you have Python 3 installed.

### Setup

1. Create a virtual environment
```bash
   python3 -m venv .venv
```

2. Activate the virtual environment
```bash
   source .venv/bin/activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

### Running the Pipeline

4. Preprocess businesses
```bash
   python scripts/run_preprocess.py
```

5. Stream join reviews
```bash
   python scripts/join_reviews.py
```

### Analysis

6. Open the analysis notebook
```
   notebooks/data_analysis.ipynb
```

## Filtering Criteria

- **City**: Philadelphia, PA
- **Minimum Reviews**: 20
- **Status**: Currently Open
- **Review Recency**: 2019 onwards
