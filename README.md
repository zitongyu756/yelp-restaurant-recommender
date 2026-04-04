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

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Preprocess Businesses**: `python scripts/run_preprocess.py`
3. **Stream Join Reviews**: `python scripts/join_reviews.py`
4. **Analysis**: Open `notebooks/data_analysis.ipynb`

## Filtering Criteria

- **City**: Philadelphia, PA
- **Minimum Reviews**: 20
- **Status**: Currently Open
- **Review Recency**: 2019 onwards
