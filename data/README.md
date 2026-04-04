# Data Cleaning & Processing Summary - Philadelphia

This directory contains the data lifecycle for the Philadelphia Restaurant Recommender.

## Directory Structure

- `data/raw/`: **DO NOT COMMIT** - Contains the original Yelp Academic Dataset JSON files (Business, Review, User, etc.).
- `data/processed/`: Contains the final, cleaned CSV files used for model training and recommendations.
- `data/interim/`: Temporary storage used during multi-step transformations (currently empty).

## Processing Steps & Filtering Criteria

To create a high-quality recommendation dataset, we applied the following refinements to the raw Yelp data:

### 1. Business Filtering (`philly_restaurants.csv`)
- **Location**: Filtered for `city == "Philadelphia"` and `state == "PA"`.
- **Category**: Inclusion of restaurant-related keywords (Food, Restaurants, Cafe, bars, etc.).
- **Reliability**: Minimum of **20 reviews** per business.
- **Status**: Only businesses marked as **Currently Open** (`is_open == 1`) were included.
- **Result**: **2,963** high-quality Philadelphia restaurants.

### 2. Review Filtering & Join (`philly_reviews.csv`)
- **Recency**: Only reviews from **2019-01-01** onwards were kept to ensure relevance to current users.
- **Memory Optimization**: Performed a **streaming join** on the 5.3GB review JSON to extract only relevant Philadelphia records without exceeding system memory.
- **Result**: **137,755** matched reviews mapped to the filtered business list.

## Usage

These datasets are the primary inputs for the embedding and recommendation modules. 
- Business metadata provides the "Profiles".
- Review text provides the "Sentiment and Context" for deeper matching.
