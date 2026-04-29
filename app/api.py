import os
import json
import logging
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our recommender logic
from src.retrieve import retrieve
from src.rerank import rerank
from src.explain import add_explanations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# We let retrieve() handle caching internally
logger.info("FastAPI backend initialized.")

@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    min_stars: float = Query(0.0),
    top_k: int = Query(30)
):
    try:
        # 1. Vector Search
        # Fetch up to 100 candidates so the reranker has a deep pool to find the most reviewed items.
        candidates = retrieve(q, top_k=100)
        
        # 2. Rerank
        reranked = rerank(candidates, q)
        
        # 3. Filter by stars
        if min_stars > 0:
            reranked = reranked[reranked["stars"] >= min_stars]
            
        # 4. Explanations
        top_results = reranked.head(top_k).copy()
        top_results = add_explanations(top_results, q)
        
        # Format the output for the frontend
        results = []
        for _, row in top_results.iterrows():
            # Format price
            price_val = row.get("price_range")
            try:
                price_display = {1: "$", 2: "$$", 3: "$$$", 4: "$$$$"}.get(int(round(float(price_val))), "")
            except:
                price_display = ""
                
            match_pct = int(row.get("final_score", 0) * 100)
            semantic_score = int(row.get("similarity_score", 0) * 100)
            
            results.append({
                "id": str(row.get("business_id")),
                "name": str(row.get("name")),
                "stars": float(row.get("stars", 0.0)),
                "review_count": int(row.get("review_count", 0)),
                "price": price_display,
                "neighborhood": str(row.get("neighborhood", "")),
                "address": str(row.get("address", "")),
                "city": str(row.get("city", "")),
                "categories": str(row.get("categories", "")),
                "match_pct": match_pct,
                "semantic_score": semantic_score,
                "explanation": str(row.get("explanation", ""))
            })
            
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"status": "error", "message": str(e)}

# Serve static frontend
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
