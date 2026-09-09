from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from recommendation import (
    get_movie,
    recommend,
    recommend_for_user,
    search_movies,
)

app = FastAPI(
    title="Hybrid Movie Recommendation API",
    description=(
        "Personalized movie recommendations using content similarity, "
        "latent-factor collaborative filtering, and quality-aware ranking."
    ),
    version="2.0",
)

# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# HOME
# -------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Hybrid Movie Recommendation API is running",
        "version": "2.0",
        "models": [
            "TF-IDF content model",
            "latent-factor collaborative filtering",
            "hybrid ranker",
        ],
    }


# -------------------------------------------------------------------
# SEARCH MOVIES
# -------------------------------------------------------------------

@app.get("/search")
def search(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
):
    return {
        "query": query,
        "results": search_movies(query, limit),
    }


# -------------------------------------------------------------------
# MOVIE DETAILS
# -------------------------------------------------------------------

@app.get("/movie/{movie_id}")
def movie_details(movie_id: int):
    movie = get_movie(movie_id)

    if movie is None:
        raise HTTPException(
            status_code=404,
            detail="Movie not found",
        )

    return movie


# -------------------------------------------------------------------
# PERSONALIZED USER RECOMMENDATIONS
# IMPORTANT: This route MUST come before /recommend/{movie_name:path}
# -------------------------------------------------------------------

@app.get("/recommend/user/{user_id}")
def personalized_recommendations(
    user_id: int,
    limit: int = Query(default=20, ge=1, le=50),
):
    recommendations = recommend_for_user(
        user_id,
        limit=limit,
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="User not found or user has no model history",
        )

    return {
        "user_id": user_id,
        "recommendation_type": "personalized_hybrid",
        "recommendations": recommendations,
    }


# -------------------------------------------------------------------
# MOVIE-BASED RECOMMENDATIONS
# -------------------------------------------------------------------

@app.get("/recommend/{movie_name:path}")
def get_recommendations(
    movie_name: str,
    user_id: int | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=50),
):
    recommendations = recommend(
        movie_name,
        user_id=user_id,
        limit=limit,
    )

    if not recommendations:
        raise HTTPException(
            status_code=404,
            detail="Movie not found",
        )

    return {
        "movie": movie_name,
        "user_id": user_id,
        "recommendation_type": (
            "hybrid"
            if user_id is not None
            else "content_plus_quality"
        ),
        "recommendations": recommendations,
    }