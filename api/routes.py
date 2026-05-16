from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime
from .schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RatingRequest,
    MovieDetailsResponse,
    HealthResponse,
    SimilarMoviesResponse
)
from .dependencies import get_recommender
from config import get_settings

router = APIRouter()
settings = get_settings()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment
    }

@router.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    recommender = Depends(get_recommender)
):
    """Get movie recommendations for a user."""
    try:
        recommendations = recommender.recommend(
            user_id=request.user_id,
            n_recommendations=request.n_recommendations
        )
        
        return {
            "user_id": request.user_id,
            "algorithm": request.algorithm,
            "recommendations": [
                {"movie_id": mid, "score": float(score)}
                for mid, score in recommendations
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ratings")
async def add_rating(request: RatingRequest):
    """Add or update a user's movie rating."""
    try:
        return {
            "user_id": request.user_id,
            "movie_id": request.movie_id,
            "rating": request.rating,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/movies/{movie_id}")
async def get_movie_details(movie_id: int):
    """Get movie details."""
    # This would connect to database
    return {
        "movie_id": movie_id,
        "title": "Movie Title",
        "genres": ["Action", "Sci-Fi"],
        "release_year": 2020,
        "average_rating": 4.5,
        "num_ratings": 1000
    }

@router.get("/movies/{movie_id}/similar", response_model=SimilarMoviesResponse)
async def get_similar_movies(
    movie_id: int,
    n: int = Query(10, ge=1, le=100)
):
    """Get movies similar to given movie."""
    try:
        return {
            "movie_id": movie_id,
            "title": "Movie Title",
            "similar_movies": [
                {"movie_id": mid, "similarity_score": float(score)}
                for mid, score in []
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
