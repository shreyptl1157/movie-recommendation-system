from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: int = Field(..., gt=0)
    n_recommendations: int = Field(10, ge=1, le=100)
    algorithm: str = Field("hybrid", regex="^(user_cf|item_cf|svd|content|hybrid)$")

class RecommendationItem(BaseModel):
    """Single recommendation item."""
    movie_id: int
    score: float

class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    user_id: int
    algorithm: str
    recommendations: List[RecommendationItem]
    timestamp: str

class RatingRequest(BaseModel):
    """Request model for adding/updating ratings."""
    user_id: int = Field(..., gt=0)
    movie_id: int = Field(..., gt=0)
    rating: float = Field(..., ge=0.5, le=5.0)
    timestamp: Optional[int] = None

class MovieDetailsResponse(BaseModel):
    """Movie details response."""
    movie_id: int
    title: str
    genres: List[str]
    release_year: int
    average_rating: float
    num_ratings: int

class SimilarMovie(BaseModel):
    """Similar movie item."""
    movie_id: int
    similarity_score: float

class SimilarMoviesResponse(BaseModel):
    """Response model for similar movies."""
    movie_id: int
    title: str
    similar_movies: List[SimilarMovie]
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    environment: str
