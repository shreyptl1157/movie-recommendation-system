from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd

class BaseRecommender(ABC):
    """Abstract base class for all recommenders."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.user_item_matrix = None
        self.movies_df = None
        self.users_df = None
    
    @abstractmethod
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None) -> None:
        """Fit the recommender to data.
        
        Args:
            ratings_df: DataFrame with columns [user_id, movie_id, rating, timestamp]
            movies_df: DataFrame with columns [movie_id, title, genres, release_year]
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations to return
            
        Returns:
            List of (movie_id, score) tuples
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        pass
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Get items similar to given item.
        
        Args:
            item_id: Item ID
            n_similar: Number of similar items to return
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        raise NotImplementedError(f"{self.name} does not support similar items")
    
    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Get user profile information.
        
        Args:
            user_id: User ID
            
        Returns:
            User profile dictionary
        """
        raise NotImplementedError(f"{self.name} does not support user profiles")
    
    def update_rating(self, user_id: int, movie_id: int, rating: float) -> None:
        """Update a user's rating for a movie.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            rating: New rating
        """
        raise NotImplementedError(f"{self.name} does not support rating updates")
