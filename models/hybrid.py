import numpy as np
import pandas as pd
from typing import List, Tuple
from .base import BaseRecommender
from .collaborative import UserBasedCF, ItemBasedCF
from .svd import SVDRecommender
from .content_based import ContentBasedRecommender

class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining multiple approaches."""
    
    def __init__(self, weights: dict = None):
        super().__init__("Hybrid")
        
        # Default weights
        if weights is None:
            weights = {
                'user_cf': 0.2,
                'item_cf': 0.2,
                'svd': 0.3,
                'content': 0.3
            }
        self.weights = weights
        
        # Initialize sub-recommenders
        self.user_cf = UserBasedCF()
        self.item_cf = ItemBasedCF()
        self.svd = SVDRecommender()
        self.content = ContentBasedRecommender()
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None) -> None:
        """Fit all sub-recommenders."""
        self.user_cf.fit(ratings_df, movies_df)
        self.item_cf.fit(ratings_df, movies_df)
        self.svd.fit(ratings_df, movies_df)
        self.content.fit(ratings_df, movies_df)
        
        self.movies_df = movies_df
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get ensemble recommendations."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Get recommendations from each model
        recs = {}
        
        # User-based CF
        for movie_id, score in self.user_cf.recommend(user_id, n_recommendations * 2):
            recs[movie_id] = recs.get(movie_id, 0) + score * self.weights['user_cf']
        
        # Item-based CF
        for movie_id, score in self.item_cf.recommend(user_id, n_recommendations * 2):
            recs[movie_id] = recs.get(movie_id, 0) + score * self.weights['item_cf']
        
        # SVD
        for movie_id, score in self.svd.recommend(user_id, n_recommendations * 2):
            recs[movie_id] = recs.get(movie_id, 0) + score * self.weights['svd']
        
        # Sort and return top N
        recommendations = sorted(recs.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Ensemble prediction."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        prediction = (
            self.user_cf.predict(user_id, movie_id) * self.weights['user_cf'] +
            self.item_cf.predict(user_id, movie_id) * self.weights['item_cf'] +
            self.svd.predict(user_id, movie_id) * self.weights['svd']
        )
        return prediction
