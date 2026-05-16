import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from typing import List, Tuple
from .base import BaseRecommender

class SVDRecommender(BaseRecommender):
    """SVD-based recommender using matrix factorization."""
    
    def __init__(self, n_factors: int = 100, n_epochs: int = 20, 
                 learning_rate: float = 0.01, regularization: float = 0.02,
                 random_state: int = 42):
        super().__init__("SVD")
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state
        
        self.user_factors = None
        self.item_factors = None
        self.user_means = None
        self.global_mean = None
        self.user_id_map = None
        self.movie_id_map = None
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None) -> None:
        """Fit SVD model to ratings data."""
        self.global_mean = ratings_df['rating'].mean()
        
        # Create ID mappings
        users = ratings_df['user_id'].unique()
        movies = ratings_df['movie_id'].unique()
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(users)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(movies)}
        
        n_users = len(users)
        n_movies = len(movies)
        
        # Initialize factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_movies, self.n_factors))
        
        # SGD training
        for epoch in range(self.n_epochs):
            for _, row in ratings_df.iterrows():
                user_idx = self.user_id_map[row['user_id']]
                movie_idx = self.movie_id_map[row['movie_id']]
                rating = row['rating']
                
                # Predict
                pred = self._predict_internal(user_idx, movie_idx)
                error = rating - pred
                
                # Update factors
                user_factor = self.user_factors[user_idx]
                item_factor = self.item_factors[movie_idx]
                
                self.user_factors[user_idx] += self.learning_rate * (
                    error * item_factor - self.regularization * user_factor
                )
                self.item_factors[movie_idx] += self.learning_rate * (
                    error * user_factor - self.regularization * item_factor
                )
        
        self.movies_df = movies_df
        self.is_fitted = True
    
    def _predict_internal(self, user_idx: int, movie_idx: int) -> float:
        """Internal prediction using indices."""
        return self.global_mean + np.dot(
            self.user_factors[user_idx],
            self.item_factors[movie_idx]
        )
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for user-movie pair."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return self.global_mean
        
        user_idx = self.user_id_map[user_id]
        movie_idx = self.movie_id_map[movie_id]
        
        return self._predict_internal(user_idx, movie_idx)
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for a user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in self.user_id_map:
            return []
        
        user_idx = self.user_id_map[user_id]
        scores = np.dot(self.user_factors[user_idx], self.item_factors.T) + self.global_mean
        
        # Get top N
        top_indices = np.argsort(scores)[-n_recommendations:][::-1]
        
        # Map back to movie IDs
        idx_to_movie = {v: k for k, v in self.movie_id_map.items()}
        recommendations = [(idx_to_movie[idx], scores[idx]) for idx in top_indices]
        
        return recommendations
