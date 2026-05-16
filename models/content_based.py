import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from .base import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using movie features."""
    
    def __init__(self, min_df: int = 5, max_df: float = 0.8, max_features: int = 5000):
        super().__init__("Content-Based")
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        
        self.vectorizer = None
        self.tfidf_matrix = None
        self.movie_similarity = None
        self.movies_df = None
        self.movie_ids = None
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None) -> None:
        """Fit content-based model."""
        if movies_df is None:
            raise ValueError("Content-based recommender requires movies_df")
        
        self.movies_df = movies_df.copy()
        self.movie_ids = movies_df['movie_id'].tolist()
        
        # Prepare content features (genres and other metadata)
        content = movies_df['genres'].fillna('')
        
        # TF-IDF vectorization
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 3),
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(content)
        
        # Calculate movie similarity
        self.movie_similarity = cosine_similarity(self.tfidf_matrix)
        
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations based on user history."""
        # This would require user's watched movies
        # For now, return empty - would need to integrate with ratings
        return []
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating based on content similarity."""
        return 0.0  # Would need user history
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Get content-similar movies."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if item_id not in self.movie_ids:
            return []
        
        movie_idx = self.movie_ids.index(item_id)
        similarities = self.movie_similarity[movie_idx]
        
        # Get top N similar (excluding the movie itself)
        top_indices = np.argsort(similarities)[-n_similar-1:-1][::-1]
        
        recommendations = [
            (self.movie_ids[idx], similarities[idx])
            for idx in top_indices
        ]
        return recommendations
