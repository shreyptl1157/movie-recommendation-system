import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
from .base import BaseRecommender

class UserBasedCF(BaseRecommender):
    """User-based Collaborative Filtering recommender."""
    
    def __init__(self, k_neighbors: int = 10, min_common_items: int = 2):
        super().__init__("User-Based CF")
        self.k_neighbors = k_neighbors
        self.min_common_items = min_common_items
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_means = None
        self.users_list = None
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None) -> None:
        """Fit the recommender to data."""
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
        self.users_list = self.user_item_matrix.index.tolist()
        self.movies_df = movies_df
        
        # Calculate user means for normalization
        self.user_means = self.user_item_matrix.mean(axis=1)
        
        # Normalize ratings
        normalized = self.user_item_matrix.sub(self.user_means, axis=0)
        
        # Calculate user similarity
        self.user_similarity = cosine_similarity(normalized)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.users_list,
            columns=self.users_list
        )
        
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for a user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in self.users_list:
            return []
        
        # Get similar users
        similar_users = self.user_similarity[user_id].drop(user_id).nlargest(self.k_neighbors)
        similar_users = similar_users[similar_users > 0]
        
        # Get movies rated by similar users but not by target user
        user_rated = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
        
        recommendations = {}
        for sim_user, similarity in similar_users.items():
            sim_user_ratings = self.user_item_matrix.loc[sim_user]
            sim_user_rated = set(sim_user_ratings[sim_user_ratings > 0].index)
            candidate_movies = sim_user_rated - user_rated
            
            for movie_id in candidate_movies:
                score = sim_user_ratings[movie_id] * similarity
                if movie_id in recommendations:
                    recommendations[movie_id] += score
                else:
                    recommendations[movie_id] = score
        
        # Sort and return top N
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for user-movie pair."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in self.users_list:
            return self.user_means.mean()
        
        # Get similar users who rated this movie
        similar_users = self.user_similarity[user_id].drop(user_id).nlargest(self.k_neighbors)
        similar_users = similar_users[similar_users > 0]
        
        ratings = []
        for sim_user, similarity in similar_users.items():
            if movie_id in self.user_item_matrix.columns:
                rating = self.user_item_matrix.loc[sim_user, movie_id]
                if rating > 0:
                    ratings.append((rating, similarity))
        
        if not ratings:
            return self.user_means[user_id]
        
        # Weighted average
        total_similarity = sum(sim for _, sim in ratings)
        prediction = sum(rating * sim for rating, sim in ratings) / total_similarity
        return prediction


class ItemBasedCF(BaseRecommender):
    """Item-based Collaborative Filtering recommender."""
    
    def __init__(self, k_neighbors: int = 10):
        super().__init__("Item-Based CF")
        self.k_neighbors = k_neighbors
        self.item_similarity = None
        self.user_item_matrix = None
        self.items_list = None
    
    def fit(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame = None) -> None:
        """Fit the recommender to data."""
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
        self.items_list = self.user_item_matrix.columns.tolist()
        self.movies_df = movies_df
        
        # Calculate item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.items_list,
            columns=self.items_list
        )
        
        self.is_fitted = True
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """Get recommendations for a user."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get movies rated by user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
        
        if not rated_movies:
            return []
        
        # Calculate scores for unrated movies
        recommendations = {}
        for rated_movie in rated_movies:
            similar_items = self.item_similarity[rated_movie].drop(rated_movie).nlargest(self.k_neighbors)
            user_rating = user_ratings[rated_movie]
            
            for movie_id, similarity in similar_items.items():
                if user_ratings[movie_id] == 0:  # Unrated
                    score = user_rating * similarity
                    if movie_id in recommendations:
                        recommendations[movie_id] += score
                    else:
                        recommendations[movie_id] = score
        
        # Sort and return top N
        recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict rating for user-movie pair."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if user_id not in self.user_item_matrix.index:
            return self.user_item_matrix.mean().mean()
        
        # Get movies rated by user
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index.tolist()
        
        if not rated_movies:
            return self.user_item_matrix.mean().mean()
        
        # Get similar items
        similar_items = self.item_similarity[movie_id].drop(movie_id).nlargest(self.k_neighbors)
        
        ratings = []
        for item_id, similarity in similar_items.items():
            if item_id in rated_movies:
                rating = user_ratings[item_id]
                ratings.append((rating, similarity))
        
        if not ratings:
            return self.user_item_matrix.mean().mean()
        
        # Weighted average
        total_similarity = sum(sim for _, sim in ratings)
        prediction = sum(rating * sim for rating, sim in ratings) / total_similarity
        return prediction
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """Get items similar to given item."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        if item_id not in self.items_list:
            return []
        
        similar = self.item_similarity[item_id].drop(item_id).nlargest(n_similar)
        return list(zip(similar.index, similar.values))
