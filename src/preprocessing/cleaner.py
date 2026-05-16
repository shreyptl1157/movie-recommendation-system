import pandas as pd
import numpy as np
from typing import List, Tuple
from src.utils.logger import get_logger

logger = get_logger(__name__)

class DataCleaner:
    """Data cleaning utilities"""
    
    def __init__(self, min_rating_count: int = 5, min_user_ratings: int = 1):
        """
        Initialize cleaner.
        
        Args:
            min_rating_count: Minimum number of ratings per movie
            min_user_ratings: Minimum number of ratings per user
        """
        self.min_rating_count = min_rating_count
        self.min_user_ratings = min_user_ratings
    
    def clean_ratings(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Clean ratings data.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            Cleaned ratings dataframe
        """
        logger.info(f'Original ratings shape: {ratings.shape}')
        
        # Remove duplicates
        ratings = ratings.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
        logger.info(f'Removed duplicates: {ratings.shape}')
        
        # Remove null values
        ratings = ratings.dropna(subset=['user_id', 'movie_id', 'rating'])
        logger.info(f'Removed nulls: {ratings.shape}')
        
        # Filter by minimum ratings per movie
        movie_counts = ratings['movie_id'].value_counts()
        valid_movies = movie_counts[movie_counts >= self.min_rating_count].index
        ratings = ratings[ratings['movie_id'].isin(valid_movies)]
        logger.info(f'Filtered by min movie ratings ({self.min_rating_count}): {ratings.shape}')
        
        # Filter by minimum ratings per user
        user_counts = ratings['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_ratings].index
        ratings = ratings[ratings['user_id'].isin(valid_users)]
        logger.info(f'Filtered by min user ratings ({self.min_user_ratings}): {ratings.shape}')
        
        return ratings
    
    def clean_movies(self, movies: pd.DataFrame, valid_movie_ids: List[int]) -> pd.DataFrame:
        """
        Clean movies data.
        
        Args:
            movies: Movies dataframe
            valid_movie_ids: List of valid movie IDs from cleaned ratings
            
        Returns:
            Cleaned movies dataframe
        """
        logger.info(f'Original movies shape: {movies.shape}')
        
        # Remove null titles
        movies = movies.dropna(subset=['title'])
        logger.info(f'Removed null titles: {movies.shape}')
        
        # Keep only movies with ratings
        movies = movies[movies['movie_id'].isin(valid_movie_ids)]
        logger.info(f'Kept only movies with ratings: {movies.shape}')
        
        return movies
    
    def normalize_ratings(self, ratings: pd.DataFrame, min_val: float = 0, max_val: float = 5) -> pd.DataFrame:
        """
        Normalize rating values to specified range.
        
        Args:
            ratings: Ratings dataframe
            min_val: Minimum rating value
            max_val: Maximum rating value
            
        Returns:
            Normalized ratings dataframe
        """
        original_min = ratings['rating'].min()
        original_max = ratings['rating'].max()
        
        ratings['rating'] = (
            (ratings['rating'] - original_min) / (original_max - original_min)
        ) * (max_val - min_val) + min_val
        
        logger.info(f'Normalized ratings to range [{min_val}, {max_val}]')
        return ratings
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
        """
        Handle missing values.
        
        Args:
            df: Input dataframe
            strategy: 'drop', 'mean', 'median', 'ffill'
            
        Returns:
            Dataframe with handled missing values
        """
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'ffill':
            df = df.fillna(method='ffill')
        
        logger.info(f'Handled missing values with strategy: {strategy}')
        return df
