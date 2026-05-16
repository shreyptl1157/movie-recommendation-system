import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple
from src.utils.logger import get_logger
from src.config import get_settings

logger = get_logger(__name__)
settings = get_settings()

class FeatureEngineer:
    """Feature engineering utilities"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.genre_vectors = None
        self.description_vectors = None
    
    def extract_genres(self, movies: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and encode genres.
        
        Args:
            movies: Movies dataframe
            
        Returns:
            Dataframe with genre features
        """
        logger.info('Extracting genre features')
        
        # Get all unique genres
        all_genres = set()
        for genres in movies['genres'].dropna():
            if isinstance(genres, str):
                all_genres.update(genres.split('|'))
        
        # Create binary features for each genre
        for genre in sorted(all_genres):
            movies[f'genre_{genre}'] = movies['genres'].apply(
                lambda x: 1 if isinstance(x, str) and genre in x else 0
            )
        
        logger.info(f'Extracted {len(all_genres)} genres')
        return movies
    
    def tfidf_description(self, movies: pd.DataFrame) -> np.ndarray:
        """
        Create TF-IDF vectors from movie descriptions.
        
        Args:
            movies: Movies dataframe
            
        Returns:
            TF-IDF matrix
        """
        logger.info('Computing TF-IDF vectors from descriptions')
        
        descriptions = movies['description'].fillna('')
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=settings.TF_IDF_MAX_FEATURES,
            min_df=settings.TF_IDF_MIN_DF,
            max_df=settings.TF_IDF_MAX_DF,
            stop_words='english'
        )
        
        self.description_vectors = self.tfidf_vectorizer.fit_transform(descriptions)
        logger.info(f'Created TF-IDF vectors with shape {self.description_vectors.shape}')
        
        return self.description_vectors
    
    def create_user_features(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Create user-based features.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            User features dataframe
        """
        logger.info('Creating user features')
        
        user_features = ratings.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()
        
        user_features.columns = ['user_id', 'avg_rating', 'std_rating', 
                                  'num_ratings', 'min_rating', 'max_rating']
        
        # Handle NaN std values
        user_features['std_rating'].fillna(0, inplace=True)
        
        logger.info(f'Created features for {len(user_features)} users')
        return user_features
    
    def create_movie_features(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Create movie-based features.
        
        Args:
            ratings: Ratings dataframe
            
        Returns:
            Movie features dataframe
        """
        logger.info('Creating movie features')
        
        movie_features = ratings.groupby('movie_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max']
        }).reset_index()
        
        movie_features.columns = ['movie_id', 'avg_rating', 'std_rating', 
                                   'num_ratings', 'min_rating', 'max_rating']
        
        # Handle NaN std values
        movie_features['std_rating'].fillna(0, inplace=True)
        
        logger.info(f'Created features for {len(movie_features)} movies')
        return movie_features
    
    def engineer_all_features(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Engineer all features.
        
        Args:
            ratings: Ratings dataframe
            movies: Movies dataframe
            
        Returns:
            Tuple of (ratings, movies with genres, movie features)
        """
        logger.info('Engineering all features')
        
        # Extract genres
        movies = self.extract_genres(movies)
        
        # Create TF-IDF description vectors
        self.tfidf_description(movies)
        
        # Create user and movie features
        user_features = self.create_user_features(ratings)
        movie_features = self.create_movie_features(ratings)
        
        logger.info('Feature engineering complete')
        return ratings, movies, movie_features
