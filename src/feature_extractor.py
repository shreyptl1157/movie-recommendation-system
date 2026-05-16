import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and engineer features for recommendation models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.genre_encoder = MultiLabelBinarizer()
        self.feature_names = []
    
    def extract_movie_features(
        self,
        movies_df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from movie metadata.
        
        Args:
            movies_df: Movies dataframe with metadata
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []
        feature_names = []
        
        # Movie ID
        features.append(movies_df['movie_id'].values.reshape(-1, 1))
        feature_names.append('movie_id')
        
        # Year
        if 'year' in movies_df.columns:
            year_features = movies_df['year'].values.reshape(-1, 1)
            year_features = self.scaler.fit_transform(year_features)
            features.append(year_features)
            feature_names.append('year')
        
        # Genre (one-hot encoding)
        if 'genres' in movies_df.columns:
            genres_list = [str(g).split('|') for g in movies_df['genres']]
            genre_features = self.genre_encoder.fit_transform(genres_list)
            features.append(genre_features)
            feature_names.extend(
                [f'genre_{g}' for g in self.genre_encoder.classes_]
            )
        
        # Combine features
        feature_matrix = np.hstack(features)
        logger.info(f"Extracted movie features: {feature_matrix.shape}")
        
        return feature_matrix, feature_names
    
    def extract_user_features(
        self,
        ratings_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from user data and ratings.
        
        Args:
            ratings_df: Ratings dataframe
            users_df: Optional users dataframe with metadata
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []
        feature_names = []
        
        # Get unique users
        unique_users = sorted(ratings_df['user_id'].unique())
        user_features_dict = {}
        
        # Rating behavior features
        for user_id in unique_users:
            user_ratings = ratings_df[ratings_df['user_id'] == user_id]['rating']
            
            user_features = [
                user_id,
                len(user_ratings),  # number of ratings
                user_ratings.mean(),  # average rating
                user_ratings.std(),  # rating variance
                user_ratings.min(),  # minimum rating
                user_ratings.max(),  # maximum rating
            ]
            user_features_dict[user_id] = user_features
        
        # Convert to array
        feature_array = np.array([user_features_dict[uid] for uid in unique_users])
        feature_names = [
            'user_id',
            'num_ratings',
            'mean_rating',
            'std_rating',
            'min_rating',
            'max_rating'
        ]
        
        # Scale numeric features
        numeric_features = feature_array[:, 1:]
        numeric_features = self.scaler.fit_transform(numeric_features)
        
        feature_matrix = np.hstack([
            feature_array[:, 0].reshape(-1, 1),
            numeric_features
        ])
        
        logger.info(f"Extracted user features: {feature_matrix.shape}")
        
        return feature_matrix, feature_names
    
    def create_interaction_features(
        self,
        ratings_df: pd.DataFrame,
        movies_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Create interaction features from ratings and movies.
        
        Args:
            ratings_df: Ratings dataframe
            movies_df: Optional movies metadata
            
        Returns:
            Interaction features dataframe
        """
        features_df = ratings_df.copy()
        
        # Temporal features
        if 'timestamp' in features_df.columns:
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
            features_df['month'] = features_df['timestamp'].dt.month
            features_df['year'] = features_df['timestamp'].dt.year
        
        # Merge movie features
        if movies_df is not None:
            features_df = features_df.merge(
                movies_df,
                on='movie_id',
                how='left'
            )
        
        # Rating level features
        features_df['is_high_rating'] = (features_df['rating'] >= 4).astype(int)
        features_df['is_low_rating'] = (features_df['rating'] <= 2).astype(int)
        
        logger.info(f"Created interaction features: {features_df.shape}")
        
        return features_df
