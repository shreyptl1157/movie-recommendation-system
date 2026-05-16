import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataProcessor:
    """Data preprocessing and transformation utilities."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rating_stats = {}
        self.user_stats = {}
        self.item_stats = {}
    
    def load_data(
        self,
        ratings_path: str,
        movies_path: Optional[str] = None,
        users_path: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files.
        
        Args:
            ratings_path: Path to ratings CSV
            movies_path: Path to movies CSV
            users_path: Path to users CSV
            
        Returns:
            Dictionary of loaded dataframes
        """
        data = {}
        
        # Load ratings
        logger.info(f"Loading ratings from {ratings_path}")
        data['ratings'] = pd.read_csv(ratings_path)
        logger.info(f"Loaded {len(data['ratings'])} ratings")
        
        # Load movies metadata
        if movies_path and Path(movies_path).exists():
            logger.info(f"Loading movies from {movies_path}")
            data['movies'] = pd.read_csv(movies_path)
            logger.info(f"Loaded {len(data['movies'])} movies")
        
        # Load user data
        if users_path and Path(users_path).exists():
            logger.info(f"Loading users from {users_path}")
            data['users'] = pd.read_csv(users_path)
            logger.info(f"Loaded {len(data['users'])} users")
        
        return data
    
    def clean_ratings(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Clean ratings data.
        
        Args:
            ratings_df: Input ratings dataframe
            
        Returns:
            Cleaned ratings dataframe
        """
        original_len = len(ratings_df)
        logger.info(f"Cleaning ratings data: {original_len} records")
        
        # Remove duplicates
        ratings_df = ratings_df.drop_duplicates(
            subset=['user_id', 'movie_id'],
            keep='last'
        )
        logger.info(f"Removed duplicates: {original_len - len(ratings_df)} records")
        
        # Remove NaN values
        ratings_df = ratings_df.dropna(subset=['user_id', 'movie_id', 'rating'])
        logger.info(f"Removed NaN values: {len(ratings_df)} records remaining")
        
        # Validate ratings are in expected range
        ratings_df = ratings_df[
            (ratings_df['rating'] >= 0) & (ratings_df['rating'] <= 5)
        ]
        logger.info(f"Validated ratings range: {len(ratings_df)} records remaining")
        
        return ratings_df.reset_index(drop=True)
    
    def filter_by_user_activity(
        self,
        ratings_df: pd.DataFrame,
        min_ratings: int = 5
    ) -> pd.DataFrame:
        """Filter users by minimum number of ratings.
        
        Args:
            ratings_df: Input ratings dataframe
            min_ratings: Minimum number of ratings per user
            
        Returns:
            Filtered ratings dataframe
        """
        original_len = len(ratings_df)
        user_counts = ratings_df.groupby('user_id').size()
        valid_users = user_counts[user_counts >= min_ratings].index
        
        ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
        logger.info(
            f"Filtered by user activity (min_ratings={min_ratings}): "
            f"{original_len - len(ratings_df)} records removed"
        )
        
        return ratings_df.reset_index(drop=True)
    
    def filter_by_item_popularity(
        self,
        ratings_df: pd.DataFrame,
        min_ratings: int = 2
    ) -> pd.DataFrame:
        """Filter items by minimum number of ratings.
        
        Args:
            ratings_df: Input ratings dataframe
            min_ratings: Minimum number of ratings per item
            
        Returns:
            Filtered ratings dataframe
        """
        original_len = len(ratings_df)
        item_counts = ratings_df.groupby('movie_id').size()
        valid_items = item_counts[item_counts >= min_ratings].index
        
        ratings_df = ratings_df[ratings_df['movie_id'].isin(valid_items)]
        logger.info(
            f"Filtered by item popularity (min_ratings={min_ratings}): "
            f"{original_len - len(ratings_df)} records removed"
        )
        
        return ratings_df.reset_index(drop=True)
    
    def calculate_statistics(self, ratings_df: pd.DataFrame) -> None:
        """Calculate and store data statistics.
        
        Args:
            ratings_df: Ratings dataframe
        """
        # Rating statistics
        self.rating_stats = {
            'mean': ratings_df['rating'].mean(),
            'std': ratings_df['rating'].std(),
            'min': ratings_df['rating'].min(),
            'max': ratings_df['rating'].max(),
            'median': ratings_df['rating'].median(),
        }
        
        # User statistics
        user_rating_counts = ratings_df.groupby('user_id').size()
        self.user_stats = {
            'count': len(user_rating_counts),
            'mean_ratings': user_rating_counts.mean(),
            'std_ratings': user_rating_counts.std(),
            'min_ratings': user_rating_counts.min(),
            'max_ratings': user_rating_counts.max(),
        }
        
        # Item statistics
        item_rating_counts = ratings_df.groupby('movie_id').size()
        self.item_stats = {
            'count': len(item_rating_counts),
            'mean_ratings': item_rating_counts.mean(),
            'std_ratings': item_rating_counts.std(),
            'min_ratings': item_rating_counts.min(),
            'max_ratings': item_rating_counts.max(),
        }
        
        logger.info("Statistics calculated")
    
    def print_statistics(self) -> None:
        """Print data statistics."""
        logger.info("=" * 50)
        logger.info("Rating Statistics")
        logger.info("=" * 50)
        for key, value in self.rating_stats.items():
            logger.info(f"{key}: {value:.4f}")
        
        logger.info("\n" + "=" * 50)
        logger.info("User Statistics")
        logger.info("=" * 50)
        for key, value in self.user_stats.items():
            logger.info(f"{key}: {value:.4f}")
        
        logger.info("\n" + "=" * 50)
        logger.info("Item Statistics")
        logger.info("=" * 50)
        for key, value in self.item_stats.items():
            logger.info(f"{key}: {value:.4f}")
    
    def split_data(
        self,
        ratings_df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            ratings_df: Ratings dataframe
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert train_ratio + val_ratio + test_ratio == 1.0
        
        df = ratings_df.sample(
            frac=1,
            random_state=self.random_state
        ).reset_index(drop=True)
        
        train_idx = int(len(df) * train_ratio)
        val_idx = train_idx + int(len(df) * val_ratio)
        
        train_df = df[:train_idx]
        val_df = df[train_idx:val_idx]
        test_df = df[val_idx:]
        
        logger.info(
            f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def create_user_item_matrix(
        self,
        ratings_df: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict, Dict]:
        """Create user-item interaction matrix.
        
        Args:
            ratings_df: Ratings dataframe
            
        Returns:
            Tuple of (matrix, user_id_map, item_id_map)
        """
        user_ids = ratings_df['user_id'].unique()
        item_ids = ratings_df['movie_id'].unique()
        
        user_id_map = {uid: idx for idx, uid in enumerate(sorted(user_ids))}
        item_id_map = {iid: idx for idx, iid in enumerate(sorted(item_ids))}
        
        matrix = np.zeros((len(user_ids), len(item_ids)))
        
        for _, row in ratings_df.iterrows():
            user_idx = user_id_map[row['user_id']]
            item_idx = item_id_map[row['movie_id']]
            matrix[user_idx, item_idx] = row['rating']
        
        logger.info(f"Created user-item matrix: {matrix.shape}")
        
        return matrix, user_id_map, item_id_map
