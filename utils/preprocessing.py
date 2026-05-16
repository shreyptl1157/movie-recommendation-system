import pandas as pd
import numpy as np
from typing import Tuple

def load_ratings(filepath: str) -> pd.DataFrame:
    """Load ratings data from file."""
    ratings = pd.read_csv(
        filepath,
        sep='::' if 'movielens' in filepath else ',',
        engine='python',
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        dtype={'user_id': int, 'movie_id': int, 'rating': float, 'timestamp': int}
    )
    return ratings

def load_movies(filepath: str) -> pd.DataFrame:
    """Load movies data from file."""
    movies = pd.read_csv(
        filepath,
        sep='::' if 'movielens' in filepath else ',',
        engine='python',
        names=['movie_id', 'title', 'genres'],
        dtype={'movie_id': int, 'title': str, 'genres': str}
    )
    return movies

def create_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """Create user-item rating matrix."""
    matrix = ratings_df.pivot_table(
        index='user_id',
        columns='movie_id',
        values='rating'
    )
    return matrix

def normalize_ratings(ratings_df: pd.DataFrame, scale: Tuple[float, float] = (1.0, 5.0)) -> pd.DataFrame:
    """Normalize ratings to specified scale."""
    min_val, max_val = scale
    current_min = ratings_df['rating'].min()
    current_max = ratings_df['rating'].max()
    
    if current_min == current_max:
        return ratings_df.copy()
    
    ratings_df = ratings_df.copy()
    ratings_df['rating'] = min_val + (ratings_df['rating'] - current_min) * \
                          (max_val - min_val) / (current_max - current_min)
    return ratings_df

def extract_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and process genre information."""
    movies_df = movies_df.copy()
    movies_df['genres'] = movies_df['genres'].fillna('')
    return movies_df

def train_test_split(ratings_df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train and test sets."""
    np.random.seed(random_state)
    indices = np.random.rand(len(ratings_df)) < (1 - test_size)
    train = ratings_df[indices]
    test = ratings_df[~indices]
    return train, test
