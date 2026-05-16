import os
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pickle
import json

logger = logging.getLogger(__name__)


def setup_logger(name: str, level: str = 'INFO') -> logging.Logger:
    """Setup logger with handlers.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
    """
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.info(f"Saved to {filepath}")


def load_pickle(filepath: str) -> Any:
    """Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    logger.info(f"Loaded from {filepath}")
    return obj


def save_json(data: Dict, filepath: str) -> None:
    """Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved to {filepath}")


def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded from {filepath}")
    return data


def normalize_ratings(ratings: np.ndarray, min_rating: float = 0, max_rating: float = 5) -> np.ndarray:
    """Normalize ratings to [0, 5] scale.
    
    Args:
        ratings: Ratings array
        min_rating: Minimum rating value
        max_rating: Maximum rating value
        
    Returns:
        Normalized ratings
    """
    ratings = np.array(ratings, dtype=float)
    current_min = np.nanmin(ratings)
    current_max = np.nanmax(ratings)
    
    if current_min == current_max:
        return np.ones_like(ratings) * (min_rating + max_rating) / 2
    
    normalized = (ratings - current_min) / (current_max - current_min)
    normalized = normalized * (max_rating - min_rating) + min_rating
    
    return normalized


def denormalize_ratings(ratings: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
    """Denormalize ratings from [0, 5] to original scale.
    
    Args:
        ratings: Normalized ratings
        original_min: Original minimum value
        original_max: Original maximum value
        
    Returns:
        Denormalized ratings
    """
    ratings = np.array(ratings, dtype=float)
    denormalized = (ratings - 0) / 5 * (original_max - original_min) + original_min
    return denormalized


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.
    
    Args:
        df: Input dataframe
        train_ratio: Training set ratio
        test_ratio: Test set ratio
        val_ratio: Validation set ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert train_ratio + test_ratio + val_ratio == 1.0, "Ratios must sum to 1.0"
    
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    train_idx = int(len(df) * train_ratio)
    val_idx = train_idx + int(len(df) * val_ratio)
    
    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]
    
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_df, val_df, test_df


def create_user_item_matrix(
    ratings_df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'movie_id',
    rating_col: str = 'rating'
) -> Tuple[np.ndarray, Dict, Dict]:
    """Create user-item interaction matrix.
    
    Args:
        ratings_df: Ratings dataframe
        user_col: User column name
        item_col: Item column name
        rating_col: Rating column name
        
    Returns:
        Tuple of (matrix, user_id_map, item_id_map)
    """
    user_ids = ratings_df[user_col].unique()
    item_ids = ratings_df[item_col].unique()
    
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
    
    matrix = np.zeros((len(user_ids), len(item_ids)))
    
    for _, row in ratings_df.iterrows():
        user_idx = user_id_map[row[user_col]]
        item_idx = item_id_map[row[item_col]]
        matrix[user_idx, item_idx] = row[rating_col]
    
    logger.info(f"Created user-item matrix: {matrix.shape}")
    
    return matrix, user_id_map, item_id_map


def get_sparsity(matrix: np.ndarray) -> float:
    """Calculate matrix sparsity.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Sparsity ratio (0-1)
    """
    total_elements = matrix.size
    non_zero_elements = np.count_nonzero(matrix)
    sparsity = 1 - (non_zero_elements / total_elements)
    return sparsity


def get_density(matrix: np.ndarray) -> float:
    """Calculate matrix density.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Density ratio (0-1)
    """
    return 1 - get_sparsity(matrix)


def get_top_k(
    scores: np.ndarray,
    k: int = 10,
    indices: Optional[np.ndarray] = None
) -> List[Tuple[int, float]]:
    """Get top k items by score.
    
    Args:
        scores: Score array
        k: Number of top items
        indices: Optional item indices mapping
        
    Returns:
        List of (item_id, score) tuples
    """
    if len(scores) == 0:
        return []
    
    k = min(k, len(scores))
    top_indices = np.argsort(scores)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            item_id = indices[idx] if indices is not None else idx
            results.append((item_id, float(scores[idx])))
    
    return results


def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray, method: str = 'cosine') -> float:
    """Calculate similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        method: Similarity method ('cosine', 'euclidean', 'pearson')
        
    Returns:
        Similarity score
    """
    if method == 'cosine':
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    elif method == 'euclidean':
        distance = np.linalg.norm(vec1 - vec2)
        return float(1 / (1 + distance))
    
    elif method == 'pearson':
        if len(vec1) < 2 or len(vec2) < 2:
            return 0.0
        return float(np.corrcoef(vec1, vec2)[0, 1] or 0)
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def get_timestamp() -> str:
    """Get current timestamp as string.
    
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def print_config(config: Dict) -> None:
    """Print configuration in readable format.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=" * 50)
    logger.info("Configuration")
    logger.info("=" * 50)
    for key, value in sorted(config.items()):
        if 'password' not in key.lower() and 'secret' not in key.lower():
            logger.info(f"{key}: {value}")
    logger.info("=" * 50)
