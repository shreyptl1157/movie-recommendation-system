import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from src.config import get_settings
from src.utils.logger import get_logger
from src.utils.helpers import ensure_dirs
import zipfile
import urllib.request

logger = get_logger(__name__)
settings = get_settings()

def download_movielens(size: str = '1m', extract_path: Optional[str] = None) -> str:
    """
    Download MovieLens dataset.
    
    Args:
        size: Dataset size ('100k', '1m', '10m', '25m')
        extract_path: Path to extract files (default: data/raw/)
        
    Returns:
        Path to extracted dataset
    """
    if extract_path is None:
        extract_path = os.path.join(settings.DATA_PATH, 'raw')
    
    ensure_dirs(extract_path)
    
    urls = {
        '100k': 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        '1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
        '10m': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
        '25m': 'http://files.grouplens.org/datasets/movielens/ml-25m.zip',
    }
    
    if size not in urls:
        raise ValueError(f'Invalid size: {size}. Must be one of {list(urls.keys())}')
    
    url = urls[size]
    zip_path = os.path.join(extract_path, f'ml-{size}.zip')
    
    logger.info(f'Downloading MovieLens {size} dataset...')
    urllib.request.urlretrieve(url, zip_path)
    logger.info(f'Downloaded to {zip_path}')
    
    logger.info(f'Extracting {zip_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    logger.info(f'Extracted to {extract_path}')
    
    # Return the extracted directory path
    if size == '100k':
        return os.path.join(extract_path, 'ml-latest-small')
    else:
        return os.path.join(extract_path, f'ml-{size}')

def load_movielens(size: str = '1m', data_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens dataset.
    
    Args:
        size: Dataset size ('100k', '1m', '10m', '25m')
        data_path: Path to dataset
        
    Returns:
        Tuple of (ratings, movies, tags)
    """
    if data_path is None:
        data_path = os.path.join(settings.DATA_PATH, 'raw', f'ml-{size}')
    
    logger.info(f'Loading MovieLens {size} dataset from {data_path}')
    
    # Load ratings
    ratings_path = os.path.join(data_path, 'ratings.dat')
    ratings = pd.read_csv(
        ratings_path,
        sep='::',
        header=None,
        names=['user_id', 'movie_id', 'rating', 'timestamp'],
        engine='python'
    )
    logger.info(f'Loaded {len(ratings)} ratings')
    
    # Load movies
    movies_path = os.path.join(data_path, 'movies.dat')
    movies = pd.read_csv(
        movies_path,
        sep='::',
        header=None,
        names=['movie_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )
    logger.info(f'Loaded {len(movies)} movies')
    
    # Load tags if available
    tags_path = os.path.join(data_path, 'tags.dat')
    if os.path.exists(tags_path):
        tags = pd.read_csv(
            tags_path,
            sep='::',
            header=None,
            names=['user_id', 'movie_id', 'tag', 'timestamp'],
            engine='python'
        )
        logger.info(f'Loaded {len(tags)} tags')
    else:
        tags = None
    
    return ratings, movies, tags

def load_custom_data(ratings_csv: str, movies_csv: str, tags_csv: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load custom dataset from CSV files.
    
    Args:
        ratings_csv: Path to ratings CSV
        movies_csv: Path to movies CSV
        tags_csv: Optional path to tags CSV
        
    Returns:
        Tuple of (ratings, movies, tags)
    """
    logger.info(f'Loading custom datasets')
    
    ratings = pd.read_csv(ratings_csv)
    logger.info(f'Loaded {len(ratings)} ratings from {ratings_csv}')
    
    movies = pd.read_csv(movies_csv)
    logger.info(f'Loaded {len(movies)} movies from {movies_csv}')
    
    tags = None
    if tags_csv and os.path.exists(tags_csv):
        tags = pd.read_csv(tags_csv)
        logger.info(f'Loaded {len(tags)} tags from {tags_csv}')
    
    return ratings, movies, tags
