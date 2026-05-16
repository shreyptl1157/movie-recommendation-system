import os
import json
import pickle
import time
from functools import wraps
from typing import Any, Callable
from src.utils.logger import get_logger

logger = get_logger(__name__)

def ensure_dirs(*dirs: str) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        *dirs: Directory paths to create
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f'Ensured directory exists: {directory}')

def load_pickle(filepath: str) -> Any:
    """
    Load a pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Unpickled object
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f'Loaded pickle file: {filepath}')
        return data
    except Exception as e:
        logger.error(f'Error loading pickle file {filepath}: {str(e)}')
        raise

def save_pickle(data: Any, filepath: str) -> None:
    """
    Save object to pickle file.
    
    Args:
        data: Object to pickle
        filepath: Path to save pickle file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f'Saved pickle file: {filepath}')
    except Exception as e:
        logger.error(f'Error saving pickle file {filepath}: {str(e)}')
        raise

def load_json(filepath: str) -> Any:
    """
    Load a JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON object
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f'Loaded JSON file: {filepath}')
        return data
    except Exception as e:
        logger.error(f'Error loading JSON file {filepath}: {str(e)}')
        raise

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save object to JSON file.
    
    Args:
        data: Object to serialize
        filepath: Path to save JSON file
        indent: JSON indentation level
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f'Saved JSON file: {filepath}')
    except Exception as e:
        logger.error(f'Error saving JSON file {filepath}: {str(e)}')
        raise

def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logger.debug(f'{func.__name__} took {elapsed:.2f} seconds')
        return result
    return wrapper
