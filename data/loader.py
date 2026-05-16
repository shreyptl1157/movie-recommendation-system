import pandas as pd
import os
from pathlib import Path
from typing import Tuple, Optional

class DataLoader:
    """Load and manage datasets."""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def load_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        return pd.read_csv(filepath, **kwargs)
    
    def load_ratings_movies(
        self,
        ratings_path: str,
        movies_path: str,
        sep: str = ','
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load ratings and movies data."""
        ratings = pd.read_csv(ratings_path, sep=sep)
        movies = pd.read_csv(movies_path, sep=sep)
        return ratings, movies
    
    def get_file_path(self, filename: str) -> str:
        """Get full path for file in data directory."""
        return str(self.data_path / filename)
    
    def file_exists(self, filename: str) -> bool:
        """Check if file exists in data directory."""
        return (self.data_path / filename).exists()
