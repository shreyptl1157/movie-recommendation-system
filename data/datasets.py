import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

class MovieLensDataset:
    """MovieLens dataset handler."""
    
    URLS = {
        'movielens-100k': 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip',
        'movielens-1m': 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'movielens-10m': 'http://files.grouplens.org/datasets/movielens/ml-10m.zip',
        'movielens-25m': 'http://files.grouplens.org/datasets/movielens/ml-25m.zip',
    }
    
    @staticmethod
    def download(dataset: str = 'movielens-100k', data_path: str = './data') -> bool:
        """Download MovieLens dataset."""
        if dataset not in MovieLensDataset.URLS:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        url = MovieLensDataset.URLS[dataset]
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        
        zip_path = data_path / f"{dataset}.zip"
        
        print(f"Downloading {dataset}...")
        try:
            urllib.request.urlretrieve(url, zip_path)
            
            print(f"Extracting {dataset}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            
            os.remove(zip_path)
            return True
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    @staticmethod
    def load(dataset: str = 'movielens-100k', data_path: str = './data'):
        """Load MovieLens dataset ratings and movies."""
        data_path = Path(data_path)
        
        if dataset == 'movielens-100k':
            dataset_dir = data_path / 'ml-latest-small'
            ratings_file = dataset_dir / 'ratings.csv'
            movies_file = dataset_dir / 'movies.csv'
            sep = ','
        elif dataset == 'movielens-1m':
            dataset_dir = data_path / 'ml-1m'
            ratings_file = dataset_dir / 'ratings.dat'
            movies_file = dataset_dir / 'movies.dat'
            sep = '::'
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
        
        if not ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
        
        # Load ratings
        if sep == ',':
            ratings = pd.read_csv(ratings_file)
        else:
            ratings = pd.read_csv(
                ratings_file,
                sep=sep,
                engine='python',
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
        
        # Load movies
        if sep == ',':
            movies = pd.read_csv(movies_file)
        else:
            movies = pd.read_csv(
                movies_file,
                sep=sep,
                engine='python',
                names=['movie_id', 'title', 'genres']
            )
        
        return ratings, movies
