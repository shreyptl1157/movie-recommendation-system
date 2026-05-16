import pytest
import pandas as pd
import numpy as np
from models import UserBasedCF, ItemBasedCF, SVDRecommender

@pytest.fixture
def sample_ratings():
    """Create sample ratings data."""
    return pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'movie_id': [1, 2, 3, 1, 2, 4, 2, 3, 4],
        'rating': [5, 4, 3, 5, 3, 4, 4, 5, 2],
        'timestamp': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    })

@pytest.fixture
def sample_movies():
    """Create sample movies data."""
    return pd.DataFrame({
        'movie_id': [1, 2, 3, 4],
        'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4'],
        'genres': ['Action', 'Comedy', 'Drama', 'Sci-Fi']
    })

def test_user_based_cf_fit(sample_ratings, sample_movies):
    """Test user-based CF fitting."""
    cf = UserBasedCF()
    cf.fit(sample_ratings, sample_movies)
    assert cf.is_fitted
    assert cf.user_item_matrix is not None

def test_user_based_cf_recommend(sample_ratings, sample_movies):
    """Test user-based CF recommendations."""
    cf = UserBasedCF()
    cf.fit(sample_ratings, sample_movies)
    
    recs = cf.recommend(user_id=1, n_recommendations=5)
    assert isinstance(recs, list)

def test_item_based_cf_fit(sample_ratings, sample_movies):
    """Test item-based CF fitting."""
    cf = ItemBasedCF()
    cf.fit(sample_ratings, sample_movies)
    assert cf.is_fitted

def test_svd_fit(sample_ratings, sample_movies):
    """Test SVD fitting."""
    svd = SVDRecommender(n_epochs=5, n_factors=10)
    svd.fit(sample_ratings, sample_movies)
    assert svd.is_fitted

def test_svd_predict(sample_ratings, sample_movies):
    """Test SVD prediction."""
    svd = SVDRecommender(n_epochs=5, n_factors=10)
    svd.fit(sample_ratings, sample_movies)
    
    pred = svd.predict(user_id=1, movie_id=4)
    assert isinstance(pred, (float, np.floating))
