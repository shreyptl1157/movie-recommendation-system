from fastapi import Depends
from config import get_settings
from models import HybridRecommender, UserBasedCF, ItemBasedCF, SVDRecommender, ContentBasedRecommender

# Initialize recommenders (would normally load from cache/database)
recommenders = {}

def get_recommender(algorithm: str = "hybrid"):
    """Dependency injection for recommender."""
    if algorithm not in recommenders:
        if algorithm == "user_cf":
            recommenders[algorithm] = UserBasedCF()
        elif algorithm == "item_cf":
            recommenders[algorithm] = ItemBasedCF()
        elif algorithm == "svd":
            recommenders[algorithm] = SVDRecommender()
        elif algorithm == "content":
            recommenders[algorithm] = ContentBasedRecommender()
        else:
            recommenders[algorithm] = HybridRecommender()
    
    return recommenders[algorithm]
