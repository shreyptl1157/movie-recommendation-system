from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
import os

class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    api_workers: int = 4
    
    # Database Configuration
    database_url: str = "postgresql://postgres:password@localhost:5432/movie_recommender"
    database_pool_size: int = 20
    database_max_overflow: int = 40
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600
    
    # Model Configuration
    svd_n_factors: int = 100
    svd_n_epochs: int = 20
    svd_lr_all: float = 0.01
    svd_reg_all: float = 0.02
    svd_random_state: int = 42
    
    # Content-Based Configuration
    content_min_df: int = 5
    content_max_df: float = 0.8
    content_max_features: int = 5000
    
    # Collaborative Configuration
    collab_k_neighbors: int = 10
    collab_min_common_items: int = 2
    
    # Data Configuration
    dataset: str = "movielens-1m"
    data_path: str = "./data"
    data_download: bool = True
    
    # Feature Engineering
    feature_engineer_enable: bool = True
    feature_engineer_genres: bool = True
    feature_engineer_temporal: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    log_format: str = "json"
    
    # Evaluation
    eval_k_values: List[int] = [5, 10, 20]
    eval_threshold: float = 3.5
    
    # Performance
    max_recommendations: int = 100
    default_n_recommendations: int = 10
    recommendation_timeout: int = 30
    
    # Security
    secret_key: str = "your-secret-key-change-this"
    algorithm: str = "HS256"
    token_expire_minutes: int = 1440
    
    # Environment
    environment: str = "development"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
