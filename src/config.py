import os
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application configuration settings"""
    
    # Application
    APP_NAME: str = os.getenv('APP_NAME', 'Movie Recommendation System')
    APP_VERSION: str = os.getenv('APP_VERSION', '1.0.0')
    DEBUG: bool = os.getenv('API_DEBUG', 'True').lower() == 'true'
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Database
    DATABASE_URL: str = os.getenv(
        'DATABASE_URL',
        'postgresql://user:password@localhost:5432/movie_recommendation'
    )
    DATABASE_ECHO: bool = os.getenv('DATABASE_ECHO', 'False').lower() == 'true'
    DATABASE_POOL_SIZE: int = int(os.getenv('DATABASE_POOL_SIZE', '20'))
    DATABASE_MAX_OVERFLOW: int = int(os.getenv('DATABASE_MAX_OVERFLOW', '10'))
    
    # Redis
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    REDIS_TTL: int = int(os.getenv('REDIS_TTL', '3600'))
    REDIS_SOCKET_TIMEOUT: int = int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
    
    # API
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    API_WORKERS: int = int(os.getenv('API_WORKERS', '4'))
    API_RELOAD: bool = os.getenv('API_DEBUG', 'True').lower() == 'true'
    
    # CORS
    ALLOWED_ORIGINS: list = [
        'http://localhost:3000',
        'http://localhost:8000',
        'http://127.0.0.1:3000',
        'http://127.0.0.1:8000',
    ]
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: list = ['*']
    ALLOW_HEADERS: list = ['*']
    
    # Models
    MODEL_TYPE: str = os.getenv('MODEL_TYPE', 'hybrid')
    CF_ALGORITHM: str = os.getenv('CF_ALGORITHM', 'svd')
    N_FACTORS: int = int(os.getenv('N_FACTORS', '50'))
    N_EPOCHS: int = int(os.getenv('N_EPOCHS', '100'))
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '32'))
    LEARNING_RATE: float = float(os.getenv('LEARNING_RATE', '0.001'))
    DROPOUT_RATE: float = float(os.getenv('DROPOUT_RATE', '0.2'))
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH: str = os.path.join(BASE_DIR, os.getenv('DATA_PATH', 'data'))
    MODEL_PATH: str = os.path.join(BASE_DIR, os.getenv('MODEL_PATH', 'data/models'))
    LOG_PATH: str = os.path.join(BASE_DIR, 'logs')
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/app.log')
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Recommendations
    DEFAULT_N_RECOMMENDATIONS: int = int(os.getenv('DEFAULT_N_RECOMMENDATIONS', '10'))
    MIN_RATING_COUNT: int = int(os.getenv('MIN_RATING_COUNT', '5'))
    
    # Evaluation
    TEST_SIZE: float = float(os.getenv('TEST_SIZE', '0.2'))
    RANDOM_STATE: int = int(os.getenv('RANDOM_STATE', '42'))
    N_FOLDS: int = int(os.getenv('N_FOLDS', '5'))
    
    # Feature Engineering
    TF_IDF_MAX_FEATURES: int = int(os.getenv('TF_IDF_MAX_FEATURES', '1000'))
    TF_IDF_MIN_DF: int = int(os.getenv('TF_IDF_MIN_DF', '2'))
    TF_IDF_MAX_DF: float = float(os.getenv('TF_IDF_MAX_DF', '0.8'))
    
    # Similarity
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.0'))
    
    # Cache
    ENABLE_CACHE: bool = os.getenv('ENABLE_CACHE', 'True').lower() == 'true'
    CACHE_RECOMMENDATIONS: bool = os.getenv('CACHE_RECOMMENDATIONS', 'True').lower() == 'true'
    
    # Security
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '30'))
    
    class Config:
        env_file = '.env'
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)"""
    return Settings()

# Ensure required directories exist
def ensure_directories():
    """Create required directories if they don't exist"""
    settings = get_settings()
    for directory in [settings.DATA_PATH, settings.MODEL_PATH, settings.LOG_PATH]:
        os.makedirs(directory, exist_ok=True)

ensure_directories()
