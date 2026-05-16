import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    
    # Flask
    SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
    TESTING = os.getenv('API_TESTING', 'False') == 'True'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'postgresql://localhost/movie_recommendations'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ECHO = os.getenv('DATABASE_ECHO', 'False') == 'True'
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': int(os.getenv('DATABASE_POOL_SIZE', 10)),
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': int(os.getenv('DATABASE_MAX_OVERFLOW', 20)),
    }
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TTL = int(os.getenv('REDIS_CACHE_TTL', 3600))
    RECOMMENDATION_TTL = int(os.getenv('REDIS_RECOMMENDATION_TTL', 1800))
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-secret-key')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(
        seconds=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600))
    )
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', 10485760))
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 10))
    
    # Models
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/saved_models')
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'hybrid')
    NUM_RECOMMENDATIONS = int(os.getenv('NUM_RECOMMENDATIONS', 10))
    MIN_RATINGS_FOR_CF = int(os.getenv('MIN_RATINGS_FOR_CF', 5))
    COLD_START_STRATEGY = os.getenv('COLD_START_STRATEGY', 'popularity')
    
    # Data
    DATA_PATH = os.getenv('DATA_PATH', 'data')
    TRAIN_TEST_SPLIT = float(os.getenv('TRAIN_TEST_SPLIT', 0.2))
    VALIDATION_SPLIT = float(os.getenv('VALIDATION_SPLIT', 0.1))
    RANDOM_SEED = int(os.getenv('RANDOM_SEED', 42))
    
    # Training
    EPOCHS = int(os.getenv('EPOCHS', 50))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 128))
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.001))
    EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', 5))
    
    # Monitoring
    MONITORING_ENABLED = os.getenv('MONITORING_ENABLED', 'True') == 'True'
    SENTRY_DSN = os.getenv('SENTRY_DSN', '')
    PROMETHEUS_ENABLED = os.getenv('PROMETHEUS_ENABLED', 'True') == 'True'
    
    # API
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_WORKERS = int(os.getenv('API_WORKERS', 4))
    
    # Environment
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_ECHO = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_ECHO = False
    LOG_LEVEL = 'WARNING'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    CACHE_TTL = 10
    RECOMMENDATION_TTL = 10


def get_config():
    """Get configuration based on environment"""
    env = os.getenv('ENVIRONMENT', 'development')
    
    if env == 'production':
        return ProductionConfig()
    elif env == 'testing':
        return TestingConfig()
    else:
        return DevelopmentConfig()
