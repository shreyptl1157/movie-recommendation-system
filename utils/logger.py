import logging
import sys
from pathlib import Path
from loguru import logger as loguru_logger

def setup_logger(name: str = "movie-recommender", level: str = "INFO", log_file: str = None):
    """Set up logging configuration."""
    
    # Remove default handlers
    loguru_logger.remove()
    
    # Add console handler
    loguru_logger.add(
        sys.stdout,
        level=level,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler if specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        loguru_logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="500 MB",
            retention="7 days"
        )
    
    return loguru_logger
