from sqlalchemy.orm import Session
from src.database.models import User, Movie, Rating
from src.utils.logger import get_logger
from typing import List, Optional

logger = get_logger(__name__)

# User CRUD Operations

def get_user(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username"""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()

def get_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
    """Get all users with pagination"""
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, username: str, email: str, hashed_password: str) -> User:
    """Create a new user"""
    user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f'Created user: {username}')
    return user

def update_user(db: Session, user_id: int, **kwargs) -> Optional[User]:
    """Update user"""
    user = get_user(db, user_id)
    if user:
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        db.commit()
        db.refresh(user)
        logger.info(f'Updated user: {user_id}')
    return user

# Movie CRUD Operations

def get_movie(db: Session, movie_id: int) -> Optional[Movie]:
    """Get movie by ID"""
    return db.query(Movie).filter(Movie.id == movie_id).first()

def get_movies(db: Session, skip: int = 0, limit: int = 100) -> List[Movie]:
    """Get all movies with pagination"""
    return db.query(Movie).offset(skip).limit(limit).all()

def search_movies(db: Session, query: str, skip: int = 0, limit: int = 100) -> List[Movie]:
    """Search movies by title or description"""
    search_pattern = f'%{query}%'
    return db.query(Movie).filter(
        (Movie.title.ilike(search_pattern)) | 
        (Movie.description.ilike(search_pattern))
    ).offset(skip).limit(limit).all()

def create_movie(db: Session, title: str, description: str = None, genres: str = None,
                 cast: str = None, director: str = None, release_year: int = None,
                 runtime: int = None, rating: float = None, poster_url: str = None) -> Movie:
    """Create a new movie"""
    movie = Movie(
        title=title,
        description=description,
        genres=genres,
        cast=cast,
        director=director,
        release_year=release_year,
        runtime=runtime,
        rating=rating,
        poster_url=poster_url
    )
    db.add(movie)
    db.commit()
    db.refresh(movie)
    logger.info(f'Created movie: {title}')
    return movie

def update_movie(db: Session, movie_id: int, **kwargs) -> Optional[Movie]:
    """Update movie"""
    movie = get_movie(db, movie_id)
    if movie:
        for key, value in kwargs.items():
            if hasattr(movie, key):
                setattr(movie, key, value)
        db.commit()
        db.refresh(movie)
        logger.info(f'Updated movie: {movie_id}')
    return movie

# Rating CRUD Operations

def get_rating(db: Session, rating_id: int) -> Optional[Rating]:
    """Get rating by ID"""
    return db.query(Rating).filter(Rating.id == rating_id).first()

def get_user_ratings(db: Session, user_id: int) -> List[Rating]:
    """Get all ratings by a user"""
    return db.query(Rating).filter(Rating.user_id == user_id).all()

def get_movie_ratings(db: Session, movie_id: int) -> List[Rating]:
    """Get all ratings for a movie"""
    return db.query(Rating).filter(Rating.movie_id == movie_id).all()

def get_user_movie_rating(db: Session, user_id: int, movie_id: int) -> Optional[Rating]:
    """Get rating from a user for a specific movie"""
    return db.query(Rating).filter(
        (Rating.user_id == user_id) & (Rating.movie_id == movie_id)
    ).first()

def create_rating(db: Session, user_id: int, movie_id: int, rating: float) -> Rating:
    """Create a new rating"""
    # Check if rating already exists
    existing = get_user_movie_rating(db, user_id, movie_id)
    if existing:
        existing.rating = rating
        db.commit()
        db.refresh(existing)
        logger.info(f'Updated rating: user_id={user_id}, movie_id={movie_id}')
        return existing
    
    new_rating = Rating(user_id=user_id, movie_id=movie_id, rating=rating)
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)
    logger.info(f'Created rating: user_id={user_id}, movie_id={movie_id}')
    return new_rating

def delete_rating(db: Session, rating_id: int) -> bool:
    """Delete a rating"""
    rating = get_rating(db, rating_id)
    if rating:
        db.delete(rating)
        db.commit()
        logger.info(f'Deleted rating: {rating_id}')
        return True
    return False
