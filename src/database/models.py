from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from src.database.db import Base

class User(Base):
    """User model"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    ratings = relationship('Rating', back_populates='user', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'

class Movie(Base):
    """Movie model"""
    __tablename__ = 'movies'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500), nullable=False, index=True)
    description = Column(Text)
    genres = Column(String(500))  # Comma-separated genres
    cast = Column(Text)  # Comma-separated cast
    director = Column(String(255))
    release_year = Column(Integer, index=True)
    runtime = Column(Integer)  # In minutes
    rating = Column(Float)  # IMDb rating
    poster_url = Column(String(1000))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    ratings = relationship('Rating', back_populates='movie', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Movie {self.title}>'

class Rating(Base):
    """Movie rating model"""
    __tablename__ = 'ratings'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    movie_id = Column(Integer, ForeignKey('movies.id', ondelete='CASCADE'), nullable=False, index=True)
    rating = Column(Float, nullable=False)  # Rating from 0-5
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship('User', back_populates='ratings')
    movie = relationship('Movie', back_populates='ratings')
    
    def __repr__(self):
        return f'<Rating user_id={self.user_id}, movie_id={self.movie_id}, rating={self.rating}>'
