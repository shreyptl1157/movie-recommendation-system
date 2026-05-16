# Movie Recommendation System

A comprehensive, production-ready movie recommendation system built with Python, FastAPI, and machine learning algorithms.

## Features

- **Multiple Recommendation Algorithms**
  - Collaborative Filtering (User-based, Item-based)
  - Singular Value Decomposition (SVD)
  - Content-Based Filtering
  - Hybrid Approach (Ensemble)

- **REST API** with FastAPI
- **Database Support** (PostgreSQL)
- **Redis Caching** for performance
- **Comprehensive Evaluation Metrics**
- **Docker & Docker Compose** support
- **Full Test Suite**
- **Production-Ready**

## Quick Start

### Using Docker (Recommended)

```bash
docker-compose up -d
```

Access the API at `http://localhost:8000`
Swagger documentation at `http://localhost:8000/docs`

### Local Installation

```bash
# Clone the repository
git clone https://github.com/shreyptl1157/movie-recommendation-system.git
cd movie-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Run the API
python -m api.main
```

## Project Structure

```
movie-recommendation-system/
├── models/                      # ML models
│   ├── __init__.py
│   ├── base.py                 # Abstract base class
│   ├── collaborative.py        # CF algorithms
│   ├── content_based.py        # Content-based filtering
│   ├── svd.py                  # SVD implementation
│   └── hybrid.py               # Ensemble approach
├── api/                         # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # App entry point
│   ├── routes.py               # API endpoints
│   ├── schemas.py              # Pydantic models
│   └── dependencies.py         # Dependency injection
├── data/                        # Data handling
│   ├── __init__.py
│   ├── loader.py               # Data loaders
│   └── datasets.py             # Dataset management
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── preprocessing.py        # Data preprocessing
│   ├── evaluation.py           # Evaluation metrics
│   ├── cache.py                # Caching layer
│   └── logger.py               # Logging setup
├── config/                      # Configuration
│   ├── __init__.py
│   └── settings.py             # Settings management
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_data.py
├── docker-compose.yml          # Docker composition
├── Dockerfile                  # Docker image
├── requirements.txt            # Python dependencies
├── .env.example                # Environment template
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## API Endpoints

### Get Recommendations

```bash
POST /api/v1/recommendations

Request:
{
  "user_id": 1,
  "n_recommendations": 10,
  "algorithm": "hybrid"
}

Response:
{
  "user_id": 1,
  "recommendations": [
    {
      "movie_id": 150,
      "title": "Apollo 13",
      "score": 4.5
    },
    ...
  ]
}
```

### Get Similar Movies

```bash
GET /api/v1/movies/{movie_id}/similar?n=10

Response:
{
  "movie_id": 1,
  "title": "Toy Story",
  "similar_movies": [
    {
      "movie_id": 3,
      "title": "Grumpier Old Men",
      "similarity_score": 0.85
    },
    ...
  ]
}
```

### Add/Update Rating

```bash
POST /api/v1/ratings

Request:
{
  "user_id": 1,
  "movie_id": 150,
  "rating": 4.5,
  "timestamp": 1234567890
}

Response:
{
  "user_id": 1,
  "movie_id": 150,
  "rating": 4.5,
  "status": "success"
}
```

### Get Movie Details

```bash
GET /api/v1/movies/{movie_id}

Response:
{
  "movie_id": 1,
  "title": "Toy Story",
  "genres": ["Adventure", "Animation", "Children"],
  "release_year": 1995,
  "average_rating": 4.15,
  "num_ratings": 215
}
```

### Health Check

```bash
GET /api/v1/health

Response:
{
  "status": "healthy",
  "timestamp": "2026-05-16T10:30:00Z"
}
```

## Available Algorithms

### 1. User-Based Collaborative Filtering
Recommends movies liked by similar users.

**Use when:**
- You have user-user similarity data
- Users with similar preferences exist
- Dataset has sufficient user ratings

### 2. Item-Based Collaborative Filtering
Recommends movies similar to ones the user has rated highly.

**Use when:**
- Movie-movie similarity is important
- You want stable recommendations
- User history is limited

### 3. SVD (Singular Value Decomposition)
Matrix factorization approach that uncovers latent factors.

**Use when:**
- You need high accuracy
- You have sparse rating matrices
- Computational resources are available

### 4. Content-Based Filtering
Recommends movies based on movie attributes (genre, cast, etc.).

**Use when:**
- Movie metadata is rich
- You have many new movies
- User has few ratings

### 5. Hybrid
Combines all approaches with weighted ensemble.

**Use when:**
- You want best results
- You can afford computation
- Production quality is critical

## Configuration

Edit `.env` to customize:

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/movies
REDIS_URL=redis://localhost:6379/0

# Model Configuration
SVD_N_FACTORS=100
SVD_LR=0.01
SVD_REG=0.02

# Data
DATASET=movielens-1m
DATA_PATH=./data

# Logging
LOG_LEVEL=INFO
```

## Evaluation Metrics

The system includes comprehensive evaluation metrics:

- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **Precision@K** - Precision of top-K recommendations
- **Recall@K** - Recall of top-K recommendations
- **NDCG** - Normalized Discounted Cumulative Gain
- **Coverage** - Percentage of catalog recommended
- **Diversity** - Variety in recommendations

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_models.py
```

## Performance Optimization

1. **Caching**: Redis caches recommendations and movie data
2. **Database Indexing**: Proper indexes on user/movie IDs
3. **Matrix Factorization**: Fast computation of latent factors
4. **Batch Processing**: Process multiple requests efficiently

## Deployment

### Docker

```bash
# Build image
docker build -t movie-recommender .

# Run container
docker run -p 8000:8000 movie-recommender
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Cloud Deployment

- **AWS**: Use ECS, RDS for PostgreSQL, ElastiCache for Redis
- **Google Cloud**: Cloud Run, Cloud SQL, Memorystore
- **Azure**: App Service, Azure Database, Azure Cache
- **Heroku**: One-click deployment with add-ons

## Troubleshooting

### Port Already in Use
```bash
# Change port in .env
API_PORT=8001
```

### Database Connection Issues
```bash
# Check database is running
# Update DATABASE_URL in .env
# Run migrations if needed
```

### Out of Memory
```bash
# Reduce SVD_N_FACTORS
# Use streaming data loader
# Implement pagination
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Author

Shrey Patel (@shreyptl1157)

## Acknowledgments

- MovieLens Dataset - University of Minnesota
- Surprise Library - Scikit-Surprise contributors
- FastAPI - Sebastián Ramírez

## Support

For issues, questions, or suggestions:
1. Check existing issues
2. Create a new issue with details
3. Include error messages and reproduction steps

## Roadmap

- [ ] Deep Learning models (Neural Collaborative Filtering)
- [ ] Real-time streaming recommendations
- [ ] Explainable AI features
- [ ] A/B testing framework
- [ ] Advanced user segmentation
- [ ] Mobile app integration
- [ ] GraphQL API
- [ ] Analytics dashboard
