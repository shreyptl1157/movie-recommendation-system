# Movie Recommendation System

A production-ready movie recommendation system using Python, FastAPI, and machine learning.

## Features

- **5 ML Algorithms**: User-CF, Item-CF, SVD, Content-Based, Hybrid
- **REST API**: FastAPI with full documentation
- **Database**: PostgreSQL with optimized schema
- **Caching**: Redis for performance
- **Docker**: Complete containerization

## Quick Start

### Using Docker

```bash
docker-compose up -d
```

API: `http://localhost:8000`  
Docs: `http://localhost:8000/docs`

### Local Setup

```bash
git clone https://github.com/shreyptl1157/movie-recommendation-system.git
cd movie-recommendation-system
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m api.main
```

## Project Structure

```
movie-recommendation-system/
├── models/              # ML models (base, collaborative, svd, content, hybrid)
├── api/                 # FastAPI app (main, routes, schemas, dependencies)
├── data/                # Data loading (loader, datasets)
├── utils/               # Utilities (preprocessing, evaluation, cache, logger)
├── config/              # Settings management
├── tests/               # Unit tests
├── docker-compose.yml   # Services orchestration
├── Dockerfile           # Docker image
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
└── init.sql            # Database schema
```

## API Endpoints

```bash
# Get recommendations
POST /api/v1/recommendations
{
  "user_id": 1,
  "n_recommendations": 10,
  "algorithm": "hybrid"
}

# Health check
GET /api/v1/health

# Add rating
POST /api/v1/ratings
{
  "user_id": 1,
  "movie_id": 100,
  "rating": 4.5
}

# Movie details
GET /api/v1/movies/{movie_id}

# Similar movies
GET /api/v1/movies/{movie_id}/similar?n=10
```

## Algorithms

| Algorithm | Best For | Accuracy | Speed |
|-----------|----------|----------|-------|
| User-CF | Similar users | Good | Slow |
| Item-CF | Similar items | Good | Medium |
| SVD | Accuracy | Excellent | Fast |
| Content-Based | New items | Good | Fast |
| Hybrid | Best results | Excellent | Medium |

## Configuration

Edit `.env`:

```env
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=postgresql://postgres:postgres@db:5432/movie_recommender
REDIS_URL=redis://redis:6379/0
SVD_N_FACTORS=100
LOG_LEVEL=INFO
```

## Evaluation Metrics

- RMSE / MAE - Prediction accuracy
- Precision@K / Recall@K - Recommendation quality
- NDCG - Ranking quality
- Coverage - Catalog diversity
- Diversity - Recommendation variety

## Testing

```bash
pytest
pytest --cov=.
pytest tests/test_models.py
```

## Deployment

### Docker

```bash
docker build -t movie-recommender .
docker run -p 8000:8000 movie-recommender
```

### Cloud

- **AWS**: ECS + RDS + ElastiCache
- **GCP**: Cloud Run + Cloud SQL + Memorystore
- **Azure**: App Service + Azure Database + Azure Cache

## License

MIT License

## Author

Shrey Patel (@shreyptl1157)
