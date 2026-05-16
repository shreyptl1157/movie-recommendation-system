# Movie Recommendation System

A comprehensive movie recommendation system using multiple algorithms (Collaborative Filtering, Content-Based, Hybrid, and Deep Learning approaches).

## Features

- **Collaborative Filtering**: User-based and item-based recommendations
- **Content-Based Filtering**: Genre, cast, and metadata-driven recommendations
- **Hybrid Approach**: Combines multiple algorithms for better results
- **Deep Learning**: Neural network-based recommendations
- **REST API**: FastAPI endpoints for easy integration
- **Database**: PostgreSQL for user data and ratings
- **Caching**: Redis for fast recommendation retrieval
- **Evaluation Metrics**: RMSE, MAE, Precision@K, Recall@K, NDCG
- **Web UI**: Interactive frontend for testing recommendations

## Project Structure

```
movie-recommendation-system/
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data
│   └── models/                 # Trained models
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration settings
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db.py              # Database connection
│   │   ├── models.py          # SQLAlchemy models
│   │   └── crud.py            # CRUD operations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── collaborative.py   # Collaborative filtering
│   │   ├── content_based.py   # Content-based filtering
│   │   ├── hybrid.py          # Hybrid approach
│   │   └── neural_network.py  # Deep learning model
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_loader.py     # Load and parse data
│   │   ├── cleaner.py         # Data cleaning
│   │   └── feature_engineer.py # Feature engineering
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI app
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── recommendations.py
│   │   │   ├── movies.py
│   │   │   ├── users.py
│   │   │   └── ratings.py
│   │   └── schemas.py         # Pydantic schemas
│   ├── cache/
│   │   ├── __init__.py
│   │   └── redis_cache.py     # Redis caching
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── validator.py       # Cross-validation
│   └── utils/
│       ├── __init__.py
│       ├── logger.py          # Logging utilities
│       └── helpers.py         # Helper functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based_filtering.ipynb
│   ├── 04_hybrid_model.ipynb
│   ├── 05_deep_learning_model.ipynb
│   └── 06_model_evaluation.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_api.py
│   ├── test_preprocessing.py
│   └── test_evaluation.py
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── SearchMovie.jsx
│   │   │   ├── RecommendationList.jsx
│   │   │   └── RatingForm.jsx
│   │   └── styles/
│   │       └── App.css
│   └── package.json
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- PostgreSQL 12+
- Redis 6+
- Docker & Docker Compose (optional)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/shreyptl1157/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize database**
   ```bash
   python -m src.database.db init_db
   ```

6. **Download dataset**
   ```bash
   python -m src.preprocessing.data_loader download_movielens
   ```

### Using Docker

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Usage

### Training Models

```python
from src.models.collaborative import CollaborativeFilter
from src.preprocessing.data_loader import load_movielens

# Load data
ratings_df = load_movielens('ml-1m')

# Train collaborative filtering model
cf_model = CollaborativeFilter(algorithm='svd')
cf_model.fit(ratings_df)
cf_model.save('data/models/collaborative_svd.pkl')
```

### Getting Recommendations

```python
from src.models.hybrid import HybridRecommender

# Initialize hybrid model
recommender = HybridRecommender()
recommender.load_models()

# Get recommendations
user_id = 1
recommendations = recommender.recommend(user_id, n_recommendations=10)
print(recommendations)
```

### API Endpoints

Start the API server:
```bash
uvicorn src.api.main:app --reload
```

Available endpoints:
- `GET /api/recommendations/{user_id}` - Get recommendations for a user
- `GET /api/movies/{movie_id}` - Get movie details
- `POST /api/ratings` - Submit a movie rating
- `GET /api/movies/search` - Search movies
- `GET /api/health` - Health check

## Algorithms

### 1. Collaborative Filtering
- **User-Based**: k-NN approach
- **Item-Based**: Cosine similarity
- **Matrix Factorization**: SVD, NMF

### 2. Content-Based Filtering
- TF-IDF vectorization of movie descriptions
- Genre and cast matching
- Cosine similarity ranking

### 3. Hybrid Model
- Weighted combination of CF and content-based
- Addresses cold-start problem
- Improves diversity

### 4. Deep Learning
- Neural collaborative filtering
- Embedding-based model
- Dropout and batch normalization

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Precision@K** (Proportion of relevant recommendations in top-K)
- **Recall@K** (Proportion of relevant items in top-K)
- **NDCG** (Normalized Discounted Cumulative Gain)
- **Coverage** (Percentage of catalog covered)
- **Diversity** (Variety in recommendations)

## Configuration

Edit `src/config.py` for:
- Database connection
- Redis connection
- Model hyperparameters
- API settings
- Logging configuration

## Data

Default dataset: MovieLens 1M
- 1,000,209 ratings
- 3,706 movies
- 6,040 users

Alternatives:
- MovieLens 10M
- MovieLens 25M
- Custom dataset

## Performance

Benchmarks on MovieLens 1M:
- Collaborative Filtering: ~0.87 RMSE
- Content-Based: ~0.92 RMSE
- Hybrid: ~0.85 RMSE
- Deep Learning: ~0.82 RMSE

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file

## Author

[Your Name]

## Support

For issues, questions, or suggestions, please open a GitHub issue.
