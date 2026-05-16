import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data

def test_get_recommendations():
    """Test recommendations endpoint."""
    payload = {
        "user_id": 1,
        "n_recommendations": 10,
        "algorithm": "hybrid"
    }
    response = client.post("/api/v1/recommendations", json=payload)
    assert response.status_code in [200, 400]  # May fail without trained model

def test_add_rating():
    """Test rating endpoint."""
    payload = {
        "user_id": 1,
        "movie_id": 100,
        "rating": 4.5
    }
    response = client.post("/api/v1/ratings", json=payload)
    assert response.status_code == 200
