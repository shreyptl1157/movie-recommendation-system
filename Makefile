.PHONY: help install dev test clean lint format type-check migrate db-seed db-reset train evaluate api dashboard docker-build docker-up docker-down

help:
	@echo "Movie Recommendation System - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install           Install dependencies"
	@echo "  make dev               Start development server"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate        Run database migrations"
	@echo "  make db-seed           Seed database with sample data"
	@echo "  make db-reset          Reset database (WARNING: deletes all data)"
	@echo ""
	@echo "Development:"
	@echo "  make lint              Run linting checks"
	@echo "  make format            Format code with black"
	@echo "  make type-check        Run type checking with mypy"
	@echo "  make test              Run test suite"
	@echo "  make test-cov          Run tests with coverage report"
	@echo ""
	@echo "Models:"
	@echo "  make train             Train all models"
	@echo "  make evaluate          Evaluate all models"
	@echo ""
	@echo "Services:"
	@echo "  make api               Start API server"
	@echo "  make dashboard         Start Streamlit dashboard"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build      Build Docker images"
	@echo "  make docker-up         Start Docker containers"
	@echo "  make docker-down       Stop Docker containers"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             Clean up generated files"

install:
	pip install -r requirements.txt

dev:
	flask --app api.app run --reload --debug

lint:
	flake8 src/ api/ tests/
	pylint src/ api/

format:
	black src/ api/ tests/ scripts/
	isort src/ api/ tests/ scripts/

type-check:
	mypy src/ api/ --ignore-missing-imports

test:
	pytest -v

test-cov:
	pytest --cov=src --cov=api --cov-report=html --cov-report=term

db-migrate:
	flask --app api.app db upgrade

db-seed:
	python scripts/seed_database.py

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Continue? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
	  flake --app api.app db downgrade; \
	  flake --app api.app db upgrade; \
	  python scripts/seed_database.py; \
	fi

train:
	python scripts/train_models.py

evaluate:
	python scripts/evaluate_models.py

api:
	python -m api.app

dashboard:
	streamlit run dashboard/app.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
