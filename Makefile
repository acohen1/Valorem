.PHONY: help install install-dev test test-unit test-integration test-system coverage \
        lint format typecheck clean clean-data \
        ingest surface features train backtest paper

help:
	@echo "Rhubarb v2.0 - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies with uv"
	@echo "  make install-dev    Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-system    Run system tests only"
	@echo "  make coverage       Run tests with coverage report"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Format code with black + ruff"
	@echo "  make typecheck      Run mypy type checker"
	@echo ""
	@echo "Data pipeline:"
	@echo "  make ingest         Ingest raw data from Databento"
	@echo "  make surface        Build surface snapshots"
	@echo "  make features       Build feature dataset"
	@echo "  make train          Train ML model"
	@echo "  make backtest       Run backtest"
	@echo "  make paper          Run paper trading"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean          Remove build artifacts and cache"
	@echo "  make clean-data     Remove data/ directory"

# Installation
install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[dev]"

# Testing
test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-system:
	uv run pytest tests/system/ -v

coverage:
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run black src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

typecheck:
	uv run mypy src/

# Data pipeline
ingest:
	uv run python scripts/ingest_raw.py

surface:
	uv run python scripts/build_surface.py

features:
	uv run python scripts/build_features.py

train:
	uv run python scripts/train_model.py

backtest:
	uv run python scripts/run_backtest.py

paper:
	uv run python scripts/run_paper_trading.py

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage

clean-data:
	rm -rf data/
	mkdir -p data/{parquet,manifest}
