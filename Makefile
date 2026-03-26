.PHONY: help install install-dev test test-unit test-integration coverage \
        lint format typecheck clean clean-data reset \
        ingest features train backtest paper db-list db-clear

help:
	@echo "Valorem - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install dependencies with uv"
	@echo "  make install-dev    Install with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make coverage       Run tests with coverage report"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Format code with ruff"
	@echo "  make typecheck      Run mypy type checker"
	@echo ""
	@echo "Data pipeline:"
	@echo "  make ingest         Run ingest_raw.py (requires --start-date, --end-date)"
	@echo "  make features       Run build_features.py (requires --start-date, --end-date, --surface-version, --feature-version)"
	@echo "  make train          Train model (default: ensemble)"
	@echo "  make train-synthetic  Quick smoke test with synthetic data"
	@echo "  make backtest       Run backtest (requires --start-date, --end-date, --checkpoint)"
	@echo "  make paper          Run paper trading"
	@echo ""
	@echo "Database:"
	@echo "  make db-list        List all tables with row counts"
	@echo "  make db-clear       Clear derived tables (surfaces + node_panel)"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean          Remove build artifacts and cache"
	@echo "  make clean-data     Remove data/ directory"
	@echo "  make reset          Remove ALL generated state (data + artifacts + cache)"

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

coverage:
	uv run pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Code quality
lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

typecheck:
	uv run mypy src/

# Data pipeline (pass additional args via ARGS, e.g. make ingest ARGS="--start-date 2024-01-01 --end-date 2024-02-01")
ingest:
	uv run python scripts/ingest_raw.py $(ARGS)

features:
	uv run python scripts/build_features.py $(ARGS)

train:
	uv run python scripts/train_model.py $(ARGS)

train-synthetic:
	uv run python scripts/train_model.py --synthetic

backtest:
	uv run python scripts/run_backtest.py $(ARGS)

paper:
	uv run python scripts/run_paper_trading.py $(ARGS)

# Database management
db-list:
	uv run python scripts/manage_data.py list

db-clear:
	uv run python scripts/manage_data.py clear --derived --yes

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

reset:
	@echo "This will delete ALL generated state:"
	@echo "  - data/          (database, manifests)"
	@echo "  - artifacts/     (model checkpoints, logs, backtest reports)"
	@echo "  - caches         (__pycache__, .pytest_cache, .mypy_cache, etc.)"
	@echo ""
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || { echo "Aborted."; exit 1; }
	$(MAKE) clean
	$(MAKE) clean-data
	rm -rf artifacts/
	@echo "Reset complete."
