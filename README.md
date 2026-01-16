# Rhubarb v2.0

**Production-Grade Volatility Arbitrage Trading System**

A modular, scalable system for predicting and exploiting implied vs realized variance mispricing in SPY options using advanced machine learning (PatchTST + Graph Neural Networks).

## Overview

Rhubarb forecasts volatility mispricing at the (timestamp, tenor, delta_bucket) level and translates predictions into bounded-risk option trades with strict risk controls and realistic execution modeling.

## Key Features

- **Modular Architecture**: Independently testable, replaceable components
- **Provider-Agnostic**: Clean abstractions for data providers (Databento, FRED)
- **Type-Safe**: Pydantic-based configuration with strict validation
- **ML Pipeline**: PatchTST temporal model + Graph Neural Network for options surface
- **Risk Management**: Pre-trade checks, stress testing, kill switches
- **Hybrid Storage**: SQLite for queries + Parquet for archival

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository
2. Copy environment template:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. Install dependencies:
   ```bash
   make install-dev
   ```

### Development Workflow

```bash
# Run tests
make test

# Run linter
make lint

# Format code
make format

# Type check
make typecheck

# View all commands
make help
```

## Project Structure

```
Rhubarb/
├── config/           # YAML configuration files
├── src/              # Source code
│   ├── config/       # Configuration management
│   ├── data/         # Data abstraction + ingestion
│   ├── surface/      # Surface engine (IV, Greeks, buckets)
│   ├── features/     # Feature engineering
│   ├── models/       # ML pipeline
│   ├── strategy/     # Strategy + execution
│   ├── risk/         # Risk management
│   └── live/         # Live trading infrastructure
├── scripts/          # Pipeline scripts
├── tests/            # Test suite
├── docs/             # Documentation
│   ├── SPEC.md       # System specification
│   └── ROADMAP.md    # Implementation milestones
└── data/             # Data artifacts (not committed)
```

## Data Pipeline

```bash
# 1. Ingest raw data from Databento
make ingest

# 2. Build surface snapshots (IV, Greeks, buckets)
make surface

# 3. Build feature dataset
make features

# 4. Train ML model
make train

# 5. Run backtest
make backtest

# 6. Run paper trading (future)
make paper
```

## Documentation

- [SPEC.md](docs/SPEC.md) - Comprehensive system specification
- [ROADMAP.md](docs/ROADMAP.md) - Implementation milestones
- [CONFIG_SCHEMA.md](docs/CONFIG_SCHEMA.md) - Configuration reference

## Testing

```bash
# Run all tests
make test

# Run specific test suites
make test-unit
make test-integration
make test-system

# Generate coverage report
make coverage
```

## Code Quality

- **Type Safety**: Full type hints + mypy in strict mode
- **Linting**: Ruff for code quality
- **Formatting**: Black for consistent style
- **Testing**: pytest with >80% coverage target

## Current Status

This is a complete rewrite of Rhubarb, prioritizing **modularity, organization, and scalability**.

See [ROADMAP.md](docs/ROADMAP.md) for implementation progress.

## License

Private project - All rights reserved
