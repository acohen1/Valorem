"""Unit tests for configuration schema validation."""

from datetime import date, time

import pytest
from pydantic import ValidationError

from src.config.schema import (
    ConfigSchema,
    DataConfig,
    DatasetSplitsConfig,
    DeltaBucketsConfig,
    ExecutionConfig,
    PricingConfig,
    RiskConfig,
    RiskCapsConfig,
)


class TestDeltaBucketsConfig:
    """Test delta bucket configuration validation."""

    def test_atm_bucket_must_have_four_elements(self):
        """ATM bucket must have exactly 4 elements."""
        # Valid: 4 elements
        valid_config = DeltaBucketsConfig(ATM=[-0.45, -0.55, 0.45, 0.55])
        assert len(valid_config.ATM) == 4

        # Invalid: 2 elements
        with pytest.raises(ValidationError, match="ATM bucket must have exactly 4 elements"):
            DeltaBucketsConfig(ATM=[-0.50, 0.50])

        # Invalid: 3 elements
        with pytest.raises(ValidationError, match="ATM bucket must have exactly 4 elements"):
            DeltaBucketsConfig(ATM=[-0.45, -0.55, 0.55])


class TestDataConfig:
    """Test data configuration validation."""

    def test_databento_dataset_must_not_be_empty(self):
        """Databento dataset fields must not be empty."""
        # Valid config
        config = DataConfig()
        assert config.providers.databento.dataset_equities == "GLBX.MDP3"
        assert config.providers.databento.dataset_options == "OPRA"

        # Invalid: empty dataset_equities
        with pytest.raises(ValidationError, match="dataset_equities must not be empty"):
            DataConfig.model_validate({
                "providers": {
                    "databento": {
                        "dataset_equities": "",
                        "dataset_options": "OPRA",
                    }
                },
                "ingestion": {}
            })


class TestDatasetSplitsConfig:
    """Test dataset splits validation."""

    def test_splits_must_be_chronologically_ordered(self):
        """Dataset splits must be in chronological order."""
        # Valid: chronologically ordered
        valid_splits = DatasetSplitsConfig(
            train_start=date(2020, 1, 1),
            train_end=date(2021, 12, 31),
            val_start=date(2022, 1, 1),
            val_end=date(2022, 6, 30),
            test_start=date(2022, 7, 1),
            test_end=date(2022, 12, 31),
        )
        assert valid_splits.train_start < valid_splits.train_end

        # Invalid: train_start >= train_end
        with pytest.raises(ValidationError, match="must be before"):
            DatasetSplitsConfig(
                train_start=date(2022, 1, 1),
                train_end=date(2020, 1, 1),  # Before train_start
                val_start=date(2022, 6, 1),
                val_end=date(2022, 9, 1),
                test_start=date(2022, 10, 1),
                test_end=date(2022, 12, 31),
            )

        # Invalid: val_start before train_end
        with pytest.raises(ValidationError, match="must be before"):
            DatasetSplitsConfig(
                train_start=date(2020, 1, 1),
                train_end=date(2022, 12, 31),
                val_start=date(2022, 1, 1),  # Before train_end
                val_end=date(2022, 6, 30),
                test_start=date(2022, 7, 1),
                test_end=date(2022, 12, 31),
            )


class TestRiskConfig:
    """Test risk configuration validation."""

    def test_risk_caps_must_be_positive(self):
        """Risk caps must be positive values."""
        # Valid config
        config = RiskConfig()
        assert config.caps.max_portfolio_delta > 0
        assert config.caps.max_portfolio_vega > 0

        # Invalid: negative max_portfolio_delta
        with pytest.raises(ValidationError):
            RiskCapsConfig(max_portfolio_delta=-100.0)

        # Invalid: zero max_position_size_usd
        with pytest.raises(ValidationError):
            RiskCapsConfig(max_position_size_usd=0.0)


class TestExecutionConfig:
    """Test execution configuration validation."""

    def test_buy_at_must_be_ask_in_v1(self):
        """Buy execution must be at ask (no mid pricing in v1)."""
        # Valid: buy_at="ask"
        config = ExecutionConfig()
        assert config.pricing.buy_at == "ask"

        # buy_at is Literal["ask"], so other values are type errors
        # Pydantic will reject at schema level


class TestConfigSchema:
    """Test root configuration schema."""

    def test_version_must_be_v1(self):
        """Config version must be v1."""
        # Valid: version="v1"
        config = ConfigSchema(
            dataset={
                "splits": {
                    "train_start": date(2020, 1, 1),
                    "train_end": date(2021, 12, 31),
                    "val_start": date(2022, 1, 1),
                    "val_end": date(2022, 6, 30),
                    "test_start": date(2022, 7, 1),
                    "test_end": date(2022, 12, 31),
                }
            },
            backtest={"start_date": date(2023, 1, 1), "end_date": date(2023, 12, 31)},
        )
        assert config.version == "v1"

        # Invalid: version="v2" (not supported)
        with pytest.raises(ValidationError):
            ConfigSchema(
                version="v2",
                dataset={
                    "splits": {
                        "train_start": date(2020, 1, 1),
                        "train_end": date(2021, 12, 31),
                        "val_start": date(2022, 1, 1),
                        "val_end": date(2022, 6, 30),
                        "test_start": date(2022, 7, 1),
                        "test_end": date(2022, 12, 31),
                    }
                },
                backtest={"start_date": date(2023, 1, 1), "end_date": date(2023, 12, 31)},
            )

    def test_extra_fields_are_forbidden(self):
        """Unknown configuration fields should raise validation error."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ConfigSchema(
                unknown_field="value",
                dataset={
                    "splits": {
                        "train_start": date(2020, 1, 1),
                        "train_end": date(2021, 12, 31),
                        "val_start": date(2022, 1, 1),
                        "val_end": date(2022, 6, 30),
                        "test_start": date(2022, 7, 1),
                        "test_end": date(2022, 12, 31),
                    }
                },
                backtest={"start_date": date(2023, 1, 1), "end_date": date(2023, 12, 31)},
            )

    def test_minimal_valid_config(self):
        """Test that minimal required fields create valid config."""
        config = ConfigSchema(
            dataset={
                "splits": {
                    "train_start": date(2020, 1, 1),
                    "train_end": date(2021, 12, 31),
                    "val_start": date(2022, 1, 1),
                    "val_end": date(2022, 6, 30),
                    "test_start": date(2022, 7, 1),
                    "test_end": date(2022, 12, 31),
                }
            },
            backtest={"start_date": date(2023, 1, 1), "end_date": date(2023, 12, 31)},
        )

        # Check defaults are set
        assert config.project.name == "rhubarb"
        assert config.universe.underlying == "SPY"
        assert config.training.device == "cuda"
