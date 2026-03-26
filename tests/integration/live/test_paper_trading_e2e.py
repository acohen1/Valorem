"""End-to-end integration tests for paper trading with all Phase 2 components.

Tests the complete flow: surface → features → signals → orders → fills
with all providers integrated.
"""

import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.config import EnvironmentConfig, TradingMode, validate_cli_config
from src.config.schema import (
    ExecutionConfig,
    FeeConfig,
    KillSwitchConfig,
    PaperConfig,
    PerTradeRiskConfig,
    PositionManagementConfig,
    RiskCapsConfig,
    RiskConfig,
    SlippageConfig,
)
from src.live import (
    Fill,
    LoopState,
    MockFeatureProvider,
    PaperOrderRouter,
    RollingFeatureProvider,
    TradingLoop,
)
from src.live.symbols import MockSymbolProvider, save_symbols_manifest
from src.strategy.types import Signal, SignalType


class MockSurfaceProvider:
    """Mock surface provider that generates realistic synthetic surfaces."""

    TENORS = [7, 14, 30, 45, 60, 90]
    BUCKETS = ["P40", "P25", "P10", "ATM", "C10", "C25", "C40"]

    def __init__(
        self,
        underlying_price: float = 500.0,
        base_iv: float = 0.20,
        iv_volatility: float = 0.02,
        seed: int | None = None,
    ) -> None:
        self._underlying_price = underlying_price
        self._base_iv = base_iv
        self._iv_volatility = iv_volatility
        self._rng = np.random.default_rng(seed)
        self._call_count = 0

    def get_latest_surface(self) -> pd.DataFrame:
        """Generate synthetic options surface."""
        self._call_count += 1
        # Small random walk for underlying
        self._underlying_price *= 1 + self._rng.normal(0, 0.001)
        return self._generate_surface()

    def _generate_surface(self) -> pd.DataFrame:
        """Generate synthetic options surface with all required columns."""
        as_of_date = date.today()
        records = []

        for tenor in self.TENORS:
            expiry = as_of_date + timedelta(days=tenor)

            for bucket in self.BUCKETS:
                # Map bucket to delta
                delta_map = {
                    "P40": -0.40, "P25": -0.25, "P10": -0.10,
                    "ATM": 0.50,
                    "C10": 0.10, "C25": 0.25, "C40": 0.40,
                }
                target_delta = delta_map[bucket]

                # Calculate strike from delta (simplified)
                if target_delta < 0:  # Put
                    strike = self._underlying_price * (1 + target_delta * 0.5)
                    right = "P"
                elif target_delta > 0 and target_delta < 0.5:  # OTM call
                    strike = self._underlying_price * (1 + target_delta * 0.5)
                    right = "C"
                else:  # ATM
                    strike = self._underlying_price
                    right = "C"

                strike = round(strike, 2)

                # Add IV volatility
                iv = self._base_iv * (1 + self._rng.normal(0, self._iv_volatility))
                iv = max(0.05, min(1.0, iv))  # Bound IV

                # Simple option pricing
                time_value = iv * np.sqrt(tenor / 365) * self._underlying_price * 0.4
                intrinsic = max(
                    0,
                    (self._underlying_price - strike)
                    if right == "C"
                    else (strike - self._underlying_price),
                )
                mid_price = intrinsic + time_value * abs(target_delta)

                bid = max(0.01, mid_price * 0.98)
                ask = mid_price * 1.02

                # Generate OCC symbol
                expiry_str = expiry.strftime("%y%m%d")
                strike_str = f"{int(strike * 1000):08d}"
                symbol = f"SPY{expiry_str}{right}{strike_str}"

                records.append({
                    "option_symbol": symbol,
                    "tenor_days": tenor,
                    "delta_bucket": bucket,
                    "strike": strike,
                    "expiry": expiry,
                    "right": right,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "mid_price": round(mid_price, 2),
                    "delta": round(target_delta, 4),
                    "gamma": round(0.02 / (1 + abs(target_delta) * 2), 4),
                    "vega": round(0.3 * np.sqrt(tenor / 30), 4),
                    "theta": round(-0.02 * (30 / tenor), 4),
                    "iv": round(iv, 4),
                    "underlying_price": self._underlying_price,
                })

        return pd.DataFrame(records)


class MockSignalGenerator:
    """Mock signal generator with controllable output."""

    def __init__(
        self,
        signal_probability: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self._signal_probability = signal_probability
        self._rng = np.random.default_rng(seed)
        self._call_count = 0

    def generate_signals(
        self,
        surface: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> list[Signal]:
        self._call_count += 1

        # Always check if features are provided (but don't require them)
        has_features = features is not None and not features.empty

        if self._rng.random() > self._signal_probability:
            return []

        valid_buckets = ["P25", "ATM", "C25"]
        valid_nodes = surface[surface["delta_bucket"].isin(valid_buckets)]
        if valid_nodes.empty:
            return []

        node = valid_nodes.sample(1, random_state=self._call_count).iloc[0]

        signal_type = self._rng.choice([
            SignalType.TERM_ANOMALY,
            SignalType.DIRECTIONAL_VOL,
        ])

        # If features available, use them to modify edge
        edge = float(self._rng.uniform(0.03, 0.08))
        if has_features:
            # Boost edge if we have feature data
            edge *= 1.1

        return [Signal(
            signal_type=signal_type,
            edge=edge,
            confidence=float(self._rng.uniform(0.6, 0.9)),
            tenor_days=int(node["tenor_days"]),
            delta_bucket=str(node["delta_bucket"]),
            timestamp=datetime.now(timezone.utc),
        )]


@pytest.fixture
def paper_config(tmp_path) -> PaperConfig:
    """Paper config for E2E testing."""
    return PaperConfig(
        loop_interval_seconds=0,
        halt_on_error=True,
        max_loop_iterations=10,
        lookback_minutes=5,
        save_state_interval=1,
        state_dir=str(tmp_path / "state"),
    )


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Execution config for E2E testing."""
    return ExecutionConfig(
        slippage=SlippageConfig(fixed_bps=5),
        fees=FeeConfig(per_contract=0.50, per_trade_minimum=1.00),
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    """Risk config for E2E testing."""
    return RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=1000.0, max_contracts=10),
        caps=RiskCapsConfig(
            max_abs_delta=200.0,
            max_abs_vega=2000.0,
            max_portfolio_loss=5000.0,
        ),
        kill_switch=KillSwitchConfig(
            halt_on_daily_loss=True,
            max_daily_loss=3000.0,
        ),
        position_management=PositionManagementConfig(),
    )


class TestPaperTradingE2E:
    """End-to-end tests for paper trading with all components."""

    def test_e2e_with_mock_providers(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test complete E2E flow with all mock providers."""
        # Setup providers
        surface_provider = MockSurfaceProvider(seed=42)
        feature_provider = RollingFeatureProvider(lookback_days=5, min_history=2)
        signal_generator = MockSignalGenerator(signal_probability=0.5, seed=42)
        order_router = PaperOrderRouter(execution_config)

        # Warm up feature history
        for _ in range(3):
            surface = surface_provider.get_latest_surface()
            feature_provider.update(surface)

        # Reset surface provider call count
        surface_provider._call_count = 0

        # Create loop
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
            feature_provider=feature_provider,
        )

        # Run
        loop.start()

        # Verify
        assert loop.metrics.total_iterations == 10
        assert surface_provider._call_count == 10
        assert signal_generator._call_count == 10

    def test_e2e_with_rolling_feature_provider(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test E2E with RollingFeatureProvider building history."""
        surface_provider = MockSurfaceProvider(seed=123)
        feature_provider = RollingFeatureProvider(
            lookback_days=20,
            min_history=5,  # Needs 5 surfaces before returning features
        )
        signal_generator = MockSignalGenerator(signal_probability=0.3, seed=123)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
            feature_provider=feature_provider,
        )

        loop.start()

        # Feature provider should have history
        assert feature_provider.history_count == 10
        assert loop.feature_provider is feature_provider

    def test_e2e_without_feature_provider(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test E2E still works without feature provider (backward compatible)."""
        surface_provider = MockSurfaceProvider(seed=456)
        signal_generator = MockSignalGenerator(signal_probability=0.3, seed=456)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
            # No feature_provider
        )

        loop.start()

        assert loop.metrics.total_iterations == 10
        assert loop.feature_provider is None

    def test_e2e_with_mock_feature_provider(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test E2E with MockFeatureProvider for testing scenarios."""
        surface_provider = MockSurfaceProvider(seed=789)
        feature_provider = MockFeatureProvider(
            default_zscore=0.5,
            default_change=0.02,
        )
        signal_generator = MockSignalGenerator(signal_probability=0.5, seed=789)
        order_router = PaperOrderRouter(execution_config)

        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
            feature_provider=feature_provider,
        )

        loop.start()

        assert loop.metrics.total_iterations == 10


class TestEnvironmentConfigIntegration:
    """Integration tests for EnvironmentConfig with trading components."""

    def test_mock_mode_config_validation(self) -> None:
        """Test EnvironmentConfig works in mock mode."""
        config = EnvironmentConfig.from_env(mode="mock")
        config.validate()  # Should not raise

        assert config.mode == TradingMode.MOCK
        assert config.is_live_data is False
        assert config.is_simulated_execution is True

    def test_cli_config_validation_mock(self) -> None:
        """Test CLI config validation for mock mode."""
        results = validate_cli_config(mode="mock")

        assert results["valid"] is True
        assert results["config"] is not None
        assert results["config"].mode == TradingMode.MOCK

    def test_cli_config_validation_paper_live_without_key(self) -> None:
        """Test CLI config validation fails without API key."""
        # Temporarily unset API key
        original = os.environ.pop("DATABENTO_API_KEY", None)
        try:
            results = validate_cli_config(mode="paper_live")
            assert results["valid"] is False
            assert any("DATABENTO_API_KEY" in e for e in results["errors"])
        finally:
            if original:
                os.environ["DATABENTO_API_KEY"] = original


class TestSymbolProviderIntegration:
    """Integration tests for symbol providers."""

    def test_mock_symbol_provider_integration(self) -> None:
        """Test MockSymbolProvider generates valid symbols."""
        provider = MockSymbolProvider(generate_count=50)
        symbols = provider.get_option_symbols(
            "SPY",
            min_dte=7,
            max_dte=30,
            moneyness_range=(0.95, 1.05),
        )

        assert len(symbols) > 0
        assert len(symbols) <= 50
        assert all(s.startswith("SPY") for s in symbols)

    def test_manifest_roundtrip(self, tmp_path) -> None:
        """Test saving and loading manifest."""
        from src.live.symbols import ManifestSymbolProvider

        # Generate symbols
        provider = MockSymbolProvider(generate_count=20)
        symbols = provider.get_option_symbols("SPY", min_dte=7, max_dte=60)

        # Save manifest
        manifest_path = tmp_path / "test_manifest.json"
        save_symbols_manifest(
            symbols,
            manifest_path,
            underlying="SPY",
            metadata={"source": "test"},
        )

        # Load manifest
        loaded_provider = ManifestSymbolProvider(manifest_path)
        loaded_symbols = loaded_provider.get_option_symbols("SPY", min_dte=0, max_dte=365)

        assert len(loaded_symbols) == len(symbols)


class TestFeatureProviderIntegration:
    """Integration tests for feature providers with trading loop."""

    def test_rolling_feature_provider_computes_features(self) -> None:
        """Test RollingFeatureProvider computes features correctly."""
        surface_provider = MockSurfaceProvider(seed=42)
        feature_provider = RollingFeatureProvider(lookback_days=10, min_history=3)

        # Build history
        for _ in range(5):
            surface = surface_provider.get_latest_surface()
            feature_provider.update(surface)

        # Get features
        current_surface = surface_provider.get_latest_surface()
        features = feature_provider.get_features(current_surface)

        assert not features.empty
        assert "iv_zscore_21d" in features.columns
        assert "term_slope" in features.columns
        assert "spread_pct" in features.columns

    def test_feature_provider_handles_insufficient_history(self) -> None:
        """Test feature provider returns empty when insufficient history."""
        surface_provider = MockSurfaceProvider(seed=42)
        feature_provider = RollingFeatureProvider(lookback_days=10, min_history=5)

        # Only 2 surfaces (less than min_history=5)
        for _ in range(2):
            surface = surface_provider.get_latest_surface()
            feature_provider.update(surface)

        current_surface = surface_provider.get_latest_surface()
        features = feature_provider.get_features(current_surface)

        assert features.empty


class TestAllComponentsIntegration:
    """Integration tests verifying all Phase 2 components work together."""

    def test_all_phase2_components(
        self,
        paper_config: PaperConfig,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
        tmp_path,
    ) -> None:
        """Test all Phase 2 components integrated together."""
        # 1. Environment Config
        env_config = EnvironmentConfig.from_env(mode="mock")
        env_config.validate()

        # 2. Symbol Provider
        symbol_provider = MockSymbolProvider(generate_count=100)
        symbols = symbol_provider.get_option_symbols("SPY", min_dte=7, max_dte=90)
        assert len(symbols) > 0

        # Save to manifest
        manifest_path = tmp_path / "symbols.json"
        save_symbols_manifest(symbols, manifest_path, underlying="SPY")

        # 3. Surface Provider
        surface_provider = MockSurfaceProvider(seed=42)

        # 4. Feature Provider
        feature_provider = RollingFeatureProvider(lookback_days=10, min_history=3)

        # Warm up
        for _ in range(5):
            surface = surface_provider.get_latest_surface()
            feature_provider.update(surface)

        # Reset counter
        surface_provider._call_count = 0

        # 5. Signal Generator
        signal_generator = MockSignalGenerator(signal_probability=0.5, seed=42)

        # 6. Order Router
        order_router = PaperOrderRouter(execution_config)

        # 7. Trading Loop with all components
        loop = TradingLoop(
            paper_config=paper_config,
            execution_config=execution_config,
            risk_config=risk_config,
            surface_provider=surface_provider,
            signal_generator=signal_generator,
            order_router=order_router,
            feature_provider=feature_provider,
        )

        # Run
        loop.start()

        # Verify all components worked
        assert loop.metrics.total_iterations == paper_config.max_loop_iterations
        assert loop.feature_provider is not None
        assert feature_provider.history_count > 0
