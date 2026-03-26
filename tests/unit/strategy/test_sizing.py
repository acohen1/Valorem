"""Unit tests for PositionSizer."""

import pytest

from src.config.schema import SizingConfig
from src.strategy.sizing import PositionSizer, SizingResult
from src.strategy.types import Signal, SignalType


@pytest.fixture
def default_sizer() -> PositionSizer:
    """Default position sizer."""
    return PositionSizer()


@pytest.fixture
def fixed_sizer() -> PositionSizer:
    """Position sizer with fixed method."""
    config = SizingConfig(
        method="fixed",
        base_contracts=2,
        max_contracts_per_trade=10,
        scale_by_confidence=False,
        scale_by_liquidity=False,
    )
    return PositionSizer(config)


@pytest.fixture
def risk_parity_sizer() -> PositionSizer:
    """Position sizer with risk_parity method."""
    config = SizingConfig(
        method="risk_parity",
        max_loss_per_trade=500.0,
        max_contracts_per_trade=10,
        scale_by_confidence=False,
        scale_by_liquidity=False,
    )
    return PositionSizer(config)


@pytest.fixture
def kelly_sizer() -> PositionSizer:
    """Position sizer with kelly_fraction method."""
    config = SizingConfig(
        method="kelly_fraction",
        kelly_fraction=0.25,
        max_loss_per_trade=500.0,
        max_contracts_per_trade=10,
        scale_by_confidence=False,
        scale_by_liquidity=False,
    )
    return PositionSizer(config)


@pytest.fixture
def sample_signal() -> Signal:
    """Sample trading signal."""
    return Signal(
        signal_type=SignalType.DIRECTIONAL_VOL,
        edge=0.05,
        confidence=0.8,
        tenor_days=30,
        delta_bucket="ATM",
    )


class TestSizingResultDataclass:
    """Tests for SizingResult dataclass."""

    def test_sizing_result_creation(self) -> None:
        """Test SizingResult creation."""
        result = SizingResult(
            quantity_multiplier=2,
            base_contracts=2,
            adjusted_contracts=2,
            confidence_factor=1.0,
            liquidity_factor=1.0,
            risk_factor=1.0,
            reason="Method: fixed, base=2, final=2",
        )
        assert result.quantity_multiplier == 2
        assert result.base_contracts == 2
        assert result.adjusted_contracts == 2
        assert result.confidence_factor == 1.0
        assert result.liquidity_factor == 1.0
        assert result.risk_factor == 1.0


class TestPositionSizerInit:
    """Tests for PositionSizer initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        sizer = PositionSizer()
        assert sizer is not None
        assert sizer._config.method == "fixed"

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = SizingConfig(method="risk_parity")
        sizer = PositionSizer(config)
        assert sizer._config.method == "risk_parity"


class TestPositionSizerFixedMethod:
    """Tests for fixed sizing method."""

    def test_fixed_returns_base_contracts(
        self, fixed_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test fixed method returns base contracts."""
        result = fixed_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,
        )
        assert result.quantity_multiplier == 2
        assert result.base_contracts == 2

    def test_fixed_ignores_max_loss(
        self, fixed_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test fixed method ignores max loss per contract."""
        result = fixed_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=1000.0,  # Would cause risk_parity to scale down
        )
        assert result.base_contracts == 2

    def test_fixed_with_different_base(self, sample_signal: Signal) -> None:
        """Test fixed method with different base contracts."""
        config = SizingConfig(
            method="fixed",
            base_contracts=5,
            scale_by_confidence=False,
            scale_by_liquidity=False,
        )
        sizer = PositionSizer(config)
        result = sizer.compute_size(signal=sample_signal, max_loss_per_contract=100.0)
        assert result.base_contracts == 5


class TestPositionSizerRiskParityMethod:
    """Tests for risk_parity sizing method."""

    def test_risk_parity_scales_by_max_loss(
        self, risk_parity_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test risk_parity sizes based on max loss ratio."""
        # max_loss_per_trade = 500, max_loss_per_contract = 100
        # => contracts = 500 / 100 = 5
        result = risk_parity_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,
        )
        assert result.base_contracts == 5

    def test_risk_parity_smaller_loss(
        self, risk_parity_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test risk_parity with smaller max loss per contract."""
        # max_loss_per_trade = 500, max_loss_per_contract = 50
        # => contracts = 500 / 50 = 10
        result = risk_parity_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=50.0,
        )
        assert result.base_contracts == 10

    def test_risk_parity_zero_max_loss(
        self, risk_parity_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test risk_parity with zero max loss falls back to base."""
        result = risk_parity_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=0.0,
        )
        assert result.base_contracts == 1  # Default base_contracts

    def test_risk_parity_minimum_one_contract(
        self, risk_parity_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test risk_parity returns at least 1 contract."""
        # Very high max loss would result in fraction
        result = risk_parity_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=1000.0,
        )
        assert result.base_contracts >= 1


class TestPositionSizerKellyMethod:
    """Tests for kelly_fraction sizing method."""

    def test_kelly_basic(
        self, kelly_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test Kelly method produces valid result."""
        result = kelly_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,
        )
        assert result.base_contracts >= 1

    def test_kelly_higher_confidence(self, kelly_sizer: PositionSizer) -> None:
        """Test Kelly sizes up with higher confidence."""
        high_conf_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.10,
            confidence=0.9,
            tenor_days=30,
            delta_bucket="ATM",
        )
        low_conf_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.10,
            confidence=0.5,
            tenor_days=30,
            delta_bucket="ATM",
        )
        high_result = kelly_sizer.compute_size(
            signal=high_conf_signal, max_loss_per_contract=100.0
        )
        low_result = kelly_sizer.compute_size(
            signal=low_conf_signal, max_loss_per_contract=100.0
        )
        # Higher confidence should result in equal or larger size
        assert high_result.base_contracts >= low_result.base_contracts

    def test_kelly_zero_edge(self, kelly_sizer: PositionSizer) -> None:
        """Test Kelly with zero edge falls back to base."""
        zero_edge_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.0,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )
        result = kelly_sizer.compute_size(
            signal=zero_edge_signal, max_loss_per_contract=100.0
        )
        assert result.base_contracts == 1  # Default base


class TestPositionSizerConfidenceScaling:
    """Tests for confidence scaling."""

    def test_confidence_scaling_enabled(self, sample_signal: Signal) -> None:
        """Test confidence scaling when enabled."""
        config = SizingConfig(
            method="fixed",
            base_contracts=2,
            scale_by_confidence=True,
            scale_by_liquidity=False,
        )
        sizer = PositionSizer(config)
        result = sizer.compute_size(signal=sample_signal, max_loss_per_contract=100.0)

        # confidence=0.8 => factor = 0.5 + 0.8 = 1.3
        assert result.confidence_factor == pytest.approx(1.3)

    def test_confidence_scaling_disabled(
        self, fixed_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test confidence scaling when disabled."""
        result = fixed_sizer.compute_size(
            signal=sample_signal, max_loss_per_contract=100.0
        )
        assert result.confidence_factor == 1.0

    def test_confidence_scaling_low_confidence(self) -> None:
        """Test confidence scaling with low confidence."""
        config = SizingConfig(
            method="fixed",
            base_contracts=4,
            scale_by_confidence=True,
            scale_by_liquidity=False,
        )
        sizer = PositionSizer(config)
        low_conf_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.2,  # Low confidence
            tenor_days=30,
            delta_bucket="ATM",
        )
        result = sizer.compute_size(signal=low_conf_signal, max_loss_per_contract=100.0)

        # confidence=0.2 => factor = max(0.5, 0.5 + 0.2) = 0.7
        assert result.confidence_factor == pytest.approx(0.7)


class TestPositionSizerLiquidityScaling:
    """Tests for liquidity scaling."""

    def test_liquidity_scaling_enabled(self, sample_signal: Signal) -> None:
        """Test liquidity scaling when enabled."""
        config = SizingConfig(
            method="fixed",
            base_contracts=2,
            scale_by_confidence=False,
            scale_by_liquidity=True,
            min_liquidity_contracts=100,
        )
        sizer = PositionSizer(config)
        result = sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,
            liquidity=50,  # Below threshold
        )

        # liquidity=50, threshold=100 => factor = 50/100 = 0.5
        assert result.liquidity_factor == pytest.approx(0.5)

    def test_liquidity_scaling_above_threshold(self, sample_signal: Signal) -> None:
        """Test liquidity scaling at or above threshold."""
        config = SizingConfig(
            method="fixed",
            base_contracts=2,
            scale_by_confidence=False,
            scale_by_liquidity=True,
            min_liquidity_contracts=100,
        )
        sizer = PositionSizer(config)
        result = sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,
            liquidity=150,  # Above threshold
        )
        assert result.liquidity_factor == 1.0

    def test_liquidity_scaling_none(
        self, fixed_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test liquidity scaling with None liquidity."""
        result = fixed_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,
            liquidity=None,
        )
        assert result.liquidity_factor == 1.0

    def test_liquidity_scaling_minimum_factor(self, sample_signal: Signal) -> None:
        """Test liquidity scaling has minimum factor of 0.1."""
        config = SizingConfig(
            method="fixed",
            base_contracts=2,
            scale_by_confidence=False,
            scale_by_liquidity=True,
            min_liquidity_contracts=100,
        )
        sizer = PositionSizer(config)
        result = sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,
            liquidity=5,  # Very low
        )

        # liquidity=5, threshold=100 => factor = max(0.1, 5/100) = 0.1
        assert result.liquidity_factor == pytest.approx(0.1)


class TestPositionSizerRiskFactor:
    """Tests for risk factor scaling."""

    def test_risk_factor_high_max_loss(self, sample_signal: Signal) -> None:
        """Test risk factor scales down for high max loss per contract."""
        config = SizingConfig(
            method="fixed",
            base_contracts=2,
            max_loss_per_trade=500.0,
            scale_by_confidence=False,
            scale_by_liquidity=False,
        )
        sizer = PositionSizer(config)
        result = sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=1000.0,  # Exceeds max_loss_per_trade
        )

        # max_loss_per_contract=1000 > max_loss_per_trade=500
        # => risk_factor = 500/1000 = 0.5
        assert result.risk_factor == pytest.approx(0.5)

    def test_risk_factor_normal(
        self, fixed_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test risk factor is 1.0 when max loss is acceptable."""
        result = fixed_sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,  # Well below limit
        )
        assert result.risk_factor == 1.0


class TestPositionSizerMaxContracts:
    """Tests for max contracts cap."""

    def test_capped_at_max_contracts(self, sample_signal: Signal) -> None:
        """Test result is capped at max_contracts_per_trade."""
        config = SizingConfig(
            method="fixed",
            base_contracts=100,  # Would exceed max
            max_contracts_per_trade=5,
            scale_by_confidence=False,
            scale_by_liquidity=False,
        )
        sizer = PositionSizer(config)
        result = sizer.compute_size(signal=sample_signal, max_loss_per_contract=10.0)

        assert result.adjusted_contracts == 5
        assert result.quantity_multiplier == 5

    def test_minimum_one_contract(self, sample_signal: Signal) -> None:
        """Test result is at least 1 contract."""
        config = SizingConfig(
            method="fixed",
            base_contracts=1,
            max_contracts_per_trade=10,
            scale_by_confidence=True,
            scale_by_liquidity=True,
            min_liquidity_contracts=1000,
        )
        sizer = PositionSizer(config)

        # Very low confidence and liquidity
        low_signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.01,
            confidence=0.1,
            tenor_days=30,
            delta_bucket="ATM",
        )
        result = sizer.compute_size(
            signal=low_signal,
            max_loss_per_contract=100.0,
            liquidity=10,  # Very low
        )

        assert result.adjusted_contracts >= 1
        assert result.quantity_multiplier >= 1


class TestPositionSizerReason:
    """Tests for reason string generation."""

    def test_reason_contains_method(
        self, fixed_sizer: PositionSizer, sample_signal: Signal
    ) -> None:
        """Test reason contains method name."""
        result = fixed_sizer.compute_size(
            signal=sample_signal, max_loss_per_contract=100.0
        )
        assert "fixed" in result.reason

    def test_reason_contains_factors(self, sample_signal: Signal) -> None:
        """Test reason contains non-1.0 factors."""
        config = SizingConfig(
            method="fixed",
            base_contracts=2,
            scale_by_confidence=True,
            scale_by_liquidity=True,
            min_liquidity_contracts=100,
        )
        sizer = PositionSizer(config)
        result = sizer.compute_size(
            signal=sample_signal,
            max_loss_per_contract=100.0,
            liquidity=50,
        )

        assert "conf_factor" in result.reason
        assert "liq_factor" in result.reason
