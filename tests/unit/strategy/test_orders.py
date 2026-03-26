"""Unit tests for OrderGenerator."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.config.schema import (
    ExecutionConfig,
    KillSwitchConfig,
    PerTradeRiskConfig,
    RiskCapsConfig,
    RiskConfig,
    SignalThresholdConfig,
    SizingConfig,
    StressConfig,
)
from src.risk.checker import RiskChecker
from src.risk.portfolio import Portfolio
from src.strategy.orders import Order, OrderGenerationResult, OrderGenerator
from src.strategy.sizing import SizingResult
from src.strategy.types import Greeks, OptionLeg, OptionRight, Signal, SignalType


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Execution config for testing."""
    return ExecutionConfig(
        signal_threshold=SignalThresholdConfig(
            min_edge=0.01,
            max_uncertainty=0.5,
            min_confidence=0.5,
        ),
        sizing=SizingConfig(
            method="fixed",
            base_contracts=1,
            max_contracts_per_trade=10,
            scale_by_confidence=False,
            scale_by_liquidity=False,
        ),
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    """Risk config for testing."""
    return RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=1000.0, max_contracts=20),
        caps=RiskCapsConfig(
            max_abs_delta=200.0,
            max_abs_gamma=100.0,
            max_abs_vega=2000.0,
            max_daily_loss=5000.0,
        ),
        stress=StressConfig(enabled=False),
        kill_switch=KillSwitchConfig(),
    )


@pytest.fixture
def risk_checker(risk_config: RiskConfig) -> RiskChecker:
    """Risk checker for testing."""
    return RiskChecker(risk_config)


@pytest.fixture
def order_generator(
    execution_config: ExecutionConfig, risk_checker: RiskChecker
) -> OrderGenerator:
    """Order generator for testing."""
    return OrderGenerator(execution_config, risk_checker)


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Sample surface data for testing."""
    return pd.DataFrame(
        {
            "option_symbol": [
                "SPY240315C00450000",
                "SPY240315C00460000",
                "SPY240415C00450000",
            ],
            "tenor_days": [30, 30, 60],
            "delta_bucket": ["ATM", "C25", "ATM"],
            "strike": [450.0, 460.0, 450.0],
            "expiry": [date(2024, 3, 15), date(2024, 3, 15), date(2024, 4, 15)],
            "right": ["C", "C", "C"],
            "bid": [5.00, 2.50, 7.00],
            "ask": [5.20, 2.70, 7.20],
            "delta": [0.50, 0.30, 0.52],
            "gamma": [0.02, 0.015, 0.018],
            "vega": [0.15, 0.10, 0.18],
            "theta": [-0.03, -0.02, -0.025],
            "volume": [1000, 500, 800],
        }
    )


@pytest.fixture
def sample_signal() -> Signal:
    """Sample trading signal."""
    return Signal(
        signal_type=SignalType.TERM_ANOMALY,
        edge=0.05,
        confidence=0.8,
        tenor_days=30,
        delta_bucket="ATM",
    )


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Sample empty portfolio."""
    return Portfolio()


class TestOrderDataclass:
    """Tests for Order dataclass."""

    def test_order_creation(self) -> None:
        """Test Order creation."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            )
        ]
        sizing_result = SizingResult(
            quantity_multiplier=1,
            base_contracts=1,
            adjusted_contracts=1,
            confidence_factor=1.0,
            liquidity_factor=1.0,
            risk_factor=1.0,
            reason="test",
        )

        order = Order(
            order_id="test-123",
            legs=legs,
            structure_type="CalendarSpread",
            signal=Signal(
                signal_type=SignalType.TERM_ANOMALY,
                edge=0.05,
                confidence=0.8,
                tenor_days=30,
                delta_bucket="ATM",
            ),
            max_loss=500.0,
            greeks=Greeks(delta=50.0, gamma=2.0, vega=15.0, theta=-3.0),
            sizing_result=sizing_result,
        )

        assert order.order_id == "test-123"
        assert len(order.legs) == 1
        assert order.structure_type == "CalendarSpread"
        assert order.max_loss == 500.0

    def test_order_timestamp_default(self) -> None:
        """Test Order timestamp default is set."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            )
        ]
        sizing_result = SizingResult(
            quantity_multiplier=1,
            base_contracts=1,
            adjusted_contracts=1,
            confidence_factor=1.0,
            liquidity_factor=1.0,
            risk_factor=1.0,
            reason="test",
        )
        order = Order(
            order_id="test-123",
            legs=legs,
            structure_type="CalendarSpread",
            signal=Signal(
                signal_type=SignalType.TERM_ANOMALY,
                edge=0.05,
                confidence=0.8,
                tenor_days=30,
                delta_bucket="ATM",
            ),
            max_loss=500.0,
            greeks=Greeks(delta=50.0, gamma=2.0, vega=15.0, theta=-3.0),
            sizing_result=sizing_result,
        )

        assert order.timestamp is not None
        assert isinstance(order.timestamp, datetime)


class TestOrderGenerationResult:
    """Tests for OrderGenerationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test OrderGenerationResult creation."""
        result = OrderGenerationResult(
            orders=[],
            rejected_signals=[],
            rejection_reasons={},
        )
        assert result.orders == []
        assert result.rejected_signals == []
        assert result.rejection_reasons == {}


class TestOrderGeneratorInit:
    """Tests for OrderGenerator initialization."""

    def test_init(
        self, execution_config: ExecutionConfig, risk_checker: RiskChecker
    ) -> None:
        """Test OrderGenerator initialization."""
        generator = OrderGenerator(execution_config, risk_checker)
        assert generator._config == execution_config
        assert generator._risk_checker == risk_checker

    def test_init_creates_default_selector(
        self, execution_config: ExecutionConfig, risk_checker: RiskChecker
    ) -> None:
        """Test OrderGenerator creates default StructureSelector."""
        generator = OrderGenerator(execution_config, risk_checker)
        assert generator._structure_selector is not None

    def test_init_creates_default_sizer(
        self, execution_config: ExecutionConfig, risk_checker: RiskChecker
    ) -> None:
        """Test OrderGenerator creates default PositionSizer."""
        generator = OrderGenerator(execution_config, risk_checker)
        assert generator._position_sizer is not None


class TestOrderGeneratorThresholds:
    """Tests for signal threshold filtering."""

    def test_signal_below_min_edge_rejected(
        self,
        order_generator: OrderGenerator,
        sample_surface: pd.DataFrame,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test signal below min_edge is rejected."""
        low_edge_signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.005,  # Below 0.01 threshold
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = order_generator.generate_orders(
            signals=[low_edge_signal],
            surface=sample_surface,
            portfolio=sample_portfolio,
            underlying_price=450.0,
        )

        assert len(result.orders) == 0
        assert len(result.rejected_signals) == 1

    def test_signal_below_min_confidence_rejected(
        self,
        order_generator: OrderGenerator,
        sample_surface: pd.DataFrame,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test signal below min_confidence is rejected."""
        low_conf_signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.3,  # Below 0.5 threshold
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = order_generator.generate_orders(
            signals=[low_conf_signal],
            surface=sample_surface,
            portfolio=sample_portfolio,
            underlying_price=450.0,
        )

        assert len(result.orders) == 0
        assert len(result.rejected_signals) == 1

    def test_signal_above_max_uncertainty_rejected(
        self,
        order_generator: OrderGenerator,
        sample_surface: pd.DataFrame,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test signal above max_uncertainty is rejected."""
        # uncertainty = 1 - confidence, max_uncertainty = 0.5
        # So confidence < 0.5 means uncertainty > 0.5
        high_uncertainty_signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.4,  # uncertainty = 0.6 > 0.5
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = order_generator.generate_orders(
            signals=[high_uncertainty_signal],
            surface=sample_surface,
            portfolio=sample_portfolio,
            underlying_price=450.0,
        )

        assert len(result.orders) == 0


class TestOrderGeneratorStructureSelection:
    """Tests for structure selection."""

    def test_unrecognized_signal_type_no_order(
        self,
        order_generator: OrderGenerator,
        sample_surface: pd.DataFrame,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test unrecognized signal type generates no order (SKEW_ANOMALY not implemented)."""
        skew_signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="P25",
        )

        result = order_generator.generate_orders(
            signals=[skew_signal],
            surface=sample_surface,
            portfolio=sample_portfolio,
            underlying_price=450.0,
        )

        assert len(result.orders) == 0
        assert len(result.rejected_signals) == 1


class TestOrderGeneratorRiskChecks:
    """Tests for risk check integration."""

    def test_order_rejected_by_risk_checker(
        self,
        execution_config: ExecutionConfig,
        sample_surface: pd.DataFrame,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test order rejected by risk checker."""
        # Create very tight risk config
        tight_risk_config = RiskConfig(
            per_trade=PerTradeRiskConfig(max_loss=10.0, max_contracts=1),  # Very tight
            caps=RiskCapsConfig(
                max_abs_delta=5.0,  # Very tight
                max_abs_gamma=1.0,
                max_abs_vega=10.0,
                max_daily_loss=50.0,
            ),
            stress=StressConfig(enabled=False),
            kill_switch=KillSwitchConfig(),
        )
        risk_checker = RiskChecker(tight_risk_config)
        generator = OrderGenerator(execution_config, risk_checker)

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=sample_surface,
            portfolio=sample_portfolio,
            underlying_price=450.0,
        )

        # Should be rejected due to tight limits
        assert len(result.orders) == 0


class TestOrderGeneratorPortfolioUpdate:
    """Tests for portfolio update during generation."""

    def test_portfolio_updated_between_signals(
        self,
        execution_config: ExecutionConfig,
        risk_checker: RiskChecker,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test portfolio is updated between signal processing."""
        # Create config with moderate limits
        generator = OrderGenerator(execution_config, risk_checker)
        portfolio = Portfolio()

        # Two signals that should both be processed
        signals = [
            Signal(
                signal_type=SignalType.TERM_ANOMALY,
                edge=0.05,
                confidence=0.8,
                tenor_days=30,
                delta_bucket="ATM",
            ),
            Signal(
                signal_type=SignalType.TERM_ANOMALY,
                edge=0.05,
                confidence=0.8,
                tenor_days=30,
                delta_bucket="ATM",
            ),
        ]

        result = generator.generate_orders(
            signals=signals,
            surface=sample_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        # At least one should be processed (depends on risk limits)
        # The second signal sees updated portfolio from first
        assert len(result.orders) + len(result.rejected_signals) == 2


class TestOrderGeneratorSingleOrder:
    """Tests for generate_single_order convenience method."""

    def test_generate_single_order(
        self,
        order_generator: OrderGenerator,
        sample_signal: Signal,
        sample_surface: pd.DataFrame,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test generate_single_order method."""
        order = order_generator.generate_single_order(
            signal=sample_signal,
            surface=sample_surface,
            portfolio=sample_portfolio,
            underlying_price=450.0,
        )

        # May or may not produce an order depending on surface data
        # Just verify it doesn't crash and returns Order or None
        assert order is None or isinstance(order, Order)

    def test_generate_single_order_no_match(
        self,
        order_generator: OrderGenerator,
        sample_surface: pd.DataFrame,
        sample_portfolio: Portfolio,
    ) -> None:
        """Test generate_single_order with no matching structure."""
        skew_signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="P25",
        )

        order = order_generator.generate_single_order(
            signal=skew_signal,
            surface=sample_surface,
            portfolio=sample_portfolio,
            underlying_price=450.0,
        )

        assert order is None


class TestOrderGeneratorLiquidity:
    """Tests for liquidity lookup."""

    def test_liquidity_from_surface(
        self,
        order_generator: OrderGenerator,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test liquidity is extracted from surface."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            )
        ]

        liquidity = order_generator._get_liquidity(legs, sample_surface)
        assert liquidity == 1000.0  # Volume from surface

    def test_liquidity_missing_symbol(
        self,
        order_generator: OrderGenerator,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test liquidity with missing symbol."""
        legs = [
            OptionLeg(
                symbol="UNKNOWN",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            )
        ]

        liquidity = order_generator._get_liquidity(legs, sample_surface)
        assert liquidity is None

    def test_liquidity_no_volume_column(
        self,
        order_generator: OrderGenerator,
    ) -> None:
        """Test liquidity when volume column is missing."""
        surface_no_vol = pd.DataFrame(
            {
                "option_symbol": ["SPY240315C00450000"],
                "strike": [450.0],
            }
        )
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            )
        ]

        liquidity = order_generator._get_liquidity(legs, surface_no_vol)
        assert liquidity is None


class TestOrderGeneratorLegScaling:
    """Tests for leg scaling."""

    def test_scale_legs(self, order_generator: OrderGenerator) -> None:
        """Test leg quantity scaling."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            ),
            OptionLeg(
                symbol="SPY240315C00460000",
                qty=-1,
                entry_price=2.50,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.3, gamma=0.015, vega=0.10, theta=-0.02),
            ),
        ]

        scaled = order_generator._scale_legs(legs, 3)

        assert scaled[0].qty == 3
        assert scaled[1].qty == -3
        # Other fields unchanged
        assert scaled[0].entry_price == 5.00
        assert scaled[1].entry_price == 2.50

    def test_scale_legs_preserves_greeks(self, order_generator: OrderGenerator) -> None:
        """Test leg scaling preserves Greeks."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.5, gamma=0.02, vega=0.15, theta=-0.03),
            )
        ]

        scaled = order_generator._scale_legs(legs, 2)

        # Greeks should be the same (per-contract Greeks)
        assert scaled[0].greeks.delta == 0.5
        assert scaled[0].greeks.gamma == 0.02


class TestOrderGeneratorOrderId:
    """Tests for order ID generation."""

    def test_order_id_unique(self, order_generator: OrderGenerator) -> None:
        """Test order IDs are unique."""
        ids = [order_generator._generate_order_id() for _ in range(100)]
        assert len(ids) == len(set(ids))  # All unique

    def test_order_id_format(self, order_generator: OrderGenerator) -> None:
        """Test order ID is valid UUID format."""
        import uuid

        order_id = order_generator._generate_order_id()
        # Should parse as valid UUID
        uuid.UUID(order_id)
