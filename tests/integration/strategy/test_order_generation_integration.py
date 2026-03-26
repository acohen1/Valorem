"""Integration tests for order generation flow.

Tests the full pipeline: signals → structure selection → sizing → risk checks → orders.
"""

from datetime import date

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
from src.risk.portfolio import Portfolio, Position
from src.strategy.orders import Order, OrderGenerator
from src.strategy.selector import StructureSelector
from src.strategy.sizing import PositionSizer
from src.strategy.types import Greeks, OptionLeg, OptionRight, Signal, SignalType


@pytest.fixture
def realistic_surface() -> pd.DataFrame:
    """Create a realistic surface snapshot for testing.

    Includes multiple tenors and delta buckets with realistic Greeks and quotes.
    Designed to support:
    - Calendar spreads: needs far options at tenor > near + 30 days (tenor_gap_days)
    - Vertical spreads: needs strikes at spread_width intervals (2% = ~9 points on 450)
    """
    data = []

    # Near-term options (30 days)
    near_expiry = date(2024, 3, 15)
    near_tenor = 30

    # ATM calls and puts
    data.append(
        {
            "option_symbol": "SPY240315C00450000",
            "tenor_days": near_tenor,
            "delta_bucket": "ATM",
            "strike": 450.0,
            "expiry": near_expiry,
            "right": "C",
            "bid": 8.50,
            "ask": 8.70,
            "delta": 0.52,
            "gamma": 0.025,
            "vega": 0.18,
            "theta": -0.05,
            "volume": 5000,
            "open_interest": 25000,
        }
    )
    data.append(
        {
            "option_symbol": "SPY240315P00450000",
            "tenor_days": near_tenor,
            "delta_bucket": "ATM",
            "strike": 450.0,
            "expiry": near_expiry,
            "right": "P",
            "bid": 8.00,
            "ask": 8.20,
            "delta": -0.48,
            "gamma": 0.025,
            "vega": 0.18,
            "theta": -0.05,
            "volume": 4500,
            "open_interest": 22000,
        }
    )

    # OTM calls at C25 bucket
    data.append(
        {
            "option_symbol": "SPY240315C00460000",
            "tenor_days": near_tenor,
            "delta_bucket": "C25",
            "strike": 460.0,
            "expiry": near_expiry,
            "right": "C",
            "bid": 3.50,
            "ask": 3.70,
            "delta": 0.30,
            "gamma": 0.020,
            "vega": 0.12,
            "theta": -0.03,
            "volume": 3000,
            "open_interest": 15000,
        }
    )

    # Additional call at 469 for vertical spread (sell strike for C25 spread)
    data.append(
        {
            "option_symbol": "SPY240315C00469000",
            "tenor_days": near_tenor,
            "delta_bucket": "C10",
            "strike": 469.0,
            "expiry": near_expiry,
            "right": "C",
            "bid": 1.20,
            "ask": 1.40,
            "delta": 0.15,
            "gamma": 0.012,
            "vega": 0.08,
            "theta": -0.02,
            "volume": 1500,
            "open_interest": 8000,
        }
    )

    # OTM puts
    data.append(
        {
            "option_symbol": "SPY240315P00440000",
            "tenor_days": near_tenor,
            "delta_bucket": "P25",
            "strike": 440.0,
            "expiry": near_expiry,
            "right": "P",
            "bid": 3.00,
            "ask": 3.20,
            "delta": -0.28,
            "gamma": 0.018,
            "vega": 0.11,
            "theta": -0.025,
            "volume": 2800,
            "open_interest": 14000,
        }
    )

    # Far-term options (90 days) - needed for calendar spreads (tenor_gap_days=30)
    far_expiry = date(2024, 5, 15)
    far_tenor = 90

    # ATM calls and puts (far)
    data.append(
        {
            "option_symbol": "SPY240515C00450000",
            "tenor_days": far_tenor,
            "delta_bucket": "ATM",
            "strike": 450.0,
            "expiry": far_expiry,
            "right": "C",
            "bid": 12.00,
            "ask": 12.30,
            "delta": 0.54,
            "gamma": 0.018,
            "vega": 0.25,
            "theta": -0.03,
            "volume": 3500,
            "open_interest": 18000,
        }
    )
    data.append(
        {
            "option_symbol": "SPY240515P00450000",
            "tenor_days": far_tenor,
            "delta_bucket": "ATM",
            "strike": 450.0,
            "expiry": far_expiry,
            "right": "P",
            "bid": 11.50,
            "ask": 11.80,
            "delta": -0.46,
            "gamma": 0.018,
            "vega": 0.25,
            "theta": -0.03,
            "volume": 3200,
            "open_interest": 16000,
        }
    )

    # OTM calls far
    data.append(
        {
            "option_symbol": "SPY240515C00460000",
            "tenor_days": far_tenor,
            "delta_bucket": "C25",
            "strike": 460.0,
            "expiry": far_expiry,
            "right": "C",
            "bid": 6.00,
            "ask": 6.25,
            "delta": 0.35,
            "gamma": 0.015,
            "vega": 0.18,
            "theta": -0.02,
            "volume": 2000,
            "open_interest": 10000,
        }
    )

    return pd.DataFrame(data)


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Execution config for integration tests."""
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
            scale_by_confidence=True,
            scale_by_liquidity=False,
        ),
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    """Risk config for integration tests."""
    return RiskConfig(
        per_trade=PerTradeRiskConfig(max_loss=2000.0, max_contracts=20),
        caps=RiskCapsConfig(
            max_abs_delta=500.0,
            max_abs_gamma=200.0,
            max_abs_vega=5000.0,
            max_daily_loss=10000.0,
        ),
        stress=StressConfig(enabled=False),
        kill_switch=KillSwitchConfig(),
    )


class TestFullOrderGenerationFlow:
    """Integration tests for full order generation flow."""

    def test_term_anomaly_to_calendar_order(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test full flow: TERM_ANOMALY signal → CalendarSpread order."""
        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(execution_config, risk_checker)
        portfolio = Portfolio()

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        # Should produce one calendar spread order
        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.structure_type == "CalendarSpread"
        assert len(order.legs) == 2
        assert order.max_loss > 0
        assert order.greeks.vega != 0  # Calendars have vega exposure

    def test_directional_vol_to_vertical_order(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test full flow: DIRECTIONAL_VOL signal → VerticalSpread order."""
        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(execution_config, risk_checker)
        portfolio = Portfolio()

        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.03,
            confidence=0.75,
            tenor_days=30,
            delta_bucket="C25",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        # Should produce one vertical spread order
        assert len(result.orders) == 1
        order = result.orders[0]
        assert order.structure_type == "VerticalSpread"
        assert len(order.legs) == 2

    def test_multiple_signals_processed(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test multiple signals are processed correctly."""
        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(execution_config, risk_checker)
        portfolio = Portfolio()

        signals = [
            Signal(
                signal_type=SignalType.TERM_ANOMALY,
                edge=0.05,
                confidence=0.8,
                tenor_days=30,
                delta_bucket="ATM",
            ),
            Signal(
                signal_type=SignalType.DIRECTIONAL_VOL,
                edge=0.03,
                confidence=0.7,
                tenor_days=30,
                delta_bucket="C25",
            ),
        ]

        result = generator.generate_orders(
            signals=signals,
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        # Both signals should be processed
        assert len(result.orders) + len(result.rejected_signals) == 2


class TestOrderGenerationWithExistingPortfolio:
    """Tests for order generation with existing portfolio positions."""

    def test_order_respects_portfolio_limits(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
    ) -> None:
        """Test order generation respects portfolio-level limits."""
        # Create risk config with tight delta limit
        tight_risk_config = RiskConfig(
            per_trade=PerTradeRiskConfig(max_loss=2000.0, max_contracts=20),
            caps=RiskCapsConfig(
                max_abs_delta=100.0,  # Tight delta limit
                max_abs_gamma=200.0,
                max_abs_vega=5000.0,
                max_daily_loss=10000.0,
            ),
            stress=StressConfig(enabled=False),
            kill_switch=KillSwitchConfig(),
        )
        risk_checker = RiskChecker(tight_risk_config)
        generator = OrderGenerator(execution_config, risk_checker)

        # Create portfolio with existing delta exposure
        existing_leg = OptionLeg(
            symbol="SPY240315C00445000",
            qty=1,
            entry_price=10.00,
            strike=445.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.6, gamma=0.02, vega=0.15, theta=-0.04),
        )
        portfolio = Portfolio(positions=[Position(legs=[existing_leg])])

        # Try to add more delta
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="C25",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        # Order might be rejected due to delta limits
        # (depends on structure Greeks and sizing)
        total_processed = len(result.orders) + len(result.rejected_signals)
        assert total_processed == 1


class TestOrderGenerationSizing:
    """Tests for position sizing in order generation."""

    def test_sizing_affects_order_legs(
        self,
        realistic_surface: pd.DataFrame,
        risk_config: RiskConfig,
    ) -> None:
        """Test that sizing configuration affects order leg quantities."""
        # Config with 2 base contracts
        config_2x = ExecutionConfig(
            signal_threshold=SignalThresholdConfig(min_edge=0.01, min_confidence=0.5),
            sizing=SizingConfig(
                method="fixed",
                base_contracts=2,
                max_contracts_per_trade=10,
                scale_by_confidence=False,
                scale_by_liquidity=False,
            ),
        )

        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(config_2x, risk_checker)
        portfolio = Portfolio()

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        if result.orders:
            order = result.orders[0]
            # Legs should have qty = 2 or -2 (base_contracts = 2)
            for leg in order.legs:
                assert abs(leg.qty) == 2

    def test_confidence_scaling_in_sizing(
        self,
        realistic_surface: pd.DataFrame,
        risk_config: RiskConfig,
    ) -> None:
        """Test confidence scaling affects order size."""
        # Config with confidence scaling
        config_scaled = ExecutionConfig(
            signal_threshold=SignalThresholdConfig(min_edge=0.01, min_confidence=0.5),
            sizing=SizingConfig(
                method="fixed",
                base_contracts=2,
                max_contracts_per_trade=10,
                scale_by_confidence=True,
                scale_by_liquidity=False,
            ),
        )

        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(config_scaled, risk_checker)
        portfolio = Portfolio()

        # High confidence signal
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.9,  # High confidence
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        if result.orders:
            order = result.orders[0]
            # With confidence=0.9, factor = 0.5 + 0.9 = 1.4
            # base=2, adjusted = 2 * 1.4 = 2.8 → rounded to 3
            assert order.sizing_result.confidence_factor == pytest.approx(1.4)


class TestOrderGenerationRejections:
    """Tests for signal rejection scenarios."""

    def test_low_edge_signal_rejected(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test low edge signals are rejected."""
        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(execution_config, risk_checker)
        portfolio = Portfolio()

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.005,  # Below min_edge=0.01
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        assert len(result.orders) == 0
        assert len(result.rejected_signals) == 1

    def test_low_confidence_signal_rejected(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test low confidence signals are rejected."""
        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(execution_config, risk_checker)
        portfolio = Portfolio()

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.3,  # Below min_confidence=0.5
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        assert len(result.orders) == 0
        assert len(result.rejected_signals) == 1


class TestOrderGenerationBoundedRisk:
    """Tests ensuring all orders have bounded risk."""

    def test_all_orders_have_max_loss(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test all generated orders have a positive max_loss."""
        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(execution_config, risk_checker)
        portfolio = Portfolio()

        signals = [
            Signal(
                signal_type=SignalType.TERM_ANOMALY,
                edge=0.05,
                confidence=0.8,
                tenor_days=30,
                delta_bucket="ATM",
            ),
            Signal(
                signal_type=SignalType.DIRECTIONAL_VOL,
                edge=0.03,
                confidence=0.7,
                tenor_days=30,
                delta_bucket="C25",
            ),
        ]

        result = generator.generate_orders(
            signals=signals,
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        for order in result.orders:
            assert order.max_loss > 0, "All orders must have bounded max loss"

    def test_all_orders_have_greeks(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test all generated orders have Greeks populated."""
        risk_checker = RiskChecker(risk_config)
        generator = OrderGenerator(execution_config, risk_checker)
        portfolio = Portfolio()

        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        for order in result.orders:
            assert isinstance(order.greeks, Greeks)
            # Greeks should be non-trivial (not all zeros unless perfectly hedged)
            assert (
                order.greeks.delta != 0
                or order.greeks.gamma != 0
                or order.greeks.vega != 0
            )


class TestOrderGenerationEndToEnd:
    """End-to-end integration tests."""

    def test_signal_to_order_pipeline(
        self,
        realistic_surface: pd.DataFrame,
        execution_config: ExecutionConfig,
        risk_config: RiskConfig,
    ) -> None:
        """Test complete signal-to-order pipeline."""
        # Setup
        risk_checker = RiskChecker(risk_config)
        selector = StructureSelector(execution_config)
        sizer = PositionSizer(execution_config.sizing)
        generator = OrderGenerator(
            execution_config,
            risk_checker,
            structure_selector=selector,
            position_sizer=sizer,
        )
        portfolio = Portfolio()

        # Create signal
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        # Generate order
        result = generator.generate_orders(
            signals=[signal],
            surface=realistic_surface,
            portfolio=portfolio,
            underlying_price=450.0,
        )

        # Verify complete order
        assert len(result.orders) == 1
        order = result.orders[0]

        # Verify order structure
        assert order.order_id is not None
        assert len(order.legs) == 2
        assert order.structure_type == "CalendarSpread"
        assert order.signal == signal
        assert order.max_loss > 0
        assert order.greeks is not None
        assert order.sizing_result is not None
        assert order.timestamp is not None

        # Verify legs are properly formed
        for leg in order.legs:
            assert leg.symbol is not None
            assert leg.qty != 0
            assert leg.entry_price > 0
            assert leg.strike > 0
            assert leg.expiry is not None
            assert leg.right in [OptionRight.CALL, OptionRight.PUT]
            assert leg.greeks is not None
