"""Unit tests for OrderRouter."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.config.schema import (
    ExecutionConfig,
    FeeConfig,
    PricingConfig,
    SlippageConfig,
)
from src.live.router import Fill, PaperOrderRouter
from src.strategy.orders import Order
from src.strategy.sizing import SizingResult
from src.strategy.types import Greeks, OptionLeg, OptionRight, Signal, SignalType


def _make_sizing_result(qty: int = 1) -> SizingResult:
    """Helper to create SizingResult with correct fields."""
    return SizingResult(
        quantity_multiplier=qty,
        base_contracts=qty,
        adjusted_contracts=qty,
        confidence_factor=1.0,
        liquidity_factor=1.0,
        risk_factor=1.0,
        reason="fixed",
    )


@pytest.fixture
def execution_config() -> ExecutionConfig:
    """Default execution configuration."""
    return ExecutionConfig(
        pricing=PricingConfig(),
        slippage=SlippageConfig(fixed_bps=10),  # 10 bps slippage
        fees=FeeConfig(per_contract=0.65, per_trade_minimum=1.00),
    )


@pytest.fixture
def paper_router(execution_config: ExecutionConfig) -> PaperOrderRouter:
    """Create paper order router."""
    return PaperOrderRouter(execution_config)


@pytest.fixture
def sample_surface() -> pd.DataFrame:
    """Create sample options surface."""
    return pd.DataFrame(
        [
            {
                "option_symbol": "SPY240315C00450000",
                "strike": 450.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 5.00,
                "ask": 5.10,
                "delta": 0.45,
            },
            {
                "option_symbol": "SPY240315C00455000",
                "strike": 455.0,
                "expiry": date(2024, 3, 15),
                "right": "C",
                "bid": 3.00,
                "ask": 3.08,
                "delta": 0.35,
            },
            {
                "option_symbol": "SPY240315P00445000",
                "strike": 445.0,
                "expiry": date(2024, 3, 15),
                "right": "P",
                "bid": 4.50,
                "ask": 4.60,
                "delta": -0.40,
            },
        ]
    )


@pytest.fixture
def sample_order() -> Order:
    """Create sample order."""
    signal = Signal(
        signal_type=SignalType.DIRECTIONAL_VOL,
        edge=0.05,
        confidence=0.8,
        tenor_days=30,
        delta_bucket="ATM",
        timestamp=datetime.now(timezone.utc),
    )

    legs = [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.05,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
        ),
        OptionLeg(
            symbol="SPY240315C00455000",
            qty=-1,
            entry_price=3.04,
            strike=455.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.35, gamma=0.018, vega=0.25, theta=-0.04),
        ),
    ]

    return Order(
        order_id="test_order_001",
        legs=legs,
        structure_type="VerticalSpread",
        signal=signal,
        max_loss=200.0,
        greeks=Greeks(delta=0.10, gamma=0.002, vega=0.05, theta=-0.01),
        sizing_result=_make_sizing_result(1),
    )


class TestPaperOrderRouterInit:
    """Tests for PaperOrderRouter initialization."""

    def test_init_with_config(self, execution_config: ExecutionConfig) -> None:
        """Test initialization with config."""
        router = PaperOrderRouter(execution_config)
        assert router is not None
        assert router._config == execution_config

    def test_is_available(self, paper_router: PaperOrderRouter) -> None:
        """Test paper router is always available."""
        assert paper_router.is_available() is True


class TestPaperOrderRouterRouteOrder:
    """Tests for PaperOrderRouter.route_order."""

    def test_route_long_order(
        self,
        paper_router: PaperOrderRouter,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test routing a long order (buy)."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.05,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            )
        ]

        order = Order(
            order_id="long_order_001",
            legs=legs,
            structure_type="SingleLeg",
            signal=signal,
            max_loss=510.0,
            greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            sizing_result=_make_sizing_result(1),
        )

        fill = paper_router.route_order(order, sample_surface)

        assert fill is not None
        assert fill.order_id == "long_order_001"
        assert len(fill.legs) == 1
        # Long orders buy at ask + slippage
        # Ask = 5.10, slippage = 10 bps
        # Fill price = 5.10 * 1.001 = 5.1051
        assert fill.fill_prices["SPY240315C00450000"] == pytest.approx(5.1051, rel=0.01)
        # Gross premium = 1 * 5.1051 * 100 = 510.51
        assert fill.gross_premium == pytest.approx(510.51, rel=0.01)

    def test_route_short_order(
        self,
        paper_router: PaperOrderRouter,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test routing a short order (sell)."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=-1,
                entry_price=5.05,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            )
        ]

        order = Order(
            order_id="short_order_001",
            legs=legs,
            structure_type="SingleLeg",
            signal=signal,
            max_loss=500.0,
            greeks=Greeks(delta=-0.45, gamma=-0.02, vega=-0.30, theta=0.05),
            sizing_result=_make_sizing_result(1),
        )

        fill = paper_router.route_order(order, sample_surface)

        assert fill is not None
        assert fill.order_id == "short_order_001"
        # Short orders sell at bid - slippage
        # Bid = 5.00, slippage = 10 bps
        # Fill price = 5.00 * 0.999 = 4.995
        assert fill.fill_prices["SPY240315C00450000"] == pytest.approx(4.995, rel=0.01)
        # Gross premium = -1 * 4.995 * 100 = -499.5 (receive money)
        assert fill.gross_premium == pytest.approx(-499.5, rel=0.01)

    def test_route_spread_order(
        self,
        paper_router: PaperOrderRouter,
        sample_surface: pd.DataFrame,
        sample_order: Order,
    ) -> None:
        """Test routing a spread order (buy one, sell one)."""
        fill = paper_router.route_order(sample_order, sample_surface)

        assert fill is not None
        assert fill.order_id == "test_order_001"
        assert len(fill.legs) == 2
        assert "SPY240315C00450000" in fill.fill_prices
        assert "SPY240315C00455000" in fill.fill_prices
        assert fill.fees > 0
        assert fill.slippage > 0

    def test_route_order_missing_quote(
        self,
        paper_router: PaperOrderRouter,
    ) -> None:
        """Test routing when quote is missing - use entry price."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = [
            OptionLeg(
                symbol="MISSING_SYMBOL",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            )
        ]

        order = Order(
            order_id="missing_quote_order",
            legs=legs,
            structure_type="SingleLeg",
            signal=signal,
            max_loss=500.0,
            greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            sizing_result=_make_sizing_result(1),
        )

        empty_surface = pd.DataFrame(columns=["option_symbol", "bid", "ask"])
        fill = paper_router.route_order(order, empty_surface)

        assert fill is not None
        assert fill.fill_prices["MISSING_SYMBOL"] == 5.00

    def test_fill_id_increments(
        self,
        paper_router: PaperOrderRouter,
        sample_surface: pd.DataFrame,
        sample_order: Order,
    ) -> None:
        """Test that fill IDs are unique and increment."""
        fill1 = paper_router.route_order(sample_order, sample_surface)
        fill2 = paper_router.route_order(sample_order, sample_surface)

        assert fill1 is not None
        assert fill2 is not None
        assert fill1.fill_id != fill2.fill_id
        assert "paper_fill_" in fill1.fill_id
        assert "paper_fill_" in fill2.fill_id


class TestPaperOrderRouterFees:
    """Tests for fee calculation."""

    def test_per_contract_fees(
        self,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test per-contract fee calculation."""
        router = PaperOrderRouter(
            ExecutionConfig(
                fees=FeeConfig(per_contract=0.65, per_trade_minimum=0.50),
            )
        )

        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=2,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            )
        ]

        order = Order(
            order_id="fee_test_order",
            legs=legs,
            structure_type="SingleLeg",
            signal=signal,
            max_loss=1000.0,
            greeks=Greeks(delta=0.90, gamma=0.04, vega=0.60, theta=-0.10),
            sizing_result=_make_sizing_result(2),
        )

        fill = router.route_order(order, sample_surface)

        assert fill is not None
        # 2 contracts * $0.65 = $1.30 (> minimum $0.50)
        assert fill.fees == pytest.approx(1.30, rel=0.01)

    def test_minimum_fee_enforced(
        self,
        sample_surface: pd.DataFrame,
    ) -> None:
        """Test minimum per-trade fee is enforced."""
        router = PaperOrderRouter(
            ExecutionConfig(
                fees=FeeConfig(per_contract=0.25, per_trade_minimum=1.00),
            )
        )

        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            )
        ]

        order = Order(
            order_id="min_fee_test_order",
            legs=legs,
            structure_type="SingleLeg",
            signal=signal,
            max_loss=500.0,
            greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            sizing_result=_make_sizing_result(1),
        )

        fill = router.route_order(order, sample_surface)

        assert fill is not None
        assert fill.fees == pytest.approx(1.00, rel=0.01)


class TestFillDataclass:
    """Tests for Fill dataclass."""

    def test_fill_attributes(self) -> None:
        """Test Fill has all required attributes."""
        legs = [
            OptionLeg(
                symbol="TEST",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
            )
        ]

        fill = Fill(
            order_id="order_123",
            legs=legs,
            gross_premium=500.0,
            slippage=0.50,
            fees=0.65,
            net_premium=498.85,
            fill_prices={"TEST": 5.00},
            timestamp=datetime.now(timezone.utc),
            fill_id="fill_001",
        )

        assert fill.order_id == "order_123"
        assert len(fill.legs) == 1
        assert fill.gross_premium == 500.0
        assert fill.slippage == 0.50
        assert fill.fees == 0.65
        assert fill.net_premium == 498.85
        assert fill.fill_prices["TEST"] == 5.00
        assert fill.fill_id == "fill_001"
