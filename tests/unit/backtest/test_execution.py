"""Unit tests for execution simulation."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.backtest.execution import CONTRACT_MULTIPLIER, ExecutionSimulator, FillResult
from src.config.schema import ExecutionConfig, FeeConfig, PricingConfig, SlippageConfig
from src.strategy.types import Greeks, OptionLeg, OptionRight


@pytest.fixture
def execution_config():
    """Create execution config for testing."""
    return ExecutionConfig(
        pricing=PricingConfig(buy_at="ask", sell_at="bid"),
        slippage=SlippageConfig(model="fixed_bps", fixed_bps=10.0),  # 10 bps
        fees=FeeConfig(per_contract=0.65, per_trade_minimum=0.0),
    )


@pytest.fixture
def execution_simulator(execution_config):
    """Create execution simulator for testing."""
    return ExecutionSimulator(execution_config)


@pytest.fixture
def sample_surface():
    """Create sample surface for testing."""
    return pd.DataFrame([
        {
            "option_symbol": "SPY240315C00450000",
            "strike": 450.0,
            "expiry": date(2024, 3, 15),
            "right": "C",
            "bid": 5.00,
            "ask": 5.20,
            "delta": 0.50,
        },
        {
            "option_symbol": "SPY240315C00455000",
            "strike": 455.0,
            "expiry": date(2024, 3, 15),
            "right": "C",
            "bid": 3.00,
            "ask": 3.15,
            "delta": 0.40,
        },
    ])


@pytest.fixture
def sample_greeks():
    """Sample Greeks for testing."""
    return Greeks(delta=0.5, gamma=0.05, vega=0.3, theta=-0.02)


class TestExecutionSimulatorInit:
    """Tests for ExecutionSimulator initialization."""

    def test_init_with_config(self, execution_config):
        """Test initialization with config."""
        sim = ExecutionSimulator(execution_config)
        assert sim is not None

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        config = ExecutionConfig()
        sim = ExecutionSimulator(config)
        assert sim is not None


class TestEntryFillSimulation:
    """Tests for entry fill simulation."""

    def test_long_position_executes_at_ask(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test that long positions buy at ask price."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,  # Long
                entry_price=5.00,  # Will be updated by fill
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_entry_fill(legs, sample_surface, timestamp)

        # Should execute at ask (5.20) + slippage (10 bps = 0.052%)
        expected_price = 5.20 * (1 + 10 / 10000)  # 5.2052
        assert result.fill_prices["SPY240315C00450000"] == pytest.approx(expected_price, abs=0.01)

    def test_short_position_executes_at_bid(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test that short positions sell at bid price."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=-1,  # Short
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_entry_fill(legs, sample_surface, timestamp)

        # Should execute at bid (5.00) - slippage (10 bps)
        expected_price = 5.00 * (1 - 10 / 10000)  # 4.995
        assert result.fill_prices["SPY240315C00450000"] == pytest.approx(expected_price, abs=0.01)

    def test_spread_entry(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test spread entry with long and short legs."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,  # Long
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            ),
            OptionLeg(
                symbol="SPY240315C00455000",
                qty=-1,  # Short
                entry_price=3.00,
                strike=455.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            ),
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_entry_fill(legs, sample_surface, timestamp)

        # Long leg at ask, short leg at bid (both with slippage)
        assert len(result.legs) == 2
        assert result.fees == pytest.approx(0.65 * 2, abs=0.01)  # 2 contracts

    def test_missing_option_uses_entry_price(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test that missing options use entry price as fallback."""
        legs = [
            OptionLeg(
                symbol="MISSING_SYMBOL",
                qty=1,
                entry_price=10.00,
                strike=460.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_entry_fill(legs, sample_surface, timestamp)

        assert result.fill_prices["MISSING_SYMBOL"] == 10.00


class TestExitFillSimulation:
    """Tests for exit fill simulation."""

    def test_close_long_sells_at_bid(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test closing long position sells at bid."""
        # Closing leg has reversed qty
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=-1,  # Closing a long (sell to close)
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_exit_fill(legs, sample_surface, timestamp)

        # Should execute at bid (5.00) - slippage
        expected_price = 5.00 * (1 - 10 / 10000)
        assert result.fill_prices["SPY240315C00450000"] == pytest.approx(expected_price, abs=0.01)

    def test_close_short_buys_at_ask(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test closing short position buys at ask."""
        # Closing leg has reversed qty
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,  # Closing a short (buy to close)
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_exit_fill(legs, sample_surface, timestamp)

        # Should execute at ask (5.20) + slippage
        expected_price = 5.20 * (1 + 10 / 10000)
        assert result.fill_prices["SPY240315C00450000"] == pytest.approx(expected_price, abs=0.01)


class TestFeeCalculation:
    """Tests for fee calculation."""

    def test_per_contract_fee(self, execution_simulator, sample_surface, sample_greeks):
        """Test per-contract fee calculation."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=3,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_entry_fill(legs, sample_surface, timestamp)

        # 3 contracts * $0.65 = $1.95
        assert result.fees == pytest.approx(1.95, abs=0.01)

    def test_minimum_fee(self, sample_surface, sample_greeks):
        """Test minimum fee is applied."""
        config = ExecutionConfig(
            fees=FeeConfig(per_contract=0.10, per_trade_minimum=1.00)
        )
        sim = ExecutionSimulator(config)

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = sim.simulate_entry_fill(legs, sample_surface, timestamp)

        # 1 contract * $0.10 = $0.10, but minimum is $1.00
        assert result.fees == pytest.approx(1.00, abs=0.01)


class TestSlippageCalculation:
    """Tests for slippage calculation."""

    def test_slippage_applied_to_price(self, execution_config, sample_surface, sample_greeks):
        """Test slippage is applied to execution price."""
        # Use 100 bps slippage for clear testing
        config = ExecutionConfig(
            slippage=SlippageConfig(model="fixed_bps", fixed_bps=100.0)
        )
        sim = ExecutionSimulator(config)

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = sim.simulate_entry_fill(legs, sample_surface, timestamp)

        # Ask is 5.20, 100 bps = 1%, so 5.20 * 1.01 = 5.252
        expected_price = 5.20 * 1.01
        assert result.fill_prices["SPY240315C00450000"] == pytest.approx(expected_price, abs=0.01)

    def test_zero_slippage(self, sample_surface, sample_greeks):
        """Test with zero slippage."""
        config = ExecutionConfig(
            slippage=SlippageConfig(model="fixed_bps", fixed_bps=0.0)
        )
        sim = ExecutionSimulator(config)

        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = sim.simulate_entry_fill(legs, sample_surface, timestamp)

        # Ask is 5.20, no slippage
        assert result.fill_prices["SPY240315C00450000"] == pytest.approx(5.20, abs=0.01)


class TestGrossPremiumCalculation:
    """Tests for gross premium calculation."""

    def test_long_premium_is_positive(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test that long positions have positive premium (pay money)."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_entry_fill(legs, sample_surface, timestamp)

        # qty * price * 100 = 1 * 5.2052 * 100 ≈ 520
        assert result.gross_premium > 0

    def test_short_premium_is_negative(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test that short positions have negative premium (receive money)."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=-1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_entry_fill(legs, sample_surface, timestamp)

        # qty * price * 100 = -1 * 4.995 * 100 ≈ -500
        assert result.gross_premium < 0


class TestNetPremiumCalculation:
    """Tests for net premium calculation."""

    def test_net_premium_includes_fees(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test that net premium accounts for fees."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        result = execution_simulator.simulate_entry_fill(legs, sample_surface, timestamp)

        # net = gross + fees (fees increase the cost of any trade)
        assert result.net_premium == pytest.approx(result.gross_premium + result.fees, abs=0.01)


class TestCostEstimation:
    """Tests for cost estimation methods."""

    def test_estimate_execution_cost(
        self, execution_simulator, sample_surface, sample_greeks
    ):
        """Test execution cost estimation."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=2,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        costs = execution_simulator.estimate_execution_cost(legs, sample_surface)

        assert "slippage" in costs
        assert "fees" in costs
        assert "total" in costs
        assert costs["total"] == pytest.approx(costs["slippage"] + costs["fees"], abs=0.01)

    def test_calculate_fee_cost(self, execution_simulator, sample_greeks):
        """Test fee cost calculation."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=5,
                entry_price=5.00,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        fee = execution_simulator.calculate_fee_cost(legs)

        # 5 contracts * $0.65
        assert fee == pytest.approx(3.25, abs=0.01)


class TestFillResultDataclass:
    """Tests for FillResult dataclass."""

    def test_fill_result_creation(self, sample_greeks):
        """Test FillResult creation."""
        legs = [
            OptionLeg(
                symbol="SPY240315C00450000",
                qty=1,
                entry_price=5.20,
                strike=450.0,
                expiry=date(2024, 3, 15),
                right=OptionRight.CALL,
                greeks=sample_greeks,
            )
        ]

        result = FillResult(
            legs=legs,
            gross_premium=520.0,
            slippage=5.20,
            fees=0.65,
            net_premium=519.35,
            fill_prices={"SPY240315C00450000": 5.20},
            timestamp=datetime.now(timezone.utc),
        )

        assert result.gross_premium == 520.0
        assert result.fees == 0.65
        assert result.net_premium == 519.35
