"""Unit tests for rebalancing engine."""

from datetime import date, datetime, timezone

import pytest

from src.config.schema import DriftBandsConfig, RebalancingConfig
from src.risk.portfolio import Portfolio, Position, PositionState
from src.strategy.positions.lifecycle import ManagedPosition
from src.strategy.positions.rebalance import DriftResult, DriftStatus, RebalanceEngine
from src.strategy.types import (
    ExitSignalType,
    Greeks,
    OptionLeg,
    OptionRight,
    Signal,
    SignalType,
)


@pytest.fixture
def rebalancing_config():
    """Create rebalancing config for testing."""
    return RebalancingConfig(
        enabled=True,
        check_interval_seconds=60,
        strategy="close_first",
        max_trades_per_rebalance=3,
    )


@pytest.fixture
def drift_bands_config():
    """Create drift bands config for testing."""
    return DriftBandsConfig(
        delta_target=0.0,
        delta_max_drift=50.0,
        vega_target=0.0,
        vega_max_drift=500.0,
        gamma_max_drift=25.0,
    )


@pytest.fixture
def rebalance_engine(rebalancing_config, drift_bands_config):
    """Create rebalance engine for testing."""
    return RebalanceEngine(
        rebalancing_config=rebalancing_config,
        drift_bands_config=drift_bands_config,
    )


@pytest.fixture
def sample_greeks():
    """Sample Greeks for testing."""
    return Greeks(delta=0.5, gamma=0.05, vega=0.3, theta=-0.02)


@pytest.fixture
def sample_signal():
    """Sample entry signal for testing."""
    return Signal(
        signal_type=SignalType.DIRECTIONAL_VOL,
        edge=0.05,
        confidence=0.75,
        tenor_days=30,
        delta_bucket="ATM",
    )


def create_portfolio_with_greeks(
    net_delta: float = 0.0,
    net_vega: float = 0.0,
    net_gamma: float = 0.0,
) -> Portfolio:
    """Create a portfolio with specified aggregate Greeks."""
    # Create a single position that produces the desired aggregate Greeks
    # Note: Portfolio aggregates position Greeks * CONTRACT_MULTIPLIER
    # So we need to set position Greeks at 1/100 of desired
    legs = [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(
                delta=net_delta / 100,  # Contract multiplier adjustment
                gamma=net_gamma / 100,
                vega=net_vega / 100,
                theta=-0.02,
            ),
        )
    ]

    return Portfolio(positions=[Position(legs=legs)])


class TestDriftDetection:
    """Tests for drift detection."""

    def test_no_drift_within_bands(self, rebalance_engine):
        """Test that no drift is detected when within bands."""
        portfolio = create_portfolio_with_greeks(
            net_delta=20.0,  # Within 50 max
            net_vega=200.0,  # Within 500 max
            net_gamma=10.0,  # Within 25 max
        )

        result = rebalance_engine.check_drift(portfolio)

        assert result.needs_rebalance is False
        assert result.most_breached is None

    def test_delta_drift_detected(self, rebalance_engine):
        """Test detection of delta drift breach."""
        portfolio = create_portfolio_with_greeks(
            net_delta=60.0,  # Exceeds 50 max
            net_vega=100.0,
            net_gamma=10.0,
        )

        result = rebalance_engine.check_drift(portfolio)

        assert result.needs_rebalance is True
        assert result.drifts["delta"].breached is True
        assert result.drifts["delta"].drift == 60.0  # abs(60 - 0)
        assert result.most_breached == "delta"

    def test_vega_drift_detected(self, rebalance_engine):
        """Test detection of vega drift breach."""
        portfolio = create_portfolio_with_greeks(
            net_delta=20.0,
            net_vega=600.0,  # Exceeds 500 max
            net_gamma=10.0,
        )

        result = rebalance_engine.check_drift(portfolio)

        assert result.needs_rebalance is True
        assert result.drifts["vega"].breached is True
        assert result.most_breached == "vega"

    def test_gamma_drift_detected(self, rebalance_engine):
        """Test detection of gamma drift breach."""
        portfolio = create_portfolio_with_greeks(
            net_delta=20.0,
            net_vega=100.0,
            net_gamma=30.0,  # Exceeds 25 max
        )

        result = rebalance_engine.check_drift(portfolio)

        assert result.needs_rebalance is True
        assert result.drifts["gamma"].breached is True

    def test_multiple_drifts_detected(self, rebalance_engine):
        """Test detection of multiple drift breaches."""
        portfolio = create_portfolio_with_greeks(
            net_delta=80.0,  # Exceeds 50 (160% of limit)
            net_vega=600.0,  # Exceeds 500 (120% of limit)
            net_gamma=30.0,  # Exceeds 25 (120% of limit)
        )

        result = rebalance_engine.check_drift(portfolio)

        assert result.needs_rebalance is True
        assert result.drifts["delta"].breached is True
        assert result.drifts["vega"].breached is True
        assert result.drifts["gamma"].breached is True
        # Most breached should be delta (160% vs 120%)
        assert result.most_breached == "delta"

    def test_negative_delta_drift(self, rebalance_engine):
        """Test drift detection for negative delta."""
        portfolio = create_portfolio_with_greeks(
            net_delta=-70.0,  # Negative drift, exceeds 50 max
            net_vega=100.0,
            net_gamma=10.0,
        )

        result = rebalance_engine.check_drift(portfolio)

        assert result.needs_rebalance is True
        assert result.drifts["delta"].breached is True
        assert result.drifts["delta"].drift == 70.0  # abs(-70 - 0)


class TestRebalanceSignalGeneration:
    """Tests for rebalance signal generation."""

    def test_no_signals_when_disabled(
        self, rebalancing_config, drift_bands_config, sample_greeks, sample_signal
    ):
        """Test that no signals generated when rebalancing disabled."""
        rebalancing_config.enabled = False
        engine = RebalanceEngine(rebalancing_config, drift_bands_config)

        portfolio = create_portfolio_with_greeks(net_delta=80.0)

        drift_result = DriftResult(
            needs_rebalance=True,
            drifts={
                "delta": DriftStatus(
                    greek_name="delta",
                    current=80.0,
                    target=0.0,
                    drift=80.0,
                    max_allowed=50.0,
                    breached=True,
                )
            },
            most_breached="delta",
        )

        # Create managed positions
        positions = [
            ManagedPosition(
                position_id="pos-001",
                state=PositionState.OPEN,
                legs=[
                    OptionLeg(
                        symbol="SPY240315C00450000",
                        qty=1,
                        entry_price=5.00,
                        strike=450.0,
                        expiry=date(2024, 3, 15),
                        right=OptionRight.CALL,
                        greeks=sample_greeks,
                    )
                ],
                structure_type="LongCall",
                entry_signal=sample_signal,
                entry_time=datetime.now(timezone.utc),
                entry_price=-500.0,
                entry_greeks=sample_greeks,
                max_loss=500.0,
                current_greeks=Greeks(delta=40.0, gamma=5.0, vega=30.0, theta=-2.0),
            )
        ]

        signals = engine.generate_rebalance_signals(drift_result, positions, portfolio)

        assert len(signals) == 0

    def test_no_signals_when_no_rebalance_needed(
        self, rebalance_engine, sample_greeks, sample_signal
    ):
        """Test that no signals generated when no rebalancing needed."""
        portfolio = create_portfolio_with_greeks(net_delta=20.0)

        drift_result = DriftResult(
            needs_rebalance=False,
            drifts={},
            most_breached=None,
        )

        positions = []

        signals = rebalance_engine.generate_rebalance_signals(
            drift_result, positions, portfolio
        )

        assert len(signals) == 0

    def test_signal_generated_for_drifted_position(
        self, rebalance_engine, sample_greeks, sample_signal
    ):
        """Test that signal generated for position contributing to drift."""
        portfolio = create_portfolio_with_greeks(net_delta=80.0)

        drift_result = DriftResult(
            needs_rebalance=True,
            drifts={
                "delta": DriftStatus(
                    greek_name="delta",
                    current=80.0,
                    target=0.0,
                    drift=80.0,
                    max_allowed=50.0,
                    breached=True,
                )
            },
            most_breached="delta",
        )

        # Position with positive delta contributing to drift
        positions = [
            ManagedPosition(
                position_id="pos-001",
                state=PositionState.OPEN,
                legs=[
                    OptionLeg(
                        symbol="SPY240315C00450000",
                        qty=1,
                        entry_price=5.00,
                        strike=450.0,
                        expiry=date(2024, 3, 15),
                        right=OptionRight.CALL,
                        greeks=sample_greeks,
                    )
                ],
                structure_type="LongCall",
                entry_signal=sample_signal,
                entry_time=datetime.now(timezone.utc),
                entry_price=-500.0,
                entry_greeks=sample_greeks,
                max_loss=500.0,
                current_greeks=Greeks(delta=40.0, gamma=5.0, vega=30.0, theta=-2.0),
            )
        ]

        signals = rebalance_engine.generate_rebalance_signals(
            drift_result, positions, portfolio
        )

        assert len(signals) == 1
        assert signals[0].exit_type == ExitSignalType.REBALANCE
        assert signals[0].position_id == "pos-001"
        assert "delta" in signals[0].reason.lower()

    def test_max_trades_limit_respected(
        self, rebalancing_config, drift_bands_config, sample_greeks, sample_signal
    ):
        """Test that max trades per rebalance is respected."""
        rebalancing_config.max_trades_per_rebalance = 2
        engine = RebalanceEngine(rebalancing_config, drift_bands_config)

        portfolio = create_portfolio_with_greeks(net_delta=150.0)

        drift_result = DriftResult(
            needs_rebalance=True,
            drifts={
                "delta": DriftStatus(
                    greek_name="delta",
                    current=150.0,
                    target=0.0,
                    drift=150.0,
                    max_allowed=50.0,
                    breached=True,
                )
            },
            most_breached="delta",
        )

        # Create 5 positions all contributing to drift
        positions = []
        for i in range(5):
            positions.append(
                ManagedPosition(
                    position_id=f"pos-{i:03d}",
                    state=PositionState.OPEN,
                    legs=[
                        OptionLeg(
                            symbol=f"SPY240315C0045{i}000",
                            qty=1,
                            entry_price=5.00,
                            strike=450.0 + i,
                            expiry=date(2024, 3, 15),
                            right=OptionRight.CALL,
                            greeks=sample_greeks,
                        )
                    ],
                    structure_type="LongCall",
                    entry_signal=sample_signal,
                    entry_time=datetime.now(timezone.utc),
                    entry_price=-500.0,
                    entry_greeks=sample_greeks,
                    max_loss=500.0,
                    current_greeks=Greeks(delta=30.0, gamma=3.0, vega=20.0, theta=-1.5),
                )
            )

        signals = engine.generate_rebalance_signals(drift_result, positions, portfolio)

        # Should be limited to 2 trades
        assert len(signals) == 2

    def test_skip_non_open_positions(
        self, rebalance_engine, sample_greeks, sample_signal
    ):
        """Test that non-OPEN positions are skipped."""
        portfolio = create_portfolio_with_greeks(net_delta=80.0)

        drift_result = DriftResult(
            needs_rebalance=True,
            drifts={
                "delta": DriftStatus(
                    greek_name="delta",
                    current=80.0,
                    target=0.0,
                    drift=80.0,
                    max_allowed=50.0,
                    breached=True,
                )
            },
            most_breached="delta",
        )

        # Position that's already closing
        positions = [
            ManagedPosition(
                position_id="pos-001",
                state=PositionState.CLOSING,  # Not OPEN
                legs=[
                    OptionLeg(
                        symbol="SPY240315C00450000",
                        qty=1,
                        entry_price=5.00,
                        strike=450.0,
                        expiry=date(2024, 3, 15),
                        right=OptionRight.CALL,
                        greeks=sample_greeks,
                    )
                ],
                structure_type="LongCall",
                entry_signal=sample_signal,
                entry_time=datetime.now(timezone.utc),
                entry_price=-500.0,
                entry_greeks=sample_greeks,
                max_loss=500.0,
                current_greeks=Greeks(delta=40.0, gamma=5.0, vega=30.0, theta=-2.0),
            )
        ]

        signals = rebalance_engine.generate_rebalance_signals(
            drift_result, positions, portfolio
        )

        assert len(signals) == 0


class TestRebalanceEngineHelpers:
    """Tests for helper methods."""

    def test_enabled_property(self, rebalance_engine):
        """Test enabled property."""
        assert rebalance_engine.enabled is True

    def test_is_within_bands_true(self, rebalance_engine):
        """Test is_within_bands returns True when within bands."""
        portfolio = create_portfolio_with_greeks(
            net_delta=20.0,
            net_vega=200.0,
            net_gamma=10.0,
        )

        assert rebalance_engine.is_within_bands(portfolio) is True

    def test_is_within_bands_false(self, rebalance_engine):
        """Test is_within_bands returns False when breached."""
        portfolio = create_portfolio_with_greeks(
            net_delta=80.0,  # Exceeds 50 max
        )

        assert rebalance_engine.is_within_bands(portfolio) is False

    def test_get_current_drift(self, rebalance_engine):
        """Test getting current drift values."""
        portfolio = create_portfolio_with_greeks(
            net_delta=30.0,
            net_vega=100.0,
            net_gamma=15.0,
        )

        drift = rebalance_engine.get_current_drift(portfolio)

        assert drift["delta"] == 30.0
        assert drift["vega"] == 100.0
        assert drift["gamma"] == 15.0


class TestDriftStatusDataclass:
    """Tests for DriftStatus dataclass."""

    def test_drift_status_creation(self):
        """Test DriftStatus creation."""
        status = DriftStatus(
            greek_name="delta",
            current=60.0,
            target=0.0,
            drift=60.0,
            max_allowed=50.0,
            breached=True,
        )

        assert status.greek_name == "delta"
        assert status.current == 60.0
        assert status.target == 0.0
        assert status.drift == 60.0
        assert status.max_allowed == 50.0
        assert status.breached is True


class TestDriftResultDataclass:
    """Tests for DriftResult dataclass."""

    def test_drift_result_creation(self):
        """Test DriftResult creation."""
        delta_status = DriftStatus(
            greek_name="delta",
            current=60.0,
            target=0.0,
            drift=60.0,
            max_allowed=50.0,
            breached=True,
        )

        result = DriftResult(
            needs_rebalance=True,
            drifts={"delta": delta_status},
            most_breached="delta",
        )

        assert result.needs_rebalance is True
        assert "delta" in result.drifts
        assert result.most_breached == "delta"
