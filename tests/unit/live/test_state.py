"""Unit tests for StateManager."""

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from src.config.schema import PaperConfig
from src.live.router import Fill
from src.live.state import StateManager, TradingState
from src.risk.portfolio import Portfolio, Position, PositionState
from src.strategy.orders import Order
from src.strategy.sizing import SizingResult
from src.strategy.types import Greeks, OptionLeg, OptionRight, Signal, SignalType


@pytest.fixture
def temp_state_dir(tmp_path: Path) -> str:
    """Create temporary state directory."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return str(state_dir)


@pytest.fixture
def paper_config(temp_state_dir: str) -> PaperConfig:
    """Create paper config with temp state dir."""
    return PaperConfig(
        state_dir=temp_state_dir,
        save_state_interval=1,
        loop_interval_seconds=0,
    )


@pytest.fixture
def sample_portfolio() -> Portfolio:
    """Create sample portfolio with positions."""
    legs = [
        OptionLeg(
            symbol="SPY240315C00450000",
            qty=1,
            entry_price=5.00,
            strike=450.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
        ),
        OptionLeg(
            symbol="SPY240315C00455000",
            qty=-1,
            entry_price=3.00,
            strike=455.0,
            expiry=date(2024, 3, 15),
            right=OptionRight.CALL,
            greeks=Greeks(delta=0.35, gamma=0.018, vega=0.25, theta=-0.04),
        ),
    ]

    portfolio = Portfolio(daily_pnl=100.0, max_acceptable_loss=5000.0)
    return portfolio.add_position(
        legs=legs,
        position_id="test_pos_001",
        structure_type="VerticalSpread",
        max_loss=200.0,
    )


@pytest.fixture
def sample_fill() -> Fill:
    """Create sample fill."""
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
    ]
    return Fill(
        order_id="order_001",
        legs=legs,
        gross_premium=-505.0,
        slippage=0.50,
        fees=0.65,
        net_premium=-506.15,
        fill_prices={"SPY240315C00450000": 5.05},
        timestamp=datetime.now(timezone.utc),
        fill_id="fill_001",
    )


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
def sample_order() -> Order:
    """Create sample order."""
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
        ),
    ]

    return Order(
        order_id="order_001",
        legs=legs,
        structure_type="SingleLeg",
        signal=signal,
        max_loss=510.0,
        greeks=Greeks(delta=0.45, gamma=0.02, vega=0.30, theta=-0.05),
        sizing_result=_make_sizing_result(1),
    )


class TestStateManagerInit:
    """Tests for StateManager initialization."""

    def test_init_creates_state_dir(self, paper_config: PaperConfig) -> None:
        """Test that state directory is created."""
        manager = StateManager(paper_config)
        assert Path(paper_config.state_dir).exists()
        assert manager.portfolio is not None

    def test_init_with_provided_portfolio(
        self, paper_config: PaperConfig, sample_portfolio: Portfolio
    ) -> None:
        """Test initialization with provided portfolio."""
        manager = StateManager(paper_config, portfolio=sample_portfolio)
        assert manager.portfolio == sample_portfolio
        assert len(manager.portfolio.positions) == 1

    def test_init_loads_existing_state(
        self, paper_config: PaperConfig, sample_portfolio: Portfolio
    ) -> None:
        """Test that existing state is loaded on init."""
        # Save state first
        manager1 = StateManager(paper_config, portfolio=sample_portfolio)
        manager1.save_snapshot(iteration=5)

        # Create new manager - should load state
        manager2 = StateManager(paper_config)
        assert len(manager2.portfolio.positions) == 1
        assert manager2.last_iteration == 5


class TestStateManagerRecordAndSave:
    """Tests for recording orders/fills and saving state."""

    def test_record_order(
        self, paper_config: PaperConfig, sample_order: Order
    ) -> None:
        """Test recording an order."""
        manager = StateManager(paper_config)
        manager.record_order(sample_order)

        assert len(manager.order_history) == 1
        assert manager.order_history[0]["order_id"] == "order_001"

    def test_record_fill(
        self, paper_config: PaperConfig, sample_fill: Fill
    ) -> None:
        """Test recording a fill."""
        manager = StateManager(paper_config)
        manager.record_fill(sample_fill)

        assert len(manager.fill_history) == 1
        assert manager.fill_history[0]["fill_id"] == "fill_001"

    def test_save_snapshot(
        self, paper_config: PaperConfig, sample_portfolio: Portfolio
    ) -> None:
        """Test saving state snapshot."""
        manager = StateManager(paper_config, portfolio=sample_portfolio)
        path = manager.save_snapshot(iteration=10)

        assert path.exists()
        assert "state_" in path.name
        assert path.suffix == ".json"

        # Verify content
        with open(path) as f:
            data = json.load(f)
        assert data["iteration"] == 10
        assert len(data["portfolio"]["positions"]) == 1

    def test_save_creates_multiple_snapshots(
        self, paper_config: PaperConfig, sample_portfolio: Portfolio
    ) -> None:
        """Test that multiple snapshots can be saved."""
        manager = StateManager(paper_config, portfolio=sample_portfolio)
        manager.save_snapshot(iteration=1)
        manager.save_snapshot(iteration=2)
        manager.save_snapshot(iteration=3)

        assert manager.get_snapshot_count() == 3


class TestStateManagerLoadState:
    """Tests for loading state."""

    def test_load_state_returns_none_when_empty(
        self, paper_config: PaperConfig
    ) -> None:
        """Test load_state returns None when no snapshots exist."""
        manager = StateManager(paper_config)
        state = manager.load_state()
        assert state is None

    def test_load_state_returns_latest(
        self, paper_config: PaperConfig, sample_portfolio: Portfolio
    ) -> None:
        """Test that load_state returns the most recent snapshot."""
        manager = StateManager(paper_config, portfolio=sample_portfolio)
        manager.save_snapshot(iteration=1)

        # Update portfolio
        new_portfolio = sample_portfolio.add_position(
            legs=[
                OptionLeg(
                    symbol="SPY240315P00445000",
                    qty=-1,
                    entry_price=4.00,
                    strike=445.0,
                    expiry=date(2024, 3, 15),
                    right=OptionRight.PUT,
                    greeks=Greeks(delta=-0.40, gamma=0.02, vega=0.28, theta=-0.04),
                )
            ],
            position_id="test_pos_002",
            structure_type="SingleLeg",
        )
        manager.update_portfolio(new_portfolio)
        manager.save_snapshot(iteration=2)

        # Load state
        state = manager.load_state()
        assert state is not None
        assert state.iteration == 2
        assert len(state.portfolio.positions) == 2

    def test_load_state_preserves_history(
        self,
        paper_config: PaperConfig,
        sample_portfolio: Portfolio,
        sample_order: Order,
        sample_fill: Fill,
    ) -> None:
        """Test that order/fill history is preserved."""
        manager1 = StateManager(paper_config, portfolio=sample_portfolio)
        manager1.record_order(sample_order)
        manager1.record_fill(sample_fill)
        manager1.save_snapshot(iteration=5)

        # Load in new manager
        manager2 = StateManager(paper_config)
        assert len(manager2.order_history) == 1
        assert len(manager2.fill_history) == 1
        assert manager2.order_history[0]["order_id"] == "order_001"
        assert manager2.fill_history[0]["fill_id"] == "fill_001"


class TestStateManagerClear:
    """Tests for clearing state."""

    def test_clear_state(
        self, paper_config: PaperConfig, sample_portfolio: Portfolio
    ) -> None:
        """Test clearing all state."""
        manager = StateManager(paper_config, portfolio=sample_portfolio)
        manager.save_snapshot(iteration=1)
        manager.save_snapshot(iteration=2)

        assert manager.get_snapshot_count() == 2

        manager.clear_state()

        assert manager.get_snapshot_count() == 0
        assert len(manager.portfolio.positions) == 0
        assert len(manager.order_history) == 0
        assert len(manager.fill_history) == 0


class TestTradingState:
    """Tests for TradingState dataclass."""

    def test_to_dict(self, sample_portfolio: Portfolio) -> None:
        """Test TradingState serialization."""
        state = TradingState(
            portfolio=sample_portfolio,
            order_history=[{"order_id": "test"}],
            fill_history=[{"fill_id": "test"}],
            iteration=10,
            daily_pnl=150.0,
        )

        data = state.to_dict()
        assert data["iteration"] == 10
        assert data["daily_pnl"] == 150.0
        assert len(data["portfolio"]["positions"]) == 1
        assert len(data["order_history"]) == 1

    def test_from_dict(self, sample_portfolio: Portfolio) -> None:
        """Test TradingState deserialization."""
        state = TradingState(
            portfolio=sample_portfolio,
            order_history=[{"order_id": "test"}],
            fill_history=[{"fill_id": "test"}],
            iteration=10,
            daily_pnl=150.0,
        )

        data = state.to_dict()
        restored = TradingState.from_dict(data)

        assert restored.iteration == 10
        assert restored.daily_pnl == 150.0
        assert len(restored.portfolio.positions) == 1
        assert len(restored.order_history) == 1
