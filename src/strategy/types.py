"""Core data types for trade structures.

This module defines the fundamental data types used throughout the strategy
package: Greeks, OptionLeg, Signal, and supporting enums.
"""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum
from typing import Any


class SignalType(str, Enum):
    """Types of trading signals from model predictions."""

    TERM_ANOMALY = "term_anomaly"  # Term structure mispricing → Calendar
    DIRECTIONAL_VOL = "directional_vol"  # Strong vol direction → Vertical
    SKEW_ANOMALY = "skew_anomaly"  # Skew mispricing → Skew trade
    ELEVATED_IV = "elevated_iv"  # High IV, low confidence → Iron condor


class ExitSignalType(str, Enum):
    """Types of exit signals for position management.

    These signals trigger position closure or rebalancing.
    """

    MODEL_REVERSAL = "model_reversal"  # Model predicts opposite direction
    EDGE_DECAY = "edge_decay"  # Edge dropped below threshold
    STOP_LOSS = "stop_loss"  # Position hit stop-loss threshold
    TAKE_PROFIT = "take_profit"  # Position hit profit target
    TIME_DECAY = "time_decay"  # Approaching expiration
    REBALANCE = "rebalance"  # Greek drift requires adjustment
    MANUAL = "manual"  # Manual override / user-initiated


class OptionRight(str, Enum):
    """Option type (call or put)."""

    CALL = "C"
    PUT = "P"


@dataclass(frozen=True)
class Greeks:
    """Option Greeks for a position.

    This is an immutable dataclass representing the four primary Greeks:
    delta, gamma, vega, and theta. Supports arithmetic operations for
    portfolio aggregation.

    Attributes:
        delta: Sensitivity to underlying price (∂V/∂S)
        gamma: Rate of change of delta (∂²V/∂S²)
        vega: Sensitivity to volatility (∂V/∂σ)
        theta: Time decay (∂V/∂t), typically negative for long positions
    """

    delta: float
    gamma: float
    vega: float
    theta: float

    def scale(self, multiplier: float) -> "Greeks":
        """Scale all Greeks by a multiplier.

        Useful for adjusting by quantity or contract size.

        Args:
            multiplier: Factor to scale all Greeks by

        Returns:
            New Greeks instance with scaled values
        """
        return Greeks(
            delta=self.delta * multiplier,
            gamma=self.gamma * multiplier,
            vega=self.vega * multiplier,
            theta=self.theta * multiplier,
        )

    def __add__(self, other: "Greeks") -> "Greeks":
        """Add two Greeks together for portfolio aggregation.

        Args:
            other: Another Greeks instance to add

        Returns:
            New Greeks instance with summed values
        """
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
        )

    def to_dict(self) -> dict[str, float]:
        """Serialize Greeks to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Greeks":
        """Deserialize Greeks from dictionary."""
        return cls(
            delta=float(data["delta"]),
            gamma=float(data["gamma"]),
            vega=float(data["vega"]),
            theta=float(data["theta"]),
        )


@dataclass
class OptionLeg:
    """Single leg of a trade structure.

    Represents one option position within a multi-leg trade structure.

    Attributes:
        symbol: OSI-format option symbol (e.g., "SPY240315C00450000")
        qty: Position quantity (+1 long, -1 short)
        entry_price: Execution price per contract
        strike: Strike price
        expiry: Expiration date
        right: Option type (CALL or PUT)
        greeks: Greeks at entry time
    """

    symbol: str
    qty: int
    entry_price: float
    strike: float
    expiry: date
    right: OptionRight
    greeks: Greeks

    def to_dict(self) -> dict[str, Any]:
        """Serialize OptionLeg to dictionary."""
        return {
            "symbol": self.symbol,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "right": self.right.value,
            "greeks": self.greeks.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OptionLeg":
        """Deserialize OptionLeg from dictionary."""
        return cls(
            symbol=data["symbol"],
            qty=int(data["qty"]),
            entry_price=float(data["entry_price"]),
            strike=float(data["strike"]),
            expiry=date.fromisoformat(data["expiry"]),
            right=OptionRight(data["right"]),
            greeks=Greeks.from_dict(data["greeks"]),
        )


@dataclass
class Signal:
    """Trading signal from model predictions.

    Represents a trading opportunity identified by the model at a specific
    node on the volatility surface.

    Attributes:
        signal_type: Type of signal (determines structure selection)
        edge: Expected edge (positive = buy vol, negative = sell vol)
        confidence: Confidence level from 0.0 to 1.0
        tenor_days: Days to expiration for the target node
        delta_bucket: Delta bucket identifier (e.g., "P25", "ATM", "C25")
        timestamp: When the signal was generated
    """

    signal_type: SignalType
    edge: float
    confidence: float
    tenor_days: int
    delta_bucket: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize Signal to dictionary."""
        return {
            "signal_type": self.signal_type.value,
            "edge": self.edge,
            "confidence": self.confidence,
            "tenor_days": self.tenor_days,
            "delta_bucket": self.delta_bucket,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Signal":
        """Deserialize Signal from dictionary."""
        return cls(
            signal_type=SignalType(data["signal_type"]),
            edge=float(data["edge"]),
            confidence=float(data["confidence"]),
            tenor_days=int(data["tenor_days"]),
            delta_bucket=data["delta_bucket"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class ExitSignal:
    """Exit signal for position management.

    Represents a signal to close or adjust an existing position.

    Attributes:
        exit_type: Type of exit signal (determines handling)
        position_id: ID of the position to exit
        urgency: Priority level from 0.0 (low) to 1.0 (immediate)
        reason: Human-readable explanation for the exit
        current_pnl_pct: Current P&L as percentage of max loss
        timestamp: When the signal was generated
    """

    exit_type: ExitSignalType
    position_id: str
    urgency: float
    reason: str
    current_pnl_pct: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize ExitSignal to dictionary."""
        return {
            "exit_type": self.exit_type.value,
            "position_id": self.position_id,
            "urgency": self.urgency,
            "reason": self.reason,
            "current_pnl_pct": self.current_pnl_pct,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExitSignal":
        """Deserialize ExitSignal from dictionary."""
        return cls(
            exit_type=ExitSignalType(data["exit_type"]),
            position_id=data["position_id"],
            urgency=float(data["urgency"]),
            reason=data["reason"],
            current_pnl_pct=float(data.get("current_pnl_pct", 0.0)),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
