"""Exit signal generation for position management.

This module provides the ExitSignalGenerator class for evaluating
open positions and generating exit signals based on model predictions
and rule-based safety nets.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ...config.schema import ExitSignalsConfig
from ...risk.portfolio import PositionState
from ..types import ExitSignal, ExitSignalType
from .lifecycle import ManagedPosition

logger = logging.getLogger(__name__)


class ExitSignalGenerator:
    """Generate exit signals for open positions.

    Evaluates positions against both model-driven exit triggers
    (edge reversal, edge decay) and rule-based safety nets
    (stop-loss, take-profit, time decay).

    Exit checks are evaluated in priority order:
    1. Stop-loss (highest priority - risk protection)
    2. Model reversal (model predicts opposite direction)
    3. Edge decay (edge fell below minimum)
    4. Take-profit (capture gains)
    5. Time decay (approaching expiration)

    Only one exit signal is generated per position per evaluation cycle.
    """

    def __init__(self, config: ExitSignalsConfig) -> None:
        """Initialize exit signal generator.

        Args:
            config: Exit signals configuration with thresholds
        """
        self._config = config

    def evaluate_positions(
        self,
        positions: list[ManagedPosition],
        surface: pd.DataFrame,
        model_predictions: Optional[pd.DataFrame] = None,
    ) -> list[ExitSignal]:
        """Evaluate all positions for exit triggers.

        Args:
            positions: Open positions to evaluate
            surface: Current surface snapshot for mark-to-market
            model_predictions: Current model predictions with 'edge' column
                               (optional - if None, model-driven exits are skipped)

        Returns:
            List of exit signals (one per position that triggered)
        """
        exit_signals: list[ExitSignal] = []

        for position in positions:
            if position.state != PositionState.OPEN:
                continue

            signal = self._evaluate_single_position(
                position=position,
                surface=surface,
                model_predictions=model_predictions,
            )

            if signal is not None:
                exit_signals.append(signal)

        return exit_signals

    def _evaluate_single_position(
        self,
        position: ManagedPosition,
        surface: pd.DataFrame,
        model_predictions: Optional[pd.DataFrame],
    ) -> Optional[ExitSignal]:
        """Evaluate a single position for exit triggers.

        Args:
            position: Position to evaluate
            surface: Current surface snapshot
            model_predictions: Current model predictions

        Returns:
            ExitSignal if triggered, None otherwise
        """
        # Priority 1: Stop-loss (highest priority - risk protection)
        signal = self._check_stop_loss(position)
        if signal is not None:
            return signal

        # Priority 2: Model reversal (if predictions available)
        if model_predictions is not None:
            signal = self._check_model_reversal(position, model_predictions)
            if signal is not None:
                return signal

            # Priority 3: Edge decay
            signal = self._check_edge_decay(position, model_predictions)
            if signal is not None:
                return signal

        # Priority 4: Take-profit
        signal = self._check_take_profit(position)
        if signal is not None:
            return signal

        # Priority 5: Time decay
        signal = self._check_time_decay(position)
        if signal is not None:
            return signal

        return None

    def _check_stop_loss(self, position: ManagedPosition) -> Optional[ExitSignal]:
        """Check structure-level stop-loss trigger.

        Triggers when unrealized loss reaches configured percentage of max loss.

        Args:
            position: Position to check

        Returns:
            ExitSignal if stop-loss triggered, None otherwise
        """
        if position.max_loss <= 0:
            return None

        # Calculate loss as percentage of max loss
        # unrealized_pnl is negative when losing money
        if position.unrealized_pnl >= 0:
            return None  # Not losing money

        loss_pct = abs(position.unrealized_pnl) / position.max_loss

        if loss_pct >= self._config.stop_loss_pct:
            return ExitSignal(
                exit_type=ExitSignalType.STOP_LOSS,
                position_id=position.position_id,
                urgency=0.95,  # High urgency for stop-loss
                reason=f"Stop-loss triggered: {loss_pct:.1%} of max loss reached "
                f"(threshold: {self._config.stop_loss_pct:.1%})",
                current_pnl_pct=-loss_pct,
            )

        return None

    def _check_model_reversal(
        self,
        position: ManagedPosition,
        predictions: pd.DataFrame,
    ) -> Optional[ExitSignal]:
        """Check if model edge has reversed direction.

        Triggers when the current edge has the opposite sign from entry
        with sufficient magnitude.

        Args:
            position: Position to check
            predictions: DataFrame with model predictions

        Returns:
            ExitSignal if reversal detected, None otherwise
        """
        current_edge = self._get_current_edge(position, predictions)
        if current_edge is None:
            return None

        entry_edge = position.entry_signal.edge

        # Check for sign flip
        # Entry positive, current negative (or vice versa)
        edges_reversed = (entry_edge > 0 and current_edge < 0) or (
            entry_edge < 0 and current_edge > 0
        )

        if not edges_reversed:
            return None

        # Check if reversal is significant (at least 50% of entry magnitude)
        min_reversal = abs(entry_edge) * 0.5
        if abs(current_edge) < min_reversal:
            return None

        # Calculate urgency based on reversal magnitude
        urgency = min(abs(current_edge) / 0.05, 1.0)

        return ExitSignal(
            exit_type=ExitSignalType.MODEL_REVERSAL,
            position_id=position.position_id,
            urgency=urgency,
            reason=f"Edge reversed: {entry_edge:.3f} -> {current_edge:.3f}",
            current_pnl_pct=position.pnl_pct_of_max_loss(),
        )

    def _check_edge_decay(
        self,
        position: ManagedPosition,
        predictions: pd.DataFrame,
    ) -> Optional[ExitSignal]:
        """Check if edge has decayed below minimum threshold.

        Triggers when current edge magnitude falls below configured
        retention ratio of entry edge.

        Args:
            position: Position to check
            predictions: DataFrame with model predictions

        Returns:
            ExitSignal if edge decayed, None otherwise
        """
        current_edge = self._get_current_edge(position, predictions)
        if current_edge is None:
            return None

        entry_edge = position.entry_signal.edge

        # Calculate edge retention ratio
        # If entry edge was 0.05 and current is 0.01, retention is 0.2 (20%)
        if abs(entry_edge) < 1e-6:
            return None  # Avoid division by zero

        retention = abs(current_edge) / abs(entry_edge)

        if retention < self._config.min_edge_retention:
            return ExitSignal(
                exit_type=ExitSignalType.EDGE_DECAY,
                position_id=position.position_id,
                urgency=0.6,  # Moderate urgency
                reason=f"Edge decayed: {abs(current_edge):.3f} is {retention:.1%} of "
                f"entry edge {abs(entry_edge):.3f} (min retention: "
                f"{self._config.min_edge_retention:.1%})",
                current_pnl_pct=position.pnl_pct_of_max_loss(),
            )

        return None

    def _check_take_profit(self, position: ManagedPosition) -> Optional[ExitSignal]:
        """Check structure-level take-profit trigger.

        Triggers when unrealized profit reaches configured percentage
        of max loss (used as reference for profit sizing).

        Args:
            position: Position to check

        Returns:
            ExitSignal if take-profit triggered, None otherwise
        """
        if position.max_loss <= 0:
            return None

        # Calculate profit as percentage of max loss
        if position.unrealized_pnl <= 0:
            return None  # Not making money

        profit_pct = position.unrealized_pnl / position.max_loss

        if profit_pct >= self._config.take_profit_pct:
            return ExitSignal(
                exit_type=ExitSignalType.TAKE_PROFIT,
                position_id=position.position_id,
                urgency=0.5,  # Lower urgency - can let profits run
                reason=f"Take-profit triggered: {profit_pct:.1%} profit of max loss "
                f"(threshold: {self._config.take_profit_pct:.1%})",
                current_pnl_pct=profit_pct,
            )

        return None

    def _check_time_decay(self, position: ManagedPosition) -> Optional[ExitSignal]:
        """Check time-based exit trigger.

        Triggers when position approaches expiration to avoid
        gamma risk and assignment risk near expiry.

        Args:
            position: Position to check

        Returns:
            ExitSignal if time decay triggered, None otherwise
        """
        if position.days_to_expiry <= self._config.min_dte_exit:
            # Calculate urgency - higher as we get closer to expiry
            urgency = min(
                self._config.min_dte_exit / max(position.days_to_expiry, 1),
                1.0,
            )

            return ExitSignal(
                exit_type=ExitSignalType.TIME_DECAY,
                position_id=position.position_id,
                urgency=urgency,
                reason=f"Time decay: {position.days_to_expiry} DTE <= "
                f"{self._config.min_dte_exit} day threshold",
                current_pnl_pct=position.pnl_pct_of_max_loss(),
            )

        return None

    def _get_current_edge(
        self,
        position: ManagedPosition,
        predictions: pd.DataFrame,
    ) -> Optional[float]:
        """Get current model edge for position's node.

        Looks up the current edge prediction for the same tenor/delta
        bucket as the position's entry signal.

        Args:
            position: Position to look up
            predictions: DataFrame with tenor_days, delta_bucket, edge columns

        Returns:
            Current edge value, or None if not found
        """
        if predictions.empty:
            return None

        # Match by tenor and delta bucket from entry signal
        tenor_days = position.entry_signal.tenor_days
        delta_bucket = position.entry_signal.delta_bucket

        node_pred = predictions[
            (predictions["tenor_days"] == tenor_days)
            & (predictions["delta_bucket"] == delta_bucket)
        ]

        if node_pred.empty:
            logger.debug(
                f"No prediction found for position {position.position_id} "
                f"at {tenor_days}d {delta_bucket}"
            )
            return None

        edge = node_pred.iloc[0].get("edge")
        if pd.isna(edge):
            return None

        return float(edge)

    def create_manual_exit(
        self,
        position_id: str,
        reason: str = "Manual exit requested",
    ) -> ExitSignal:
        """Create a manual exit signal.

        Args:
            position_id: Position to exit
            reason: Reason for manual exit

        Returns:
            ExitSignal with MANUAL type
        """
        return ExitSignal(
            exit_type=ExitSignalType.MANUAL,
            position_id=position_id,
            urgency=1.0,  # Immediate
            reason=reason,
            current_pnl_pct=0.0,  # Will be updated when processing
        )
