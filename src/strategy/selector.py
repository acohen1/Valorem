"""Structure selection logic.

This module provides the StructureSelector class that maps trading signals
to appropriate trade structures based on signal characteristics.
"""

from typing import Optional

import pandas as pd

from ..config.schema import ExecutionConfig
from .structures import (
    CalendarSpread,
    IronCondor,
    SkewTrade,
    TradeStructure,
    VerticalSpread,
)
from .types import Signal, SignalType


class StructureSelector:
    """Select appropriate trade structure based on signal characteristics.

    Maps signal types to trade structures following the logic:
    - TERM_ANOMALY → CalendarSpread (term structure mispricing)
    - DIRECTIONAL_VOL (high confidence) → VerticalSpread (strong vol direction)
    - SKEW_ANOMALY → SkewTrade (skew mispricing between puts and calls)
    - ELEVATED_IV (low confidence) → IronCondor (sell premium, profit from IV crush)

    Attributes:
        _config: Execution configuration
        _structures: Mapping of structure names to instances
    """

    # Confidence thresholds for structure selection
    HIGH_CONFIDENCE_THRESHOLD = 0.7
    LOW_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self, config: Optional[ExecutionConfig] = None) -> None:
        """Initialize structure selector.

        Args:
            config: Execution configuration (optional, uses defaults if not provided)
        """
        self._config = config or ExecutionConfig()

        # Initialize available structures
        self._structures: dict[str, TradeStructure] = {
            "calendar": CalendarSpread(),
            "vertical": VerticalSpread(),
            "skew": SkewTrade(),
            "iron_condor": IronCondor(),
        }

    def select_structure(
        self,
        signal: Signal,
        surface: Optional[pd.DataFrame] = None,
    ) -> Optional[TradeStructure]:
        """Select structure based on signal type and confidence.

        The selection logic follows these rules:
        1. TERM_ANOMALY → Calendar spread (exploit term structure mispricing)
        2. DIRECTIONAL_VOL with high confidence → Vertical spread
        3. SKEW_ANOMALY → Skew trade (risk reversal with wings)
        4. ELEVATED_IV → Iron condor (sell premium on both sides)
        5. Default → None (no trade)

        Args:
            signal: Trading signal with type and confidence
            surface: Optional surface data (for future use in dynamic selection)

        Returns:
            Selected TradeStructure instance, or None if no suitable structure
        """
        # Term structure signal → Calendar
        if signal.signal_type == SignalType.TERM_ANOMALY:
            return self._structures["calendar"]

        # Strong directional vol signal → Vertical
        if (
            signal.signal_type == SignalType.DIRECTIONAL_VOL
            and signal.confidence >= self.HIGH_CONFIDENCE_THRESHOLD
        ):
            return self._structures["vertical"]

        # Medium confidence directional vol → also use vertical (more conservative)
        if (
            signal.signal_type == SignalType.DIRECTIONAL_VOL
            and signal.confidence >= self.LOW_CONFIDENCE_THRESHOLD
        ):
            return self._structures["vertical"]

        # Skew anomaly → Skew trade (risk reversal with wings)
        if signal.signal_type == SignalType.SKEW_ANOMALY:
            return self._structures["skew"]

        # Elevated IV → Iron condor (sell premium, profit from IV crush)
        # Works best with low confidence (neutral view)
        if signal.signal_type == SignalType.ELEVATED_IV:
            return self._structures["iron_condor"]

        # Default: no trade for unrecognized patterns
        return None

    def get_available_structures(self) -> list[str]:
        """Get list of available structure names.

        Returns:
            List of structure name strings
        """
        return list(self._structures.keys())

    def get_structure_by_name(self, name: str) -> Optional[TradeStructure]:
        """Get a specific structure by name.

        Args:
            name: Structure name (e.g., "calendar", "vertical")

        Returns:
            TradeStructure instance or None if not found
        """
        return self._structures.get(name)
