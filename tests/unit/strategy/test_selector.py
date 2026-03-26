"""Unit tests for StructureSelector."""

import pytest

from src.config.schema import ExecutionConfig
from src.strategy.selector import StructureSelector
from src.strategy.structures import CalendarSpread, IronCondor, SkewTrade, VerticalSpread
from src.strategy.types import Signal, SignalType


@pytest.fixture
def selector() -> StructureSelector:
    """Default structure selector."""
    return StructureSelector()


@pytest.fixture
def selector_with_config() -> StructureSelector:
    """Structure selector with custom config."""
    config = ExecutionConfig()
    return StructureSelector(config)


class TestStructureSelectorInit:
    """Tests for StructureSelector initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        selector = StructureSelector()
        assert selector is not None
        assert len(selector.get_available_structures()) == 4  # calendar, vertical, skew, iron_condor

    def test_init_with_config(self) -> None:
        """Test initialization with config."""
        config = ExecutionConfig()
        selector = StructureSelector(config)
        assert selector._config == config

    def test_available_structures(self, selector: StructureSelector) -> None:
        """Test available structures."""
        structures = selector.get_available_structures()
        assert "calendar" in structures
        assert "vertical" in structures
        assert "skew" in structures
        assert "iron_condor" in structures


class TestStructureSelectorTermAnomaly:
    """Tests for TERM_ANOMALY signal type."""

    def test_term_anomaly_returns_calendar(self, selector: StructureSelector) -> None:
        """Test TERM_ANOMALY signal returns CalendarSpread."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, CalendarSpread)

    def test_term_anomaly_low_confidence(self, selector: StructureSelector) -> None:
        """Test TERM_ANOMALY with low confidence still returns CalendarSpread."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="ATM",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, CalendarSpread)

    def test_term_anomaly_negative_edge(self, selector: StructureSelector) -> None:
        """Test TERM_ANOMALY with negative edge returns CalendarSpread."""
        signal = Signal(
            signal_type=SignalType.TERM_ANOMALY,
            edge=-0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, CalendarSpread)


class TestStructureSelectorDirectionalVol:
    """Tests for DIRECTIONAL_VOL signal type."""

    def test_directional_vol_high_confidence_returns_vertical(
        self, selector: StructureSelector
    ) -> None:
        """Test DIRECTIONAL_VOL with high confidence returns VerticalSpread."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="C25",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, VerticalSpread)

    def test_directional_vol_exactly_high_threshold(
        self, selector: StructureSelector
    ) -> None:
        """Test DIRECTIONAL_VOL at exactly high threshold returns VerticalSpread."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.7,  # Exactly HIGH_CONFIDENCE_THRESHOLD
            tenor_days=30,
            delta_bucket="C25",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, VerticalSpread)

    def test_directional_vol_medium_confidence_returns_vertical(
        self, selector: StructureSelector
    ) -> None:
        """Test DIRECTIONAL_VOL with medium confidence returns VerticalSpread."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.55,  # Between LOW and HIGH thresholds
            tenor_days=30,
            delta_bucket="C25",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, VerticalSpread)

    def test_directional_vol_exactly_low_threshold(
        self, selector: StructureSelector
    ) -> None:
        """Test DIRECTIONAL_VOL at exactly low threshold returns VerticalSpread."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.5,  # Exactly LOW_CONFIDENCE_THRESHOLD
            tenor_days=30,
            delta_bucket="C25",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, VerticalSpread)

    def test_directional_vol_low_confidence_returns_none(
        self, selector: StructureSelector
    ) -> None:
        """Test DIRECTIONAL_VOL with low confidence returns None."""
        signal = Signal(
            signal_type=SignalType.DIRECTIONAL_VOL,
            edge=0.05,
            confidence=0.3,  # Below LOW_CONFIDENCE_THRESHOLD
            tenor_days=30,
            delta_bucket="C25",
        )
        structure = selector.select_structure(signal)
        assert structure is None


class TestStructureSelectorSkewAnomaly:
    """Tests for SKEW_ANOMALY signal type."""

    def test_skew_anomaly_returns_skew_trade(self, selector: StructureSelector) -> None:
        """Test SKEW_ANOMALY returns SkewTrade."""
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="P25",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, SkewTrade)

    def test_skew_anomaly_low_confidence_returns_skew_trade(
        self, selector: StructureSelector
    ) -> None:
        """Test SKEW_ANOMALY with low confidence still returns SkewTrade."""
        signal = Signal(
            signal_type=SignalType.SKEW_ANOMALY,
            edge=0.05,
            confidence=0.3,
            tenor_days=30,
            delta_bucket="P25",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, SkewTrade)


class TestStructureSelectorElevatedIV:
    """Tests for ELEVATED_IV signal type."""

    def test_elevated_iv_returns_iron_condor(
        self, selector: StructureSelector
    ) -> None:
        """Test ELEVATED_IV returns IronCondor."""
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.3,  # Low confidence = neutral view, ideal for iron condor
            tenor_days=30,
            delta_bucket="ATM",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, IronCondor)

    def test_elevated_iv_high_confidence_returns_iron_condor(
        self, selector: StructureSelector
    ) -> None:
        """Test ELEVATED_IV with high confidence still returns IronCondor."""
        signal = Signal(
            signal_type=SignalType.ELEVATED_IV,
            edge=0.05,
            confidence=0.8,
            tenor_days=30,
            delta_bucket="ATM",
        )
        structure = selector.select_structure(signal)
        assert isinstance(structure, IronCondor)


class TestStructureSelectorGetByName:
    """Tests for get_structure_by_name method."""

    def test_get_calendar_by_name(self, selector: StructureSelector) -> None:
        """Test getting calendar structure by name."""
        structure = selector.get_structure_by_name("calendar")
        assert isinstance(structure, CalendarSpread)

    def test_get_vertical_by_name(self, selector: StructureSelector) -> None:
        """Test getting vertical structure by name."""
        structure = selector.get_structure_by_name("vertical")
        assert isinstance(structure, VerticalSpread)

    def test_get_unknown_by_name(self, selector: StructureSelector) -> None:
        """Test getting unknown structure returns None."""
        structure = selector.get_structure_by_name("unknown")
        assert structure is None

    def test_get_iron_condor_by_name(self, selector: StructureSelector) -> None:
        """Test getting iron_condor by name."""
        structure = selector.get_structure_by_name("iron_condor")
        assert isinstance(structure, IronCondor)

    def test_get_skew_by_name(self, selector: StructureSelector) -> None:
        """Test getting skew structure by name."""
        structure = selector.get_structure_by_name("skew")
        assert isinstance(structure, SkewTrade)


class TestStructureSelectorThresholds:
    """Tests for confidence thresholds."""

    def test_high_confidence_threshold_value(self) -> None:
        """Test HIGH_CONFIDENCE_THRESHOLD value."""
        assert StructureSelector.HIGH_CONFIDENCE_THRESHOLD == 0.7

    def test_low_confidence_threshold_value(self) -> None:
        """Test LOW_CONFIDENCE_THRESHOLD value."""
        assert StructureSelector.LOW_CONFIDENCE_THRESHOLD == 0.5
