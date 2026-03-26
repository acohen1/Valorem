"""Unit tests for validation utilities."""

from typing import Protocol, runtime_checkable

import pytest

# Import directly from module to avoid circular imports via __init__
from src.utils.validation import (
    validate_in_range,
    validate_non_negative,
    validate_not_none,
    validate_positive,
    validate_protocol,
)


# Test protocols
@runtime_checkable
class SimpleProtocol(Protocol):
    """Simple protocol with one method."""

    def do_something(self) -> str:
        ...


@runtime_checkable
class MultiMethodProtocol(Protocol):
    """Protocol with multiple methods."""

    def method_a(self) -> int:
        ...

    def method_b(self, x: str) -> str:
        ...


class ValidImplementation:
    """Valid implementation of SimpleProtocol."""

    def do_something(self) -> str:
        return "done"


class InvalidImplementation:
    """Invalid implementation - missing method."""

    def wrong_method(self) -> str:
        return "wrong"


class PartialImplementation:
    """Partial implementation of MultiMethodProtocol."""

    def method_a(self) -> int:
        return 1

    # Missing method_b


class ValidMultiImplementation:
    """Valid implementation of MultiMethodProtocol."""

    def method_a(self) -> int:
        return 1

    def method_b(self, x: str) -> str:
        return x.upper()


class TestValidateProtocol:
    """Tests for validate_protocol function."""

    def test_valid_implementation_passes(self):
        """Test that valid implementation passes validation."""
        obj = ValidImplementation()
        # Should not raise
        validate_protocol(obj, SimpleProtocol, "test_obj")

    def test_invalid_implementation_raises(self):
        """Test that invalid implementation raises TypeError."""
        obj = InvalidImplementation()
        with pytest.raises(TypeError, match="must implement SimpleProtocol"):
            validate_protocol(obj, SimpleProtocol, "test_obj")

    def test_error_includes_object_name(self):
        """Test that error message includes the object name."""
        obj = InvalidImplementation()
        with pytest.raises(TypeError, match="my_provider must implement"):
            validate_protocol(obj, SimpleProtocol, "my_provider")

    def test_error_lists_missing_methods(self):
        """Test that error message lists missing methods."""
        obj = PartialImplementation()
        with pytest.raises(TypeError, match="method_b"):
            validate_protocol(obj, MultiMethodProtocol, "test_obj")

    def test_multi_method_valid_implementation(self):
        """Test multi-method protocol with valid implementation."""
        obj = ValidMultiImplementation()
        # Should not raise
        validate_protocol(obj, MultiMethodProtocol, "test_obj")

    def test_none_object_raises(self):
        """Test that None object raises TypeError."""
        with pytest.raises(TypeError):
            validate_protocol(None, SimpleProtocol, "test_obj")


class TestValidateNotNone:
    """Tests for validate_not_none function."""

    def test_non_none_passes(self):
        """Test that non-None value passes."""
        validate_not_none("value", "test_param")
        validate_not_none(0, "test_param")
        validate_not_none([], "test_param")
        validate_not_none(False, "test_param")

    def test_none_raises(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="test_param cannot be None"):
            validate_not_none(None, "test_param")

    def test_error_includes_param_name(self):
        """Test that error message includes parameter name."""
        with pytest.raises(ValueError, match="my_value cannot be None"):
            validate_not_none(None, "my_value")


class TestValidatePositive:
    """Tests for validate_positive function."""

    def test_positive_int_passes(self):
        """Test that positive int passes."""
        validate_positive(1, "test_param")
        validate_positive(100, "test_param")

    def test_positive_float_passes(self):
        """Test that positive float passes."""
        validate_positive(0.1, "test_param")
        validate_positive(99.9, "test_param")

    def test_zero_raises(self):
        """Test that zero raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(0, "test_param")

    def test_negative_raises(self):
        """Test that negative value raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive(-1, "test_param")

    def test_error_includes_value(self):
        """Test that error message includes actual value."""
        with pytest.raises(ValueError, match="got -5"):
            validate_positive(-5, "test_param")


class TestValidateNonNegative:
    """Tests for validate_non_negative function."""

    def test_positive_passes(self):
        """Test that positive value passes."""
        validate_non_negative(1, "test_param")
        validate_non_negative(0.1, "test_param")

    def test_zero_passes(self):
        """Test that zero passes."""
        validate_non_negative(0, "test_param")
        validate_non_negative(0.0, "test_param")

    def test_negative_raises(self):
        """Test that negative value raises ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_non_negative(-1, "test_param")

    def test_error_includes_value(self):
        """Test that error message includes actual value."""
        with pytest.raises(ValueError, match="got -3.5"):
            validate_non_negative(-3.5, "test_param")


class TestValidateInRange:
    """Tests for validate_in_range function."""

    def test_value_in_range_passes(self):
        """Test that value within range passes."""
        validate_in_range(5, "test_param", min_val=0, max_val=10)
        validate_in_range(0, "test_param", min_val=0, max_val=10)
        validate_in_range(10, "test_param", min_val=0, max_val=10)

    def test_below_min_raises(self):
        """Test that value below min raises ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_in_range(-1, "test_param", min_val=0, max_val=10)

    def test_above_max_raises(self):
        """Test that value above max raises ValueError."""
        with pytest.raises(ValueError, match="must be <= 10"):
            validate_in_range(11, "test_param", min_val=0, max_val=10)

    def test_min_only(self):
        """Test validation with only min bound."""
        validate_in_range(100, "test_param", min_val=0)
        with pytest.raises(ValueError):
            validate_in_range(-1, "test_param", min_val=0)

    def test_max_only(self):
        """Test validation with only max bound."""
        validate_in_range(-100, "test_param", max_val=10)
        with pytest.raises(ValueError):
            validate_in_range(11, "test_param", max_val=10)

    def test_no_bounds_passes(self):
        """Test that any value passes with no bounds."""
        validate_in_range(-1000, "test_param")
        validate_in_range(1000, "test_param")

    def test_float_bounds(self):
        """Test with float bounds."""
        validate_in_range(0.5, "test_param", min_val=0.0, max_val=1.0)
        with pytest.raises(ValueError):
            validate_in_range(1.5, "test_param", min_val=0.0, max_val=1.0)
