"""Protocol validation utilities.

This module provides runtime validation for protocol implementations,
catching integration bugs early and providing clear error messages.

Usage:
    from src.utils.validation import validate_protocol

    # Validate at construction time
    validate_protocol(provider, MarketDataProvider, "market_data_provider")
"""

from typing import Any, Type, get_type_hints


def validate_protocol(
    obj: Any,
    protocol: Type,
    name: str = "object",
) -> None:
    """Validate that an object implements a protocol.

    Uses isinstance() check for runtime_checkable protocols to verify
    that the object has all required methods with correct signatures.

    Args:
        obj: Object to validate
        protocol: Protocol class to validate against (must be @runtime_checkable)
        name: Name of the object for error messages

    Raises:
        TypeError: If obj does not implement the protocol

    Example:
        >>> from src.data.providers.protocol import MarketDataProvider
        >>> validate_protocol(my_provider, MarketDataProvider, "market_data_provider")

    Note:
        This function requires the protocol to be decorated with @runtime_checkable.
        It performs structural subtyping checks - the object must have all methods
        defined in the protocol with compatible signatures.
    """
    if not isinstance(obj, protocol):
        # Build helpful error message listing missing methods
        missing = _get_missing_methods(obj, protocol)
        if missing:
            methods_str = ", ".join(missing)
            raise TypeError(
                f"{name} must implement {protocol.__name__}. "
                f"Missing methods: {methods_str}"
            )
        else:
            raise TypeError(
                f"{name} must implement {protocol.__name__}"
            )


def _get_missing_methods(obj: Any, protocol: Type) -> list[str]:
    """Get list of protocol methods missing from object.

    Args:
        obj: Object to check
        protocol: Protocol to check against

    Returns:
        List of method names that are missing or not callable
    """
    missing = []

    # Get protocol's method names (excluding dunder methods)
    for attr_name in dir(protocol):
        if attr_name.startswith("_"):
            continue

        protocol_attr = getattr(protocol, attr_name, None)
        if not callable(protocol_attr):
            continue

        # Check if object has this method
        obj_attr = getattr(obj, attr_name, None)
        if obj_attr is None or not callable(obj_attr):
            missing.append(attr_name)

    return missing


def validate_not_none(
    value: Any,
    name: str,
) -> None:
    """Validate that a value is not None.

    Args:
        value: Value to check
        name: Name of the value for error messages

    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def validate_positive(
    value: float | int,
    name: str,
) -> None:
    """Validate that a numeric value is positive.

    Args:
        value: Value to check
        name: Name of the value for error messages

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(
    value: float | int,
    name: str,
) -> None:
    """Validate that a numeric value is non-negative.

    Args:
        value: Value to check
        name: Name of the value for error messages

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_in_range(
    value: float | int,
    name: str,
    min_val: float | int | None = None,
    max_val: float | int | None = None,
) -> None:
    """Validate that a value is within a range.

    Args:
        value: Value to check
        name: Name of the value for error messages
        min_val: Minimum allowed value (inclusive), or None for no minimum
        max_val: Maximum allowed value (inclusive), or None for no maximum

    Raises:
        ValueError: If value is outside the allowed range
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {value}")
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} must be <= {max_val}, got {value}")
