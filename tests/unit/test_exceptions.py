"""Unit tests for domain exception hierarchy."""

import pytest

from src.exceptions import (
    ConfigError,
    DataError,
    DataReadError,
    DataValidationError,
    DataWriteError,
    ExecutionError,
    ProviderError,
    ValoremError,
    SignalError,
    StructureError,
)


class TestExceptionHierarchy:
    """Test that exception hierarchy is correct."""

    def test_valorem_error_is_base_exception(self):
        """Test that ValoremError inherits from Exception."""
        assert issubclass(ValoremError, Exception)

    def test_data_error_inherits_from_valorem_error(self):
        """Test that DataError inherits from ValoremError."""
        assert issubclass(DataError, ValoremError)
        assert issubclass(DataError, Exception)

    def test_data_write_error_inherits_from_data_error(self):
        """Test that DataWriteError inherits from DataError."""
        assert issubclass(DataWriteError, DataError)
        assert issubclass(DataWriteError, ValoremError)

    def test_data_read_error_inherits_from_data_error(self):
        """Test that DataReadError inherits from DataError."""
        assert issubclass(DataReadError, DataError)
        assert issubclass(DataReadError, ValoremError)

    def test_data_validation_error_inherits_from_data_error(self):
        """Test that DataValidationError inherits from DataError."""
        assert issubclass(DataValidationError, DataError)
        assert issubclass(DataValidationError, ValoremError)

    def test_config_error_inherits_from_valorem_error(self):
        """Test that ConfigError inherits from ValoremError."""
        assert issubclass(ConfigError, ValoremError)

    def test_provider_error_inherits_from_valorem_error(self):
        """Test that ProviderError inherits from ValoremError."""
        assert issubclass(ProviderError, ValoremError)

    def test_signal_error_inherits_from_valorem_error(self):
        """Test that SignalError inherits from ValoremError."""
        assert issubclass(SignalError, ValoremError)

    def test_structure_error_inherits_from_valorem_error(self):
        """Test that StructureError inherits from ValoremError."""
        assert issubclass(StructureError, ValoremError)

    def test_execution_error_inherits_from_valorem_error(self):
        """Test that ExecutionError inherits from ValoremError."""
        assert issubclass(ExecutionError, ValoremError)


class TestExceptionCatching:
    """Test that exceptions can be caught at appropriate levels."""

    def test_catch_all_valorem_errors(self):
        """Test that all domain exceptions can be caught as ValoremError."""
        exceptions = [
            DataWriteError("write failed"),
            DataReadError("read failed"),
            DataValidationError("validation failed"),
            ConfigError("config error"),
            ProviderError("provider error"),
            SignalError("signal error"),
            StructureError("structure error"),
            ExecutionError("execution error"),
        ]

        for exc in exceptions:
            with pytest.raises(ValoremError):
                raise exc

    def test_catch_data_errors_specifically(self):
        """Test that data errors can be caught separately from other errors."""
        data_exceptions = [
            DataWriteError("write failed"),
            DataReadError("read failed"),
            DataValidationError("validation failed"),
        ]

        for exc in data_exceptions:
            with pytest.raises(DataError):
                raise exc

        # Config error should not be caught as DataError
        with pytest.raises(ConfigError):
            raise ConfigError("config error")

    def test_catch_write_vs_read_errors(self):
        """Test that write and read errors can be distinguished."""
        with pytest.raises(DataWriteError):
            raise DataWriteError("write failed")

        with pytest.raises(DataReadError):
            raise DataReadError("read failed")

        # Write error should not be caught as read error
        try:
            raise DataWriteError("write failed")
        except DataReadError:
            pytest.fail("DataWriteError should not be caught as DataReadError")
        except DataWriteError:
            pass  # Expected


class TestExceptionMessages:
    """Test that exception messages are preserved."""

    def test_exception_message_preserved(self):
        """Test that exception message is accessible via str()."""
        msg = "Database connection failed"
        exc = DataWriteError(msg)
        assert str(exc) == msg

    def test_exception_chaining(self):
        """Test that exceptions can be properly chained."""
        original = ValueError("Original error")

        try:
            try:
                raise original
            except ValueError as e:
                raise DataWriteError("Wrapped error") from e
        except DataWriteError as chained:
            assert chained.__cause__ is original
            assert str(chained) == "Wrapped error"
            assert str(chained.__cause__) == "Original error"

    def test_exception_with_formatted_message(self):
        """Test that formatted messages work correctly."""
        table_name = "raw_underlying_bars"
        exc = DataWriteError(f"Failed to write to {table_name}")
        assert "raw_underlying_bars" in str(exc)
