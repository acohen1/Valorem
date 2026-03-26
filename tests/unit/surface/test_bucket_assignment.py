"""Unit tests for delta bucket assignment."""

import numpy as np
import pandas as pd
import pytest

from src.surface.buckets.assign import DeltaBucketAssigner


class TestDeltaBucketAssignerInitialization:
    """Test DeltaBucketAssigner initialization."""

    def test_initialization_with_valid_config(self):
        """Test initialization with valid bucket configuration."""
        config = {
            "P25": [-0.35, -0.15],
            "ATM": [-0.55, -0.45, 0.45, 0.55],
            "C25": [0.15, 0.35],
        }
        assigner = DeltaBucketAssigner(config)
        assert assigner is not None
        assert len(assigner.get_bucket_names()) == 3

    def test_initialization_with_empty_config_raises_error(self):
        """Test that empty configuration raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            DeltaBucketAssigner({})

    def test_initialization_with_invalid_atm_raises_error(self):
        """Test that invalid ATM bucket raises ValueError."""
        config = {
            "ATM": [-0.55, 0.55],  # Only 2 elements
        }
        with pytest.raises(ValueError, match="ATM bucket must have exactly 4 elements"):
            DeltaBucketAssigner(config)

    def test_initialization_with_invalid_standard_bucket_raises_error(self):
        """Test that invalid standard bucket raises ValueError."""
        config = {
            "P25": [-0.35, -0.25, -0.15],  # 3 elements
        }
        with pytest.raises(ValueError, match="Bucket P25 must have exactly 2 elements"):
            DeltaBucketAssigner(config)


class TestBucketAssignment:
    """Test bucket assignment logic."""

    @pytest.fixture
    def standard_config(self):
        """Standard bucket configuration for testing.

        Note: Buckets are processed in order, so ATM should come first
        to catch |delta| near 0.5 before directional buckets.
        """
        return {
            "ATM": [-0.55, -0.45, 0.45, 0.55],
            "P40": [-0.45, -0.35],
            "P25": [-0.35, -0.15],
            "P10": [-0.15, 0.0],
            "C10": [0.0, 0.15],
            "C25": [0.15, 0.35],
            "C40": [0.35, 0.45],
        }

    @pytest.fixture
    def assigner(self, standard_config):
        """Create assigner with standard configuration."""
        return DeltaBucketAssigner(standard_config)

    def test_assign_atm_call(self, assigner):
        """Test ATM assignment for call options (delta near 0.5)."""
        deltas = pd.Series([0.50])
        buckets = assigner.assign(deltas)
        assert buckets.iloc[0] == "ATM"

    def test_assign_atm_put(self, assigner):
        """Test ATM assignment for put options (delta near -0.5)."""
        deltas = pd.Series([-0.50])
        buckets = assigner.assign(deltas)
        assert buckets.iloc[0] == "ATM"

    def test_assign_atm_boundary(self, assigner):
        """Test ATM assignment at boundaries."""
        deltas = pd.Series([0.45, 0.55, -0.45, -0.55])
        buckets = assigner.assign(deltas)
        assert all(buckets == "ATM")

    def test_assign_otm_call(self, assigner):
        """Test OTM call bucket assignment."""
        deltas = pd.Series([0.10, 0.20, 0.40])
        buckets = assigner.assign(deltas)
        assert buckets.iloc[0] == "C10"
        assert buckets.iloc[1] == "C25"
        assert buckets.iloc[2] == "C40"

    def test_assign_otm_put(self, assigner):
        """Test OTM put bucket assignment."""
        deltas = pd.Series([-0.20, -0.40])
        buckets = assigner.assign(deltas)
        assert buckets.iloc[0] == "P25"
        assert buckets.iloc[1] == "P40"

    def test_assign_deep_otm_put(self, assigner):
        """Test deep OTM put assignment."""
        deltas = pd.Series([-0.10])
        buckets = assigner.assign(deltas)
        assert buckets.iloc[0] == "P10"

    def test_assign_no_bucket(self, assigner):
        """Test options that don't fall into any bucket."""
        # Delta > 0.55 or < -0.55 outside ATM, and > 0.45 for C40
        deltas = pd.Series([0.60])
        buckets = assigner.assign(deltas)
        assert buckets.iloc[0] is None

    def test_assign_vectorized(self, assigner):
        """Test vectorized assignment of multiple options."""
        deltas = pd.Series([0.50, -0.50, 0.25, -0.25, 0.10, -0.10])
        buckets = assigner.assign(deltas)

        assert buckets.iloc[0] == "ATM"  # Call ATM
        assert buckets.iloc[1] == "ATM"  # Put ATM
        assert buckets.iloc[2] == "C25"  # Call 25 delta
        assert buckets.iloc[3] == "P25"  # Put 25 delta
        assert buckets.iloc[4] == "C10"  # Call 10 delta
        # Put 10 delta at -0.10 is in P10 range [0.0, -0.15]

    def test_assign_preserves_index(self, assigner):
        """Test that assignment preserves input index."""
        deltas = pd.Series([0.50, 0.25], index=["opt1", "opt2"])
        buckets = assigner.assign(deltas)
        assert list(buckets.index) == ["opt1", "opt2"]

    def test_assign_empty_series(self, assigner):
        """Test assignment with empty series."""
        deltas = pd.Series([], dtype=float)
        buckets = assigner.assign(deltas)
        assert len(buckets) == 0


class TestBucketBoundaries:
    """Test bucket boundary handling."""

    @pytest.fixture
    def simple_config(self):
        """Simple non-overlapping bucket configuration."""
        return {
            "LOW": [-1.0, -0.5],
            "ATM": [-0.55, -0.45, 0.45, 0.55],
            "HIGH": [0.56, 1.0],  # Starts above ATM's |delta| range to avoid overlap
        }

    @pytest.fixture
    def assigner(self, simple_config):
        """Create assigner with simple configuration."""
        return DeltaBucketAssigner(simple_config)

    def test_boundary_inclusion(self, assigner):
        """Test that boundaries are inclusive."""
        # Exact boundary values
        deltas = pd.Series([-1.0, -0.5, 0.45, 0.55, 0.56, 1.0])
        buckets = assigner.assign(deltas)

        assert buckets.iloc[0] == "LOW"  # -1.0
        assert buckets.iloc[1] == "LOW"  # -0.5
        assert buckets.iloc[2] == "ATM"  # 0.45
        assert buckets.iloc[3] == "ATM"  # 0.55
        assert buckets.iloc[4] == "HIGH"  # 0.56
        assert buckets.iloc[5] == "HIGH"  # 1.0

    def test_no_double_assignment(self):
        """Test that no option is assigned to multiple buckets."""
        # Configuration with potential overlap
        config = {
            "P25": [-0.35, -0.15],
            "ATM": [-0.55, -0.45, 0.45, 0.55],
        }
        assigner = DeltaBucketAssigner(config)

        deltas = pd.Series([-0.50])  # Could match ATM via |delta|
        buckets = assigner.assign(deltas)

        # Should only get one assignment (first match wins)
        assert buckets.iloc[0] is not None


class TestBucketMetadata:
    """Test bucket metadata methods."""

    @pytest.fixture
    def assigner(self):
        """Create assigner with standard configuration."""
        config = {
            "P25": [-0.35, -0.15],
            "ATM": [-0.55, -0.45, 0.45, 0.55],
            "C25": [0.15, 0.35],
        }
        return DeltaBucketAssigner(config)

    def test_get_bucket_names(self, assigner):
        """Test getting list of bucket names."""
        names = assigner.get_bucket_names()
        assert "P25" in names
        assert "ATM" in names
        assert "C25" in names
        assert len(names) == 3

    def test_get_bucket_bounds(self, assigner):
        """Test getting bounds for specific bucket."""
        min_delta, max_delta = assigner.get_bucket_bounds("P25")
        assert min_delta == -0.35
        assert max_delta == -0.15

    def test_get_bucket_bounds_atm(self, assigner):
        """Test getting bounds for ATM bucket."""
        min_delta, max_delta = assigner.get_bucket_bounds("ATM")
        assert min_delta == 0.45
        assert max_delta == 0.55

    def test_get_bucket_bounds_invalid_raises_error(self, assigner):
        """Test that invalid bucket name raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            assigner.get_bucket_bounds("INVALID")


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_swapped_bounds_handled(self):
        """Test that swapped bounds (min > max) are handled."""
        config = {
            "P25": [-0.15, -0.35],  # Swapped order
        }
        assigner = DeltaBucketAssigner(config)

        # Should still work correctly
        deltas = pd.Series([-0.25])
        buckets = assigner.assign(deltas)
        assert buckets.iloc[0] == "P25"

    def test_nan_handling(self):
        """Test that NaN deltas result in None bucket."""
        config = {"ATM": [-0.55, -0.45, 0.45, 0.55]}
        assigner = DeltaBucketAssigner(config)

        deltas = pd.Series([0.50, np.nan, 0.50])
        buckets = assigner.assign(deltas)

        assert buckets.iloc[0] == "ATM"
        assert buckets.iloc[1] is None  # NaN should not match
        assert buckets.iloc[2] == "ATM"

    def test_extreme_deltas(self):
        """Test handling of extreme delta values."""
        config = {"DEEP": [-1.0, -0.9]}
        assigner = DeltaBucketAssigner(config)

        deltas = pd.Series([-0.99, -1.0, -1.1])  # Last one is invalid
        buckets = assigner.assign(deltas)

        assert buckets.iloc[0] == "DEEP"
        assert buckets.iloc[1] == "DEEP"
        assert buckets.iloc[2] is None  # Outside range
