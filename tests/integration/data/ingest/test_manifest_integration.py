"""Integration tests for manifest generation.

These tests verify the end-to-end manifest generation workflow including:
- File I/O operations
- Deterministic output across runs
- Integration with spot reference lookup
- Full pipeline from raw symbols to manifest file
"""

import json
import tempfile
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from src.config.schema import ConfigSchema, DatasetConfig, DatasetSplitsConfig
from src.data.ingest.manifest import (
    Manifest,
    ManifestGenerator,
    get_spot_reference,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def config():
    """Create test configuration."""
    return ConfigSchema(
        dataset=DatasetConfig(
            splits=DatasetSplitsConfig(
                train_start=date(2020, 1, 1),
                train_end=date(2022, 12, 31),
                val_start=date(2023, 1, 1),
                val_end=date(2023, 6, 30),
                test_start=date(2023, 7, 1),
                test_end=date(2023, 12, 31),
            ),
            min_dte=7,
            max_dte=90,
        ),
        backtest={"start_date": date(2023, 7, 1), "end_date": date(2023, 12, 31)},
    )


@pytest.fixture
def temp_manifest_dir():
    """Create temporary directory for manifest files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def realistic_symbols():
    """Generate a realistic set of option symbols.

    Creates symbols for multiple expiries with various strikes
    similar to what would be available for SPY options.
    """
    symbols = []
    spot = 450.0

    # Generate strikes from 80% to 120% of spot
    strikes = [int(spot * ratio) for ratio in [0.80, 0.85, 0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10, 1.15, 1.20]]

    # Multiple expiry dates (weekly and monthly)
    expiries = [
        ("231229", 8),   # Dec 29, 2023 - DTE 8
        ("240105", 15),  # Jan 5, 2024 - DTE 15
        ("240119", 29),  # Jan 19, 2024 - DTE 29 (monthly)
        ("240126", 36),  # Jan 26, 2024 - DTE 36
        ("240216", 57),  # Feb 16, 2024 - DTE 57 (monthly)
        ("240315", 85),  # Mar 15, 2024 - DTE 85 (monthly)
        ("240419", 120), # Apr 19, 2024 - DTE 120 (outside max_dte=90)
        ("240621", 183), # Jun 21, 2024 - DTE 183 (outside max_dte=90)
    ]

    for expiry_str, _ in expiries:
        for strike in strikes:
            # Format strike as 8-digit integer (strike * 1000)
            strike_str = f"{strike * 1000:08d}"
            symbols.append(f"SPY{expiry_str}C{strike_str}")
            symbols.append(f"SPY{expiry_str}P{strike_str}")

    return symbols


@pytest.fixture
def underlying_data():
    """Create underlying price data for spot reference."""
    # Simulate minute bars for a trading day
    timestamps = pd.date_range(
        start="2023-12-21 09:30:00",
        end="2023-12-21 16:00:00",
        freq="1min",
        tz="UTC",
    )

    # Simulate price movement during the day
    import numpy as np
    np.random.seed(42)
    prices = 448.0 + np.cumsum(np.random.randn(len(timestamps)) * 0.1)

    return pd.DataFrame({
        "ts_utc": timestamps,
        "open": prices,
        "high": prices + 0.05,
        "low": prices - 0.05,
        "close": prices,
        "volume": np.random.randint(1000, 10000, len(timestamps)),
    })


# ============================================================================
# Determinism Tests
# ============================================================================


class TestManifestDeterminism:
    """Test that manifest generation is deterministic."""

    def test_multiple_generations_identical(self, config, realistic_symbols):
        """Verify that multiple generations produce identical results."""
        generator = ManifestGenerator(config=config)
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifests = []
        for _ in range(5):
            manifest = generator.generate_manifest(
                available_symbols=realistic_symbols,
                spot_reference=spot,
                as_of_date=as_of,
                dte_min=7,
                dte_max=90,
                moneyness_min=0.85,
                moneyness_max=1.15,
                options_per_expiry_side=20,
            )
            manifests.append(manifest)

        # All manifests should be identical
        first = manifests[0]
        for i, m in enumerate(manifests[1:], 2):
            assert m.symbols == first.symbols, f"Manifest {i} differs in symbols"
            assert m.metadata.symbols_count == first.metadata.symbols_count
            assert m.metadata.config_hash == first.metadata.config_hash

    def test_config_hash_stability(self, config, realistic_symbols):
        """Verify config hash is stable across runs."""
        generator = ManifestGenerator(config=config)
        as_of = date(2023, 12, 21)
        spot = 450.0

        hashes = []
        for _ in range(10):
            manifest = generator.generate_manifest(
                available_symbols=realistic_symbols,
                spot_reference=spot,
                as_of_date=as_of,
            )
            hashes.append(manifest.metadata.config_hash)

        # All hashes should be identical
        assert len(set(hashes)) == 1, "Config hash varies across runs"


# ============================================================================
# File I/O Tests
# ============================================================================


class TestManifestFileIO:
    """Test manifest file operations."""

    def test_write_read_roundtrip(self, config, realistic_symbols, temp_manifest_dir):
        """Test complete write and read roundtrip."""
        generator = ManifestGenerator(config=config)
        as_of = date(2023, 12, 21)
        spot = 450.0

        # Generate manifest
        original = generator.generate_manifest(
            available_symbols=realistic_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=90,
            moneyness_min=0.85,
            moneyness_max=1.15,
            options_per_expiry_side=20,
        )

        # Write to file
        path = temp_manifest_dir / "test_manifest.json"
        generator.write_manifest(original, path)

        # Verify file exists and is valid JSON
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "symbols" in data
        assert "metadata" in data

        # Load and compare
        loaded = generator.load_manifest(path)

        assert loaded.symbols == original.symbols
        assert loaded.metadata.spot_reference == original.metadata.spot_reference
        assert loaded.metadata.dte_min == original.metadata.dte_min
        assert loaded.metadata.dte_max == original.metadata.dte_max
        assert loaded.metadata.symbols_count == original.metadata.symbols_count

    def test_manifest_file_structure(self, config, realistic_symbols, temp_manifest_dir):
        """Verify manifest file has correct structure."""
        generator = ManifestGenerator(config=config)

        manifest = generator.generate_manifest(
            available_symbols=realistic_symbols,
            spot_reference=450.0,
            as_of_date=date(2023, 12, 21),
        )

        path = temp_manifest_dir / "structure_test.json"
        generator.write_manifest(manifest, path)

        with open(path) as f:
            data = json.load(f)

        # Verify required fields
        assert isinstance(data["symbols"], list)
        assert isinstance(data["metadata"], dict)
        assert isinstance(data["symbols_by_expiry"], dict)

        # Verify metadata fields
        metadata = data["metadata"]
        required_fields = [
            "as_of_ts_utc",
            "spot_reference",
            "dte_min",
            "dte_max",
            "moneyness_min",
            "moneyness_max",
            "options_per_expiry_side",
            "underlying",
            "generated_at",
            "config_hash",
            "symbols_count",
            "expiries_count",
        ]
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"

    def test_creates_parent_directories(self, config, realistic_symbols, temp_manifest_dir):
        """Test that write_manifest creates parent directories."""
        generator = ManifestGenerator(config=config)

        manifest = generator.generate_manifest(
            available_symbols=realistic_symbols,
            spot_reference=450.0,
            as_of_date=date(2023, 12, 21),
        )

        # Use nested path that doesn't exist
        path = temp_manifest_dir / "nested" / "dirs" / "manifest.json"
        assert not path.parent.exists()

        generator.write_manifest(manifest, path)

        assert path.exists()
        assert path.parent.exists()


# ============================================================================
# Spot Reference Integration Tests
# ============================================================================


class TestSpotReferenceIntegration:
    """Test spot reference lookup with manifest generation."""

    def test_spot_from_underlying_data(self, config, realistic_symbols, underlying_data):
        """Test getting spot reference from underlying data."""
        # Get spot at market close
        as_of_ts = datetime(2023, 12, 21, 16, 0, 0)
        spot = get_spot_reference(underlying_data, as_of_ts)

        assert spot > 0
        assert isinstance(spot, float)

        # Generate manifest with this spot
        generator = ManifestGenerator(config=config)
        manifest = generator.generate_manifest(
            available_symbols=realistic_symbols,
            spot_reference=spot,
            as_of_date=as_of_ts.date(),
        )

        assert manifest.metadata.spot_reference == spot
        assert len(manifest.symbols) > 0

    def test_spot_at_different_times(self, config, underlying_data):
        """Test spot reference at different intraday times."""
        spots = []
        times = [
            datetime(2023, 12, 21, 10, 0, 0),
            datetime(2023, 12, 21, 12, 0, 0),
            datetime(2023, 12, 21, 14, 0, 0),
            datetime(2023, 12, 21, 16, 0, 0),
        ]

        for as_of_ts in times:
            spot = get_spot_reference(underlying_data, as_of_ts)
            spots.append(spot)

        # Spots at different times should generally differ
        # (unless by coincidence the prices are the same)
        assert len(spots) == 4
        # At minimum, verify all are valid positive numbers
        for spot in spots:
            assert spot > 0


# ============================================================================
# Full Pipeline Tests
# ============================================================================


class TestFullPipeline:
    """Test complete manifest generation pipeline."""

    def test_end_to_end_workflow(self, config, realistic_symbols, underlying_data, temp_manifest_dir):
        """Test complete workflow from symbols to persisted manifest."""
        generator = ManifestGenerator(config=config)

        # Step 1: Get spot reference
        as_of_ts = datetime(2023, 12, 21, 16, 0, 0)
        spot = get_spot_reference(underlying_data, as_of_ts)

        # Step 2: Generate manifest
        manifest = generator.generate_manifest(
            available_symbols=realistic_symbols,
            spot_reference=spot,
            as_of_date=as_of_ts.date(),
            dte_min=7,
            dte_max=90,
            moneyness_min=0.85,
            moneyness_max=1.15,
            options_per_expiry_side=10,
        )

        # Verify manifest properties
        assert len(manifest.symbols) > 0
        assert manifest.metadata.symbols_count == len(manifest.symbols)
        assert manifest.metadata.spot_reference == spot

        # Step 3: Persist manifest
        manifest_path = temp_manifest_dir / f"manifest_SPY_{as_of_ts.date().isoformat()}.json"
        generator.write_manifest(manifest, manifest_path)

        # Step 4: Reload and verify
        loaded = generator.load_manifest(manifest_path)
        assert loaded.symbols == manifest.symbols

        # Step 5: Verify all symbols are within expected DTE range
        for sym in loaded.symbols:
            info = generator.parse_option_symbol(sym)
            dte = (info.expiry_date - as_of_ts.date()).days
            assert 7 <= dte <= 90, f"Symbol {sym} has DTE {dte} outside range"

    def test_multiple_dates_workflow(self, config, realistic_symbols, temp_manifest_dir):
        """Test generating manifests for multiple dates."""
        generator = ManifestGenerator(config=config)
        spot = 450.0

        dates = [
            date(2023, 12, 20),
            date(2023, 12, 21),
            date(2023, 12, 22),
        ]

        manifests = {}
        for as_of in dates:
            manifest = generator.generate_manifest(
                available_symbols=realistic_symbols,
                spot_reference=spot,
                as_of_date=as_of,
            )

            path = temp_manifest_dir / f"manifest_SPY_{as_of.isoformat()}.json"
            generator.write_manifest(manifest, path)
            manifests[as_of] = manifest

        # Verify all manifests were created
        assert len(manifests) == 3

        # Manifests for different dates may have different symbols
        # due to DTE filtering
        for as_of, manifest in manifests.items():
            # Verify DTE constraints hold
            for sym in manifest.symbols:
                info = generator.parse_option_symbol(sym)
                dte = (info.expiry_date - as_of).days
                assert dte >= 7  # min_dte from config
                assert dte <= 90  # max_dte from config


# ============================================================================
# Performance Tests
# ============================================================================


class TestManifestPerformance:
    """Test manifest generation performance."""

    def test_large_symbol_set(self, config):
        """Test performance with large symbol set."""
        import time

        # Generate 10,000 symbols (realistic for SPY options)
        symbols = []
        spot = 450.0

        for strike_offset in range(-100, 101, 2):  # 101 strikes
            strike = int((spot + strike_offset) * 1000)
            for expiry_str in [f"24{month:02d}{day:02d}" for month in range(1, 13) for day in [5, 12, 19, 26]]:
                symbols.append(f"SPY{expiry_str}C{strike:08d}")
                symbols.append(f"SPY{expiry_str}P{strike:08d}")

        generator = ManifestGenerator(config=config)

        start = time.time()
        manifest = generator.generate_manifest(
            available_symbols=symbols[:5000],  # Use subset
            spot_reference=spot,
            as_of_date=date(2023, 12, 21),
            options_per_expiry_side=20,
        )
        elapsed = time.time() - start

        # Should complete in reasonable time (< 2 seconds)
        assert elapsed < 2.0, f"Manifest generation took {elapsed:.2f}s"
        assert len(manifest.symbols) > 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestIntegrationEdgeCases:
    """Test edge cases in integration scenarios."""

    def test_manifest_with_minimal_symbols(self, config, temp_manifest_dir):
        """Test manifest with very few symbols."""
        generator = ManifestGenerator(config=config)

        symbols = [
            "SPY240119C00450000",
            "SPY240119P00450000",
        ]

        manifest = generator.generate_manifest(
            available_symbols=symbols,
            spot_reference=450.0,
            as_of_date=date(2023, 12, 21),
        )

        path = temp_manifest_dir / "minimal.json"
        generator.write_manifest(manifest, path)

        loaded = generator.load_manifest(path)
        assert len(loaded.symbols) == 2

    def test_varying_moneyness_ranges(self, config, realistic_symbols):
        """Test different moneyness ranges produce expected results."""
        generator = ManifestGenerator(config=config)
        spot = 450.0
        as_of = date(2023, 12, 21)

        # Narrow range (near ATM)
        narrow = generator.generate_manifest(
            available_symbols=realistic_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            moneyness_min=0.98,
            moneyness_max=1.02,
        )

        # Wide range
        wide = generator.generate_manifest(
            available_symbols=realistic_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            moneyness_min=0.80,
            moneyness_max=1.20,
        )

        # Wide range should have more or equal symbols
        assert len(wide.symbols) >= len(narrow.symbols)
