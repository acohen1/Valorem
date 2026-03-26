"""Unit tests for manifest generation."""

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
    ManifestMetadata,
    OptionSymbolInfo,
    compute_dte,
    compute_moneyness,
    get_spot_reference,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_config():
    """Create sample config for testing."""
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
            max_dte=120,
        ),
        backtest={"start_date": date(2023, 7, 1), "end_date": date(2023, 12, 31)},
    )


@pytest.fixture
def generator(sample_config):
    """Create ManifestGenerator instance."""
    return ManifestGenerator(config=sample_config)


@pytest.fixture
def sample_symbols():
    """Create sample option symbols for SPY."""
    # SPY trading around $450
    # Format: {underlying}{YYMMDD}{C/P}{strike*1000, 8 digits}
    symbols = []

    # Jan 20, 2024 expiry (DTE ~30 from Dec 21, 2023)
    # Calls
    symbols.extend([
        "SPY240120C00400000",  # $400 call
        "SPY240120C00420000",  # $420 call
        "SPY240120C00440000",  # $440 call
        "SPY240120C00450000",  # $450 call (ATM)
        "SPY240120C00460000",  # $460 call
        "SPY240120C00480000",  # $480 call
        "SPY240120C00500000",  # $500 call
    ])
    # Puts
    symbols.extend([
        "SPY240120P00400000",  # $400 put
        "SPY240120P00420000",  # $420 put
        "SPY240120P00440000",  # $440 put
        "SPY240120P00450000",  # $450 put (ATM)
        "SPY240120P00460000",  # $460 put
        "SPY240120P00480000",  # $480 put
        "SPY240120P00500000",  # $500 put
    ])

    # Feb 16, 2024 expiry (DTE ~57 from Dec 21, 2023)
    symbols.extend([
        "SPY240216C00430000",  # $430 call
        "SPY240216C00450000",  # $450 call
        "SPY240216C00470000",  # $470 call
        "SPY240216P00430000",  # $430 put
        "SPY240216P00450000",  # $450 put
        "SPY240216P00470000",  # $470 put
    ])

    # Mar 15, 2024 expiry (DTE ~85 from Dec 21, 2023)
    symbols.extend([
        "SPY240315C00440000",  # $440 call
        "SPY240315C00450000",  # $450 call
        "SPY240315C00460000",  # $460 call
        "SPY240315P00440000",  # $440 put
        "SPY240315P00450000",  # $450 put
        "SPY240315P00460000",  # $460 put
    ])

    # Jun 21, 2024 expiry (DTE ~183 from Dec 21, 2023) - outside default max_dte
    symbols.extend([
        "SPY240621C00450000",  # $450 call
        "SPY240621P00450000",  # $450 put
    ])

    # Dec 28, 2023 expiry (DTE ~7 from Dec 21, 2023) - near min_dte
    symbols.extend([
        "SPY231228C00448000",  # $448 call
        "SPY231228C00450000",  # $450 call
        "SPY231228C00452000",  # $452 call
        "SPY231228P00448000",  # $448 put
        "SPY231228P00450000",  # $450 put
        "SPY231228P00452000",  # $452 put
    ])

    return symbols


@pytest.fixture
def underlying_df():
    """Create sample underlying price data."""
    return pd.DataFrame({
        "ts_utc": pd.to_datetime([
            "2023-12-20T16:00:00",
            "2023-12-21T10:00:00",
            "2023-12-21T12:00:00",
            "2023-12-21T16:00:00",
        ], utc=True),
        "close": [448.50, 449.00, 449.50, 450.00],
    })


# ============================================================================
# OptionSymbolInfo Tests
# ============================================================================


class TestOptionSymbolInfo:
    """Tests for OptionSymbolInfo dataclass."""

    def test_is_call(self):
        """Test is_call property."""
        call = OptionSymbolInfo(
            raw_symbol="SPY240120C00450000",
            underlying="SPY",
            expiry_date=date(2024, 1, 20),
            right="C",
            strike=450.0,
        )
        assert call.is_call is True
        assert call.is_put is False

    def test_is_put(self):
        """Test is_put property."""
        put = OptionSymbolInfo(
            raw_symbol="SPY240120P00450000",
            underlying="SPY",
            expiry_date=date(2024, 1, 20),
            right="P",
            strike=450.0,
        )
        assert put.is_call is False
        assert put.is_put is True


# ============================================================================
# Symbol Parsing Tests
# ============================================================================


class TestSymbolParsing:
    """Tests for option symbol parsing."""

    def test_parse_valid_call_symbol(self, generator):
        """Test parsing a valid call option symbol."""
        info = generator.parse_option_symbol("SPY240120C00450000")

        assert info is not None
        assert info.underlying == "SPY"
        assert info.expiry_date == date(2024, 1, 20)
        assert info.right == "C"
        assert info.strike == 450.0
        assert info.is_call is True

    def test_parse_valid_put_symbol(self, generator):
        """Test parsing a valid put option symbol."""
        info = generator.parse_option_symbol("SPY240120P00400000")

        assert info is not None
        assert info.underlying == "SPY"
        assert info.expiry_date == date(2024, 1, 20)
        assert info.right == "P"
        assert info.strike == 400.0
        assert info.is_put is True

    def test_parse_fractional_strike(self, generator):
        """Test parsing symbol with fractional strike."""
        # $450.50 = 00450500
        info = generator.parse_option_symbol("SPY240120C00450500")

        assert info is not None
        assert info.strike == 450.5

    def test_parse_low_strike(self, generator):
        """Test parsing symbol with low strike price."""
        # $5.00 = 00005000
        info = generator.parse_option_symbol("SPY240120P00005000")

        assert info is not None
        assert info.strike == 5.0

    def test_parse_high_strike(self, generator):
        """Test parsing symbol with high strike price."""
        # $1000.00 = 01000000
        info = generator.parse_option_symbol("SPY240120C01000000")

        assert info is not None
        assert info.strike == 1000.0

    def test_parse_short_underlying(self, generator):
        """Test parsing symbol with short underlying name."""
        info = generator.parse_option_symbol("X240120C00025000")

        assert info is not None
        assert info.underlying == "X"
        assert info.strike == 25.0

    def test_parse_long_underlying(self, generator):
        """Test parsing symbol with 4-character underlying."""
        info = generator.parse_option_symbol("AAPL240120C00180000")

        assert info is not None
        assert info.underlying == "AAPL"
        assert info.strike == 180.0

    def test_parse_invalid_symbol_too_short(self, generator):
        """Test parsing invalid symbol (too short)."""
        info = generator.parse_option_symbol("SPY240120C")

        assert info is None

    def test_parse_invalid_symbol_bad_right(self, generator):
        """Test parsing invalid symbol (bad right indicator)."""
        info = generator.parse_option_symbol("SPY240120X00450000")

        assert info is None

    def test_parse_invalid_symbol_no_numbers(self, generator):
        """Test parsing invalid symbol (no numbers)."""
        info = generator.parse_option_symbol("SPYABCDEFGHIJ")

        assert info is None

    def test_parse_lowercase_normalized(self, generator):
        """Test that lowercase symbols are normalized."""
        info = generator.parse_option_symbol("spy240120c00450000")

        assert info is not None
        assert info.underlying == "SPY"
        assert info.right == "C"

    def test_parse_symbol_with_whitespace(self, generator):
        """Test parsing symbol with leading/trailing whitespace."""
        info = generator.parse_option_symbol("  SPY240120C00450000  ")

        assert info is not None
        assert info.underlying == "SPY"


# ============================================================================
# DTE Filter Tests
# ============================================================================


class TestDTEFiltering:
    """Tests for days-to-expiration filtering."""

    def test_filter_by_dte_basic(self, generator, sample_symbols):
        """Test basic DTE filtering."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=60,  # Only Jan and early Feb expiries
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=100,
        )

        # Should include Jan 20 (DTE=30) and Dec 28 (DTE=7), but not Feb 16 (DTE=57)
        # Actually Dec 21 + 57 = Feb 16, so it should be included
        # Let me recalculate: Dec 21 to Jan 20 = 30 days
        # Dec 21 to Feb 16 = 57 days (within 60)
        # Dec 21 to Mar 15 = 85 days (outside)
        # Dec 21 to Jun 21 = 183 days (outside)
        # Dec 21 to Dec 28 = 7 days (exactly at min)

        # Check that Mar and Jun expiries are excluded
        for sym in manifest.symbols:
            assert "240315" not in sym  # Mar 15, 2024
            assert "240621" not in sym  # Jun 21, 2024

    def test_filter_excludes_expired_options(self, generator, sample_symbols):
        """Test that expired options are excluded."""
        as_of = date(2024, 1, 25)  # After Jan 20 expiry
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=1,
            dte_max=365,
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=100,
        )

        # Jan 20 and Dec 28 expiries should be excluded (expired)
        for sym in manifest.symbols:
            assert "240120" not in sym
            assert "231228" not in sym

    def test_filter_min_dte_boundary(self, generator, sample_symbols):
        """Test DTE filter at minimum boundary."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        # DTE to Dec 28 is exactly 7 days
        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,  # Exactly matches Dec 28
            dte_max=365,
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=100,
        )

        # Dec 28 should be included (DTE=7, at boundary)
        dec28_symbols = [s for s in manifest.symbols if "231228" in s]
        assert len(dec28_symbols) > 0

    def test_filter_min_dte_excludes_below(self, generator, sample_symbols):
        """Test that options below min DTE are excluded."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=8,  # Dec 28 has DTE=7, should be excluded
            dte_max=365,
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=100,
        )

        # Dec 28 should NOT be included
        dec28_symbols = [s for s in manifest.symbols if "231228" in s]
        assert len(dec28_symbols) == 0


# ============================================================================
# Moneyness Filter Tests
# ============================================================================


class TestMoneynessFiltering:
    """Tests for moneyness filtering."""

    def test_filter_by_moneyness_basic(self, generator, sample_symbols):
        """Test basic moneyness filtering."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=1,
            dte_max=365,
            moneyness_min=0.95,  # 95% of spot = $427.50
            moneyness_max=1.05,  # 105% of spot = $472.50
            options_per_expiry_side=100,
        )

        # Only strikes between $427.50 and $472.50 should be included
        for sym in manifest.symbols:
            info = generator.parse_option_symbol(sym)
            moneyness = info.strike / spot
            assert 0.95 <= moneyness <= 1.05

    def test_filter_otm_puts(self, generator, sample_symbols):
        """Test filtering for OTM puts only."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=1,
            dte_max=365,
            moneyness_min=0.85,  # Deep OTM puts
            moneyness_max=0.99,  # Just below ATM
            options_per_expiry_side=100,
        )

        # Should only have low strike options
        for sym in manifest.symbols:
            info = generator.parse_option_symbol(sym)
            assert info.strike < spot

    def test_filter_otm_calls(self, generator, sample_symbols):
        """Test filtering for OTM calls only."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=1,
            dte_max=365,
            moneyness_min=1.01,  # Just above ATM
            moneyness_max=1.15,  # OTM calls
            options_per_expiry_side=100,
        )

        # Should only have high strike options
        for sym in manifest.symbols:
            info = generator.parse_option_symbol(sym)
            assert info.strike > spot


# ============================================================================
# Per-Expiry Cap Tests
# ============================================================================


class TestPerExpiryCaps:
    """Tests for per-expiry per-side capping."""

    def test_cap_per_expiry_side(self, generator, sample_symbols):
        """Test that per-expiry-side caps are applied."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=1,
            dte_max=365,
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=2,  # Only 2 per expiry per side
        )

        # Count per expiry per side
        counts: dict[tuple[date, str], int] = {}
        for sym in manifest.symbols:
            info = generator.parse_option_symbol(sym)
            key = (info.expiry_date, info.right)
            counts[key] = counts.get(key, 0) + 1

        # Each group should have at most 2
        for key, count in counts.items():
            assert count <= 2, f"{key} has {count} symbols, expected <= 2"

    def test_cap_selects_nearest_atm(self, generator):
        """Test that capping selects options nearest to ATM."""
        # Create symbols with varying distances from ATM
        symbols = [
            "SPY240120C00400000",  # $400 - far from ATM
            "SPY240120C00440000",  # $440 - close to ATM
            "SPY240120C00450000",  # $450 - ATM
            "SPY240120C00460000",  # $460 - close to ATM
            "SPY240120C00500000",  # $500 - far from ATM
        ]

        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=1,
            dte_max=365,
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=3,  # Take 3 closest to ATM
        )

        # Should select $440, $450, $460 (closest to $450 spot)
        strikes = sorted([generator.parse_option_symbol(s).strike for s in manifest.symbols])
        assert strikes == [440.0, 450.0, 460.0]

    def test_cap_one_per_expiry_side(self, generator, sample_symbols):
        """Test extreme case of cap=1."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=1,
            dte_max=365,
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=1,
        )

        # Should have exactly 1 call and 1 put per expiry
        calls_by_expiry: dict[date, list[str]] = {}
        puts_by_expiry: dict[date, list[str]] = {}

        for sym in manifest.symbols:
            info = generator.parse_option_symbol(sym)
            if info.is_call:
                if info.expiry_date not in calls_by_expiry:
                    calls_by_expiry[info.expiry_date] = []
                calls_by_expiry[info.expiry_date].append(sym)
            else:
                if info.expiry_date not in puts_by_expiry:
                    puts_by_expiry[info.expiry_date] = []
                puts_by_expiry[info.expiry_date].append(sym)

        for exp, syms in calls_by_expiry.items():
            assert len(syms) == 1, f"Expected 1 call for {exp}, got {len(syms)}"

        for exp, syms in puts_by_expiry.items():
            assert len(syms) == 1, f"Expected 1 put for {exp}, got {len(syms)}"


# ============================================================================
# Manifest Metadata Tests
# ============================================================================


class TestManifestMetadata:
    """Tests for manifest metadata."""

    def test_manifest_metadata_fields(self, generator, sample_symbols):
        """Test that manifest metadata contains all required fields."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        assert manifest.metadata.spot_reference == spot
        assert manifest.metadata.dte_min == 7
        assert manifest.metadata.dte_max == 120
        assert manifest.metadata.moneyness_min == 0.8
        assert manifest.metadata.moneyness_max == 1.2
        assert manifest.metadata.options_per_expiry_side == 50
        assert manifest.metadata.underlying == "SPY"
        assert manifest.metadata.symbols_count == len(manifest.symbols)
        assert manifest.metadata.expiries_count == len(manifest.symbols_by_expiry)
        assert manifest.metadata.config_hash is not None
        assert len(manifest.metadata.config_hash) == 16

    def test_manifest_symbols_by_expiry(self, generator, sample_symbols):
        """Test that symbols_by_expiry groups correctly."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=1,
            dte_max=365,
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=100,
        )

        # All symbols should be accounted for in groups
        grouped_symbols = []
        for syms in manifest.symbols_by_expiry.values():
            grouped_symbols.extend(syms)

        assert sorted(grouped_symbols) == sorted(manifest.symbols)


# ============================================================================
# Manifest Serialization Tests
# ============================================================================


class TestManifestSerialization:
    """Tests for manifest JSON serialization."""

    def test_manifest_to_dict(self, generator, sample_symbols):
        """Test manifest to_dict conversion."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        data = manifest.to_dict()

        assert "symbols" in data
        assert "metadata" in data
        assert "symbols_by_expiry" in data
        assert data["metadata"]["spot_reference"] == spot
        assert data["metadata"]["dte_min"] == 7

    def test_manifest_from_dict(self, generator, sample_symbols):
        """Test manifest from_dict roundtrip."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        original = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        data = original.to_dict()
        restored = Manifest.from_dict(data)

        assert restored.symbols == original.symbols
        assert restored.metadata.spot_reference == original.metadata.spot_reference
        assert restored.metadata.dte_min == original.metadata.dte_min
        assert restored.metadata.config_hash == original.metadata.config_hash

    def test_manifest_write_and_load(self, generator, sample_symbols):
        """Test writing and loading manifest from file."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            generator.write_manifest(manifest, path)

            assert path.exists()

            loaded = generator.load_manifest(path)

            assert loaded.symbols == manifest.symbols
            assert loaded.metadata.spot_reference == manifest.metadata.spot_reference

    def test_manifest_json_valid(self, generator, sample_symbols):
        """Test that manifest JSON is valid."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        # Should be valid JSON
        json_str = json.dumps(manifest.to_dict())
        parsed = json.loads(json_str)

        assert parsed["metadata"]["spot_reference"] == spot


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for standalone helper functions."""

    def test_get_spot_reference_basic(self, underlying_df):
        """Test basic spot reference retrieval."""
        as_of = datetime(2023, 12, 21, 14, 0, 0)

        spot = get_spot_reference(underlying_df, as_of)

        # Should get the last close at or before 14:00 = 12:00 close of 449.50
        assert spot == 449.50

    def test_get_spot_reference_exact_match(self, underlying_df):
        """Test spot reference at exact timestamp."""
        as_of = datetime(2023, 12, 21, 16, 0, 0)

        spot = get_spot_reference(underlying_df, as_of)

        assert spot == 450.0

    def test_get_spot_reference_before_first(self, underlying_df):
        """Test spot reference before first timestamp falls back to earliest bar."""
        as_of = datetime(2023, 12, 19, 12, 0, 0)

        spot = get_spot_reference(underlying_df, as_of)

        assert spot == 448.50

    def test_get_spot_reference_empty_df(self):
        """Test spot reference with empty DataFrame raises error."""
        df = pd.DataFrame(columns=["ts_utc", "close"])

        with pytest.raises(ValueError, match="No underlying data provided"):
            get_spot_reference(df, datetime(2023, 12, 21, 12, 0, 0))

    def test_compute_dte_positive(self):
        """Test DTE calculation for future expiry."""
        dte = compute_dte(date(2024, 1, 20), date(2023, 12, 21))

        assert dte == 30

    def test_compute_dte_zero(self):
        """Test DTE calculation for same day."""
        dte = compute_dte(date(2024, 1, 20), date(2024, 1, 20))

        assert dte == 0

    def test_compute_dte_negative(self):
        """Test DTE calculation for expired option."""
        dte = compute_dte(date(2023, 12, 15), date(2023, 12, 21))

        assert dte == -6

    def test_compute_moneyness_atm(self):
        """Test moneyness calculation at ATM."""
        moneyness = compute_moneyness(450.0, 450.0)

        assert moneyness == 1.0

    def test_compute_moneyness_otm_put(self):
        """Test moneyness calculation for OTM put."""
        moneyness = compute_moneyness(400.0, 450.0)

        assert moneyness == pytest.approx(0.8889, rel=0.01)

    def test_compute_moneyness_otm_call(self):
        """Test moneyness calculation for OTM call."""
        moneyness = compute_moneyness(500.0, 450.0)

        assert moneyness == pytest.approx(1.1111, rel=0.01)

    def test_compute_moneyness_zero_spot(self):
        """Test moneyness with zero spot raises error."""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            compute_moneyness(450.0, 0.0)


# ============================================================================
# Determinism Tests
# ============================================================================


class TestDeterminism:
    """Tests for manifest generation determinism."""

    def test_same_inputs_same_output(self, generator, sample_symbols):
        """Test that same inputs produce identical output."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        manifest1 = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        manifest2 = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        assert manifest1.symbols == manifest2.symbols
        assert manifest1.metadata.config_hash == manifest2.metadata.config_hash

    def test_symbol_order_does_not_affect_output(self, generator, sample_symbols):
        """Test that input symbol order doesn't affect output."""
        as_of = date(2023, 12, 21)
        spot = 450.0

        # Shuffle input symbols
        import random
        shuffled = sample_symbols.copy()
        random.shuffle(shuffled)

        manifest1 = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        manifest2 = generator.generate_manifest(
            available_symbols=shuffled,
            spot_reference=spot,
            as_of_date=as_of,
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        # Output should be sorted and identical
        assert manifest1.symbols == manifest2.symbols


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_symbols_list(self, generator):
        """Test with empty symbols list."""
        manifest = generator.generate_manifest(
            available_symbols=[],
            spot_reference=450.0,
            as_of_date=date(2023, 12, 21),
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        assert manifest.symbols == []
        assert manifest.metadata.symbols_count == 0

    def test_no_symbols_match_criteria(self, generator, sample_symbols):
        """Test when no symbols match criteria."""
        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=450.0,
            as_of_date=date(2023, 12, 21),
            dte_min=1000,  # No options with 1000+ DTE
            dte_max=2000,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        assert manifest.symbols == []

    def test_invalid_spot_reference(self, generator, sample_symbols):
        """Test with invalid spot reference."""
        with pytest.raises(ValueError, match="spot_reference must be positive"):
            generator.generate_manifest(
                available_symbols=sample_symbols,
                spot_reference=-1.0,
                as_of_date=date(2023, 12, 21),
                dte_min=7,
                dte_max=120,
                moneyness_min=0.8,
                moneyness_max=1.2,
                options_per_expiry_side=50,
            )

    def test_all_invalid_symbols(self, generator):
        """Test with all invalid symbols."""
        invalid_symbols = ["INVALID1", "NOTANOPTION", "12345"]

        manifest = generator.generate_manifest(
            available_symbols=invalid_symbols,
            spot_reference=450.0,
            as_of_date=date(2023, 12, 21),
            dte_min=7,
            dte_max=120,
            moneyness_min=0.8,
            moneyness_max=1.2,
            options_per_expiry_side=50,
        )

        assert manifest.symbols == []

    def test_mixed_valid_invalid_symbols(self, generator):
        """Test with mix of valid and invalid symbols."""
        symbols = [
            "SPY240120C00450000",  # Valid
            "INVALID",  # Invalid
            "SPY240120P00450000",  # Valid
            "NOTANOPTION",  # Invalid
        ]

        manifest = generator.generate_manifest(
            available_symbols=symbols,
            spot_reference=450.0,
            as_of_date=date(2023, 12, 21),
            dte_min=1,
            dte_max=365,
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=50,
        )

        # Should have only the 2 valid symbols
        assert len(manifest.symbols) == 2


# ============================================================================
# Config Integration Tests
# ============================================================================


class TestConfigIntegration:
    """Tests for config integration."""

    def test_uses_config_defaults(self, generator, sample_symbols):
        """Test that config defaults are used when not specified."""
        manifest = generator.generate_manifest(
            available_symbols=sample_symbols,
            spot_reference=450.0,
            as_of_date=date(2023, 12, 21),
            # Not specifying dte_min/dte_max - should use config defaults
            moneyness_min=0.5,
            moneyness_max=1.5,
            options_per_expiry_side=50,
        )

        assert manifest.metadata.dte_min == 7  # From config
        assert manifest.metadata.dte_max == 120  # From config

    def test_get_manifest_path(self, generator):
        """Test manifest path generation."""
        path = generator.get_manifest_path(date(2023, 12, 21))

        assert "manifest_SPY_2023-12-21.json" in str(path)
        assert "data/manifest" in str(path)
