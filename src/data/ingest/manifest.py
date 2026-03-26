"""Options manifest generator for deterministic symbol selection.

This module generates a manifest of option symbols to ingest based on
DTE (days to expiration), moneyness, and per-expiry caps. The manifest
is deterministic for a given configuration and reference timestamp.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.config.schema import ConfigSchema


@dataclass
class OptionSymbolInfo:
    """Parsed option symbol information."""

    raw_symbol: str
    underlying: str
    expiry_date: date
    right: str  # 'C' for call, 'P' for put
    strike: float

    @property
    def is_call(self) -> bool:
        """Check if option is a call."""
        return self.right == "C"

    @property
    def is_put(self) -> bool:
        """Check if option is a put."""
        return self.right == "P"


@dataclass
class ManifestMetadata:
    """Metadata for the generated manifest."""

    as_of_ts_utc: datetime
    spot_reference: float
    dte_min: int
    dte_max: int
    moneyness_min: float
    moneyness_max: float
    options_per_expiry_side: int
    underlying: str
    generated_at: datetime
    config_hash: str
    symbols_count: int = 0
    expiries_count: int = 0


@dataclass
class Manifest:
    """Options manifest with symbols and metadata."""

    symbols: list[str]
    metadata: ManifestMetadata
    symbols_by_expiry: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "symbols": self.symbols,
            "symbols_by_expiry": self.symbols_by_expiry,
            "metadata": {
                "as_of_ts_utc": self.metadata.as_of_ts_utc.isoformat(),
                "spot_reference": self.metadata.spot_reference,
                "dte_min": self.metadata.dte_min,
                "dte_max": self.metadata.dte_max,
                "moneyness_min": self.metadata.moneyness_min,
                "moneyness_max": self.metadata.moneyness_max,
                "options_per_expiry_side": self.metadata.options_per_expiry_side,
                "underlying": self.metadata.underlying,
                "generated_at": self.metadata.generated_at.isoformat(),
                "config_hash": self.metadata.config_hash,
                "symbols_count": self.metadata.symbols_count,
                "expiries_count": self.metadata.expiries_count,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        """Create manifest from dictionary."""
        metadata = ManifestMetadata(
            as_of_ts_utc=datetime.fromisoformat(data["metadata"]["as_of_ts_utc"]),
            spot_reference=data["metadata"]["spot_reference"],
            dte_min=data["metadata"]["dte_min"],
            dte_max=data["metadata"]["dte_max"],
            moneyness_min=data["metadata"]["moneyness_min"],
            moneyness_max=data["metadata"]["moneyness_max"],
            options_per_expiry_side=data["metadata"]["options_per_expiry_side"],
            underlying=data["metadata"]["underlying"],
            generated_at=datetime.fromisoformat(data["metadata"]["generated_at"]),
            config_hash=data["metadata"]["config_hash"],
            symbols_count=data["metadata"]["symbols_count"],
            expiries_count=data["metadata"]["expiries_count"],
        )
        return cls(
            symbols=data["symbols"],
            metadata=metadata,
            symbols_by_expiry=data.get("symbols_by_expiry", {}),
        )


class ManifestGenerator:
    """Generates deterministic option symbol manifests.

    The manifest generator selects option symbols based on:
    1. Days to expiration (DTE) filter
    2. Moneyness filter (strike/spot ratio)
    3. Per-expiry per-side cap (nearest-to-ATM ordering)

    The output is deterministic for a given config and reference timestamp.
    """

    # OSI (Options Symbology Initiative) regex pattern
    # Format: {underlying}{YYMMDD}{C/P}{strike price * 1000, 8 digits}
    # Example: SPY230120C00400000 = SPY Jan 20 2023 $400 Call
    OSI_PATTERN = re.compile(
        r"^(?P<underlying>[A-Z]{1,6})"
        r"(?P<expiry>\d{6})"
        r"(?P<right>[CP])"
        r"(?P<strike>\d{8})$"
    )

    def __init__(
        self,
        config: ConfigSchema,
        logger: logging.Logger | None = None,
    ):
        """Initialize manifest generator.

        Args:
            config: Application configuration
            logger: Optional logger instance
        """
        self._config = config
        self._logger = logger or logging.getLogger(__name__)

    def generate_manifest(
        self,
        available_symbols: list[str],
        spot_reference: float,
        as_of_date: date,
        dte_min: int | None = None,
        dte_max: int | None = None,
        moneyness_min: float = 0.8,
        moneyness_max: float = 1.2,
        options_per_expiry_side: int = 50,
    ) -> Manifest:
        """Generate option symbol manifest.

        Args:
            available_symbols: List of available option symbols (from provider)
            spot_reference: Spot price for moneyness calculation
            as_of_date: Reference date for DTE calculation
            dte_min: Minimum days to expiration (default from config)
            dte_max: Maximum days to expiration (default from config)
            moneyness_min: Minimum strike/spot ratio (e.g., 0.8 for 20% OTM puts)
            moneyness_max: Maximum strike/spot ratio (e.g., 1.2 for 20% OTM calls)
            options_per_expiry_side: Max options per expiry per side (call/put)

        Returns:
            Manifest with selected symbols and metadata

        Raises:
            ValueError: If spot_reference is invalid or no symbols match criteria
        """
        if spot_reference <= 0:
            raise ValueError(f"spot_reference must be positive, got {spot_reference}")

        # Use config defaults if not specified
        dte_min = dte_min if dte_min is not None else self._config.dataset.min_dte
        dte_max = dte_max if dte_max is not None else self._config.dataset.max_dte

        self._logger.info(
            f"Generating manifest: spot={spot_reference:.2f}, "
            f"DTE=[{dte_min}, {dte_max}], "
            f"moneyness=[{moneyness_min:.2f}, {moneyness_max:.2f}]"
        )

        # Step 1: Parse all symbols
        parsed_symbols = self._parse_symbols(available_symbols)
        self._logger.debug(f"Parsed {len(parsed_symbols)} valid option symbols")

        # Step 2: Filter by DTE
        dte_filtered = self._filter_by_dte(parsed_symbols, as_of_date, dte_min, dte_max)
        self._logger.debug(f"After DTE filter: {len(dte_filtered)} symbols")

        # Step 3: Filter by moneyness
        moneyness_filtered = self._filter_by_moneyness(
            dte_filtered, spot_reference, moneyness_min, moneyness_max
        )
        self._logger.debug(f"After moneyness filter: {len(moneyness_filtered)} symbols")

        # Step 4: Apply per-expiry-side caps
        selected = self._apply_per_expiry_caps(
            moneyness_filtered, spot_reference, options_per_expiry_side
        )
        self._logger.info(f"Selected {len(selected)} symbols after caps")

        # Build symbols list and group by expiry
        symbols = sorted([s.raw_symbol for s in selected])
        symbols_by_expiry = self._group_by_expiry(selected)

        # Create metadata
        as_of_ts = datetime.combine(as_of_date, datetime.min.time())
        metadata = ManifestMetadata(
            as_of_ts_utc=as_of_ts,
            spot_reference=spot_reference,
            dte_min=dte_min,
            dte_max=dte_max,
            moneyness_min=moneyness_min,
            moneyness_max=moneyness_max,
            options_per_expiry_side=options_per_expiry_side,
            underlying=self._config.universe.underlying,
            generated_at=datetime.now(UTC),
            config_hash=self._compute_config_hash(),
            symbols_count=len(symbols),
            expiries_count=len(symbols_by_expiry),
        )

        return Manifest(
            symbols=symbols,
            metadata=metadata,
            symbols_by_expiry=symbols_by_expiry,
        )

    def _parse_symbols(self, symbols: list[str]) -> list[OptionSymbolInfo]:
        """Parse option symbols into structured info.

        Args:
            symbols: Raw option symbol strings

        Returns:
            List of parsed OptionSymbolInfo objects (invalid symbols skipped)
        """
        parsed = []
        for symbol in symbols:
            info = self.parse_option_symbol(symbol)
            if info is not None:
                parsed.append(info)
        return parsed

    def parse_option_symbol(self, symbol: str) -> OptionSymbolInfo | None:
        """Parse a single option symbol.

        Supports OSI (OCC) format: {underlying}{YYMMDD}{C/P}{strike*1000}
        Example: SPY230120C00400000 = SPY Jan 20 2023 $400 Call

        Args:
            symbol: Option symbol string

        Returns:
            OptionSymbolInfo or None if parsing fails
        """
        # Strip all spaces for regex matching (Databento uses padded OCC format
        # like "SPY   240119C00445000", normalize to "SPY240119C00445000")
        normalized = symbol.replace(" ", "").upper()
        match = self.OSI_PATTERN.match(normalized)
        if not match:
            self._logger.debug(f"Failed to parse option symbol: {symbol}")
            return None

        underlying = match.group("underlying")
        expiry_str = match.group("expiry")  # YYMMDD
        right = match.group("right")
        strike_str = match.group("strike")

        # Parse expiry date (YYMMDD)
        try:
            year = 2000 + int(expiry_str[:2])
            month = int(expiry_str[2:4])
            day = int(expiry_str[4:6])
            expiry_date = date(year, month, day)
        except ValueError:
            self._logger.debug(f"Invalid expiry date in symbol: {symbol}")
            return None

        # Parse strike price (8 digits: 5 before decimal, 3 after)
        # e.g., 00400000 = $400.00
        try:
            strike = int(strike_str) / 1000.0
        except ValueError:
            self._logger.debug(f"Invalid strike in symbol: {symbol}")
            return None

        return OptionSymbolInfo(
            raw_symbol=symbol.strip(),  # Keep original format for API calls
            underlying=underlying,
            expiry_date=expiry_date,
            right=right,
            strike=strike,
        )

    def _filter_by_dte(
        self,
        symbols: list[OptionSymbolInfo],
        as_of_date: date,
        dte_min: int,
        dte_max: int,
    ) -> list[OptionSymbolInfo]:
        """Filter symbols by days to expiration.

        Args:
            symbols: Parsed option symbols
            as_of_date: Reference date for DTE calculation
            dte_min: Minimum DTE (inclusive)
            dte_max: Maximum DTE (inclusive)

        Returns:
            Filtered list of symbols
        """
        filtered = []
        for sym in symbols:
            dte = (sym.expiry_date - as_of_date).days
            if dte_min <= dte <= dte_max:
                filtered.append(sym)
        return filtered

    def _filter_by_moneyness(
        self,
        symbols: list[OptionSymbolInfo],
        spot: float,
        moneyness_min: float,
        moneyness_max: float,
    ) -> list[OptionSymbolInfo]:
        """Filter symbols by moneyness (strike/spot ratio).

        Args:
            symbols: Parsed option symbols
            spot: Spot price
            moneyness_min: Minimum moneyness ratio (inclusive)
            moneyness_max: Maximum moneyness ratio (inclusive)

        Returns:
            Filtered list of symbols
        """
        filtered = []
        for sym in symbols:
            moneyness = sym.strike / spot
            if moneyness_min <= moneyness <= moneyness_max:
                filtered.append(sym)
        return filtered

    def _apply_per_expiry_caps(
        self,
        symbols: list[OptionSymbolInfo],
        spot: float,
        options_per_expiry_side: int,
    ) -> list[OptionSymbolInfo]:
        """Apply per-expiry per-side caps using nearest-ATM ordering.

        For each expiry date and side (call/put), select the N options
        with strikes closest to the spot price.

        Args:
            symbols: Filtered option symbols
            spot: Spot price for ATM distance calculation
            options_per_expiry_side: Maximum options per expiry per side

        Returns:
            Capped list of symbols
        """
        # Group by (expiry, side)
        groups: dict[tuple[date, str], list[OptionSymbolInfo]] = {}
        for sym in symbols:
            key = (sym.expiry_date, sym.right)
            if key not in groups:
                groups[key] = []
            groups[key].append(sym)

        # For each group, sort by distance from ATM and take top N
        selected = []
        for (expiry, side), group_symbols in sorted(groups.items()):
            # Sort by absolute distance from spot (nearest to ATM first)
            sorted_by_atm = sorted(group_symbols, key=lambda s: abs(s.strike - spot))

            # Take top N
            selected.extend(sorted_by_atm[:options_per_expiry_side])

        return selected

    def _group_by_expiry(
        self, symbols: list[OptionSymbolInfo]
    ) -> dict[str, list[str]]:
        """Group symbols by expiry date.

        Args:
            symbols: Option symbol info objects

        Returns:
            Dictionary mapping expiry date strings to symbol lists
        """
        groups: dict[str, list[str]] = {}
        for sym in symbols:
            key = sym.expiry_date.isoformat()
            if key not in groups:
                groups[key] = []
            groups[key].append(sym.raw_symbol)

        # Sort symbols within each group
        for key in groups:
            groups[key] = sorted(groups[key])

        return dict(sorted(groups.items()))

    def _compute_config_hash(self) -> str:
        """Compute hash of relevant config parameters for reproducibility.

        Returns:
            SHA256 hash of config parameters (first 16 chars)
        """
        config_str = json.dumps(
            {
                "underlying": self._config.universe.underlying,
                "min_dte": self._config.dataset.min_dte,
                "max_dte": self._config.dataset.max_dte,
            },
            sort_keys=True,
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def write_manifest(self, manifest: Manifest, path: Path) -> None:
        """Write manifest to JSON file.

        Args:
            manifest: Manifest to write
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

        self._logger.info(f"Wrote manifest to {path} ({manifest.metadata.symbols_count} symbols)")

    def load_manifest(self, path: Path) -> Manifest:
        """Load manifest from JSON file.

        Args:
            path: Manifest file path

        Returns:
            Loaded Manifest object

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValueError: If manifest file is invalid
        """
        with open(path) as f:
            data = json.load(f)

        return Manifest.from_dict(data)

    def get_manifest_path(self, as_of_date: date) -> Path:
        """Get the expected manifest path for a given date.

        Args:
            as_of_date: Reference date

        Returns:
            Path to manifest file
        """
        manifest_dir = Path(self._config.paths.manifest_dir)
        filename = f"manifest_{self._config.universe.underlying}_{as_of_date.isoformat()}.json"
        return manifest_dir / filename


def get_spot_reference(
    underlying_df: pd.DataFrame,
    as_of_ts_utc: datetime,
) -> float:
    """Get spot reference price (last close at or before as_of timestamp).

    This is a standalone function that can be used with any underlying data source.

    Args:
        underlying_df: DataFrame with columns [ts_utc, close]
        as_of_ts_utc: Reference timestamp

    Returns:
        Last close price at or before as_of_ts_utc

    Raises:
        ValueError: If no valid price found before as_of_ts_utc
    """
    if underlying_df.empty:
        raise ValueError("No underlying data provided")

    # Ensure ts_utc is datetime
    df = underlying_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["ts_utc"]):
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    # Normalize timezone - ensure both are UTC-aware or both are naive
    as_of_ts = pd.Timestamp(as_of_ts_utc)
    if df["ts_utc"].dt.tz is None:
        # Data is tz-naive, make as_of tz-naive as well
        as_of_ts = as_of_ts.tz_localize(None) if as_of_ts.tz is not None else as_of_ts
    else:
        # Data is tz-aware, make as_of tz-aware
        if as_of_ts.tz is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        else:
            as_of_ts = as_of_ts.tz_convert("UTC")

    # Filter to timestamps at or before as_of
    valid = df[df["ts_utc"] <= as_of_ts]

    if valid.empty:
        # as_of may be a weekend/holiday before the first bar;
        # fall back to the earliest available close price
        sorted_df = df.sort_values("ts_utc")
        if sorted_df.empty:
            raise ValueError(f"No underlying data found at or before {as_of_ts_utc}")
        last_row = sorted_df.iloc[0]
    else:
        # Get the last close price at or before as_of
        last_row = valid.sort_values("ts_utc").iloc[-1]

    return float(last_row["close"])


def compute_dte(expiry_date: date, as_of_date: date) -> int:
    """Compute days to expiration.

    Args:
        expiry_date: Option expiration date
        as_of_date: Reference date

    Returns:
        Days to expiration (can be negative if expired)
    """
    return (expiry_date - as_of_date).days


def compute_moneyness(strike: float, spot: float) -> float:
    """Compute moneyness ratio.

    Args:
        strike: Option strike price
        spot: Current spot price

    Returns:
        Moneyness ratio (strike / spot)

    Raises:
        ValueError: If spot is zero or negative
    """
    if spot <= 0:
        raise ValueError(f"Spot price must be positive, got {spot}")
    return strike / spot
