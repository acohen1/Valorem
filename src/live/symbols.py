"""Symbol discovery for live trading.

This module provides symbol providers for discovering option symbols
from market data sources or static manifests.
"""

import json
import logging
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import databento as db
import pandas as pd

logger = logging.getLogger(__name__)


class SymbolProvider(Protocol):
    """Protocol for discovering option symbols."""

    def get_option_symbols(
        self,
        underlying: str,
        min_dte: int = 7,
        max_dte: int = 90,
        moneyness_range: tuple[float, float] = (0.85, 1.15),
    ) -> list[str]:
        """Get option symbols matching criteria.

        Args:
            underlying: Underlying symbol (e.g., "SPY")
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            moneyness_range: (min, max) strike/spot ratio

        Returns:
            List of OCC option symbols
        """
        ...


class DatabentoSymbolProvider:
    """Fetches option symbols from Databento definitions endpoint.

    Uses Databento's parent symbology to fetch all available option
    contracts for an underlying, then filters by DTE and moneyness.

    Example:
        provider = DatabentoSymbolProvider()
        symbols = provider.get_option_symbols(
            "SPY",
            min_dte=7,
            max_dte=30,
            moneyness_range=(0.95, 1.05),
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        underlying_price: float | None = None,
    ):
        """Initialize Databento symbol provider.

        Args:
            api_key: Databento API key. If None, loads from DATABENTO_API_KEY env var.
            underlying_price: Current underlying price for moneyness filtering.
                If None, will attempt to fetch from API.

        Raises:
            ValueError: If API key is not provided and not found in environment.
        """
        key = api_key or os.getenv("DATABENTO_API_KEY")
        if not key:
            raise ValueError(
                "Databento API key required. Provide via api_key parameter "
                "or DATABENTO_API_KEY environment variable."
            )

        self._client = db.Historical(key=key)
        self._underlying_price = underlying_price

    def get_option_symbols(
        self,
        underlying: str,
        min_dte: int = 7,
        max_dte: int = 90,
        moneyness_range: tuple[float, float] = (0.85, 1.15),
    ) -> list[str]:
        """Get option symbols from Databento matching criteria.

        Args:
            underlying: Underlying symbol (e.g., "SPY")
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            moneyness_range: (min, max) strike/spot ratio

        Returns:
            List of OCC option symbols matching filters
        """
        try:
            # Fetch definitions for the underlying
            definitions = self._fetch_definitions(underlying)

            if definitions.empty:
                logger.warning(f"No option definitions found for {underlying}")
                return []

            # Get underlying price for moneyness calculation
            spot_price = self._underlying_price
            if spot_price is None:
                spot_price = self._get_underlying_price(underlying)

            if spot_price is None:
                logger.warning(
                    f"Could not determine underlying price for {underlying}, "
                    "skipping moneyness filter"
                )
                moneyness_range = (0.0, float("inf"))

            # Filter by criteria
            filtered = self._filter_definitions(
                definitions,
                underlying=underlying,
                min_dte=min_dte,
                max_dte=max_dte,
                moneyness_range=moneyness_range,
                spot_price=spot_price,
            )

            logger.info(
                f"Found {len(filtered)} option symbols for {underlying} "
                f"(DTE: {min_dte}-{max_dte}, moneyness: {moneyness_range})"
            )

            return sorted(filtered)

        except Exception as e:
            logger.error(f"Failed to get option symbols for {underlying}: {e}")
            raise

    def _fetch_definitions(self, underlying: str) -> pd.DataFrame:
        """Fetch option definitions from Databento.

        Args:
            underlying: Underlying symbol

        Returns:
            DataFrame with option definitions
        """
        parent_symbol = f"{underlying}.OPT"
        today = datetime.now()

        logger.info(f"Fetching option definitions for {parent_symbol}")

        data = self._client.timeseries.get_range(
            dataset="OPRA.PILLAR",
            schema="definition",
            stype_in="parent",
            symbols=[parent_symbol],
            start=today.date().isoformat(),
            end=today.date().isoformat(),
        )

        return data.to_df()

    def _get_underlying_price(self, underlying: str) -> float | None:
        """Get current underlying price from API.

        Args:
            underlying: Underlying symbol

        Returns:
            Current price or None if unavailable
        """
        try:
            # Try to get latest trade from GLBX.MDP3 (equities)
            today = datetime.now()
            data = self._client.timeseries.get_range(
                dataset="XNAS.ITCH",  # NASDAQ
                schema="ohlcv-1d",
                symbols=[underlying],
                stype_in="raw_symbol",
                start=(today.date()).isoformat(),
                end=today.date().isoformat(),
            )

            df = data.to_df()
            if not df.empty and "close" in df.columns:
                return float(df["close"].iloc[-1])

        except Exception as e:
            logger.debug(f"Could not fetch underlying price from API: {e}")

        return None

    def _filter_definitions(
        self,
        definitions: pd.DataFrame,
        underlying: str,
        min_dte: int,
        max_dte: int,
        moneyness_range: tuple[float, float],
        spot_price: float | None,
    ) -> list[str]:
        """Filter option definitions by criteria.

        Args:
            definitions: DataFrame with option definitions
            underlying: Underlying symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            moneyness_range: (min, max) strike/spot ratio
            spot_price: Current underlying price

        Returns:
            List of filtered option symbols
        """
        today = date.today()
        symbols = []

        # Get symbol column
        if "raw_symbol" in definitions.columns:
            symbol_col = "raw_symbol"
        elif "symbol" in definitions.columns:
            symbol_col = "symbol"
        else:
            symbol_col = definitions.index.name or "symbol"

        for idx, row in definitions.iterrows():
            symbol = row.get(symbol_col, idx)
            if pd.isna(symbol):
                continue

            symbol = str(symbol)

            # Parse option symbol to extract expiry and strike
            parsed = self._parse_occ_symbol(symbol)
            if parsed is None:
                continue

            expiry, strike, _ = parsed

            # Filter by DTE
            dte = (expiry - today).days
            if dte < min_dte or dte > max_dte:
                continue

            # Filter by moneyness
            if spot_price is not None:
                moneyness = strike / spot_price
                if moneyness < moneyness_range[0] or moneyness > moneyness_range[1]:
                    continue

            symbols.append(symbol)

        return symbols

    def _parse_occ_symbol(
        self, symbol: str
    ) -> tuple[date, float, str] | None:
        """Parse OCC option symbol format.

        OCC format: ROOT + YYMMDD + C/P + STRIKE (8 digits, strike * 1000)
        Example: SPY240315C00450000 = SPY call, expires 2024-03-15, strike $450

        Args:
            symbol: OCC option symbol

        Returns:
            Tuple of (expiry_date, strike, right) or None if parsing fails
        """
        # OCC format regex
        # Root symbol (1-6 chars), date (6 digits), C/P, strike (8 digits)
        pattern = r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$"
        match = re.match(pattern, symbol)

        if not match:
            return None

        try:
            root, date_str, right, strike_str = match.groups()

            # Parse expiry date (YYMMDD format)
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            expiry = date(year, month, day)

            # Parse strike (stored as strike * 1000)
            strike = int(strike_str) / 1000.0

            return expiry, strike, right

        except (ValueError, IndexError):
            return None


class ManifestSymbolProvider:
    """Loads symbols from a local JSON manifest file.

    Useful for testing, working offline, or using a curated symbol list.

    Manifest format:
        {
            "symbols": ["SPY240315C00450000", "SPY240315P00440000", ...],
            "underlying": "SPY",
            "created_at": "2024-01-01T00:00:00Z"
        }

    Or simple list:
        ["SPY240315C00450000", "SPY240315P00440000", ...]

    Example:
        provider = ManifestSymbolProvider("data/manifests/spy_options.json")
        symbols = provider.get_option_symbols("SPY")
    """

    def __init__(self, manifest_path: str | Path):
        """Initialize manifest symbol provider.

        Args:
            manifest_path: Path to JSON manifest file

        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValueError: If manifest format is invalid
        """
        self._manifest_path = Path(manifest_path)

        if not self._manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        with open(self._manifest_path) as f:
            data = json.load(f)

        # Support both list and dict formats
        if isinstance(data, list):
            self._symbols = data
            self._underlying = None
        elif isinstance(data, dict):
            self._symbols = data.get("symbols", [])
            self._underlying = data.get("underlying")
        else:
            raise ValueError(f"Invalid manifest format: expected list or dict")

        if not self._symbols:
            logger.warning(f"Manifest {manifest_path} contains no symbols")

    def get_option_symbols(
        self,
        underlying: str,
        min_dte: int = 7,
        max_dte: int = 90,
        moneyness_range: tuple[float, float] = (0.85, 1.15),
    ) -> list[str]:
        """Get option symbols from manifest matching criteria.

        Note: The manifest is assumed to already contain appropriate symbols.
        DTE and moneyness filtering is applied if symbols can be parsed.

        Args:
            underlying: Underlying symbol (used to filter if manifest has mixed underlyings)
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            moneyness_range: (min, max) strike/spot ratio (requires underlying price)

        Returns:
            List of matching option symbols
        """
        # If manifest specifies underlying and it doesn't match, return empty
        if self._underlying and self._underlying != underlying:
            logger.warning(
                f"Manifest underlying ({self._underlying}) doesn't match "
                f"requested underlying ({underlying})"
            )
            return []

        # Filter by underlying prefix and DTE
        today = date.today()
        filtered = []

        for symbol in self._symbols:
            # Check underlying prefix
            if not symbol.startswith(underlying):
                continue

            # Parse and filter by DTE
            parsed = self._parse_occ_symbol(symbol)
            if parsed is None:
                # Can't parse, include symbol anyway
                filtered.append(symbol)
                continue

            expiry, strike, _ = parsed
            dte = (expiry - today).days

            if dte < min_dte or dte > max_dte:
                continue

            filtered.append(symbol)

        logger.info(
            f"Loaded {len(filtered)} symbols from manifest "
            f"(DTE: {min_dte}-{max_dte})"
        )

        return sorted(filtered)

    def _parse_occ_symbol(
        self, symbol: str
    ) -> tuple[date, float, str] | None:
        """Parse OCC option symbol format.

        Same parsing logic as DatabentoSymbolProvider.
        """
        pattern = r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$"
        match = re.match(pattern, symbol)

        if not match:
            return None

        try:
            root, date_str, right, strike_str = match.groups()

            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            expiry = date(year, month, day)

            strike = int(strike_str) / 1000.0

            return expiry, strike, right

        except (ValueError, IndexError):
            return None


class MockSymbolProvider:
    """Mock symbol provider for testing.

    Generates synthetic option symbols for a given underlying.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        generate_count: int = 100,
    ):
        """Initialize mock symbol provider.

        Args:
            symbols: Static list of symbols to return. If None, generates synthetic symbols.
            generate_count: Number of symbols to generate if symbols is None.
        """
        self._symbols = symbols
        self._generate_count = generate_count

    def get_option_symbols(
        self,
        underlying: str,
        min_dte: int = 7,
        max_dte: int = 90,
        moneyness_range: tuple[float, float] = (0.85, 1.15),
    ) -> list[str]:
        """Get mock option symbols.

        If static symbols were provided, returns those filtered by underlying.
        Otherwise generates synthetic symbols.
        """
        if self._symbols is not None:
            return [s for s in self._symbols if s.startswith(underlying)]

        return self._generate_symbols(
            underlying, min_dte, max_dte, moneyness_range
        )

    def _generate_symbols(
        self,
        underlying: str,
        min_dte: int,
        max_dte: int,
        moneyness_range: tuple[float, float],
    ) -> list[str]:
        """Generate synthetic option symbols.

        Args:
            underlying: Underlying symbol
            min_dte: Minimum DTE
            max_dte: Maximum DTE
            moneyness_range: Moneyness range

        Returns:
            List of generated symbols
        """
        from datetime import timedelta

        today = date.today()
        symbols = []

        # Assume SPY-like price of 500 for strike generation
        base_price = 500

        # Generate expirations (weekly-ish)
        expirations = []
        for dte in range(min_dte, max_dte + 1, 7):
            exp = today + timedelta(days=dte)
            expirations.append(exp)

        # Generate strikes based on moneyness range
        min_strike = int(base_price * moneyness_range[0])
        max_strike = int(base_price * moneyness_range[1])
        strikes = list(range(min_strike, max_strike + 1, 5))

        # Generate symbols
        for exp in expirations:
            date_str = exp.strftime("%y%m%d")
            for strike in strikes:
                for right in ["C", "P"]:
                    strike_str = f"{int(strike * 1000):08d}"
                    symbol = f"{underlying}{date_str}{right}{strike_str}"
                    symbols.append(symbol)

                    if len(symbols) >= self._generate_count:
                        return sorted(symbols)

        return sorted(symbols)


def save_symbols_manifest(
    symbols: list[str],
    output_path: str | Path,
    underlying: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Save symbols to a JSON manifest file.

    Args:
        symbols: List of option symbols
        output_path: Path to write manifest file
        underlying: Optional underlying symbol
        metadata: Optional additional metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "symbols": sorted(symbols),
        "count": len(symbols),
        "created_at": datetime.now().isoformat(),
    }

    if underlying:
        manifest["underlying"] = underlying

    if metadata:
        manifest.update(metadata)

    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Saved {len(symbols)} symbols to {output_path}")
