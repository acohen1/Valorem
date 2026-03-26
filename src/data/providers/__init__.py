"""Data provider protocols and implementations.

Concrete providers (DatabentoProvider, FREDProvider) are imported lazily
to avoid crashing when their API keys or packages are not available.
Use direct imports for those classes:

    from src.data.providers.databento import DatabentoProvider
    from src.data.providers.fred import FREDProvider
"""

from src.data.providers.mock import MockMacroDataProvider, MockMarketDataProvider
from src.data.providers.protocol import MacroDataProvider, MarketDataProvider

__all__ = [
    "MarketDataProvider",
    "MacroDataProvider",
    "MockMarketDataProvider",
    "MockMacroDataProvider",
]
