"""Live/Paper trading infrastructure.

This module provides the trading loop and order routing components
for paper and live trading.
"""

from .loop import (
    LoopMetrics,
    LoopState,
    TradingLoop,
)
from .features import (
    DatabaseFeatureProvider,
    FeatureProvider,
    MockFeatureProvider,
    RollingFeatureProvider,
)
from .monitoring import (
    Alert,
    AlertLevel,
    TradingMetrics,
    TradingMonitor,
)
from .positions import (
    PositionSnapshot,
    PositionTracker,
)
from .router import Fill, OrderRouter, PaperOrderRouter
from .state import (
    StateManager,
    TradingState,
)
from .signal_generator import (
    ModelSignalGenerator,
    RuleBasedSignalGenerator,
    SignalGenerator,
    SignalGeneratorBase,
)
from .surface_provider import (
    DatabaseSurfaceProvider,
    SurfaceProvider,
)
from .ingestion import (
    BaseIngestionService,
    DatabentoIngestionService,
    IngestionService,
    MockIngestionService,
)
from .symbols import (
    DatabentoSymbolProvider,
    ManifestSymbolProvider,
    MockSymbolProvider,
    SymbolProvider,
    save_symbols_manifest,
)

__all__ = [
    # Loop
    "TradingLoop",
    "LoopState",
    "LoopMetrics",
    # Protocols
    "SurfaceProvider",
    "SignalGenerator",
    "FeatureProvider",
    "IngestionService",
    # Surface Providers
    "DatabaseSurfaceProvider",
    # Ingestion Services
    "BaseIngestionService",
    "DatabentoIngestionService",
    "MockIngestionService",
    # Signal Generators
    "SignalGeneratorBase",
    "ModelSignalGenerator",
    "RuleBasedSignalGenerator",
    # Feature Providers
    "RollingFeatureProvider",
    "DatabaseFeatureProvider",
    "MockFeatureProvider",
    # Router
    "OrderRouter",
    "PaperOrderRouter",
    "Fill",
    # State Management
    "StateManager",
    "TradingState",
    # Position Tracking
    "PositionTracker",
    "PositionSnapshot",
    # Monitoring
    "TradingMonitor",
    "TradingMetrics",
    "Alert",
    "AlertLevel",
    # Symbol Providers
    "SymbolProvider",
    "DatabentoSymbolProvider",
    "ManifestSymbolProvider",
    "MockSymbolProvider",
    "save_symbols_manifest",
]
