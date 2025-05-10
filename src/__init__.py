"""
Trading System package.

This is the main package for the trading system application.
"""

__version__ = "1.0.0"

# Import key components for direct access
from .core import (
    DataPipeline,
    OrderStatus,
    OrderType,
    StockScreener,
    TradeDirection,
    TradeExecutor,
    TradeResult,
)
from .models import PatternRecognitionModel, RankingModel, FinancialSentimentModel
from .utils import get_logger, setup_logger

# Define public exports
__all__ = [
    # Version info
    "__version__",
    # Core components
    "DataPipeline",
    "StockScreener",
    "TradeExecutor",
    "OrderStatus",
    "OrderType",
    "TradeResult",
    "TradeDirection",
    # Models
    "RankingModel",
    "FinancialSentimentModel",
    "PatternRecognitionModel",
    # Utilities
    "setup_logger",
    "get_logger",
]
