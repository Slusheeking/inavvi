"""
Trading System package.

This is the main package for the trading system application.
"""

__version__ = "1.0.0"

# Import key components for direct access
from .core import DataPipeline, StockScreener, TradeExecutor
from .core import OrderStatus, OrderType, TradeResult, TradeDirection
from .models import RankingModel, SentimentModel, PatternRecognitionModel
from .utils import setup_logger, get_logger

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
    "SentimentModel",
    "PatternRecognitionModel",
    
    # Utilities
    "setup_logger",
    "get_logger",
]
