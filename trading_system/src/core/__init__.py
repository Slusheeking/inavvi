"""
Core trading system functionality.

This module contains the main trading system components:
- Data pipeline
- Position monitoring
- Stock screening
- Trade execution
"""

import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any, Callable

from .data_pipeline import DataPipeline
from .position_monitor import PositionMonitor
from .screening import StockScreener
from .trade_execution import (
    TradeExecutor,
    OrderStatus,
    OrderType,
    TradeResult,
    TradeDirection
)

# Setup module logger
logger = logging.getLogger(__name__)

class TradingSystemMode(Enum):
    """Trading system operation modes"""
    BACKTEST = auto()
    PAPER_TRADING = auto()
    LIVE_TRADING = auto()
    OPTIMIZATION = auto()

class TradingSystem:
    """
    Main trading system class that integrates all core components.
    
    This class provides a unified interface for the trading system,
    integrating data pipeline, screening, position monitoring, and
    trade execution components.
    """
    
    def __init__(
        self,
        mode: TradingSystemMode = TradingSystemMode.PAPER_TRADING,
        config_path: Optional[str] = None
    ):
        """
        Initialize the trading system.
        
        Args:
            mode: System operation mode
            config_path: Path to configuration file
        """
        self.mode = mode
        self.config_path = config_path
        
        # Initialize core components
        self.data_pipeline = DataPipeline()
        self.position_monitor = PositionMonitor()
        self.stock_screener = StockScreener()
        self.trade_executor = TradeExecutor()
        
        logger.info(f"Trading system initialized in {mode.name} mode")
        
    def setup(self) -> bool:
        """
        Set up the trading system components.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.data_pipeline.initialize()
            self.position_monitor.initialize()
            self.stock_screener.initialize()
            self.trade_executor.initialize()
            logger.info("Trading system setup complete")
            return True
        except Exception as e:
            logger.error(f"Failed to set up trading system: {e}")
            return False
            
    def run_screening(self) -> List[Dict[str, Any]]:
        """
        Run the stock screening process.
        
        Returns:
            List of screening results
        """
        data = self.data_pipeline.get_latest_market_data()
        return self.stock_screener.run(data)
        
    def execute_trades(self, trade_signals: List[Dict[str, Any]]) -> List[TradeResult]:
        """
        Execute trades based on signals.
        
        Args:
            trade_signals: List of trade signals to execute
            
        Returns:
            List of trade results
        """
        results = []
        for signal in trade_signals:
            result = self.trade_executor.execute_trade(signal)
            self.position_monitor.update(result)
            results.append(result)
        return results
        
    def get_positions_summary(self) -> Dict[str, Any]:
        """
        Get summary of current positions.
        
        Returns:
            Dictionary with position summary
        """
        return self.position_monitor.get_summary()

# Factory function to create instance with default configuration
def create_trading_system(
    mode: Union[TradingSystemMode, str] = TradingSystemMode.PAPER_TRADING,
    config_path: Optional[str] = None
) -> TradingSystem:
    """
    Factory function to create a trading system instance.
    
    Args:
        mode: System operation mode
        config_path: Path to configuration file
        
    Returns:
        Configured TradingSystem instance
    """
    if isinstance(mode, str):
        try:
            mode = TradingSystemMode[mode.upper()]
        except KeyError:
            logger.warning(f"Unknown mode: {mode}, defaulting to PAPER_TRADING")
            mode = TradingSystemMode.PAPER_TRADING
            
    system = TradingSystem(mode=mode, config_path=config_path)
    system.setup()
    return system

__all__ = [
    # Core components
    "DataPipeline",
    "PositionMonitor",
    "StockScreener",
    "TradeExecutor",
    
    # Integrated system
    "TradingSystem",
    "TradingSystemMode",
    "create_trading_system",
    
    # Trade execution enums
    "OrderStatus",
    "OrderType",
    "TradeResult",
    "TradeDirection"
]
