"""
Redis client for the trading system.

This module implements the Redis client that handles both Redis Cache and Redis Store
components as shown in the system architecture. The Redis Cache stores raw data from
data sources, while the Redis Store contains processed data from ML models and broker
interactions.
"""

import json
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import redis

from src.config.settings import settings
from src.utils.logging import setup_logger

logger = setup_logger("redis_client")


class RedisClient:
    """
    Redis client for storing and retrieving trading data.
    
    Implements both Redis Cache and Redis Store functionality:
    
    Redis Cache:
    - Raw market data from data sources (Alpha Vantage, Polygon, Yahoo)
    - Temporary storage for position data and news/sentiment
    
    Redis Store:
    - Processed market data from ML models
    - Position status from monitoring
    - Sentiment scores from analysis
    - Order status from broker
    - Trading signals
    
    Provides specialized methods for:
    - Stock data (prices, indicators)
    - Watchlists and rankings
    - Position data (active and closed)
    - Trading signals
    - System state
    """

    def __init__(self):
        """Initialize the Redis client with connection settings from configuration."""
        # Create connection parameters
        conn_params = {
            "host": settings.database.redis_host,
            "port": settings.database.redis_port,
            "decode_responses": False,  # Keep as bytes for flexibility with different data types
        }

        # Only add password if it's not empty
        if settings.database.redis_password:
            conn_params["password"] = settings.database.redis_password

        # Create connection
        self._conn = redis.Redis(**conn_params)
        logger.info(
            f"Connected to Redis at {settings.database.redis_host}:{settings.database.redis_port}"
        )

        # Test connection
        try:
            self._conn.ping()
            logger.info("Redis connection test successful")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _serialize(self, data: Any) -> bytes:
        """
        Serialize data for storage in Redis.

        Args:
            data: Data to serialize

        Returns:
            Serialized data as bytes
        """
        if isinstance(data, (str, int, float, bool)):
            return str(data).encode("utf-8")
        elif isinstance(data, (dict, list, tuple, set)):
            try:
                return json.dumps(data).encode("utf-8")
            except (TypeError, OverflowError):
                return pickle.dumps(data)
        elif isinstance(data, pd.DataFrame):
            return pickle.dumps(data)
        elif isinstance(data, np.ndarray):
            return pickle.dumps(data)
        else:
            return pickle.dumps(data)

    def _deserialize(self, data: bytes, data_type: str = "auto") -> Any:
        """
        Deserialize data from Redis.

        Args:
            data: Serialized data
            data_type: Type hint for deserialization ('auto', 'json', 'pickle', 'str')

        Returns:
            Deserialized data
        """
        if data is None:
            return None

        if data_type == "str":
            return data.decode("utf-8")

        if data_type == "json" or (data_type == "auto" and data.startswith(b"{")):
            try:
                return json.loads(data.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        if data_type == "pickle" or data_type == "auto":
            try:
                return pickle.loads(data)
            except pickle.UnpicklingError:
                pass

        # Fallback to string
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data

    # ---------- General Key-Value Operations ----------

    def set(self, key: str, value: Any, expiry: Optional[int] = None, ex: Optional[int] = None) -> bool:
        """
        Set a key-value pair in Redis.

        Args:
            key: Key to set
            value: Value to set
            expiry: Optional expiry time in seconds (deprecated, use ex instead)
            ex: Optional expiry time in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            serialized = self._serialize(value)
            # Use ex if provided, otherwise use expiry
            expiration = ex if ex is not None else expiry
            if expiration:
                return self._conn.setex(key, expiration, serialized)
            else:
                return self._conn.set(key, serialized)
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def get(self, key: str, data_type: str = "auto") -> Any:
        """
        Get a value from Redis.

        Args:
            key: Key to get
            data_type: Type hint for deserialization

        Returns:
            Value if found, None otherwise
        """
        try:
            data = self._conn.get(key)
            return self._deserialize(data, data_type)
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return None

    def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key: Key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            return bool(self._conn.delete(key))
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    # ---------- Stock Data Operations ----------

    def set_stock_data(self, symbol: str, data: Dict[str, Any], data_type: str = "price") -> bool:
        """
        Store stock data in Redis.

        Args:
            symbol: Stock symbol
            data: Data to store
            data_type: Type of data ('price', 'indicators', 'fundamentals', 'sentiment')

        Returns:
            True if successful, False otherwise
        """
        key = f"stocks:{data_type}:{symbol}"
        return self.set(key, data)

    def get_stock_data(self, symbol: str, data_type: str = "price") -> Optional[Dict[str, Any]]:
        """
        Get stock data from Redis.

        Args:
            symbol: Stock symbol
            data_type: Type of data ('price', 'indicators', 'fundamentals', 'sentiment')

        Returns:
            Stock data if found, None otherwise
        """
        key = f"stocks:{data_type}:{symbol}"
        return self.get(key)

    def update_stock_price(self, symbol: str, price_data: Dict[str, float]) -> bool:
        """
        Update real-time price data for a stock.

        Args:
            symbol: Stock symbol
            price_data: Price data (e.g., {'last': 123.45, 'bid': 123.40, 'ask': 123.50})

        Returns:
            True if successful, False otherwise
        """
        key = f"stocks:price:{symbol}"
        # Get existing data or initialize empty dict
        existing_data = self.get(key) or {}
        # Update with new price data
        existing_data.update(price_data)
        # Store timestamp
        existing_data["timestamp"] = pd.Timestamp.now().isoformat()
        return self.set(key, existing_data)

    # ---------- Watchlist Operations ----------

    def set_watchlist(self, watchlist: List[str]) -> bool:
        """
        Store the current watchlist.

        Args:
            watchlist: List of stock symbols

        Returns:
            True if successful, False otherwise
        """
        return self.set("watchlist:current", watchlist)

    def get_watchlist(self) -> List[str]:
        """
        Get the current watchlist.

        Returns:
            List of stock symbols
        """
        try:
            # Get raw data from Redis
            raw_data = self._conn.get("watchlist:current")
            if raw_data is None:
                return []

            # Try to deserialize as JSON
            try:
                watchlist = json.loads(raw_data.decode("utf-8"))
                if isinstance(watchlist, list):
                    return [str(symbol) for symbol in watchlist]
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # Fallback to pickle
            try:
                watchlist = pickle.loads(raw_data)
                if isinstance(watchlist, list):
                    return [str(symbol) for symbol in watchlist]
            except pickle.UnpicklingError:
                pass

            return []
        except Exception as e:
            logger.error(f"Error getting watchlist: {e}")
            return []

    def add_to_watchlist(self, symbol: str) -> bool:
        """
        Add a symbol to the watchlist.

        Args:
            symbol: Stock symbol to add

        Returns:
            True if successful, False otherwise
        """
        watchlist = self.get_watchlist()
        if symbol not in watchlist:
            watchlist.append(symbol)
            return self.set_watchlist(watchlist)
        return True

    def remove_from_watchlist(self, symbol: str) -> bool:
        """
        Remove a symbol from the watchlist.

        Args:
            symbol: Stock symbol to remove

        Returns:
            True if successful, False otherwise
        """
        watchlist = self.get_watchlist()
        if symbol in watchlist:
            watchlist.remove(symbol)
            return self.set_watchlist(watchlist)
        return True

    # ---------- Ranking Operations ----------

    def set_ranked_candidates(self, candidates: List[Dict[str, Any]]) -> bool:
        """
        Store ranked trading candidates.

        Args:
            candidates: List of candidate stocks with scores

        Returns:
            True if successful, False otherwise
        """
        return self.set("candidates:ranked", candidates)

    def get_ranked_candidates(self) -> List[Dict[str, Any]]:
        """
        Get ranked trading candidates.

        Returns:
            List of candidate stocks with scores
        """
        try:
            # Get raw data from Redis
            raw_data = self._conn.get("candidates:ranked")
            if raw_data is None:
                return []

            # Try to deserialize as JSON
            try:
                candidates = json.loads(raw_data.decode("utf-8"))
                if isinstance(candidates, list):
                    return candidates
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

            # Fallback to pickle
            try:
                candidates = pickle.loads(raw_data)
                if isinstance(candidates, list):
                    return candidates
            except pickle.UnpicklingError:
                pass

            return []
        except Exception as e:
            logger.error(f"Error getting ranked candidates: {e}")
            return []

    def add_candidate_score(
        self, symbol: str, score: float, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a score for a trading candidate.

        Args:
            symbol: Stock symbol
            score: Ranking score
            metadata: Optional additional data

        Returns:
            True if successful, False otherwise
        """
        candidates = self.get_ranked_candidates()

        # Check if candidate already exists
        for candidate in candidates:
            if candidate["symbol"] == symbol:
                candidate["score"] = score
                if metadata:
                    candidate.update(metadata)
                break
        else:
            # Candidate doesn't exist, add it
            new_candidate = {"symbol": symbol, "score": score}
            if metadata:
                new_candidate.update(metadata)
            candidates.append(new_candidate)

        # Sort by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return self.set_ranked_candidates(candidates)

    # ---------- Position Operations ----------

    def set_active_position(self, symbol: str, position_data: Dict[str, Any]) -> bool:
        """
        Store active position data.

        Args:
            symbol: Stock symbol
            position_data: Position data

        Returns:
            True if successful, False otherwise
        """
        key = f"positions:active:{symbol}"
        return self.set(key, position_data)

    def get_active_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get active position data.

        Args:
            symbol: Stock symbol

        Returns:
            Position data if found, None otherwise
        """
        key = f"positions:active:{symbol}"
        return self.get(key)

    def get_all_active_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active positions.

        Returns:
            Dictionary of active positions
        """
        pattern = "positions:active:*"
        keys = self._conn.keys(pattern)

        positions = {}
        for key in keys:
            symbol = key.decode("utf-8").split(":")[-1]
            position_data = self.get(key.decode("utf-8"))
            if position_data:
                positions[symbol] = position_data

        return positions

    def delete_active_position(self, symbol: str) -> bool:
        """
        Delete an active position.

        Args:
            symbol: Stock symbol

        Returns:
            True if successful, False otherwise
        """
        key = f"positions:active:{symbol}"
        return self.delete(key)

    def update_position_pnl(self, symbol: str, current_price: float) -> bool:
        """
        Update unrealized P&L for a position.

        Args:
            symbol: Stock symbol
            current_price: Current price

        Returns:
            True if successful, False otherwise
        """
        position = self.get_active_position(symbol)
        if not position:
            return False

        # Calculate unrealized P&L
        entry_price = position.get("entry_price", 0)
        quantity = position.get("quantity", 0)
        side = position.get("side", "long")

        if side.lower() == "long":
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = (current_price / entry_price - 1) * 100 if entry_price else 0
        else:  # short
            unrealized_pnl = (entry_price - current_price) * quantity
            unrealized_pnl_pct = (1 - current_price / entry_price) * 100 if entry_price else 0

        # Update position data
        position["current_price"] = current_price
        position["unrealized_pnl"] = unrealized_pnl
        position["unrealized_pnl_pct"] = unrealized_pnl_pct
        position["last_update"] = pd.Timestamp.now().isoformat()

        return self.set_active_position(symbol, position)

    # ---------- Trading Signals Operations ----------

    def add_trading_signal(
        self, symbol: str, signal_type: str, signal_data: Dict[str, Any]
    ) -> bool:
        """
        Add a trading signal.

        Args:
            symbol: Stock symbol
            signal_type: Type of signal ('entry', 'exit', 'adjust')
            signal_data: Signal data

        Returns:
            True if successful, False otherwise
        """
        # Add timestamp if not provided
        if "timestamp" not in signal_data:
            signal_data["timestamp"] = pd.Timestamp.now().isoformat()

        key = f"signals:{signal_type}:{symbol}"
        return self.set(key, signal_data)

    def get_trading_signal(self, symbol: str, signal_type: str) -> Optional[Dict[str, Any]]:
        """
        Get a trading signal.

        Args:
            symbol: Stock symbol
            signal_type: Type of signal ('entry', 'exit', 'adjust')

        Returns:
            Signal data if found, None otherwise
        """
        key = f"signals:{signal_type}:{symbol}"
        return self.get(key)

    def clear_trading_signal(self, symbol: str, signal_type: str) -> bool:
        """
        Clear a trading signal.

        Args:
            symbol: Stock symbol
            signal_type: Type of signal ('entry', 'exit', 'adjust')

        Returns:
            True if successful, False otherwise
        """
        key = f"signals:{signal_type}:{symbol}"
        return self.delete(key)

    # ---------- Closed Position Operations ----------
    
    def set_closed_position(self, symbol: str, position_data: Dict[str, Any]) -> bool:
        """
        Store closed position data.
        
        Args:
            symbol: Stock symbol
            position_data: Position data
            
        Returns:
            True if successful, False otherwise
        """
        key = f"positions:closed:{symbol}:{pd.Timestamp.now().isoformat()}"
        return self.set(key, position_data)
    
    def get_all_closed_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all closed positions.
        
        Returns:
            Dictionary of closed positions, with the most recent position for each symbol
        """
        pattern = "positions:closed:*"
        keys = self._conn.keys(pattern)
        
        positions = {}
        # Group positions by symbol
        symbol_positions = {}
        
        for key in keys:
            key_str = key.decode("utf-8")
            parts = key_str.split(":")
            symbol = parts[2]
            timestamp = ":".join(parts[3:])
            
            position_data = self.get(key_str)
            if position_data:
                if symbol not in symbol_positions:
                    symbol_positions[symbol] = []
                position_data["close_timestamp"] = timestamp
                symbol_positions[symbol].append(position_data)
        
        # For each symbol, get the most recent position
        for symbol, pos_list in symbol_positions.items():
            # Sort by close timestamp (most recent first)
            pos_list.sort(key=lambda x: x.get("close_timestamp", ""), reverse=True)
            # Use the most recent position
            positions[symbol] = pos_list[0]
        
        return positions
    
    def get_all_closed_positions_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all closed positions with full history.
        
        Returns:
            Dictionary of closed positions, with a list of all positions for each symbol
        """
        pattern = "positions:closed:*"
        keys = self._conn.keys(pattern)
        
        positions = {}
        
        for key in keys:
            key_str = key.decode("utf-8")
            parts = key_str.split(":")
            symbol = parts[2]
            timestamp = ":".join(parts[3:])
            
            position_data = self.get(key_str)
            if position_data:
                if symbol not in positions:
                    positions[symbol] = []
                position_data["close_timestamp"] = timestamp
                positions[symbol].append(position_data)
        
        return positions
    
    # ---------- Trading Signals Operations (Extended) ----------
    
    def get_all_trading_signals(self, signal_type: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all trading signals of a specific type.
        
        Args:
            signal_type: Type of signal ('entry', 'exit', 'adjust')
            
        Returns:
            Dictionary of signals by symbol
        """
        pattern = f"signals:{signal_type}:*"
        keys = self._conn.keys(pattern)
        
        signals = {}
        for key in keys:
            symbol = key.decode("utf-8").split(":")[-1]
            signal_data = self.get(key.decode("utf-8"))
            if signal_data:
                signals[symbol] = signal_data
        
        return signals
    
    # ---------- Cache Operations ----------
    
    def cache_data(self, key: str, data: Any, expiry: Optional[int] = None) -> bool:
        """
        Store data in Redis Cache with optional expiry.
        
        Args:
            key: Cache key
            data: Data to cache
            expiry: Optional expiry time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = f"cache:{key}"
        return self.set(cache_key, data, expiry=expiry)
    
    def get_cached_data(self, key: str) -> Any:
        """
        Get data from Redis Cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data if found, None otherwise
        """
        cache_key = f"cache:{key}"
        return self.get(cache_key)
    
    # ---------- ML Model Data Operations ----------
    
    def store_model_output(self, model_name: str, data_type: str, data: Any) -> bool:
        """
        Store ML model output in Redis Store.
        
        Args:
            model_name: Name of the model (e.g., 'market_analysis', 'position_monitor')
            data_type: Type of data
            data: Model output data
            
        Returns:
            True if successful, False otherwise
        """
        key = f"models:{model_name}:{data_type}"
        return self.set(key, data)
    
    def get_model_output(self, model_name: str, data_type: str) -> Any:
        """
        Get ML model output from Redis Store.
        
        Args:
            model_name: Name of the model
            data_type: Type of data
            
        Returns:
            Model output data if found, None otherwise
        """
        key = f"models:{model_name}:{data_type}"
        return self.get(key)
    
    # ---------- Broker Operations ----------
    
    def store_broker_data(self, data_type: str, data: Any, symbol: Optional[str] = None) -> bool:
        """
        Store broker data in Redis Store.
        
        Args:
            data_type: Type of data (e.g., 'order', 'position', 'account')
            data: Broker data
            symbol: Optional stock symbol
            
        Returns:
            True if successful, False otherwise
        """
        if symbol:
            key = f"broker:{data_type}:{symbol}"
        else:
            key = f"broker:{data_type}"
        return self.set(key, data)
    
    def get_broker_data(self, data_type: str, symbol: Optional[str] = None) -> Any:
        """
        Get broker data from Redis Store.
        
        Args:
            data_type: Type of data
            symbol: Optional stock symbol
            
        Returns:
            Broker data if found, None otherwise
        """
        if symbol:
            key = f"broker:{data_type}:{symbol}"
        else:
            key = f"broker:{data_type}"
        return self.get(key)
    
    # ---------- System State Operations ----------

    def set_system_state(self, state: Dict[str, Any]) -> bool:
        """
        Store system state.

        Args:
            state: System state data

        Returns:
            True if successful, False otherwise
        """
        return self.set("system:state", state)

    def get_system_state(self) -> Dict[str, Any]:
        """
        Get system state.

        Returns:
            System state data
        """
        return self.get("system:state") or {}

    def update_system_state(self, **kwargs) -> bool:
        """
        Update system state.

        Args:
            **kwargs: Key-value pairs to update

        Returns:
            True if successful, False otherwise
        """
        state = self.get_system_state()
        state.update(kwargs)
        return self.set_system_state(state)

    # ---------- Dashboard Data Operations ----------

    def set_dashboard_data(self, data_type: str, data: Any) -> bool:
        """
        Store dashboard data.

        Args:
            data_type: Type of data
            data: Data to store

        Returns:
            True if successful, False otherwise
        """
        key = f"dashboard:{data_type}"
        return self.set(key, data)

    def get_dashboard_data(self, data_type: str) -> Any:
        """
        Get dashboard data.

        Args:
            data_type: Type of data

        Returns:
            Dashboard data if found, None otherwise
        """
        key = f"dashboard:{data_type}"
        return self.get(key)


# Create global Redis client instance
redis_client = RedisClient()
