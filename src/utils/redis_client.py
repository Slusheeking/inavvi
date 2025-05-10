"""
Redis client for the trading system.
"""
import json
import pickle
from typing import Any, Dict, List, Optional

import redis
import numpy as np
import pandas as pd

from src.config.settings import settings
from src.utils.logging import setup_logger

logger = setup_logger("redis_client")

class RedisClient:
    """
    Redis client for storing and retrieving trading data.
    
    Provides specialized methods for:
    - Stock data (prices, indicators)
    - Watchlists and rankings
    - Position data
    - Trading signals
    """
    
    def __init__(self):
        """Initialize the Redis client with connection settings from configuration."""
        # Create connection parameters
        conn_params = {
            'host': settings.database.redis_host,
            'port': settings.database.redis_port,
            'decode_responses': False,  # Keep as bytes for flexibility with different data types
        }
        
        # Only add password if it's not empty
        if settings.database.redis_password:
            conn_params['password'] = settings.database.redis_password
        
        # Create connection
        self._conn = redis.Redis(**conn_params)
        logger.info(f"Connected to Redis at {settings.database.redis_host}:{settings.database.redis_port}")
        
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
            return str(data).encode('utf-8')
        elif isinstance(data, (dict, list, tuple, set)):
            try:
                return json.dumps(data).encode('utf-8')
            except (TypeError, OverflowError):
                return pickle.dumps(data)
        elif isinstance(data, pd.DataFrame):
            return pickle.dumps(data)
        elif isinstance(data, np.ndarray):
            return pickle.dumps(data)
        else:
            return pickle.dumps(data)
    
    def _deserialize(self, data: bytes, data_type: str = 'auto') -> Any:
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
        
        if data_type == 'str':
            return data.decode('utf-8')
        
        if data_type == 'json' or (data_type == 'auto' and data.startswith(b'{')):
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        if data_type == 'pickle' or data_type == 'auto':
            try:
                return pickle.loads(data)
            except pickle.UnpicklingError:
                pass
        
        # Fallback to string
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            return data
    
    # ---------- General Key-Value Operations ----------
    
    def set(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """
        Set a key-value pair in Redis.
        
        Args:
            key: Key to set
            value: Value to set
            expiry: Optional expiry time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            serialized = self._serialize(value)
            if expiry:
                return self._conn.setex(key, expiry, serialized)
            else:
                return self._conn.set(key, serialized)
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False
    
    def get(self, key: str, data_type: str = 'auto') -> Any:
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
    
    def set_stock_data(self, symbol: str, data: Dict[str, Any], data_type: str = 'price') -> bool:
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
    
    def get_stock_data(self, symbol: str, data_type: str = 'price') -> Optional[Dict[str, Any]]:
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
        existing_data['timestamp'] = pd.Timestamp.now().isoformat()
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
                watchlist = json.loads(raw_data.decode('utf-8'))
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
                candidates = json.loads(raw_data.decode('utf-8'))
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
    
    def add_candidate_score(self, symbol: str, score: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
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
            if candidate['symbol'] == symbol:
                candidate['score'] = score
                if metadata:
                    candidate.update(metadata)
                break
        else:
            # Candidate doesn't exist, add it
            new_candidate = {'symbol': symbol, 'score': score}
            if metadata:
                new_candidate.update(metadata)
            candidates.append(new_candidate)
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
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
            symbol = key.decode('utf-8').split(':')[-1]
            position_data = self.get(key.decode('utf-8'))
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
        entry_price = position.get('entry_price', 0)
        quantity = position.get('quantity', 0)
        side = position.get('side', 'long')
        
        if side.lower() == 'long':
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = (current_price / entry_price - 1) * 100 if entry_price else 0
        else:  # short
            unrealized_pnl = (entry_price - current_price) * quantity
            unrealized_pnl_pct = (1 - current_price / entry_price) * 100 if entry_price else 0
        
        # Update position data
        position['current_price'] = current_price
        position['unrealized_pnl'] = unrealized_pnl
        position['unrealized_pnl_pct'] = unrealized_pnl_pct
        position['last_update'] = pd.Timestamp.now().isoformat()
        
        return self.set_active_position(symbol, position)
    
    # ---------- Trading Signals Operations ----------
    
    def add_trading_signal(self, symbol: str, signal_type: str, signal_data: Dict[str, Any]) -> bool:
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
        if 'timestamp' not in signal_data:
            signal_data['timestamp'] = pd.Timestamp.now().isoformat()
        
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
