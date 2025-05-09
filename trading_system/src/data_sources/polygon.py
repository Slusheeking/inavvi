"""
Polygon API client for fetching market data.
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import websockets
from polygon import RESTClient, WebSocketClient

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

logger = setup_logger("polygon")

class PolygonAPI:
    """
    Client for Polygon.io APIs.
    
    Provides methods for:
    - Fetching stock data via REST API
    - Subscribing to real-time updates via WebSocket
    - Managing subscription channels
    """
    
    def __init__(self):
        """Initialize the Polygon API client with API key from settings."""
        self.api_key = settings.api.polygon_api_key
        self.rest_client = RESTClient(self.api_key)
        self.ws_client = None
        self.ws_connected = False
        self.active_symbols = set()
        
        logger.info("Polygon API client initialized")
    
    # ---------- REST API Methods ----------
    
    async def get_stock_snapshot(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a snapshot of current stock data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Stock snapshot data if successful, None otherwise
        """
        try:
            snapshot = self.rest_client.get_snapshot(ticker=symbol)
            
            # Process and return snapshot data
            if snapshot and hasattr(snapshot, 'ticker'):
                data = {
                    'symbol': snapshot.ticker.ticker,
                    'price': {
                        'last': snapshot.ticker.last.price,
                        'open': snapshot.ticker.day.open,
                        'high': snapshot.ticker.day.high,
                        'low': snapshot.ticker.day.low,
                        'close': snapshot.ticker.prevDay.close,
                        'volume': snapshot.ticker.day.volume,
                    },
                    'timestamp': pd.Timestamp.now().isoformat(),
                }
                
                # Cache in Redis
                redis_client.set_stock_data(symbol, data, 'price')
                
                return data
        except Exception as e:
            logger.error(f"Error fetching snapshot for {symbol}: {e}")
        
        return None
    
    async def get_daily_bars(self, symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
        """
        Get daily OHLCV bars for a stock.
        
        Args:
            symbol: Stock symbol
            days: Number of days to fetch
            
        Returns:
            DataFrame of daily bars if successful, None otherwise
        """
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            
            # Format dates as ISO format strings
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            # Fetch aggregates
            aggs = list(self.rest_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_str,
                to=end_str,
                limit=50000
            ))
            
            if not aggs:
                logger.warning(f"No daily bars found for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': pd.Timestamp.fromtimestamp(agg.timestamp / 1000),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
                'vwap': agg.vwap,
                'transactions': agg.transactions
            } for agg in aggs])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching daily bars for {symbol}: {e}")
        
        return None
    
    async def get_intraday_bars(
        self, 
        symbol: str, 
        minutes: int = 1, 
        days: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday OHLCV bars for a stock.
        
        Args:
            symbol: Stock symbol
            minutes: Bar interval in minutes (1, 5, 15, 30, etc.)
            days: Number of days to fetch
            
        Returns:
            DataFrame of intraday bars if successful, None otherwise
        """
        try:
            # Calculate time range
            end = datetime.now()
            start = end - timedelta(days=days)
            
            # Format dates as ISO format strings
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            # Fetch aggregates
            aggs = list(self.rest_client.get_aggs(
                ticker=symbol,
                multiplier=minutes,
                timespan="minute",
                from_=start_str,
                to=end_str,
                limit=50000
            ))
            
            if not aggs:
                logger.warning(f"No intraday bars found for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': pd.Timestamp.fromtimestamp(agg.timestamp / 1000),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume,
                'vwap': agg.vwap,
                'transactions': agg.transactions
            } for agg in aggs])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Sort by timestamp
            df.sort_index(inplace=True)
            
            # Store in Redis cache with expiry (1 hour)
            redis_key = f"stocks:intraday:{symbol}:{minutes}m"
            redis_client.set(redis_key, df, expiry=3600)
            
            return df
        except Exception as e:
            logger.error(f"Error fetching intraday bars for {symbol}: {e}")
        
        return None
    
    async def get_ticker_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a ticker.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Ticker details if successful, None otherwise
        """
        try:
            ticker_details = self.rest_client.get_ticker_details(symbol)
            
            if ticker_details:
                data = {
                    'symbol': ticker_details.ticker,
                    'name': ticker_details.name,
                    'market': ticker_details.market,
                    'locale': ticker_details.locale,
                    'primary_exchange': ticker_details.primary_exchange,
                    'type': ticker_details.type,
                    'active': ticker_details.active,
                    'currency_name': ticker_details.currency_name,
                    'cik': ticker_details.cik,
                    'composite_figi': ticker_details.composite_figi,
                    'share_class_figi': ticker_details.share_class_figi,
                    'last_updated_utc': ticker_details.last_updated_utc,
                }
                
                # Cache in Redis
                redis_client.set_stock_data(symbol, data, 'details')
                
                return data
        except Exception as e:
            logger.error(f"Error fetching ticker details for {symbol}: {e}")
        
        return None
    
    async def get_stock_universe(self, type: str = 'cs', active: bool = True) -> List[Dict[str, Any]]:
        """
        Get the universe of available stocks.
        
        Args:
            type: Type of tickers ('cs' for common stock, 'et' for ETF, etc.)
            active: Whether to include only active tickers
            
        Returns:
            List of ticker information
        """
        try:
            # Fetch all tickers
            tickers = list(self.rest_client.get_tickers(
                type=type,
                market='stocks',
                active=active
            ))
            
            # Extract relevant information
            universe = [{
                'symbol': ticker.ticker,
                'name': ticker.name,
                'market': ticker.market,
                'locale': ticker.locale,
                'primary_exchange': ticker.primary_exchange,
                'type': ticker.type,
                'active': ticker.active,
                'currency_name': ticker.currency_name,
            } for ticker in tickers]
            
            logger.info(f"Fetched {len(universe)} tickers from Polygon")
            
            # Store in Redis cache with 24-hour expiry
            redis_client.set("stocks:universe", universe, expiry=86400)
            
            return universe
        except Exception as e:
            logger.error(f"Error fetching stock universe: {e}")
        
        return []
    
    async def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status.
        
        Returns:
            Market status information
        """
        try:
            status = self.rest_client.get_market_status()
            
            data = {
                'market': status.market,
                'server_time': status.server_time,
                'exchanges': {}
            }
            
            for exchange in status.exchanges:
                data['exchanges'][exchange.name] = {
                    'name': exchange.name,
                    'type': exchange.type,
                    'market': exchange.market,
                    'status': exchange.status,
                    'session_start': exchange.session_start,
                    'session_end': exchange.session_end,
                }
            
            # Cache in Redis
            redis_client.set("market:status", data, expiry=300)  # Expire after 5 minutes
            
            return data
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
        
        return {'market': 'unknown', 'server_time': None, 'exchanges': {}}
    
    # ---------- WebSocket API Methods ----------
    
    async def connect_websocket(self):
        """
        Connect to Polygon WebSocket API.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Create WebSocket client if not already exists
            if not self.ws_client:
                self.ws_client = WebSocketClient(
                    api_key=self.api_key,
                    cluster="stocks",
                    process_message=self._process_ws_message,
                    on_close=self._on_ws_close,
                    on_error=self._on_ws_error
                )
            
            # Start connection
            self.ws_client.start()
            self.ws_connected = True
            logger.info("Connected to Polygon WebSocket")
            
            # Authentication will be handled automatically by the client
            
            return True
        except Exception as e:
            logger.error(f"Error connecting to Polygon WebSocket: {e}")
            self.ws_connected = False
            return False
    
    async def disconnect_websocket(self):
        """
        Disconnect from Polygon WebSocket API.
        
        Returns:
            True if disconnected successfully, False otherwise
        """
        try:
            if self.ws_client:
                self.ws_client.close()
                self.ws_client = None
                self.ws_connected = False
                logger.info("Disconnected from Polygon WebSocket")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Polygon WebSocket: {e}")
            return False
    
    async def subscribe_to_symbols(self, symbols: List[str]):
        """
        Subscribe to real-time updates for symbols.
        
        Args:
            symbols: List of symbols to subscribe to
            
        Returns:
            True if subscribed successfully, False otherwise
        """
        try:
            if not self.ws_connected:
                await self.connect_websocket()
            
            # Subscribe to trade updates
            channels = [f"T.{symbol}" for symbol in symbols]
            
            # Add to active symbols
            self.active_symbols.update(symbols)
            
            # Subscribe
            if self.ws_client:
                self.ws_client.subscribe(channels)
                logger.info(f"Subscribed to {len(symbols)} symbols: {', '.join(symbols[:5])}{' ...' if len(symbols) > 5 else ''}")
                return True
        except Exception as e:
            logger.error(f"Error subscribing to symbols: {e}")
        
        return False
    
    async def unsubscribe_from_symbols(self, symbols: List[str]):
        """
        Unsubscribe from real-time updates for symbols.
        
        Args:
            symbols: List of symbols to unsubscribe from
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        try:
            if not self.ws_connected:
                return False
            
            # Unsubscribe from trade updates
            channels = [f"T.{symbol}" for symbol in symbols]
            
            # Remove from active symbols
            self.active_symbols.difference_update(symbols)
            
            # Unsubscribe
            if self.ws_client:
                self.ws_client.unsubscribe(channels)
                logger.info(f"Unsubscribed from {len(symbols)} symbols")
                return True
        except Exception as e:
            logger.error(f"Error unsubscribing from symbols: {e}")
        
        return False
    
    def _process_ws_message(self, message: Dict[str, Any]):
        """
        Process incoming WebSocket message.
        
        Args:
            message: WebSocket message
        """
        try:
            # Check message type
            if isinstance(message, list) and len(message) > 0:
                for event in message:
                    self._process_single_event(event)
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _process_single_event(self, event: Dict[str, Any]):
        """
        Process a single WebSocket event.
        
        Args:
            event: WebSocket event
        """
        try:
            # Skip non-trade events for now
            if event.get('ev') != 'T':
                return
            
            # Extract trade data
            symbol = event.get('sym')
            price = event.get('p')
            size = event.get('s')
            timestamp = event.get('t')
            
            if not (symbol and price):
                return
            
            # Convert timestamp to datetime
            dt = pd.Timestamp(timestamp, unit='ms') if timestamp else pd.Timestamp.now()
            
            # Create trade data
            trade_data = {
                'symbol': symbol,
                'price': price,
                'size': size,
                'timestamp': dt.isoformat(),
            }
            
            # Update Redis cache
            redis_client.update_stock_price(symbol, {'last': price})
            
            # Publish trade event
            redis_client.set(f"events:trade:{symbol}", trade_data)
            
            # Update position P&L if active position
            position = redis_client.get_active_position(symbol)
            if position:
                redis_client.update_position_pnl(symbol, price)
        except Exception as e:
            logger.error(f"Error processing single event: {e}")
    
    def _on_ws_close(self):
        """Handle WebSocket connection close."""
        logger.info("Polygon WebSocket connection closed")
        self.ws_connected = False
    
    def _on_ws_error(self, error: Exception):
        """
        Handle WebSocket error.
        
        Args:
            error: WebSocket error
        """
        logger.error(f"Polygon WebSocket error: {error}")
        self.ws_connected = False
    
    # ---------- Screening Methods ----------
    
    async def get_gainers_losers(self, limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get top gainers and losers for the day.
        
        Args:
            limit: Number of stocks to fetch in each category
            
        Returns:
            Dictionary with 'gainers' and 'losers' lists
        """
        try:
            # Get snapshots for top stocks
            gainers = list(self.rest_client.get_snapshot_gainers_losers(direction="gainers", limit=limit))
            losers = list(self.rest_client.get_snapshot_gainers_losers(direction="losers", limit=limit))
            
            # Format gainers
            gainers_formatted = []
            for ticker in gainers:
                if hasattr(ticker, 'ticker'):
                    gainers_formatted.append({
                        'symbol': ticker.ticker.ticker,
                        'price': ticker.ticker.day.close if hasattr(ticker.ticker.day, 'close') else ticker.ticker.min.close,
                        'change': ticker.ticker.todaysChange,
                        'change_percent': ticker.ticker.todaysChangePerc,
                        'volume': ticker.ticker.day.volume if hasattr(ticker.ticker.day, 'volume') else ticker.ticker.min.volume,
                    })
            
            # Format losers
            losers_formatted = []
            for ticker in losers:
                if hasattr(ticker, 'ticker'):
                    losers_formatted.append({
                        'symbol': ticker.ticker.ticker,
                        'price': ticker.ticker.day.close if hasattr(ticker.ticker.day, 'close') else ticker.ticker.min.close,
                        'change': ticker.ticker.todaysChange,
                        'change_percent': ticker.ticker.todaysChangePerc,
                        'volume': ticker.ticker.day.volume if hasattr(ticker.ticker.day, 'volume') else ticker.ticker.min.volume,
                    })
            
            result = {
                'gainers': gainers_formatted,
                'losers': losers_formatted
            }
            
            # Cache in Redis with 5-minute expiry
            redis_client.set("market:gainers_losers", result, expiry=300)
            
            return result
        except Exception as e:
            logger.error(f"Error fetching gainers/losers: {e}")
        
        return {'gainers': [], 'losers': []}
    
    async def get_most_active(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get most active stocks for the day by volume.
        
        Args:
            limit: Number of stocks to fetch
            
        Returns:
            List of most active stocks
        """
        try:
            active = []
            
            # Get stock universe from cache or fetch if not available
            universe = redis_client.get("stocks:universe")
            if not universe:
                universe = await self.get_stock_universe()
            
            # Fetch snapshots batch by batch to avoid rate limits
            batch_size = 100
            for i in range(0, min(len(universe), 1000), batch_size):
                batch = universe[i:i+batch_size]
                symbols = [stock['symbol'] for stock in batch]
                
                # Get snapshots for batch
                snapshots = {}
                for symbol in symbols:
                    try:
                        snapshot = await self.get_stock_snapshot(symbol)
                        if snapshot:
                            snapshots[symbol] = snapshot
                    except Exception as e:
                        logger.debug(f"Error fetching pre-market data for {symbol}: {e}")
                        continue
                
                # Avoid rate limits
                await asyncio.sleep(0.5)
                
                # Extract volume information
                for symbol, snapshot in snapshots.items():
                    if 'price' in snapshot and 'volume' in snapshot['price']:
                        active.append({
                            'symbol': symbol,
                            'price': snapshot['price'].get('last', 0),
                            'volume': snapshot['price'].get('volume', 0),
                        })
            
            # Sort by volume (descending)
            active = [a for a in active if a.get('volume', 0) > 0]  # Filter out zero volume
            active.sort(key=lambda x: x.get('volume', 0), reverse=True)
            
            # Take top N
            active = active[:limit]
            
            # Cache in Redis with 5-minute expiry
            redis_client.set("market:most_active", active, expiry=300)
            
            return active
        except Exception as e:
            logger.error(f"Error fetching most active stocks: {e}")
        
        return []
    
    async def get_premarket_movers(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get pre-market movers.
        
        Args:
            limit: Number of stocks to fetch
            
        Returns:
            List of pre-market movers
        """
        try:
            # This is a more complex operation requiring multiple API calls
            # For now, we'll implement a simplified version
            
            # Get top gainers/losers and filter for pre-market activity
            gainers_losers = await self.get_gainers_losers(limit=50)
            
            candidates = gainers_losers['gainers'] + gainers_losers['losers']
            
            # Filter and get pre-market data for each symbol
            premarket_movers = []
            
            # Only run this during pre-market hours or right after market open
            now = datetime.now()
            current_hour = now.hour
            if not (4 <= current_hour < 10):  # Only run between 4am and 10am ET
                return []
            for candidate in candidates:
                symbol = candidate['symbol']
                
                # Get pre-market data using aggregates
                now = datetime.now()
                if now.hour >= 9 and now.hour < 16:  # During market hours
                    # If during market hours, use today's pre-market
                    start_str = now.strftime('%Y-%m-%d')
                    end_str = now.strftime('%Y-%m-%d')
                else:
                    # If after market hours, use next day's pre-market
                    tomorrow = now + timedelta(days=1)
                    start_str = tomorrow.strftime('%Y-%m-%d')
                    end_str = tomorrow.strftime('%Y-%m-%d')
                
                try:
                    # Get 1-minute bars for pre-market (4:00 AM - 9:30 AM ET)
                    aggs = list(self.rest_client.get_aggs(
                        ticker=symbol,
                        multiplier=1,
                        timespan="minute",
                        from_=start_str,
                        to=end_str,
                        limit=50000
                    ))
                    
                    if aggs:
                        # Filter pre-market aggregates
                        premarket_aggs = []
                        for agg in aggs:
                            dt = datetime.fromtimestamp(agg.timestamp / 1000)
                            if dt.hour >= 4 and dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
                                premarket_aggs.append(agg)
                        
                        if premarket_aggs:
                            # Calculate pre-market stats
                            open_price = premarket_aggs[0].open if premarket_aggs else None
                            last_price = premarket_aggs[-1].close if premarket_aggs else None
                            high_price = max([agg.high for agg in premarket_aggs]) if premarket_aggs else None
                            low_price = min([agg.low for agg in premarket_aggs]) if premarket_aggs else None
                            volume = sum([agg.volume for agg in premarket_aggs]) if premarket_aggs else 0
                            
                            if open_price and last_price:
                                change_pct = (last_price / open_price - 1) * 100
                                
                                premarket_movers.append({
                                    'symbol': symbol,
                                    'last_price': last_price,
                                    'open_price': open_price,
                                    'high_price': high_price,
                                    'low_price': low_price,
                                    'change_percent': change_pct,
                                    'volume': volume,
                                })