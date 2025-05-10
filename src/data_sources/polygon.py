"""
Polygon API client for fetching market data.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd
import websockets
from polygon import RESTClient, WebSocketClient

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

logger = setup_logger("polygon")

# Define custom exceptions for Polygon errors
class PolygonDataError(Exception):
    """Custom exception for errors during Polygon data fetching."""
    pass

class PolygonRateLimitError(PolygonDataError):
    """Custom exception for Polygon rate limit errors."""
    pass

class PolygonSymbolNotFoundError(PolygonDataError):
    """Custom exception for symbol not found errors."""
    pass

class PolygonInvalidResponseError(PolygonDataError):
    """Custom exception for invalid response from Polygon API."""
    pass

# Retry decorator with exponential backoff
import functools
import random

def retry_async(attempts=3, delay=1, backoff=2):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except (PolygonRateLimitError, aiohttp.ClientError, websockets.exceptions.WebSocketException) as e:
                    if attempt < attempts - 1:
                        sleep_time = delay * (backoff ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Attempt {attempt + 1}/{attempts} failed for {func.__name__}. Retrying in {sleep_time:.2f} seconds...")
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(f"All {attempts} attempts failed for {func.__name__}.")
                        raise e
                except (PolygonDataError, PolygonSymbolNotFoundError, PolygonInvalidResponseError) as e:
                     # These are not typically retryable errors, re-raise immediately
                     logger.error(f"Non-retryable Polygon error in {func.__name__}: {e}")
                     raise e
                except Exception as e:
                    logger.error(f"An unexpected error occurred during {func.__name__}: {e}")
                    raise e
            return None # Should not reach here if attempts > 0
        return wrapper
    return decorator


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
    
    @retry_async(attempts=3, delay=5)
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
            if snapshot and hasattr(snapshot, 'ticker') and snapshot.ticker:
                data = {
                    'symbol': snapshot.ticker.ticker,
                    'price': {
                        'last': snapshot.ticker.last.price if hasattr(snapshot.ticker, 'last') and snapshot.ticker.last else None,
                        'open': snapshot.ticker.day.open if hasattr(snapshot.ticker, 'day') and snapshot.ticker.day else None,
                        'high': snapshot.ticker.day.high if hasattr(snapshot.ticker, 'day') and snapshot.ticker.day else None,
                        'low': snapshot.ticker.day.low if hasattr(snapshot.ticker, 'day') and snapshot.ticker.day else None,
                        'close': snapshot.ticker.prevDay.close if hasattr(snapshot.ticker, 'prevDay') and snapshot.ticker.prevDay else None,
                        'volume': snapshot.ticker.day.volume if hasattr(snapshot.ticker, 'day') and snapshot.ticker.day else None,
                    },
                    'timestamp': pd.Timestamp.now().isoformat(),
                }
                
                # Cache in Redis
                redis_client.set_stock_data(symbol, data, 'price')
                
                return data
            elif snapshot and hasattr(snapshot, 'status') and snapshot.status == 'NOT_FOUND':
                 raise PolygonSymbolNotFoundError(f"Symbol not found: {symbol}")
            else:
                 raise PolygonInvalidResponseError(f"Invalid snapshot response for {symbol}: {snapshot}")

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching snapshot for {symbol}: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching snapshot for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching snapshot for {symbol}: {e}") from e

    
    @retry_async(attempts=3, delay=5)
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
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching daily bars for {symbol}: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching daily bars for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching daily bars for {symbol}: {e}") from e
    
    @retry_async(attempts=3, delay=5)
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
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching intraday bars for {symbol}: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching intraday bars for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching intraday bars for {symbol}: {e}") from e
    
    @retry_async(attempts=3, delay=5)
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
            
            if ticker_details and hasattr(ticker_details, 'ticker'):
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
            elif ticker_details and hasattr(ticker_details, 'status') and ticker_details.status == 'NOT_FOUND':
                 raise PolygonSymbolNotFoundError(f"Symbol not found: {symbol}")
            else:
                 raise PolygonInvalidResponseError(f"Invalid ticker details response for {symbol}: {ticker_details}")

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching ticker details for {symbol}: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching ticker details for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching ticker details for {symbol}: {e}") from e
    
    @retry_async(attempts=3, delay=10)
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
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching stock universe: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching stock universe: {e}")
            raise PolygonDataError(f"Unexpected error fetching stock universe: {e}") from e
    
    @retry_async(attempts=3, delay=5)
    async def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status.
        
        Returns:
            Market status information
        """
        try:
            status = self.rest_client.get_market_status()
            
            if status and hasattr(status, 'market'):
                data = {
                    'market': status.market,
                    'server_time': status.server_time,
                    'exchanges': {}
                }
                
                if hasattr(status, 'exchanges') and status.exchanges:
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
            else:
                 raise PolygonInvalidResponseError(f"Invalid market status response: {status}")

        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching market status: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching market status: {e}")
            raise PolygonDataError(f"Unexpected error fetching market status: {e}") from e
    
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
    
    @retry_async(attempts=3, delay=5)
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
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching gainers/losers: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching gainers/losers: {e}")
            raise PolygonDataError(f"Unexpected error fetching gainers/losers: {e}") from e
    
    @retry_async(attempts=3, delay=5)
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
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching most active stocks: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching most active stocks: {e}")
            raise PolygonDataError(f"Unexpected error fetching most active stocks: {e}") from e
    
    @retry_async(attempts=3, delay=5)
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
                except Exception as e:
                    logger.error(f"Error fetching pre-market data for {symbol}: {e}")
            
            # Sort by absolute change percentage (descending)
            premarket_movers.sort(key=lambda x: abs(x.get('change_percent', 0)), reverse=True)
            
            # Take top N
            premarket_movers = premarket_movers[:limit]
            
            # Cache in Redis with 15-minute expiry
            redis_client.set("market:pre_market_movers", premarket_movers, expiry=900)
            
            return premarket_movers
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching pre-market movers: {e}")
            raise e # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching pre-market movers: {e}")
            raise PolygonDataError(f"Unexpected error fetching pre-market movers: {e}") from e
            
        return []
    
    @retry_async(attempts=3, delay=5)
    async def get_trading_status(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading status for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Trading status information
        """
        try:
            status = self.rest_client.get_ticker_details(symbol)
            
            if status:
                data = {
                    'symbol': symbol,
                    'active': status.active,
                    'primary_exchange': status.primary_exchange,
                    'type': status.type,
                    'updated_at': pd.Timestamp.now().isoformat(),
                }
                
                # Cache in Redis
                redis_client.set(f"stocks:status:{symbol}", data, expiry=3600)
                
                return data
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching trading status for {symbol}: {e}")
            raise e  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching trading status for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching trading status for {symbol}: {e}") from e
        
        return {'symbol': symbol, 'active': False, 'updated_at': pd.Timestamp.now().isoformat()}
    
    @retry_async(attempts=3, delay=5)
    async def get_previous_close(self, symbol: str) -> Optional[float]:
        """
        Get previous day's closing price for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Previous close price if available, None otherwise
        """
        try:
            # Get snapshot which includes previous day data
            snapshot = await self.get_stock_snapshot(symbol)
            
            if snapshot and 'price' in snapshot:
                return snapshot['price'].get('close', None)
            
            # Fallback: get daily bars for last 2 days
            bars = await self.get_daily_bars(symbol, days=2)
            
            if bars is not None and not bars.empty:
                # Get second to last row (previous day)
                if len(bars) >= 2:
                    return float(bars.iloc[-2]['close'])
                # If only one day available, use that
                return float(bars.iloc[-1]['close'])
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching previous close for {symbol}: {e}")
            raise e  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching previous close for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching previous close for {symbol}: {e}") from e
        
        return None
    
    @retry_async(attempts=3, delay=5)
    async def get_historical_data(self, symbol: str, timeframe: str = 'day', multiplier: int = 1, 
                                  start_date: str = None, end_date: str = None, limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a symbol.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe ('minute', 'hour', 'day', 'week', 'month', 'quarter', 'year')
            multiplier: Multiplier for timeframe
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of results
            
        Returns:
            DataFrame with historical data if successful, None otherwise
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                # Default to 1 year ago for day+ timeframes, 5 days for minute/hour
                if timeframe in ['minute', 'hour']:
                    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
                else:
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Fetch aggregates
            aggs = list(self.rest_client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timeframe,
                from_=start_date,
                to=end_date,
                limit=limit
            ))
            
            if not aggs:
                logger.warning(f"No historical data found for {symbol} ({timeframe})")
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
            
            # Cache in Redis with appropriate expiry
            cache_key = f"stocks:history:{symbol}:{timeframe}{multiplier}"
            
            # Set expiry based on timeframe
            if timeframe in ['minute', 'hour']:
                expiry = 3600  # 1 hour for intraday data
            else:
                expiry = 86400  # 24 hours for daily+ data
            
            redis_client.set(cache_key, df, expiry=expiry)
            
            return df
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching historical data for {symbol}: {e}")
            raise e  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching historical data for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching historical data for {symbol}: {e}") from e
        
        return None
    
    @retry_async(attempts=3, delay=5)
    async def get_dividends(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get dividend history for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of dividend events
        """
        try:
            # Fetch dividends for last 2 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years
            
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch dividend data
            dividends = list(self.rest_client.get_dividends(
                ticker=symbol,
                ex_dividend_date_gte=start_str,
                ex_dividend_date_lte=end_str
            ))
            
            # Format results
            results = []
            for div in dividends:
                results.append({
                    'symbol': div.ticker,
                    'ex_date': div.ex_dividend_date,
                    'payment_date': div.payment_date,
                    'record_date': div.record_date,
                    'declared_date': div.declaration_date,
                    'amount': div.cash_amount,
                    'frequency': div.frequency,
                    'dividend_type': div.dividend_type
                })
            
            # Cache in Redis
            redis_client.set(f"stocks:dividends:{symbol}", results, expiry=86400)  # 24-hour expiry
            
            return results
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching dividends for {symbol}: {e}")
            raise e  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching dividends for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching dividends for {symbol}: {e}") from e
        
        return []
    
    @retry_async(attempts=3, delay=5)
    async def get_splits(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get stock split history for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of split events
        """
        try:
            # Fetch splits for last 5 years
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)  # 5 years
            
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch split data
            splits = list(self.rest_client.get_splits(
                ticker=symbol,
                execution_date_gte=start_str,
                execution_date_lte=end_str
            ))
            
            # Format results
            results = []
            for split in splits:
                results.append({
                    'symbol': split.ticker,
                    'execution_date': split.execution_date,
                    'split_from': split.split_from,
                    'split_to': split.split_to,
                    'ratio': split.split_ratio
                })
            
            # Cache in Redis
            redis_client.set(f"stocks:splits:{symbol}", results, expiry=86400)  # 24-hour expiry
            
            return results
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching splits for {symbol}: {e}")
            raise e  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching splits for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching splits for {symbol}: {e}") from e
        
        return []
    
    @retry_async(attempts=3, delay=5)
    async def get_earnings(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get recent earnings releases for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of earnings events
        """
        try:
            # Fetch last 4 quarterly earnings
            earnings = list(self.rest_client.get_ticker_earnings(symbol, limit=4))
            
            # Format results
            results = []
            for er in earnings:
                results.append({
                    'symbol': er.ticker,
                    'fiscal_quarter': er.fiscal_quarter,
                    'fiscal_year': er.fiscal_year,
                    'eps_actual': er.eps_actual,
                    'eps_estimate': er.eps_estimate,
                    'eps_surprise': er.eps_surprise,
                    'eps_surprise_percent': er.eps_surprise_percent,
                    'quarter_end': er.quarter_end,
                    'call_time': er.call_time,
                    'report_date': er.report_date
                })
            
            # Cache in Redis
            redis_client.set(f"stocks:earnings:{symbol}", results, expiry=86400)  # 24-hour expiry
            
            return results
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching earnings for {symbol}: {e}")
            raise e  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching earnings for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching earnings for {symbol}: {e}") from e
        
        return []
    
    @retry_async(attempts=3, delay=5)
    async def get_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent news articles for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of news articles
            
        Returns:
            List of news articles
        """
        try:
            # Get current date and 7 days ago
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            # Format dates
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch news
            news = list(self.rest_client.get_ticker_news(
                ticker=symbol,
                published_gte=start_str,
                published_lte=end_str,
                limit=limit,
                order='desc'  # Most recent first
            ))
            
            # Format results
            results = []
            for article in news:
                results.append({
                    'id': article.id,
                    'publisher': article.publisher.name if hasattr(article.publisher, 'name') else None,
                    'title': article.title,
                    'author': article.author,
                    'published_utc': article.published_utc,
                    'article_url': article.article_url,
                    'tickers': article.tickers,
                    'image_url': article.image_url,
                    'description': article.description,
                    'keywords': article.keywords,
                    'amp_url': article.amp_url
                })
            
            # Cache in Redis
            redis_client.set(f"stocks:news:{symbol}", results, expiry=3600)  # 1-hour expiry
            
            return results
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching news for {symbol}: {e}")
            raise e  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching news for {symbol}: {e}") from e
        
        return []
    
    @retry_async(attempts=3, delay=5)
    async def get_options_chain(self, symbol: str, expiration_date: str = None) -> Dict[str, Any]:
        """
        Get options chain for a symbol.
        
        Args:
            symbol: Stock symbol
            expiration_date: Options expiration date (YYYY-MM-DD)
            
        Returns:
            Dictionary with options chain data
        """
        try:
            # Get all available expirations if date not provided
            if not expiration_date:
                expirations = list(self.rest_client.get_options_expirations(symbol))
                if not expirations:
                    logger.warning(f"No options expirations found for {symbol}")
                    return {'symbol': symbol, 'expirations': [], 'calls': [], 'puts': []}
                
                # Use the nearest expiration
                expirations.sort(key=lambda x: x.expiration_date)
                expiration_date = expirations[0].expiration_date
            
            # Fetch options for the specified expiration
            options = list(self.rest_client.get_options_chain(symbol, expiration_date=expiration_date))
            
            # Separate calls and puts
            calls = []
            puts = []
            
            for option in options:
                option_data = {
                    'symbol': option.ticker,
                    'underlying': option.underlying_ticker,
                    'expiration': option.expiration_date,
                    'strike': option.strike_price,
                    'last_price': option.last_trade.price if hasattr(option, 'last_trade') and option.last_trade else None,
                    'bid': option.bid if hasattr(option, 'bid') else None,
                    'ask': option.ask if hasattr(option, 'ask') else None,
                    'volume': option.day.volume if hasattr(option, 'day') and hasattr(option.day, 'volume') else 0,
                    'open_interest': option.open_interest if hasattr(option, 'open_interest') else 0,
                    'implied_volatility': option.implied_volatility if hasattr(option, 'implied_volatility') else None,
                    'delta': option.greeks.delta if hasattr(option, 'greeks') and hasattr(option.greeks, 'delta') else None,
                    'gamma': option.greeks.gamma if hasattr(option, 'greeks') and hasattr(option.greeks, 'gamma') else None,
                    'theta': option.greeks.theta if hasattr(option, 'greeks') and hasattr(option.greeks, 'theta') else None,
                    'vega': option.greeks.vega if hasattr(option, 'greeks') and hasattr(option.greeks, 'vega') else None,
                }
                
                # Sort into calls and puts
                if 'C' in option.ticker:
                    calls.append(option_data)
                elif 'P' in option.ticker:
                    puts.append(option_data)
            
            # Sort by strike price
            calls.sort(key=lambda x: x['strike'])
            puts.sort(key=lambda x: x['strike'])
            
            # Create result
            result = {
                'symbol': symbol,
                'expiration_date': expiration_date,
                'expirations': [exp.expiration_date for exp in expirations] if 'expirations' in locals() else [],
                'calls': calls,
                'puts': puts,
                'updated_at': pd.Timestamp.now().isoformat()
            }
            
            # Cache in Redis
            redis_client.set(f"stocks:options:{symbol}:{expiration_date}", result, expiry=1800)  # 30-minute expiry
            
            return result
        except aiohttp.ClientError as e:
            logger.error(f"Network error fetching options chain for {symbol}: {e}")
            raise e  # Re-raise for retry
        except Exception as e:
            logger.error(f"Unexpected error fetching options chain for {symbol}: {e}")
            raise PolygonDataError(f"Unexpected error fetching options chain for {symbol}: {e}") from e
        
        return {'symbol': symbol, 'expirations': [], 'calls': [], 'puts': []}

# Create global instance
polygon_client = PolygonAPI()

# High-level function for compatibility with unified API
async def fetch_polygon_data(symbol: str, interval: str = 'day', **kwargs) -> Dict[str, Any]:
    """
    Fetch data from Polygon API.
    
    Args:
        symbol: Stock symbol
        interval: Time interval ('day' or 'minute')
        **kwargs: Additional parameters
            multiplier: Optional multiplier for interval (default: 1)
            days: Optional number of days to fetch (default: 30 for daily, 1 for intraday)
            
    Returns:
        Dict containing the fetched market data
    """
    try:
        multiplier = kwargs.get('multiplier', 1)
        
        # Determine if this is intraday or daily data
        if interval == 'minute':
            days = kwargs.get('days', 1)
            df = await polygon_client.get_intraday_bars(symbol, minutes=multiplier, days=days)
        else:
            days = kwargs.get('days', 30)
            df = await polygon_client.get_daily_bars(symbol, days=days)
            
        if df is None:
            logger.warning(f"No data returned from Polygon for {symbol}")
            return {}
            
        # Convert DataFrame to dict format for consistent return
        result = {
            'symbol': symbol,
            'interval': interval,
            'data': df.reset_index().to_dict('records'),
            'last_refreshed': datetime.now().isoformat(),
        }
        
        return result
    except Exception as e:
        logger.error(f"Error fetching data from Polygon: {e}")
        raise PolygonDataError(f"Failed to fetch data: {str(e)}") from e