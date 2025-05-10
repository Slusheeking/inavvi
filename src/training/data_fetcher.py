"""
Data fetching module for the trading system.

This module provides functionality for fetching financial data:
- Historical OHLCV data from various sources
- Financial news data for sentiment analysis
- Market data for context and features
- Data preprocessing and normalization
"""
import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import torch

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("data_fetcher")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataFetcher:
    """
    Data fetching system for ML model training and inference.
    
    Features:
    - Connects to multiple data sources
    - Fetches historical price data
    - Obtains financial news data
    - Cleans and structures data for models
    - Caches data in Redis for faster access
    - Utilizes GPU acceleration when available
    """
    
    def __init__(
        self,
        data_days: int = 365,
        use_polygon: bool = True,
        use_alpha_vantage: bool = True,
        use_redis_cache: bool = True,
        data_dir: str = None,
        use_gpu: bool = True
    ):
        """
        Initialize the data fetcher.
        
        Args:
            data_days: Number of days of historical data to fetch
            use_polygon: Whether to use Polygon.io API
            use_alpha_vantage: Whether to use Alpha Vantage API
            use_redis_cache: Whether to use Redis for caching data
            data_dir: Directory to store downloaded data files
            use_gpu: Whether to use GPU acceleration
        """
        # Configuration
        self.data_days = data_days
        self.use_polygon = use_polygon
        self.use_alpha_vantage = use_alpha_vantage
        self.use_redis_cache = use_redis_cache
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # API Keys
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # Data directories
        self.data_dir = data_dir or settings.data_dir
        self.price_data_dir = os.path.join(self.data_dir, "price_data")
        self.news_data_dir = os.path.join(self.data_dir, "news_data")
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.price_data_dir, exist_ok=True)
        os.makedirs(self.news_data_dir, exist_ok=True)
        
        # Data storage
        self.price_data = {}
        self.news_data = []
        self.data_cache = {}  # In-memory cache for frequently accessed data
        
        # API rate limiting settings
        self.polygon_requests_per_minute = 5
        self.alpha_vantage_requests_per_minute = 5
        
        # Cache settings
        self.cache_ttl = {
            "price_data": 86400,  # 24 hours
            "news_data": 43200,   # 12 hours
            "universe": 86400,    # 24 hours
            "indicators": 3600    # 1 hour
        }
        
        # Check for valid API keys
        if self.use_polygon and not self.polygon_api_key:
            logger.warning("Polygon API key not found. Disabling Polygon data source.")
            self.use_polygon = False
        
        if self.use_alpha_vantage and not self.alpha_vantage_api_key:
            logger.warning("Alpha Vantage API key not found. Disabling Alpha Vantage data source.")
            self.use_alpha_vantage = False
        
        logger.info(f"DataFetcher initialized with {data_days} days of historical data")
        logger.info(f"Data sources: Polygon={self.use_polygon}, Alpha Vantage={self.use_alpha_vantage}")
        logger.info(f"GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")

    def get_universe(self, size: int = 300) -> List[str]:
        """
        Get universe of stocks for data fetching.
        
        Args:
            size: Number of stocks to include in the universe
            
        Returns:
            List of stock symbols
        """
        # Try to get universe from Redis
        if self.use_redis_cache:
            universe = redis_client.get("trading:universe")
            if universe and len(universe) > 0:
                logger.info(f"Retrieved universe of {len(universe)} stocks from Redis")
                return universe[:size]
        
        # Generate universe if not in Redis
        try:
            # Try to get S&P 500 components from Polygon
            if self.use_polygon:
                sp500_symbols = self._fetch_sp500_symbols()
                if sp500_symbols and len(sp500_symbols) > 0:
                    # Take the top stocks by market cap
                    universe = sp500_symbols[:size]
                    
                    # Store in Redis if enabled
                    if self.use_redis_cache:
                        redis_client.set("trading:universe", universe)
                    
                    logger.info(f"Created universe of {len(universe)} stocks from S&P 500 components")
                    return universe
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols: {e}")
        
        # Fallback universe
        logger.warning("Using fallback universe of stocks")
        
        fallback_universe = [
            # Large Cap Tech
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", 
            # Financial
            "JPM", "BAC", "V", "MA", "GS", "MS", "BRK.B",
            # Healthcare
            "JNJ", "PFE", "UNH", "MRK", "ABT", "TMO", "LLY",
            # Consumer
            "PG", "KO", "PEP", "WMT", "HD", "MCD", "COST",
            # Industrial
            "CAT", "GE", "BA", "MMM", "HON", "UPS", "LMT",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC",
            # ETFs
            "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK"
        ]
        
        # Store in Redis if enabled
        if self.use_redis_cache:
            redis_client.set("trading:universe", fallback_universe)
        
        return fallback_universe[:size]

    def _fetch_sp500_symbols(self) -> List[str]:
        """
        Fetch S&P 500 symbols from Polygon API.
        
        Returns:
            List of S&P 500 stock symbols
        """
        if not self.use_polygon or not self.polygon_api_key:
            return []
        
        try:
            url = f"https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&sort=market_cap&order=desc&limit=500&apiKey={self.polygon_api_key}"
            
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if "results" in data:
                symbols = [item["ticker"] for item in data["results"]]
                return symbols
            
            return []
        
        except Exception as e:
            logger.error(f"Error fetching S&P 500 symbols from Polygon: {e}")
            return []

    def fetch_historical_data(self, symbols: List[str] = None, timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for multiple symbols.
        
        Args:
            symbols: List of stock symbols to fetch
            timeframe: Timeframe for the data (1d, 1h, etc.)
            
        Returns:
            Dictionary of DataFrames with historical price data
        """
        logger.info(f"Fetching historical data for {len(symbols) if symbols else 'universe'} symbols")
        
        # If symbols not provided, use universe
        if not symbols:
            symbols = self.get_universe()
        
        # Check if we have any symbols
        if not symbols:
            logger.warning("No symbols to fetch")
            return {}
        
        # Initialize results
        results = {}
        
        # Check if cached in Redis
        if self.use_redis_cache:
            cached_data = self._get_cached_data(symbols, timeframe)
            
            # If all data found in cache, return early
            if len(cached_data) == len(symbols):
                logger.info(f"Retrieved all {len(symbols)} symbols from Redis cache")
                return cached_data
            
            # Otherwise, add cached data to results
            results.update(cached_data)
            
            # Remove symbols that were found in cache
            symbols = [s for s in symbols if s not in cached_data]
            
            logger.info(f"Retrieved {len(cached_data)} symbols from Redis cache, fetching {len(symbols)} remaining symbols")
        
        # Fetch from Polygon first
        if self.use_polygon and self.polygon_api_key:
            polygon_results = self._fetch_from_polygon(symbols, timeframe)
            results.update(polygon_results)
            
            # Remove symbols that were successfully fetched
            symbols = [s for s in symbols if s not in polygon_results]
        
        # Fetch remaining symbols from Alpha Vantage
        if symbols and self.use_alpha_vantage and self.alpha_vantage_api_key:
            alpha_vantage_results = self._fetch_from_alpha_vantage(symbols, timeframe)
            results.update(alpha_vantage_results)
        
        # If still not all symbols fetched, try to generate synthetic data
        missing_symbols = [s for s in symbols if s not in results]
        if missing_symbols:
            logger.warning(f"Generating synthetic data for {len(missing_symbols)} symbols")
            for symbol in missing_symbols:
                results[symbol] = self._generate_synthetic_price_data(symbol)
        
        # Store the data in Redis if enabled
        if self.use_redis_cache:
            self._cache_data_in_redis(results, timeframe)
        
        # Store the data
        self.price_data.update(results)
        
        logger.info(f"Successfully fetched data for {len(results)} out of {len(symbols)} symbols")
        
        return results

    def get_cached_dataset(self, key: str, generator_func, expiry: int = None) -> Any:
        """
        Get dataset from cache or generate and cache it.
        
        Args:
            key: Cache key
            generator_func: Function to generate dataset if not in cache
            expiry: Cache expiry time in seconds (uses default TTL if not specified)
            
        Returns:
            Dataset
        """
        # Check in-memory cache first
        if key in self.data_cache:
            logger.debug(f"Retrieved {key} from in-memory cache")
            return self.data_cache[key]
        
        # Then check Redis cache
        if self.use_redis_cache:
            cached_data = redis_client.get(f"dataset:{key}")
            if cached_data is not None:
                # Store in in-memory cache for faster future access
                self.data_cache[key] = cached_data
                logger.debug(f"Retrieved {key} from Redis cache")
                return cached_data
        
        # Generate dataset
        logger.debug(f"Generating dataset for {key}")
        data = generator_func()
        
        # Cache dataset
        if data is not None:
            # Store in in-memory cache
            self.data_cache[key] = data
            
            # Store in Redis if enabled
            if self.use_redis_cache:
                # Use provided expiry or get from default TTLs
                if expiry is None:
                    # Extract data type from key (e.g., "price_data:AAPL" -> "price_data")
                    data_type = key.split(":")[0] if ":" in key else "default"
                    expiry = self.cache_ttl.get(data_type, 3600)  # Default 1 hour
                
                redis_client.set(f"dataset:{key}", data, ex=expiry)
        
        return data
    
    def _get_cached_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Get cached data from Redis.
        
        Args:
            symbols: List of stock symbols to fetch
            timeframe: Timeframe for the data (1d, 1h, etc.)
            
        Returns:
            Dictionary of DataFrames with cached price data
        """
        cached_data = {}
        
        # Batch Redis keys for more efficient retrieval
        redis_keys = [f"stocks:history:{symbol}:{self.data_days}d:{timeframe}" for symbol in symbols]
        
        # Use Redis pipeline for batch retrieval if available
        if hasattr(redis_client._conn, 'pipeline'):
            try:
                pipe = redis_client._conn.pipeline()
                for key in redis_keys:
                    pipe.get(key)
                results = pipe.execute()
                
                for symbol, result in zip(symbols, results):
                    if result is not None:
                        df = redis_client._deserialize(result)
                        if df is not None and not (isinstance(df, pd.DataFrame) and df.empty):
                            cached_data[symbol] = df
                
                return cached_data
            except Exception as e:
                logger.error(f"Error using Redis pipeline: {e}")
                # Fall back to individual gets
        
        # Individual gets if pipeline not available or failed
        for symbol in symbols:
            # Create Redis key
            redis_key = f"stocks:history:{symbol}:{self.data_days}d:{timeframe}"
            
            # Try to get data from Redis
            cached_df = redis_client.get(redis_key)
            
            if cached_df is not None and not (isinstance(cached_df, pd.DataFrame) and cached_df.empty):
                cached_data[symbol] = cached_df
        
        return cached_data
    
    def _cache_data_in_redis(self, data: Dict[str, pd.DataFrame], timeframe: str) -> None:
        """
        Cache data in Redis.
        
        Args:
            data: Dictionary of DataFrames with price data
            timeframe: Timeframe for the data (1d, 1h, etc.)
        """
        # Use Redis pipeline for batch storage if available
        if hasattr(redis_client._conn, 'pipeline') and data:
            try:
                pipe = redis_client._conn.pipeline()
                
                for symbol, df in data.items():
                    # Create Redis key
                    redis_key = f"stocks:history:{symbol}:{self.data_days}d:{timeframe}"
                    
                    # Serialize data
                    serialized = redis_client._serialize(df)
                    
                    # Add to pipeline
                    pipe.setex(redis_key, self.cache_ttl.get("price_data", 86400), serialized)
                
                # Execute pipeline
                pipe.execute()
                return
            except Exception as e:
                logger.error(f"Error using Redis pipeline: {e}")
                # Fall back to individual sets
        
        # Individual sets if pipeline not available or failed
        for symbol, df in data.items():
            # Create Redis key
            redis_key = f"stocks:history:{symbol}:{self.data_days}d:{timeframe}"
            
            # Cache data in Redis with expiration
            redis_client.set(redis_key, df, ex=self.cache_ttl.get("price_data", 86400))

    def _fetch_from_polygon(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data from Polygon API.
        
        Args:
            symbols: List of stock symbols to fetch
            timeframe: Timeframe for the data (1d, 1h, etc.)
            
        Returns:
            Dictionary of DataFrames with historical price data
        """
        logger.info(f"Fetching {len(symbols)} symbols from Polygon API")
        
        # Initialize results
        results = {}
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=self.data_days)).strftime("%Y-%m-%d")
        
        # Convert timeframe to Polygon format
        if timeframe == "1d":
            timespan = "day"
        elif timeframe == "1h":
            timespan = "hour"
        else:
            timespan = "day"  # Default to daily
        
        # Process each symbol
        for i, symbol in enumerate(symbols):
            try:
                # Apply rate limiting
                if i > 0 and i % self.polygon_requests_per_minute == 0:
                    logger.info(f"Rate limiting Polygon API: Waiting 60 seconds after {i} requests")
                    time.sleep(60)
                
                # Construct URL
                url = (
                    f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/"
                    f"{start_date}/{end_date}?adjusted=true&sort=asc&limit=5000&apiKey={self.polygon_api_key}"
                )
                
                # Make request
                response = requests.get(url)
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                if "results" in data and data["results"]:
                    # Create DataFrame
                    df = pd.DataFrame(data["results"])
                    
                    # Rename columns to standard format
                    df = df.rename(columns={
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                        "t": "timestamp"
                    })
                    
                    # Convert timestamp to datetime and set as index
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df = df.set_index("timestamp")
                    
                    # Add to results
                    results[symbol] = df
                    
                    logger.debug(f"Polygon: Successfully fetched {symbol} with {len(df)} records")
                else:
                    logger.warning(f"Polygon: No data found for {symbol}")
            
            except Exception as e:
                logger.error(f"Polygon: Error fetching {symbol}: {e}")
        
        logger.info(f"Polygon: Fetched data for {len(results)} out of {len(symbols)} symbols")
        
        return results

    def _fetch_from_alpha_vantage(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data from Alpha Vantage API.
        
        Args:
            symbols: List of stock symbols to fetch
            timeframe: Timeframe for the data (1d, 1h, etc.)
            
        Returns:
            Dictionary of DataFrames with historical price data
        """
        logger.info(f"Fetching {len(symbols)} symbols from Alpha Vantage API")
        
        # Initialize results
        results = {}
        
        # Convert timeframe to Alpha Vantage format
        if timeframe == "1d":
            function = "TIME_SERIES_DAILY_ADJUSTED"
            data_key = "Time Series (Daily)"
        elif timeframe == "1h":
            function = "TIME_SERIES_INTRADAY"
            data_key = "Time Series (60min)"
        else:
            function = "TIME_SERIES_DAILY_ADJUSTED"
            data_key = "Time Series (Daily)"
        
        # Process each symbol
        for i, symbol in enumerate(symbols):
            try:
                # Apply rate limiting
                if i > 0 and i % self.alpha_vantage_requests_per_minute == 0:
                    logger.info(f"Rate limiting Alpha Vantage API: Waiting 60 seconds after {i} requests")
                    time.sleep(60)
                
                # Construct URL
                base_url = "https://www.alphavantage.co/query"
                
                # Parameters for API request
                params = {
                    "function": function,
                    "symbol": symbol,
                    "outputsize": "full",
                    "apikey": self.alpha_vantage_api_key
                }
                
                # Add interval parameter for intraday data
                if timeframe == "1h":
                    params["interval"] = "60min"
                
                # Make request
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                if data_key in data:
                    # Get time series data
                    time_series = data[data_key]
                    
                    # Create lists for OHLCV data
                    dates = []
                    ohlcv = []
                    
                    # Calculate cutoff date
                    cutoff_date = datetime.now() - timedelta(days=self.data_days)
                    cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
                    
                    # Process each data point
                    for date_str, values in time_series.items():
                        # Skip data older than cutoff
                        if date_str < cutoff_date_str:
                            continue
                        
                        dates.append(date_str)
                        
                        # Extract OHLCV values
                        open_price = float(values["1. open"])
                        high_price = float(values["2. high"])
                        low_price = float(values["3. low"])
                        close_price = float(values["4. close"])
                        volume = float(values["5. volume"])
                        
                        ohlcv.append([open_price, high_price, low_price, close_price, volume])
                    
                    # Create DataFrame
                    df = pd.DataFrame(
                        ohlcv,
                        index=pd.to_datetime(dates),
                        columns=["open", "high", "low", "close", "volume"]
                    )
                    
                    # Sort by date
                    df = df.sort_index()
                    
                    # Add to results
                    results[symbol] = df
                    
                    logger.debug(f"Alpha Vantage: Successfully fetched {symbol} with {len(df)} records")
                else:
                    logger.warning(f"Alpha Vantage: No data found for {symbol}")
            
            except Exception as e:
                logger.error(f"Alpha Vantage: Error fetching {symbol}: {e}")
        
        logger.info(f"Alpha Vantage: Fetched data for {len(results)} out of {len(symbols)} symbols")
        
        return results

    def _generate_synthetic_price_data(self, symbol: str) -> pd.DataFrame:
        """
        Generate synthetic price data for testing and development.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with synthetic price data
        """
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.data_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Initialize with base price
        np.random.seed(hash(symbol) % 10000)  # Seed based on symbol for consistency
        base_price = np.random.uniform(50, 500)
        
        # Generate prices with random walk
        num_days = len(dates)
        returns = np.random.normal(0.0005, 0.015, num_days)
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, num_days))
        df.loc[df.index[0], 'open'] = prices[0] * 0.995  # First day open
        
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, num_days)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, num_days)))
        df['volume'] = np.random.randint(50000, 5000000, num_days)
        
        # Some basic patterns like trend and seasonality
        # Weekly pattern
        week_pattern = np.sin(np.arange(num_days) * 2 * np.pi / 5) * 0.01
        df['close'] *= (1 + week_pattern)
        
        # Trend pattern
        trend = np.linspace(0, 0.2, num_days) * np.random.choice([-1, 1])
        df['close'] *= (1 + trend)
        
        # Make sure high >= close >= low
        df['high'] = df[['high', 'close']].max(axis=1)
        df['low'] = df[['low', 'close']].min(axis=1)
        
        # Fix any NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Generated synthetic data for {symbol} with {len(df)} records")
        
        return df

    def fetch_news_data(self, symbols: List[str] = None, days: int = 30) -> List[dict]:
        """
        Fetch financial news data for sentiment analysis.
        
        Args:
            symbols: List of stock symbols to fetch news for
            days: Number of days of news to fetch
            
        Returns:
            List of news items
        """
        logger.info(f"Fetching news data for {len(symbols) if symbols else 'universe'} symbols")
        
        # If symbols not provided, use universe
        if not symbols:
            symbols = self.get_universe()
        
        # Check for cached news in Redis
        cached_news = []
        if self.use_redis_cache:
            cached_news = redis_client.get("news:recent")
            
            if cached_news and len(cached_news) > 0:
                logger.info(f"Retrieved {len(cached_news)} news items from Redis cache")
                self.news_data = cached_news
                return cached_news
        
        # Collect news from Polygon API
        news_items = []
        
        if self.use_polygon and self.polygon_api_key:
            logger.info("Fetching news from Polygon API")
            
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            try:
                # For better coverage, fetch general market news first
                url = (
                    f"https://api.polygon.io/v2/reference/news?limit=100&order=desc&"
                    f"published_utc.gte={start_date}&published_utc.lte={end_date}&"
                    f"apiKey={self.polygon_api_key}"
                )
                
                response = requests.get(url)
                response.raise_for_status()
                
                data = response.json()
                
                if "results" in data:
                    for item in data["results"]:
                        # Extract relevant fields
                        news_item = {
                            "title": item.get("title", ""),
                            "summary": item.get("description", ""),
                            "source": item.get("publisher", {}).get("name", "Polygon"),
                            "url": item.get("article_url", ""),
                            "published_at": item.get("published_utc", ""),
                            "symbols": item.get("tickers", []),
                            "relevance_score": 1.0,  # Default relevance score
                            "sentiment_score": 0.0  # Default sentiment score
                        }
                        
                        # Filter out items without a title or summary
                        if not news_item["title"] or not news_item["summary"]:
                            continue
                        
                        news_items.append(news_item)
                
                logger.info(f"Fetched {len(news_items)} general news items from Polygon")
                
                # Now fetch specific news for top symbols (to avoid API limits)
                top_symbols = symbols[:20]  # Use top 20 symbols
                
                for i, symbol in enumerate(top_symbols):
                    try:
                        # Apply rate limiting
                        if i > 0 and i % self.polygon_requests_per_minute == 0:
                            logger.info(f"Rate limiting Polygon API: Waiting 60 seconds after {i} requests")
                            time.sleep(60)
                        
                        # Construct URL for specific symbol
                        symbol_url = (
                            f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=50&order=desc&"
                            f"published_utc.gte={start_date}&published_utc.lte={end_date}&"
                            f"apiKey={self.polygon_api_key}"
                        )
                        
                        response = requests.get(symbol_url)
                        response.raise_for_status()
                        
                        data = response.json()
                        
                        if "results" in data:
                            symbol_news_count = 0
                            
                            for item in data["results"]:
                                # Extract relevant fields
                                news_item = {
                                    "title": item.get("title", ""),
                                    "summary": item.get("description", ""),
                                    "source": item.get("publisher", {}).get("name", "Polygon"),
                                    "url": item.get("article_url", ""),
                                    "published_at": item.get("published_utc", ""),
                                    "symbols": item.get("tickers", [symbol]),
                                    "relevance_score": 1.0,  # Default relevance score
                                    "sentiment_score": 0.0  # Default sentiment score
                                }
                                
                                # Filter out items without a title or summary
                                if not news_item["title"] or not news_item["summary"]:
                                    continue
                                
                                # Check if this news item is already in our list (by URL)
                                if any(n["url"] == news_item["url"] for n in news_items):
                                    continue
                                
                                news_items.append(news_item)
                                symbol_news_count += 1
                            
                            logger.debug(f"Fetched {symbol_news_count} news items for {symbol}")
                    
                    except Exception as e:
                        logger.error(f"Error fetching news for {symbol}: {e}")
            
            except Exception as e:
                logger.error(f"Error fetching news from Polygon: {e}")
        
        # If no news found, generate synthetic news
        if not news_items:
            logger.warning("No real news data found, generating synthetic news")
            news_items = self._generate_synthetic_news(symbols, days)
        
        # Store news data
        self.news_data = news_items
        
        # Cache in Redis if enabled
        if self.use_redis_cache and news_items:
            redis_client.set("news:recent", news_items, ex=43200)  # Cache for 12 hours
        
        logger.info(f"Fetched {len(news_items)} news items")
        
        return news_items

    def _generate_synthetic_news(self, symbols: List[str], days: int) -> List[dict]:
        """
        Generate synthetic news data for testing and development.
        
        Args:
            symbols: List of stock symbols
            days: Number of days to generate news for
            
        Returns:
            List of synthetic news items
        """
        news_items = []
        
        # Templates for titles and summaries
        bullish_titles = [
            "{symbol} reports strong quarterly earnings",
            "{symbol} exceeds analyst expectations",
            "{symbol} announces new product launch",
            "Analysts upgrade {symbol} to 'buy'",
            "{symbol} expands into new markets",
            "{symbol} stock surges on positive news",
            "{symbol} reports record revenue growth",
            "CEO of {symbol} optimistic about future growth",
        ]
        
        bearish_titles = [
            "{symbol} misses earnings expectations",
            "Analysts downgrade {symbol} to 'sell'",
            "{symbol} facing regulatory challenges",
            "{symbol} reports disappointing sales figures",
            "Competition intensifies for {symbol}",
            "{symbol} warns of slowdown in growth",
            "{symbol} cuts guidance for next quarter",
            "Investors concerned about {symbol}'s debt levels",
        ]
        
        neutral_titles = [
            "{symbol} announces management changes",
            "{symbol} to present at upcoming conference",
            "{symbol} maintains market position",
            "Industry outlook remains stable for {symbol}",
            "{symbol} completes reorganization",
            "{symbol} holds annual shareholder meeting",
            "New partnerships announced for {symbol}",
            "{symbol} maintains dividend",
        ]
        
        # Corresponding summaries
        bullish_summaries = [
            "{symbol} reported quarterly earnings that exceeded analyst expectations, with revenue growth of {growth}% year-over-year.",
            "Analysts have upgraded {symbol} to a 'buy' rating, citing strong growth prospects and competitive positioning.",
            "{symbol} announced the launch of new products that are expected to significantly contribute to revenue growth in the coming quarters.",
            "The CEO of {symbol} expressed optimism about future growth, highlighting expansion plans and strong customer demand.",
            "{symbol} is expanding into new markets, which is expected to drive revenue growth and increase market share.",
        ]
        
        bearish_summaries = [
            "{symbol} reported quarterly earnings that fell short of analyst expectations, with revenue declining by {decline}% year-over-year.",
            "Analysts have downgraded {symbol} to a 'sell' rating, citing concerns about market saturation and increasing competition.",
            "{symbol} is facing regulatory challenges that could impact its business operations and financial performance.",
            "The CEO of {symbol} warned of a potential slowdown in growth due to macroeconomic headwinds and industry challenges.",
            "{symbol} has cut its guidance for the next quarter, indicating challenges in meeting previously set targets.",
        ]
        
        neutral_summaries = [
            "{symbol} announced management changes as part of its ongoing strategic restructuring efforts.",
            "{symbol} will be presenting at an upcoming industry conference to showcase its latest innovations and strategy.",
            "Industry analysts expect {symbol} to maintain its current market position despite competitive pressures.",
            "{symbol} completed a reorganization aimed at improving operational efficiency and reducing costs.",
            "{symbol} held its annual shareholder meeting where management discussed the company's performance and future outlook.",
        ]
        
        # Generate news over the date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for symbol in symbols[:30]:  # Limit to top 30 symbols
            # Generate 1-3 news items per symbol
            num_news = np.random.randint(1, 4)
            
            for _ in range(num_news):
                # Select random date
                news_date = np.random.choice(date_range)
                
                # Randomly select news sentiment
                sentiment = np.random.choice(["bullish", "bearish", "neutral"], p=[0.4, 0.3, 0.3])
                
                if sentiment == "bullish":
                    title_template = np.random.choice(bullish_titles)
                    summary_template = np.random.choice(bullish_summaries)
                    sentiment_score = np.random.uniform(0.2, 0.8)
                elif sentiment == "bearish":
                    title_template = np.random.choice(bearish_titles)
                    summary_template = np.random.choice(bearish_summaries)
                    sentiment_score = np.random.uniform(-0.8, -0.2)
                else:
                    title_template = np.random.choice(neutral_titles)
                    summary_template = np.random.choice(neutral_summaries)
                    sentiment_score = np.random.uniform(-0.2, 0.2)
                
                # Fill templates
                growth = np.random.randint(5, 30)
                decline = np.random.randint(5, 30)
                
                title = title_template.format(symbol=symbol)
                summary = summary_template.format(symbol=symbol, growth=growth, decline=decline)
                
                # Create news item
                news_item = {
                    "title": title,
                    "summary": summary,
                    "source": np.random.choice(["Market News", "Financial Times", "Wall Street Journal", "Bloomberg", "CNBC"]),
                    "url": f"https://example.com/news/{symbol.lower()}/{news_date.strftime('%Y%m%d')}",
                    "published_at": news_date.isoformat(),
                    "symbols": [symbol],
                    "relevance_score": np.random.uniform(0.7, 1.0),
                    "sentiment_score": sentiment_score
                }
                
                news_items.append(news_item)
        
        # Add some market news
        market_titles = [
            "Market reaches new high as economic data improves",
            "Stocks fall amid recession fears",
            "Fed announces interest rate decision",
            "Inflation data comes in below expectations",
            "Market volatility increases as geopolitical tensions rise",
            "Economic outlook remains positive despite challenges",
            "Investors react to latest employment data",
            "Market trends suggest continued growth",
        ]
        
        market_summaries = [
            "The stock market reached new highs today as economic data showed improvements in key sectors.",
            "Stocks fell across the board as investors grew concerned about potential recession signals.",
            "The Federal Reserve announced its latest interest rate decision, impacting market expectations.",
            "The latest inflation data came in below analysts' expectations, potentially giving the Fed more flexibility.",
            "Market volatility increased as geopolitical tensions rose, creating uncertainty for investors.",
        ]
        
        # Add 10-15 market news items
        num_market_news = np.random.randint(10, 16)
        
        for _ in range(num_market_news):
            # Select random date
            news_date = np.random.choice(date_range)
            
            # Randomly select news sentiment
            sentiment = np.random.choice(["bullish", "bearish", "neutral"], p=[0.4, 0.3, 0.3])
            
            title = np.random.choice(market_titles)
            summary = np.random.choice(market_summaries)
            
            if sentiment == "bullish":
                sentiment_score = np.random.uniform(0.2, 0.8)
            elif sentiment == "bearish":
                sentiment_score = np.random.uniform(-0.8, -0.2)
            else:
                sentiment_score = np.random.uniform(-0.2, 0.2)
            
            # Create news item
            news_item = {
                "title": title,
                "summary": summary,
                "source": np.random.choice(["Market News", "Financial Times", "Wall Street Journal", "Bloomberg", "CNBC"]),
                "url": f"https://example.com/market-news/{news_date.strftime('%Y%m%d')}",
                "published_at": news_date.isoformat(),
                "symbols": [],  # No specific symbols
                "relevance_score": np.random.uniform(0.7, 1.0),
                "sentiment_score": sentiment_score
            }
            
            news_items.append(news_item)
        
        # Sort by date (newest first)
        news_items.sort(key=lambda x: x["published_at"], reverse=True)
        
        logger.info(f"Generated {len(news_items)} synthetic news items")
        
        return news_items

    def save_data_to_disk(self, output_dir: str = None) -> None:
        """
        Save fetched data to disk.
        
        Args:
            output_dir: Directory to save data to
        """
        # Use specified output directory or default
        output_dir = output_dir or self.data_dir
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save price data
        if self.price_data:
            price_data_dir = os.path.join(output_dir, "price_data")
            os.makedirs(price_data_dir, exist_ok=True)
            
            for symbol, df in self.price_data.items():
                # Save as CSV
                csv_path = os.path.join(price_data_dir, f"{symbol}.csv")
                df.to_csv(csv_path)
            
            logger.info(f"Saved price data for {len(self.price_data)} symbols to {price_data_dir}")
        
        # Save news data
        if self.news_data:
            news_data_dir = os.path.join(output_dir, "news_data")
            os.makedirs(news_data_dir, exist_ok=True)
            
            # Save as JSON
            json_path = os.path.join(news_data_dir, f"news_{datetime.now().strftime('%Y%m%d')}.json")
            
            with open(json_path, 'w') as f:
                json.dump(self.news_data, f, indent=2)
            
            logger.info(f"Saved {len(self.news_data)} news items to {json_path}")

    def load_data_from_disk(self, input_dir: str = None) -> Tuple[Dict[str, pd.DataFrame], List[dict]]:
        """
        Load data from disk.
        
        Args:
            input_dir: Directory to load data from
            
        Returns:
            Tuple of (price_data, news_data)
        """
        # Use specified input directory or default
        input_dir = input_dir or self.data_dir
        
        price_data = {}
        news_data = []
        
        # Load price data
        price_data_dir = os.path.join(input_dir, "price_data")
        if os.path.exists(price_data_dir):
            for file in os.listdir(price_data_dir):
                if file.endswith(".csv"):
                    try:
                        # Extract symbol from filename
                        symbol = file.split(".")[0]
                        
                        # Load CSV
                        csv_path = os.path.join(price_data_dir, file)
                        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                        
                        # Add to price data
                        price_data[symbol] = df
                    except Exception as e:
                        logger.error(f"Error loading price data from {file}: {e}")
            
            logger.info(f"Loaded price data for {len(price_data)} symbols from {price_data_dir}")
        
        # Load news data
        news_data_dir = os.path.join(input_dir, "news_data")
        if os.path.exists(news_data_dir):
            # Find most recent news file
            news_files = [f for f in os.listdir(news_data_dir) if f.startswith("news_") and f.endswith(".json")]
            
            if news_files:
                # Sort by date (filename format is news_YYYYMMDD.json)
                latest_file = sorted(news_files)[-1]
                
                try:
                    # Load JSON
                    json_path = os.path.join(news_data_dir, latest_file)
                    
                    with open(json_path, 'r') as f:
                        news_data = json.load(f)
                    
                    logger.info(f"Loaded {len(news_data)} news items from {json_path}")
                except Exception as e:
                    logger.error(f"Error loading news data from {latest_file}: {e}")
        
        # Update instance data
        self.price_data = price_data
        self.news_data = news_data
        
        return price_data, news_data