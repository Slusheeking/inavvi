"""
Data fetching module for the trading system.

This module provides functionality for fetching financial data:
- Historical OHLCV data from Polygon and Alpha Vantage
- Financial news data from various sources
- Market data for sentiment analysis and other models
- Data persistence in Redis and TimescaleDB
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import json
import asyncio
import redis
import psycopg2
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_fetching.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("data_fetcher")

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
load_dotenv()

# Import project modules
from src.config.settings import settings
from src.utils.redis_client import redis_client


class DataFetcher:
    """
    Data fetching system for the trading system ML models.
    
    Features:
    - Connects to multiple data sources
    - Fetches historical price data
    - Obtains financial news data
    - Cleans and structures data for model training
    - Caches data in Redis for faster access
    """
    
    def __init__(
        self,
        data_days: int = 365,
        use_polygon: bool = True,
        use_alpha_vantage: bool = True,
        use_redis_cache: bool = True,
        data_dir: str = None
    ):
        """
        Initialize the data fetcher.
        
        Args:
            data_days: Number of days of historical data to fetch
            use_polygon: Whether to use Polygon.io API
            use_alpha_vantage: Whether to use Alpha Vantage API
            use_redis_cache: Whether to use Redis for caching data
            data_dir: Directory to store downloaded data files
        """
        # Configuration
        self.data_days = data_days
        self.use_polygon = use_polygon
        self.use_alpha_vantage = use_alpha_vantage
        self.use_redis_cache = use_redis_cache
        
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
        
        # API rate limiting settings
        self.polygon_requests_per_minute = 5  # Adjust based on your subscription
        self.alpha_vantage_requests_per_minute = 5  # Adjust based on your subscription
        
        # Check for valid API keys
        if self.use_polygon and not self.polygon_api_key:
            logger.warning("Polygon API key not found. Disabling Polygon data source.")
            self.use_polygon = False
        
        if self.use_alpha_vantage and not self.alpha_vantage_api_key:
            logger.warning("Alpha Vantage API key not found. Disabling Alpha Vantage data source.")
            self.use_alpha_vantage = False
        
        logger.info(f"DataFetcher initialized with {data_days} days of historical data")
        logger.info(f"Data sources: Polygon={self.use_polygon}, Alpha Vantage={self.use_alpha_vantage}")

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
            # Telecom
            "T", "VZ", "TMUS", "CMCSA", "DIS", "NFLX", "CHTR",
            # ETFs
            "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLP", "XLY", "XLB", "XLU"
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
        
        # Store the data in Redis if enabled
        if self.use_redis_cache:
            self._cache_data_in_redis(results, timeframe)
        
        # Store the data
        self.price_data = results
        
        logger.info(f"Successfully fetched data for {len(results)} out of {len(symbols) + len(results)} symbols")
        
        return results

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
        
        for symbol in symbols:
            # Create Redis key
            redis_key = f"stocks:history:{symbol}:{self.data_days}d:{timeframe}"
            
            # Try to get data from Redis
            df = redis_client.get(redis_key)
            
            if df is not None and not df.empty:
                cached_data[symbol] = df
        
        return cached_data

    def _cache_data_in_redis(self, data: Dict[str, pd.DataFrame], timeframe: str) -> None:
        """
        Cache data in Redis.
        
        Args:
            data: Dictionary of DataFrames with price data
            timeframe: Timeframe for the data (1d, 1h, etc.)
        """
        for symbol, df in data.items():
            # Create Redis key
            redis_key = f"stocks:history:{symbol}:{self.data_days}d:{timeframe}"
            
            # Cache data in Redis with 24 hour expiration
            redis_client.set(redis_key, df, ex=86400)

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
                # Apply rate limiting (Alpha Vantage has strict limits)
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
        polygon_news = self._fetch_news_from_polygon(symbols, days)
        
        # Store news data
        self.news_data = polygon_news
        
        # Cache in Redis if enabled
        if self.use_redis_cache and polygon_news:
            redis_client.set("news:recent", polygon_news, ex=43200)  # Cache for 12 hours
        
        logger.info(f"Fetched {len(polygon_news)} news items")
        
        return polygon_news

    def _fetch_news_from_polygon(self, symbols: List[str], days: int) -> List[dict]:
        """
        Fetch news data from Polygon API.
        
        Args:
            symbols: List of stock symbols to fetch news for
            days: Number of days of news to fetch
            
        Returns:
            List of news items
        """
        if not self.use_polygon or not self.polygon_api_key:
            return []
        
        news_items = []
        
        # Calculate date range
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        try:
            # For better coverage, fetch general market news first
            url = (
                f"https://api.polygon.io/v2/reference/news?limit=1000&order=desc&"
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
                        "relevance_score": 1.0  # Default relevance score
                    }
                    
                    # Filter out items without a title or summary
                    if not news_item["title"] or not news_item["summary"]:
                        continue
                    
                    # Add sentiment score placeholder (will be filled by sentiment model)
                    news_item["sentiment_score"] = 0.0
                    
                    news_items.append(news_item)
            
            logger.info(f"Polygon: Fetched {len(news_items)} general news items")
            
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
                                "relevance_score": 1.0  # Default relevance score
                            }
                            
                            # Filter out items without a title or summary
                            if not news_item["title"] or not news_item["summary"]:
                                continue
                            
                            # Check if this news item is already in our list (by URL)
                            if any(n["url"] == news_item["url"] for n in news_items):
                                continue
                            
                            # Add sentiment score placeholder (will be filled by sentiment model)
                            news_item["sentiment_score"] = 0.0
                            
                            news_items.append(news_item)
                            symbol_news_count += 1
                        
                        logger.debug(f"Polygon: Fetched {symbol_news_count} news items for {symbol}")
                
                except Exception as e:
                    logger.error(f"Polygon: Error fetching news for {symbol}: {e}")
            
        except Exception as e:
            logger.error(f"Polygon: Error fetching news: {e}")
        
        logger.info(f"Polygon: Fetched a total of {len(news_items)} news items")
        
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


def fetch_data(symbols: List[str] = None, timeframe: str = "1d", data_days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical price data using the DataFetcher class.

    Args:
        symbols: List of stock symbols to fetch.
        timeframe: Timeframe for the data (1d, 1h, etc.).
        data_days: Number of days of historical data to fetch.

    Returns:
        Dictionary of DataFrames with historical price data.
    """
    fetcher = DataFetcher(data_days=data_days)
    if not symbols:
        symbols = fetcher.get_universe() # Get default universe if none provided
    return fetcher.fetch_historical_data(symbols=symbols, timeframe=timeframe)


def main():
    """
    Main function to demonstrate data fetching.
    """
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch financial data for model training")
    
    parser.add_argument("--days", type=int, default=365,
                       help="Number of days of historical data to fetch")
    
    parser.add_argument("--symbols", type=str, default="",
                       help="Comma-separated list of symbols to fetch")
    
    parser.add_argument("--timeframe", type=str, default="1d",
                       help="Timeframe for the data (1d, 1h)")
    
    parser.add_argument("--news", action="store_true",
                       help="Fetch news data")
    
    parser.add_argument("--news-days", type=int, default=30,
                       help="Number of days of news to fetch")
    
    parser.add_argument("--save", action="store_true",
                       help="Save data to disk")
    
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Directory to save data to")
    
    args = parser.parse_args()
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(
        data_days=args.days,
        use_polygon=True,
        use_alpha_vantage=True,
        use_redis_cache=True,
        data_dir=args.output_dir
    )
    
    # Get symbols
    symbols = args.symbols.split(",") if args.symbols else None
    
    if not symbols:
        # Get universe
        symbols = data_fetcher.get_universe()
    
    # Fetch historical data
    price_data = data_fetcher.fetch_historical_data(
        symbols=symbols,
        timeframe=args.timeframe
    )
    
    # Print summary of price data
    print(f"\nPrice Data Summary:")
    print(f"Total symbols: {len(price_data)}")
    
    if price_data:
        # Print a few examples
        for symbol in list(price_data.keys())[:3]:
            df = price_data[symbol]
            print(f"\n{symbol} data ({len(df)} records):")
            print(df.head())
    
    # Fetch news data if requested
    if args.news:
        news_data = data_fetcher.fetch_news_data(
            symbols=symbols,
            days=args.news_days
        )
        
        # Print summary of news data
        print(f"\nNews Data Summary:")
        print(f"Total news items: {len(news_data)}")
        
        if news_data:
            # Print a few examples
            for i, item in enumerate(news_data[:3]):
                print(f"\nNews item {i+1}:")
                print(f"Title: {item['title']}")
                print(f"Source: {item['source']}")
                print(f"Date: {item['published_at']}")
                print(f"Symbols: {item['symbols']}")
                print(f"Summary: {item['summary'][:100]}...")
    
    # Save data if requested
    if args.save:
        data_fetcher.save_data_to_disk()


if __name__ == "__main__":
    main()
