"""
Alpha Vantage API client for fetching market data, fundamentals, and news.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import pandas as pd
import requests

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

logger = setup_logger("alpha_vantage")

class AlphaVantageAPI:
    """
    Client for Alpha Vantage APIs.
    
    Provides methods for:
    - Market data (prices, indicators)
    - Fundamental data
    - News and sentiment analysis
    - Economic indicators
    """
    
    # Base URL for Alpha Vantage API
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self):
        """Initialize the Alpha Vantage API client with API key from settings."""
        self.api_key = settings.api.alpha_vantage_api_key
        self.session = None
        
        # Rate limiting parameters
        self.rate_limit = 150  # requests per minute (premium tier)
        self.calls_made = 0
        self.reset_time = datetime.now() + timedelta(minutes=1)
        
        logger.info("Alpha Vantage API client initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session.
        
        Returns:
            aiohttp.ClientSession: The session
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def _make_request(self, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Make a request to Alpha Vantage API with rate limiting.
        
        Args:
            params: Request parameters
            
        Returns:
            Response data if successful, None otherwise
        """
        # Apply rate limiting
        now = datetime.now()
        if now >= self.reset_time:
            # Reset counter
            self.calls_made = 0
            self.reset_time = now + timedelta(minutes=1)
        
        if self.calls_made >= self.rate_limit:
            wait_time = (self.reset_time - now).total_seconds()
            logger.warning(f"Rate limit reached. Waiting {wait_time:.2f} seconds.")
            await asyncio.sleep(wait_time)
            # Reset after waiting
            self.calls_made = 0
            self.reset_time = datetime.now() + timedelta(minutes=1)
        
        # Add API key to parameters
        params['apikey'] = self.api_key
        
        try:
            # Get session
            session = await self._get_session()
            
            # Make request
            async with session.get(self.BASE_URL, params=params) as response:
                self.calls_made += 1
                
                if response.status != 200:
                    logger.error(f"Error {response.status} from Alpha Vantage: {await response.text()}")
                    return None
                
                try:
                    data = await response.json()
                except aiohttp.ContentTypeError:
                    # Try to parse as text if JSON fails
                    text = await response.text()
                    logger.error(f"Invalid JSON response: {text[:200]}...")
                    return None
                
                # Check for error messages
                if "Error Message" in data:
                    logger.error(f"Alpha Vantage API error: {data['Error Message']}")
                    return None
                
                if "Note" in data and "API call frequency" in data["Note"]:
                    logger.warning(f"Alpha Vantage rate limit warning: {data['Note']}")
                
                return data
        except Exception as e:
            logger.error(f"Error making request to Alpha Vantage: {e}")
            return None
    
    # ---------- Market Data Methods ----------
    
    async def get_daily_prices(self, symbol: str, outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """
        Get daily price data for a symbol.
        
        Args:
            symbol: Stock symbol
            outputsize: 'compact' for last 100 days, 'full' for 20+ years
            
        Returns:
            DataFrame of daily prices if successful, None otherwise
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
        }
        
        data = await self._make_request(params)
        if not data or "Time Series (Daily)" not in data:
            return None
        
        # Convert to DataFrame
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Convert column names
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert data types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Cache in Redis (1 hour expiry)
        redis_client.set(f"stocks:daily:{symbol}", df, expiry=3600)
        
        return df
    
    async def get_intraday_prices(
        self, 
        symbol: str, 
        interval: str = '1min', 
        outputsize: str = 'compact'
    ) -> Optional[pd.DataFrame]:
        """
        Get intraday price data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            outputsize: 'compact' for last 100 candles, 'full' for 2000+ candles
            
        Returns:
            DataFrame of intraday prices if successful, None otherwise
        """
        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
        }
        
        data = await self._make_request(params)
        if not data or f"Time Series ({interval})" not in data:
            return None
        
        # Convert to DataFrame
        time_series = data[f"Time Series ({interval})"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        
        # Convert column names
        df.columns = [col.split('. ')[1] for col in df.columns]
        
        # Convert data types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Cache in Redis (30 min expiry)
        redis_client.set(f"stocks:intraday:{symbol}:{interval}", df, expiry=1800)
        
        return df
    
    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Quote data if successful, None otherwise
        """
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
        }
        
        data = await self._make_request(params)
        if not data or "Global Quote" not in data:
            return None
        
        quote = data["Global Quote"]
        
        # Format the quote data
        formatted_quote = {
            'symbol': quote.get('01. symbol'),
            'price': float(quote.get('05. price', 0)),
            'change': float(quote.get('09. change', 0)),
            'change_percent': float(quote.get('10. change percent', '0%').rstrip('%')),
            'volume': int(quote.get('06. volume', 0)),
            'latest_trading_day': quote.get('07. latest trading day'),
        }
        
        # Cache in Redis (5 min expiry)
        redis_client.set_stock_data(symbol, formatted_quote, 'quote')
        
        return formatted_quote
    
    # ---------- Technical Indicators Methods ----------
    
    async def get_technical_indicator(
        self, 
        symbol: str, 
        indicator: str, 
        interval: str = 'daily',
        time_period: int = 20, 
        series_type: str = 'close',
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """
        Get technical indicator data for a symbol.
        
        Args:
            symbol: Stock symbol
            indicator: Indicator function (e.g., 'SMA', 'EMA', 'RSI')
            interval: Time interval ('daily', '1min', '5min', etc.)
            time_period: Number of data points to calculate the indicator
            series_type: Price series to use ('close', 'open', 'high', 'low')
            **kwargs: Additional parameters for specific indicators
            
        Returns:
            DataFrame of indicator data if successful, None otherwise
        """
        params = {
            'function': indicator,
            'symbol': symbol,
            'interval': interval,
            'time_period': str(time_period),
            'series_type': series_type,
        }
        
        # Add additional parameters
        for key, value in kwargs.items():
            params[key] = str(value)
        
        data = await self._make_request(params)
        if not data or "Technical Analysis" not in data:
            return None
        
        # Convert to DataFrame
        technical_data = data["Technical Analysis: " + indicator]
        df = pd.DataFrame.from_dict(technical_data, orient='index')
        
        # Convert data types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
        
        # Sort by date
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Cache in Redis (1 hour expiry)
        redis_key = f"stocks:indicator:{symbol}:{indicator}:{interval}:{time_period}"
        redis_client.set(redis_key, df, expiry=3600)
        
        return df
    
    async def get_rsi(
        self, 
        symbol: str, 
        interval: str = 'daily',
        time_period: int = 14,
        series_type: str = 'close'
    ) -> Optional[pd.DataFrame]:
        """
        Get Relative Strength Index (RSI) data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval ('daily', '1min', '5min', etc.)
            time_period: Number of data points to calculate RSI
            series_type: Price series to use ('close', 'open', 'high', 'low')
            
        Returns:
            DataFrame of RSI data if successful, None otherwise
        """
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator='RSI',
            interval=interval,
            time_period=time_period,
            series_type=series_type
        )
    
    async def get_macd(
        self, 
        symbol: str, 
        interval: str = 'daily',
        series_type: str = 'close',
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> Optional[pd.DataFrame]:
        """
        Get Moving Average Convergence Divergence (MACD) data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval ('daily', '1min', '5min', etc.)
            series_type: Price series to use ('close', 'open', 'high', 'low')
            fastperiod: Fast period
            slowperiod: Slow period
            signalperiod: Signal period
            
        Returns:
            DataFrame of MACD data if successful, None otherwise
        """
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator='MACD',
            interval=interval,
            series_type=series_type,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
    
    async def get_bollinger_bands(
        self, 
        symbol: str, 
        interval: str = 'daily',
        time_period: int = 20,
        series_type: str = 'close',
        nbdevup: int = 2,
        nbdevdn: int = 2
    ) -> Optional[pd.DataFrame]:
        """
        Get Bollinger Bands data for a symbol.
        
        Args:
            symbol: Stock symbol
            interval: Time interval ('daily', '1min', '5min', etc.)
            time_period: Number of data points for the moving average
            series_type: Price series to use ('close', 'open', 'high', 'low')
            nbdevup: Standard deviation multiplier for upper band
            nbdevdn: Standard deviation multiplier for lower band
            
        Returns:
            DataFrame of Bollinger Bands data if successful, None otherwise
        """
        return await self.get_technical_indicator(
            symbol=symbol,
            indicator='BBANDS',
            interval=interval,
            time_period=time_period,
            series_type=series_type,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )
    
    # ---------- Fundamental Data Methods ----------
    
    async def get_company_overview(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company overview data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Company overview data if successful, None otherwise
        """
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol,
        }
        
        data = await self._make_request(params)
        if not data or "Symbol" not in data:
            return None
        
        # Cache in Redis (24 hour expiry - fundamentals don't change often)
        redis_client.set_stock_data(symbol, data, 'fundamentals')
        
        return data
    
    async def get_income_statement(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get income statement data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Income statement data if successful, None otherwise
        """
        params = {
            'function': 'INCOME_STATEMENT',
            'symbol': symbol,
        }
        
        data = await self._make_request(params)
        if not data or "annualReports" not in data:
            return None
        
        # Cache in Redis (24 hour expiry)
        redis_client.set(f"stocks:financials:{symbol}:income", data, expiry=86400)
        
        return data
    
    async def get_balance_sheet(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get balance sheet data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Balance sheet data if successful, None otherwise
        """
        params = {
            'function': 'BALANCE_SHEET',
            'symbol': symbol,
        }
        
        data = await self._make_request(params)
        if not data or "annualReports" not in data:
            return None
        
        # Cache in Redis (24 hour expiry)
        redis_client.set(f"stocks:financials:{symbol}:balance", data, expiry=86400)
        
        return data
    
    async def get_cash_flow(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get cash flow data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Cash flow data if successful, None otherwise
        """
        params = {
            'function': 'CASH_FLOW',
            'symbol': symbol,
        }
        
        data = await self._make_request(params)
        if not data or "annualReports" not in data:
            return None
        
        # Cache in Redis (24 hour expiry)
        redis_client.set(f"stocks:financials:{symbol}:cashflow", data, expiry=86400)
        
        return data
    
    # ---------- News & Sentiment Methods ----------
    
    async def get_news_sentiment(
        self, 
        symbols: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        limit: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Get news and sentiment data.
        
        Args:
            symbols: Optional list of stock symbols
            topics: Optional list of topics
            time_from: Start time (YYYYMMDDTHHMM format)
            time_to: End time (YYYYMMDDTHHMM format)
            limit: Maximum number of news items to return
            
        Returns:
            News and sentiment data if successful, None otherwise
        """
        params = {
            'function': 'NEWS_SENTIMENT',
        }
        
        # Add optional parameters
        if symbols:
            params['tickers'] = ','.join(symbols)
        
        if topics:
            params['topics'] = ','.join(topics)
        
        if time_from:
            params['time_from'] = time_from
        
        if time_to:
            params['time_to'] = time_to
        
        if limit:
            params['limit'] = str(limit)
        
        data = await self._make_request(params)
        if not data or "feed" not in data:
            return None
        
        # Cache in Redis (15 min expiry for news)
        cache_key = "news:sentiment"
        if symbols:
            cache_key += f":{','.join(symbols)}"
        
        redis_client.set(cache_key, data, expiry=900)
        
        return data
    
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get news for a specific symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of news items to return
            
        Returns:
            List of news items if successful, None otherwise
        """
        data = await self.get_news_sentiment(symbols=[symbol], limit=limit)
        if not data or "feed" not in data:
            return None
        
        news_items = data["feed"]
        
        # Format news items
        formatted_news = []
        for item in news_items:
            # Check if this news item is relevant to the symbol
            ticker_sentiments = item.get("ticker_sentiment", [])
            is_relevant = False
            relevance_score = 0
            sentiment_score = 0
            
            for ticker_sentiment in ticker_sentiments:
                if ticker_sentiment.get("ticker") == symbol:
                    is_relevant = True
                    relevance_score = float(ticker_sentiment.get("relevance_score", 0))
                    sentiment_score = float(ticker_sentiment.get("ticker_sentiment_score", 0))
                    break
            
            if is_relevant and relevance_score > 0.2:  # Only include somewhat relevant news
                formatted_news.append({
                    'title': item.get("title"),
                    'summary': item.get("summary"),
                    'url': item.get("url"),
                    'time_published': item.get("time_published"),
                    'authors': item.get("authors", []),
                    'relevance_score': relevance_score,
                    'sentiment_score': sentiment_score,
                    'overall_sentiment_score': float(item.get("overall_sentiment_score", 0)),
                })
        
        # Sort by relevance and recency
        formatted_news.sort(key=lambda x: (x['relevance_score'], x.get('time_published', '')), reverse=True)
        
        # Cache in Redis (15 min expiry)
        redis_client.set(f"stocks:news:{symbol}", formatted_news, expiry=900)
        
        return formatted_news
    
    # ---------- Economic Indicators Methods ----------
    
    async def get_economic_indicator(self, indicator: str) -> Optional[pd.DataFrame]:
        """
        Get economic indicator data.
        
        Args:
            indicator: Economic indicator function
                - 'REAL_GDP' - Real GDP
                - 'INFLATION' - Inflation (CPI)
                - 'UNEMPLOYMENT' - Unemployment rate
                - 'RETAIL_SALES' - Retail sales
                - 'TREASURY_YIELD' - Treasury yield
            
        Returns:
            DataFrame of economic indicator data if successful, None otherwise
        """
        params = {
            'function': indicator,
        }
        
        data = await self._make_request(params)
        if not data or "data" not in data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data["data"])
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Convert value column to numeric
        df['value'] = pd.to_numeric(df['value'])
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Cache in Redis (6 hour expiry for economic data)
        redis_client.set(f"economic:{indicator.lower()}", df, expiry=21600)
        
        return df
    
    async def get_real_gdp(self) -> Optional[pd.DataFrame]:
        """
        Get real GDP data.
        
        Returns:
            DataFrame of real GDP data if successful, None otherwise
        """
        return await self.get_economic_indicator('REAL_GDP')
    
    async def get_inflation(self) -> Optional[pd.DataFrame]:
        """
        Get inflation (CPI) data.
        
        Returns:
            DataFrame of inflation data if successful, None otherwise
        """
        return await self.get_economic_indicator('CPI')
    
    async def get_unemployment(self) -> Optional[pd.DataFrame]:
        """
        Get unemployment rate data.
        
        Returns:
            DataFrame of unemployment rate data if successful, None otherwise
        """
        return await self.get_economic_indicator('UNEMPLOYMENT')
    
    async def get_treasury_yield(self, interval: str = 'daily', maturity: str = '10year') -> Optional[pd.DataFrame]:
        """
        Get Treasury yield data.
        
        Args:
            interval: Time interval ('daily', 'weekly', 'monthly')
            maturity: Bond maturity ('3month', '2year', '5year', '10year', '30year')
            
        Returns:
            DataFrame of Treasury yield data if successful, None otherwise
        """
        params = {
            'function': 'TREASURY_YIELD',
            'interval': interval,
            'maturity': maturity,
        }
        
        data = await self._make_request(params)
        if not data or "data" not in data:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(data["data"])
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'])
        
        # Set date as index
        df.set_index('date', inplace=True)
        
        # Convert value column to numeric
        df['value'] = pd.to_numeric(df['value'])
        
        # Sort by date
        df.sort_index(inplace=True)
        
        # Cache in Redis (6 hour expiry)
        redis_key = f"economic:treasury_yield:{interval}:{maturity}"
        redis_client.set(redis_key, df, expiry=21600)
        
        return df
    
    # ---------- Sector Performance Methods ----------
    
    async def get_sector_performance(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get performance data for different sectors.
        
        Returns:
            Sector performance data if successful, None otherwise
        """
        params = {
            'function': 'SECTOR',
        }
        
        data = await self._make_request(params)
        if not data or len(data) <= 1:  # Metadata only
            return None
        
        # Process sector data
        sectors = {}
        
        # Skip metadata field
        for time_range, sector_data in data.items():
            if time_range == "Meta Data":
                continue
                
            time_label = time_range.replace("Rank ", "")
            sectors[time_label] = {}
            
            for sector, performance in sector_data.items():
                # Convert string percentage to float
                perf_value = float(performance.rstrip('%'))
                sectors[time_label][sector] = perf_value
        
        # Cache in Redis (1 hour expiry)
        redis_client.set("market:sectors", sectors, expiry=3600)
        
        return sectors
    
    # ---------- Cleanup ----------
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Alpha Vantage API session closed")

# Create global Alpha Vantage client instance
alpha_vantage_client = AlphaVantageAPI()