"""
Alpha Vantage API client for fetching market data, fundamentals, and news.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiohttp
import numpy as np
import pandas as pd
import requests

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

logger = setup_logger("alpha_vantage")

# Define custom exceptions for Alpha Vantage errors
class AlphaVantageDataError(Exception):
    """Custom exception for errors during Alpha Vantage data fetching."""
    pass

class AlphaVantageRateLimitError(AlphaVantageDataError):
    """Custom exception for Alpha Vantage rate limit errors."""
    pass

class AlphaVantageSymbolNotFoundError(AlphaVantageDataError):
    """Custom exception for symbol not found errors."""
    pass

class AlphaVantageInvalidParameterError(AlphaVantageDataError):
    """Custom exception for invalid parameter errors."""
    pass

# Simple retry decorator for Alpha Vantage requests
import functools

def retry_av(attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except (AlphaVantageRateLimitError, aiohttp.ClientError) as e:
                    if attempt < attempts - 1:
                        logger.warning(f"Attempt {attempt + 1}/{attempts} failed for {func.__name__}. Retrying in {delay} seconds...")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {attempts} attempts failed for {func.__name__}.")
                        raise e
                except Exception as e:
                    logger.error(f"An unexpected error occurred during {func.__name__}: {e}")
                    raise e
            return None # Should not reach here if attempts > 0
        return wrapper
    return decorator

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
    
    async def _make_request(self, params: Dict[str, str]) -> Dict[str, Any]:
        """
        Make a request to Alpha Vantage API with rate limiting.
        
        Args:
            params: Request parameters
            
        Returns:
            Response data if successful
            
        Raises:
            AlphaVantageRateLimitError: If rate limit is exceeded
            AlphaVantageSymbolNotFoundError: If symbol is not found
            AlphaVantageInvalidParameterError: If parameters are invalid
            AlphaVantageDataError: For other Alpha Vantage errors
            aiohttp.ClientError: For network-related errors
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
                    error_text = await response.text()
                    logger.error(f"Error {response.status} from Alpha Vantage: {error_text}")
                    
                    if response.status == 429:
                        raise AlphaVantageRateLimitError(f"Rate limit exceeded: {error_text}")
                    elif response.status == 404:
                        raise AlphaVantageSymbolNotFoundError(f"Resource not found: {error_text}")
                    elif response.status == 400:
                        raise AlphaVantageInvalidParameterError(f"Invalid parameters: {error_text}")
                    else:
                        raise AlphaVantageDataError(f"HTTP error {response.status}: {error_text}")
                
                try:
                    data = await response.json()
                except aiohttp.ContentTypeError:
                    # Try to parse as text if JSON fails
                    text = await response.text()
                    logger.error(f"Invalid JSON response: {text[:200]}...")
                    raise AlphaVantageDataError(f"Invalid JSON response: {text[:200]}...")
                
                # Check for error messages
                if "Error Message" in data:
                    error_msg = data['Error Message']
                    logger.error(f"Alpha Vantage API error: {error_msg}")
                    
                    if "Invalid API call" in error_msg:
                        raise AlphaVantageInvalidParameterError(f"Invalid API call: {error_msg}")
                    elif "not found" in error_msg.lower() or "no data" in error_msg.lower():
                        raise AlphaVantageSymbolNotFoundError(f"Symbol not found: {error_msg}")
                    else:
                        raise AlphaVantageDataError(f"API error: {error_msg}")
                
                if "Note" in data and "API call frequency" in data["Note"]:
                    note = data["Note"]
                    logger.warning(f"Alpha Vantage rate limit warning: {note}")
                    
                    # If this is the only field in the response, it's a rate limit error
                    if len(data) == 1 or (len(data) == 2 and "Meta Data" in data):
                        raise AlphaVantageRateLimitError(f"Rate limit exceeded: {note}")
                
                return data
                
        except (AlphaVantageDataError, AlphaVantageRateLimitError, 
                AlphaVantageSymbolNotFoundError, AlphaVantageInvalidParameterError, 
                aiohttp.ClientError) as e:
            # Let these exceptions propagate for retry handling
            raise e
        except Exception as e:
            logger.error(f"Unexpected error making request to Alpha Vantage: {e}")
            raise AlphaVantageDataError(f"Unexpected error: {str(e)}") from e
    
    # ---------- Market Data Methods ----------
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "Time Series (Daily)" not in data:
                logger.warning(f"No daily price data found for {symbol}")
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
            
            # Data validation - check for missing values
            if df.isnull().values.any():
                logger.warning(f"Missing values detected in daily price data for {symbol}")
                # Fill missing values with forward fill then backward fill
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Cache in Redis (1 hour expiry)
            redis_client.set(f"stocks:daily:{symbol}", df, expiry=3600)
            
            return df
            
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if f"Time Series ({interval})" not in data:
                logger.warning(f"No intraday data found for {symbol} with interval {interval}")
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
            
            # Data validation - check for missing values
            if df.isnull().values.any():
                logger.warning(f"Missing values detected in intraday data for {symbol}")
                # Fill missing values with forward fill then backward fill
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Cache in Redis (30 min expiry)
            redis_client.set(f"stocks:intraday:{symbol}:{interval}", df, expiry=1800)
            
            return df
            
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except AlphaVantageInvalidParameterError as e:
            logger.error(f"Invalid parameter for intraday prices: {e}")
            # If it's an invalid interval, we should not retry
            if "interval" in str(e).lower():
                return None
            raise e
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "Global Quote" not in data:
                logger.warning(f"No quote data found for {symbol}")
                return None
            
            quote = data["Global Quote"]
            
            # Validate quote data
            if not quote or len(quote) == 0:
                logger.warning(f"Empty quote data for {symbol}")
                return None
            
            # Format the quote data with error handling
            try:
                formatted_quote = {
                    'symbol': quote.get('01. symbol', symbol),
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': float(quote.get('10. change percent', '0%').rstrip('%')),
                    'volume': int(quote.get('06. volume', 0)),
                    'latest_trading_day': quote.get('07. latest trading day'),
                }
                
                # Cache in Redis (5 min expiry)
                redis_client.set_stock_data(symbol, formatted_quote, 'quote')
                
                return formatted_quote
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing quote data for {symbol}: {e}")
                logger.debug(f"Raw quote data: {quote}")
                raise AlphaVantageDataError(f"Error parsing quote data for {symbol}: {e}") from e
                
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    # ---------- Technical Indicators Methods ----------
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            # Check for valid technical indicator data
            if "Technical Analysis" not in data:
                logger.warning(f"No technical indicator data found for {symbol} with indicator {indicator}")
                return None
            
            # The key includes the indicator name
            technical_key = "Technical Analysis: " + indicator
            if technical_key not in data:
                logger.warning(f"Technical indicator data format unexpected for {symbol} with indicator {indicator}")
                return None
                
            technical_data = data[technical_key]
            
            # Check if we have any data points
            if not technical_data or len(technical_data) == 0:
                logger.warning(f"Empty technical indicator data for {symbol} with indicator {indicator}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(technical_data, orient='index')
            
            # Convert data types with error handling
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error converting column {col} to numeric for {symbol} indicator {indicator}: {e}")
                    # Try to clean the data before conversion
                    df[col] = df[col].replace('', np.nan).replace('-', np.nan)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by date
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Data validation - check for missing values
            if df.isnull().values.any():
                logger.warning(f"Missing values detected in technical indicator data for {symbol}")
                # Fill missing values with forward fill then backward fill
                df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Cache in Redis (1 hour expiry)
            redis_key = f"stocks:indicator:{symbol}:{indicator}:{interval}:{time_period}"
            redis_client.set(redis_key, df, expiry=3600)
            
            return df
            
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except AlphaVantageInvalidParameterError as e:
            logger.error(f"Invalid parameter for technical indicator: {e}")
            # If it's an invalid indicator or interval, we should not retry
            if "function" in str(e).lower() or "interval" in str(e).lower():
                return None
            raise e
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
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
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "Symbol" not in data:
                logger.warning(f"No company overview data found for {symbol}")
                return None
            
            # Validate that we have the expected symbol
            if data.get("Symbol") != symbol:
                logger.warning(f"Symbol mismatch in company overview data: requested {symbol}, got {data.get('Symbol')}")
            
            # Data validation - check for essential fields
            essential_fields = ["Name", "Industry", "Sector", "MarketCapitalization"]
            missing_fields = [field for field in essential_fields if field not in data or not data[field]]
            
            if missing_fields:
                logger.warning(f"Missing essential fields in company overview for {symbol}: {missing_fields}")
            
            # Cache in Redis (24 hour expiry - fundamentals don't change often)
            redis_client.set_stock_data(symbol, data, 'fundamentals')
            
            return data
            
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "annualReports" not in data:
                logger.warning(f"No income statement data found for {symbol}")
                return None
            
            # Validate that we have actual reports
            if not data["annualReports"] or len(data["annualReports"]) == 0:
                logger.warning(f"Empty income statement data for {symbol}")
                return None
            
            # Data validation - check for essential fields in the most recent report
            latest_report = data["annualReports"][0]
            essential_fields = ["fiscalDateEnding", "totalRevenue", "netIncome"]
            missing_fields = [field for field in essential_fields if field not in latest_report or not latest_report[field]]
            
            if missing_fields:
                logger.warning(f"Missing essential fields in income statement for {symbol}: {missing_fields}")
            
            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:financials:{symbol}:income", data, expiry=86400)
            
            return data
            
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "annualReports" not in data:
                logger.warning(f"No balance sheet data found for {symbol}")
                return None
            
            # Validate that we have actual reports
            if not data["annualReports"] or len(data["annualReports"]) == 0:
                logger.warning(f"Empty balance sheet data for {symbol}")
                return None
            
            # Data validation - check for essential fields in the most recent report
            latest_report = data["annualReports"][0]
            essential_fields = ["fiscalDateEnding", "totalAssets", "totalLiabilities", "totalShareholderEquity"]
            missing_fields = [field for field in essential_fields if field not in latest_report or not latest_report[field]]
            
            if missing_fields:
                logger.warning(f"Missing essential fields in balance sheet for {symbol}: {missing_fields}")
            
            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:financials:{symbol}:balance", data, expiry=86400)
            
            return data
            
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "annualReports" not in data:
                logger.warning(f"No cash flow data found for {symbol}")
                return None
            
            # Validate that we have actual reports
            if not data["annualReports"] or len(data["annualReports"]) == 0:
                logger.warning(f"Empty cash flow data for {symbol}")
                return None
            
            # Data validation - check for essential fields in the most recent report
            latest_report = data["annualReports"][0]
            essential_fields = ["fiscalDateEnding", "operatingCashflow", "cashflowFromInvestment", "cashflowFromFinancing"]
            missing_fields = [field for field in essential_fields if field not in latest_report or not latest_report[field]]
            
            if missing_fields:
                logger.warning(f"Missing essential fields in cash flow for {symbol}: {missing_fields}")
            
            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:financials:{symbol}:cashflow", data, expiry=86400)
            
            return data
            
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    # ---------- News & Sentiment Methods ----------
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "feed" not in data:
                logger.warning(f"No news sentiment data found for the specified parameters")
                return None
            
            # Validate that we have actual news items
            if not data["feed"] or len(data["feed"]) == 0:
                logger.warning(f"Empty news feed returned")
                return None
            
            # Cache in Redis (15 min expiry for news)
            cache_key = "news:sentiment"
            if symbols:
                cache_key += f":{','.join(symbols)}"
            
            redis_client.set(cache_key, data, expiry=900)
            
            return data
            
        except AlphaVantageInvalidParameterError as e:
            logger.error(f"Invalid parameter for news sentiment: {e}")
            # If it's an invalid parameter, we should not retry
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    @retry_av(attempts=3, delay=5)
    async def get_symbol_news(self, symbol: str, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get news for a specific symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of news items to return
            
        Returns:
            List of news items if successful, None otherwise
        """
        try:
            data = await self.get_news_sentiment(symbols=[symbol], limit=limit)
            
            if not data or "feed" not in data:
                logger.warning(f"No news data found for {symbol}")
                return None
            
            news_items = data["feed"]
            
            # Format news items
            formatted_news = []
            for item in news_items:
                try:
                    # Check if this news item is relevant to the symbol
                    ticker_sentiments = item.get("ticker_sentiment", [])
                    is_relevant = False
                    relevance_score = 0
                    sentiment_score = 0
                    
                    for ticker_sentiment in ticker_sentiments:
                        if ticker_sentiment.get("ticker") == symbol:
                            is_relevant = True
                            # Handle potential non-numeric values
                            try:
                                relevance_score = float(ticker_sentiment.get("relevance_score", 0))
                                sentiment_score = float(ticker_sentiment.get("ticker_sentiment_score", 0))
                            except (ValueError, TypeError):
                                logger.warning(f"Non-numeric sentiment scores for {symbol} in news item")
                                relevance_score = 0
                                sentiment_score = 0
                            break
                    
                    if is_relevant and relevance_score > 0.2:  # Only include somewhat relevant news
                        # Handle potential missing fields with defaults
                        try:
                            overall_sentiment = float(item.get("overall_sentiment_score", 0))
                        except (ValueError, TypeError):
                            overall_sentiment = 0
                            
                        formatted_news.append({
                            'title': item.get("title", "No title"),
                            'summary': item.get("summary", "No summary available"),
                            'url': item.get("url", ""),
                            'time_published': item.get("time_published", ""),
                            'authors': item.get("authors", []),
                            'relevance_score': relevance_score,
                            'sentiment_score': sentiment_score,
                            'overall_sentiment_score': overall_sentiment,
                        })
                except Exception as e:
                    logger.warning(f"Error processing news item for {symbol}: {e}")
                    continue
            
            # Sort by relevance and recency
            formatted_news.sort(key=lambda x: (x['relevance_score'], x.get('time_published', '')), reverse=True)
            
            # Cache in Redis (15 min expiry)
            redis_client.set(f"stocks:news:{symbol}", formatted_news, expiry=900)
            
            return formatted_news
            
        except AlphaVantageSymbolNotFoundError:
            logger.warning(f"Symbol not found: {symbol}")
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    # ---------- Economic Indicators Methods ----------
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "data" not in data:
                logger.warning(f"No economic indicator data found for {indicator}")
                return None
            
            # Validate that we have actual data points
            if not data["data"] or len(data["data"]) == 0:
                logger.warning(f"Empty economic indicator data for {indicator}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data["data"])
            
            # Data validation - check for required columns
            if 'date' not in df.columns or 'value' not in df.columns:
                logger.warning(f"Missing required columns in economic indicator data for {indicator}")
                return None
            
            # Convert date column with error handling
            try:
                df['date'] = pd.to_datetime(df['date'])
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting date column for {indicator}: {e}")
                # Try to clean the date column
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Drop rows with invalid dates
                df = df.dropna(subset=['date'])
                
                if df.empty:
                    logger.error(f"No valid dates in economic indicator data for {indicator}")
                    return None
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Convert value column to numeric with error handling
            try:
                df['value'] = pd.to_numeric(df['value'])
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting value column for {indicator}: {e}")
                # Try to clean the value column
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                # Fill missing values with forward fill then backward fill
                df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')
                
                if df['value'].isnull().all():
                    logger.error(f"No valid values in economic indicator data for {indicator}")
                    return None
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Cache in Redis (6 hour expiry for economic data)
            redis_client.set(f"economic:{indicator.lower()}", df, expiry=21600)
            
            return df
            
        except AlphaVantageInvalidParameterError as e:
            logger.error(f"Invalid parameter for economic indicator: {e}")
            # If it's an invalid indicator, we should not retry
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
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
    
    @retry_av(attempts=3, delay=5)
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
        
        try:
            data = await self._make_request(params)
            
            if "data" not in data:
                logger.warning(f"No treasury yield data found for {maturity} with interval {interval}")
                return None
            
            # Validate that we have actual data points
            if not data["data"] or len(data["data"]) == 0:
                logger.warning(f"Empty treasury yield data for {maturity} with interval {interval}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data["data"])
            
            # Data validation - check for required columns
            if 'date' not in df.columns or 'value' not in df.columns:
                logger.warning(f"Missing required columns in treasury yield data for {maturity}")
                return None
            
            # Convert date column with error handling
            try:
                df['date'] = pd.to_datetime(df['date'])
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting date column for treasury yield {maturity}: {e}")
                # Try to clean the date column
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Drop rows with invalid dates
                df = df.dropna(subset=['date'])
                
                if df.empty:
                    logger.error(f"No valid dates in treasury yield data for {maturity}")
                    return None
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            # Convert value column to numeric with error handling
            try:
                df['value'] = pd.to_numeric(df['value'])
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting value column for treasury yield {maturity}: {e}")
                # Try to clean the value column
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                # Fill missing values with forward fill then backward fill
                df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')
                
                if df['value'].isnull().all():
                    logger.error(f"No valid values in treasury yield data for {maturity}")
                    return None
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Cache in Redis (6 hour expiry)
            redis_key = f"economic:treasury_yield:{interval}:{maturity}"
            redis_client.set(redis_key, df, expiry=21600)
            
            return df
            
        except AlphaVantageInvalidParameterError as e:
            logger.error(f"Invalid parameter for treasury yield: {e}")
            # If it's an invalid interval or maturity, we should not retry
            if "interval" in str(e).lower() or "maturity" in str(e).lower():
                return None
            raise e
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    # ---------- Sector Performance Methods ----------
    
    @retry_av(attempts=3, delay=5)
    async def get_sector_performance(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Get performance data for different sectors.
        
        Returns:
            Sector performance data if successful, None otherwise
        """
        params = {
            'function': 'SECTOR',
        }
        
        try:
            data = await self._make_request(params)
            
            if not data or len(data) <= 1:  # Metadata only
                logger.warning("No sector performance data found")
                return None
            
            # Process sector data
            sectors = {}
            
            # Skip metadata field
            for time_range, sector_data in data.items():
                if time_range == "Meta Data":
                    continue
                    
                time_label = time_range.replace("Rank ", "")
                sectors[time_label] = {}
                
                if not isinstance(sector_data, dict):
                    logger.warning(f"Invalid sector data format for {time_range}")
                    continue
                
                for sector, performance in sector_data.items():
                    try:
                        # Convert string percentage to float with error handling
                        if not isinstance(performance, str):
                            logger.warning(f"Non-string performance value for {sector}: {performance}")
                            continue
                            
                        # Remove '%' and convert to float
                        perf_value = float(performance.rstrip('%'))
                        sectors[time_label][sector] = perf_value
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error converting performance value for {sector}: {e}")
                        continue
            
            # Validate that we have actual data
            if not sectors or all(len(time_data) == 0 for time_data in sectors.values()):
                logger.warning("No valid sector performance data after processing")
                return None
            
            # Cache in Redis (1 hour expiry)
            redis_client.set("market:sectors", sectors, expiry=3600)
            
            return sectors
            
        except AlphaVantageInvalidParameterError as e:
            logger.error(f"Invalid parameter for sector performance: {e}")
            return None
        except (AlphaVantageRateLimitError, AlphaVantageDataError, aiohttp.ClientError) as e:
            # These will be handled by the retry decorator
            raise e
    
    # ---------- Cleanup ----------
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Alpha Vantage API session closed")

# Create global Alpha Vantage client instance
alpha_vantage_client = AlphaVantageAPI()

# High-level function for compatibility with unified API
async def fetch_alpha_vantage_data(symbol: str, interval: str = 'daily', **kwargs) -> Dict[str, Any]:
    """
    Fetch data from Alpha Vantage API.
    
    Args:
        symbol: Stock symbol
        interval: Time interval ('daily' or '1min', '5min', '15min', '30min', '60min')
        **kwargs: Additional parameters
            
    Returns:
        Dict containing the fetched market data
    """
    try:
        # Determine if this is intraday or daily data
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            df = await alpha_vantage_client.get_intraday_prices(symbol, interval, kwargs.get('outputsize', 'compact'))
        else:
            df = await alpha_vantage_client.get_daily_prices(symbol, kwargs.get('outputsize', 'compact'))
            
        if df is None:
            logger.warning(f"No data returned from Alpha Vantage for {symbol}")
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
        logger.error(f"Error fetching data from Alpha Vantage: {e}")
        raise AlphaVantageDataError(f"Failed to fetch data: {str(e)}") from e
