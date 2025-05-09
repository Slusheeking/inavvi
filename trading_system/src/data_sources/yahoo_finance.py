"""
Yahoo Finance API client for fetching market data and financial information.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import yfinance as yf

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

logger = setup_logger("yahoo_finance")

class YahooFinanceAPI:
    """
    Client for Yahoo Finance data.
    
    Provides methods for:
    - Stock data
    - Financial statements
    - Analyst recommendations
    - Institutional holdings
    - Options data
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance API client."""
        self.rate_limit_delay = 0.2  # Seconds between requests to avoid rate limiting
        
        logger.info("Yahoo Finance API client initialized")
    
    async def _run_in_threadpool(self, func, *args, **kwargs):
        """
        Run synchronous Yahoo Finance functions asynchronously in a thread pool.
        
        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            lambda: func(*args, **kwargs)
        )
    
    # ---------- Stock Data Methods ----------
    
    async def get_ticker_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Ticker information if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = await self._run_in_threadpool(getattr, ticker, 'info')
            
            if not info:
                logger.warning(f"No info found for {symbol}")
                return None
            
            # Cache in Redis (6 hour expiry)
            redis_client.set_stock_data(symbol, info, 'info')
            
            return info
        except Exception as e:
            logger.error(f"Error fetching ticker info for {symbol}: {e}")
            return None
    
    async def get_historical_prices(
        self, 
        symbol: str, 
        period: str = '1mo', 
        interval: str = '1d'
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data.
        
        Args:
            symbol: Stock symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame of historical prices if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = await self._run_in_threadpool(ticker.history, period=period, interval=interval)
            
            if hist.empty:
                logger.warning(f"No historical data found for {symbol}")
                return None
            
            # Cache in Redis (1 hour expiry)
            redis_key = f"stocks:history:{symbol}:{period}:{interval}"
            redis_client.set(redis_key, hist, expiry=3600)
            
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {e}")
            return None
    
    async def get_intraday_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get today's intraday price data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame of intraday prices if successful, None otherwise
        """
        # For intraday, we use 1m interval for the past 5 days (max allowed)
        # and then filter for today
        try:
            ticker = yf.Ticker(symbol)
            hist = await self._run_in_threadpool(ticker.history, period='5d', interval='1m')
            
            if hist.empty:
                logger.warning(f"No intraday data found for {symbol}")
                return None
            
            # Filter for today
            today = pd.Timestamp.now().floor('D')
            intraday = hist[hist.index >= today]
            
            if intraday.empty:
                logger.warning(f"No today's intraday data found for {symbol}")
                return None
            
            # Cache in Redis (5 minute expiry)
            redis_key = f"stocks:intraday:{symbol}"
            redis_client.set(redis_key, intraday, expiry=300)
            
            return intraday
        except Exception as e:
            logger.error(f"Error fetching intraday prices for {symbol}: {e}")
            return None
    
    # ---------- Financial Data Methods ----------
    
    async def get_financials(self, symbol: str) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get financial statements (income statement, balance sheet, cash flow).
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of financial statements if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get financial statements
            income_stmt = await self._run_in_threadpool(getattr, ticker, 'income_stmt')
            balance_sheet = await self._run_in_threadpool(getattr, ticker, 'balance_sheet')
            cash_flow = await self._run_in_threadpool(getattr, ticker, 'cashflow')
            
            # Combine into a dictionary
            financials = {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
            
            # Check if any data was returned
            if all(df.empty for df in financials.values()):
                logger.warning(f"No financial data found for {symbol}")
                return None
            
            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:financials:{symbol}", financials, expiry=86400)
            
            return financials
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return None
    
    async def get_earnings(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get earnings data including estimates and upcoming earnings date.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Earnings data if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get earnings data
            earnings = await self._run_in_threadpool(getattr, ticker, 'earnings')
            earnings_dates = await self._run_in_threadpool(getattr, ticker, 'earnings_dates')
            calendar = await self._run_in_threadpool(getattr, ticker, 'calendar')
            
            # Combine into a dictionary
            earnings_data = {
                'earnings': earnings,
                'earnings_dates': earnings_dates,
                'calendar': calendar
            }
            
            # Cache in Redis (6 hour expiry)
            redis_client.set(f"stocks:earnings:{symbol}", earnings_data, expiry=21600)
            
            return earnings_data
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return None
    
    # ---------- Analyst Data Methods ----------
    
    async def get_analyst_recommendations(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get analyst recommendations.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame of analyst recommendations if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            recommendations = await self._run_in_threadpool(getattr, ticker, 'recommendations')
            
            if recommendations is None or recommendations.empty:
                logger.warning(f"No analyst recommendations found for {symbol}")
                return None
            
            # Cache in Redis (6 hour expiry)
            redis_client.set(f"stocks:recommendations:{symbol}", recommendations, expiry=21600)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error fetching analyst recommendations for {symbol}: {e}")
            return None
    
    async def get_analyst_price_targets(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get analyst price targets.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary of price targets if successful, None otherwise
        """
        try:
            # Get ticker info which includes price targets
            info = await self.get_ticker_info(symbol)
            if not info:
                return None
            
            # Extract price target information
            price_targets = {
                'current_price': info.get('currentPrice', 0),
                'target_low': info.get('targetLowPrice', 0),
                'target_mean': info.get('targetMeanPrice', 0),
                'target_median': info.get('targetMedianPrice', 0),
                'target_high': info.get('targetHighPrice', 0),
                'recommendation': info.get('recommendationKey', 'unknown'),
                'num_analyst_opinions': info.get('numberOfAnalystOpinions', 0)
            }
            
            # Cache in Redis (6 hour expiry)
            redis_client.set(f"stocks:price_targets:{symbol}", price_targets, expiry=21600)
            
            return price_targets
        except Exception as e:
            logger.error(f"Error fetching analyst price targets for {symbol}: {e}")
            return None
    
    # ---------- Institutional Data Methods ----------
    
    async def get_institutional_holders(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get institutional holders.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame of institutional holders if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            holders = await self._run_in_threadpool(getattr, ticker, 'institutional_holders')
            
            if holders is None or holders.empty:
                logger.warning(f"No institutional holders found for {symbol}")
                return None
            
            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:institutional_holders:{symbol}", holders, expiry=86400)
            
            return holders
        except Exception as e:
            logger.error(f"Error fetching institutional holders for {symbol}: {e}")
            return None
    
    async def get_major_holders(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get major holders breakdown.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame of major holders if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            holders = await self._run_in_threadpool(getattr, ticker, 'major_holders')
            
            if holders is None or holders.empty:
                logger.warning(f"No major holders found for {symbol}")
                return None
            
            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:major_holders:{symbol}", holders, expiry=86400)
            
            return holders
        except Exception as e:
            logger.error(f"Error fetching major holders for {symbol}: {e}")
            return None
    
    # ---------- Options Data Methods ----------
    
    async def get_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Get options chain data.
        
        Args:
            symbol: Stock symbol
            expiration_date: Options expiration date (YYYY-MM-DD format)
            
        Returns:
            Dictionary with 'calls' and 'puts' DataFrames if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates if none provided
            if expiration_date is None:
                expirations = await self._run_in_threadpool(getattr, ticker, 'options')
                if not expirations:
                    logger.warning(f"No options expirations found for {symbol}")
                    return None
                expiration_date = expirations[0]  # Use the nearest expiration
            
            # Get options chain for the expiration date
            options = await self._run_in_threadpool(ticker.option_chain, expiration_date)
            
            if options is None:
                logger.warning(f"No options data found for {symbol} expiring on {expiration_date}")
                return None
            
            # Extract calls and puts
            chain = {
                'calls': options.calls,
                'puts': options.puts
            }
            
            # Cache in Redis (1 hour expiry)
            redis_key = f"stocks:options:{symbol}:{expiration_date}"
            redis_client.set(redis_key, chain, expiry=3600)
            
            return chain
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return None
    
    async def get_option_expirations(self, symbol: str) -> Optional[List[str]]:
        """
        Get available option expiration dates.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of expiration dates if successful, None otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = await self._run_in_threadpool(getattr, ticker, 'options')
            
            if not expirations:
                logger.warning(f"No options expirations found for {symbol}")
                return None
            
            # Cache in Redis (1 hour expiry)
            redis_key = f"stocks:option_expirations:{symbol}"
            redis_client.set(redis_key, expirations, expiry=3600)
            
            return expirations
        except Exception as e:
            logger.error(f"Error fetching option expirations for {symbol}: {e}")
            return None
    
    # ---------- Screening Methods ----------
    
    async def screen_stocks(self, criteria: Dict[str, Any], limit: int = 100) -> List[str]:
        """
        Screen stocks based on criteria.
        
        Args:
            criteria: Dictionary of screening criteria
            limit: Maximum number of results to return
            
        Returns:
            List of matching stock symbols
        """
        # This is a simplified implementation as yfinance doesn't have a built-in screener
        # We'll fetch data for a pre-defined universe and filter
        try:
            # Get universe from Redis or use a default list
            universe = redis_client.get("stocks:universe")
            if not universe:
                # Use a default list of common stocks
                universe = await self._get_default_universe()
            
            # List to store matching symbols
            matches = []
            
            # Process stocks in batches to avoid rate limiting
            batch_size = 20
            for i in range(0, len(universe), batch_size):
                batch = universe[i:i+batch_size]
                symbols = [stock['symbol'] if isinstance(stock, dict) else stock for stock in batch]
                
                # Get info for each symbol
                for symbol in symbols:
                    # Add a small delay to avoid rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    # Get ticker info
                    info = await self.get_ticker_info(symbol)
                    if not info:
                        continue
                    
                    # Check if it matches criteria
                    if self._matches_criteria(info, criteria):
                        matches.append(symbol)
                        
                        # Stop if limit reached
                        if len(matches) >= limit:
                            break
                
                # Stop if limit reached
                if len(matches) >= limit:
                    break
            
            logger.info(f"Screened {len(matches)} stocks matching criteria")
            return matches
        except Exception as e:
            logger.error(f"Error screening stocks: {e}")
            return []
    
    async def _get_default_universe(self) -> List[str]:
        """
        Get a default universe of stocks.
        
        Returns:
            List of stock symbols
        """
        # For simplicity, use S&P 500 components
        try:
            sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            symbols = sp500['Symbol'].tolist()
            return symbols
        except Exception as e:
            logger.error(f"Error fetching default universe: {e}")
            # Return a small set of liquid stocks as fallback
            return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'BAC', 'V']
    
    def _matches_criteria(self, info: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """
        Check if a stock matches the given criteria.
        
        Args:
            info: Stock information
            criteria: Dictionary of screening criteria
            
        Returns:
            True if the stock matches all criteria, False otherwise
        """
        for key, value in criteria.items():
            if key not in info:
                return False
            
            if isinstance(value, dict):
                # Handle range criteria
                if 'min' in value and info[key] < value['min']:
                    return False
                if 'max' in value and info[key] > value['max']:
                    return False
            elif info[key] != value:
                return False
        
        return True

# Create global Yahoo Finance client instance
yahoo_finance_client = YahooFinanceAPI()