"""
Yahoo Finance API client for fetching market data and financial information.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

# Check if this file is being run directly
if __name__ == "__main__":
    print("Yahoo Finance API client module initialized successfully.")
    print("This module provides interfaces to the Yahoo Finance API for market data.")
    print("\nTo use this module, import it in your code:")
    print("from src.data_sources.yahoo_finance import yahoo_finance_client, fetch_yahoo_finance_data")
    print("\nExample usage:")
    print("data = await fetch_yahoo_finance_data('AAPL', interval='1d')")
    sys.exit(0)

import pandas as pd
import yfinance as yf

from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

logger = setup_logger("yahoo_finance")


# Define custom exceptions for Yahoo Finance errors
class YahooFinanceDataError(Exception):
    """Custom exception for errors during Yahoo Finance data fetching."""

    pass


class YahooFinanceRateLimitError(YahooFinanceDataError):
    """Custom exception for Yahoo Finance rate limit errors."""

    pass


class YahooFinanceSymbolNotFoundError(YahooFinanceDataError):
    """Custom exception for symbol not found errors."""

    pass


class YahooFinanceInvalidIntervalError(YahooFinanceDataError):
    """Custom exception for invalid interval errors."""

    pass


# Simple retry decorator (can be enhanced with backoff strategies)
import functools


def retry(attempts=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except (YahooFinanceRateLimitError, YahooFinanceDataError) as e:
                    if attempt < attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{attempts} failed for {func.__name__}. Retrying in {delay} seconds..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {attempts} attempts failed for {func.__name__}.")
                        raise e
                except Exception as e:
                    logger.error(f"An unexpected error occurred during {func.__name__}: {e}")
                    raise e

        return wrapper

    return decorator


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
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    # ---------- Stock Data Methods ----------

    @retry(attempts=3, delay=5)
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
            info = await self._run_in_threadpool(getattr, ticker, "info")

            if not info:
                logger.warning(f"No info found for {symbol}")
                return None

            # Cache in Redis (6 hour expiry)
            redis_client.set_stock_data(symbol, info, "info")

            return info
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching ticker info for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching ticker info for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching ticker info for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching ticker info for {symbol}: {e}"
            ) from e

    @retry(attempts=3, delay=5)
    async def get_historical_prices(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
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
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching historical prices for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "invalid interval" in str(e).lower():
                raise YahooFinanceInvalidIntervalError(
                    f"Invalid interval provided for {symbol}: {e}"
                ) from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching historical prices for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching historical prices for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching historical prices for {symbol}: {e}"
            ) from e

    @retry(attempts=3, delay=5)
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
            hist = await self._run_in_threadpool(ticker.history, period="5d", interval="1m")

            if hist.empty:
                logger.warning(f"No intraday data found for {symbol}")
                return None

            # Filter for today
            today = pd.Timestamp.now().floor("D")
            intraday = hist[hist.index >= today]

            if intraday.empty:
                logger.warning(f"No today's intraday data found for {symbol}")
                return None

            # Cache in Redis (5 minute expiry)
            redis_key = f"stocks:intraday:{symbol}"
            redis_client.set(redis_key, intraday, expiry=300)

            return intraday
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching intraday prices for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "invalid interval" in str(e).lower():
                raise YahooFinanceInvalidIntervalError(
                    f"Invalid interval provided for {symbol}: {e}"
                ) from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching intraday prices for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching intraday prices for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching intraday prices for {symbol}: {e}"
            ) from e

    # ---------- Financial Data Methods ----------

    @retry(attempts=3, delay=10)
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
            income_stmt = await self._run_in_threadpool(getattr, ticker, "income_stmt")
            balance_sheet = await self._run_in_threadpool(getattr, ticker, "balance_sheet")
            cash_flow = await self._run_in_threadpool(getattr, ticker, "cashflow")

            # Combine into a dictionary
            financials = {
                "income_statement": income_stmt,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
            }

            # Check if any data was returned
            if all(df.empty for df in financials.values()):
                logger.warning(f"No financial data found for {symbol}")
                return None

            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:financials:{symbol}", financials, expiry=86400)

            return financials
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching financials for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching financials for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching financials for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching financials for {symbol}: {e}"
            ) from e

    @retry(attempts=3, delay=10)
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
            earnings = await self._run_in_threadpool(getattr, ticker, "earnings")
            earnings_dates = await self._run_in_threadpool(getattr, ticker, "earnings_dates")
            calendar = await self._run_in_threadpool(getattr, ticker, "calendar")

            # Combine into a dictionary
            earnings_data = {
                "earnings": earnings,
                "earnings_dates": earnings_dates,
                "calendar": calendar,
            }

            # Cache in Redis (6 hour expiry)
            redis_client.set(f"stocks:earnings:{symbol}", earnings_data, expiry=21600)

            return earnings_data
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching earnings for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching earnings for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching earnings for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching earnings for {symbol}: {e}"
            ) from e

    # ---------- Analyst Data Methods ----------

    @retry(attempts=3, delay=5)
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
            recommendations = await self._run_in_threadpool(getattr, ticker, "recommendations")

            if recommendations is None or recommendations.empty:
                logger.warning(f"No analyst recommendations found for {symbol}")
                return None

            # Cache in Redis (6 hour expiry)
            redis_client.set(f"stocks:recommendations:{symbol}", recommendations, expiry=21600)

            return recommendations
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching analyst recommendations for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching analyst recommendations for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching analyst recommendations for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching analyst recommendations for {symbol}: {e}"
            ) from e

    @retry(attempts=3, delay=5)
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
                "current_price": info.get("currentPrice", 0),
                "target_low": info.get("targetLowPrice", 0),
                "target_mean": info.get("targetMeanPrice", 0),
                "target_median": info.get("targetMedianPrice", 0),
                "target_high": info.get("targetHighPrice", 0),
                "recommendation": info.get("recommendationKey", "unknown"),
                "num_analyst_opinions": info.get("numberOfAnalystOpinions", 0),
            }

            # Cache in Redis (6 hour expiry)
            redis_client.set(f"stocks:price_targets:{symbol}", price_targets, expiry=21600)

            return price_targets
        except (
            YahooFinanceDataError,
            YahooFinanceRateLimitError,
            YahooFinanceSymbolNotFoundError,
        ) as e:
            # Re-raise specific exceptions from get_ticker_info
            logger.error(f"Error fetching ticker info for analyst price targets for {symbol}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error fetching analyst price targets for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching analyst price targets for {symbol}: {e}"
            ) from e

    # ---------- Institutional Data Methods ----------

    @retry(attempts=3, delay=10)
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
            holders = await self._run_in_threadpool(getattr, ticker, "institutional_holders")

            if holders is None or holders.empty:
                logger.warning(f"No institutional holders found for {symbol}")
                return None

            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:institutional_holders:{symbol}", holders, expiry=86400)

            return holders
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching institutional holders for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching institutional holders for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching institutional holders for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching institutional holders for {symbol}: {e}"
            ) from e

    @retry(attempts=3, delay=10)
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
            holders = await self._run_in_threadpool(getattr, ticker, "major_holders")

            if holders is None or holders.empty:
                logger.warning(f"No major holders found for {symbol}")
                return None

            # Cache in Redis (24 hour expiry)
            redis_client.set(f"stocks:major_holders:{symbol}", holders, expiry=86400)

            return holders
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching major holders for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching major holders for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching major holders for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching major holders for {symbol}: {e}"
            ) from e

    # ---------- Options Data Methods ----------

    @retry(attempts=3, delay=5)
    async def get_options_chain(
        self, symbol: str, expiration_date: Optional[str] = None
    ) -> Optional[Dict[str, pd.DataFrame]]:
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
                expirations = await self._run_in_threadpool(getattr, ticker, "options")
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
            chain = {"calls": options.calls, "puts": options.puts}

            # Cache in Redis (1 hour expiry)
            redis_key = f"stocks:options:{symbol}:{expiration_date}"
            redis_client.set(redis_key, chain, expiry=3600)

            return chain
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching options chain for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching options chain for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching options chain for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching options chain for {symbol}: {e}"
            ) from e

    @retry(attempts=3, delay=5)
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
            expirations = await self._run_in_threadpool(getattr, ticker, "options")

            if not expirations:
                logger.warning(f"No options expirations found for {symbol}")
                return None

            # Cache in Redis (1 hour expiry)
            redis_key = f"stocks:option_expirations:{symbol}"
            redis_client.set(redis_key, expirations, expiry=3600)

            return expirations
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching option expirations for {symbol}: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(f"Rate limit exceeded for {symbol}: {e}") from e
            elif "cannot find or fetch ticker" in str(e).lower():
                raise YahooFinanceSymbolNotFoundError(
                    f"Symbol not found or cannot be fetched: {symbol}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching option expirations for {symbol}: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching option expirations for {symbol}: {e}")
            raise YahooFinanceDataError(
                f"Unexpected error fetching option expirations for {symbol}: {e}"
            ) from e

    # ---------- Screening Methods ----------

    @retry(attempts=3, delay=10)
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

            # Process stocks concurrently in batches
            batch_size = 50  # Increased batch size for concurrent processing
            for i in range(0, len(universe), batch_size):
                batch = universe[i : i + batch_size]
                symbols = [stock["symbol"] if isinstance(stock, dict) else stock for stock in batch]

                # Create tasks for concurrent fetching of ticker info
                tasks = [self.get_ticker_info(symbol) for symbol in symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for symbol, info in zip(symbols, results):
                    # Check if the result is an exception
                    if isinstance(info, Exception):
                        if isinstance(info, YahooFinanceSymbolNotFoundError):
                            logger.debug(f"Symbol not found during screening: {symbol}")
                        elif isinstance(info, YahooFinanceRateLimitError):
                            logger.warning(
                                f"Rate limit hit during screening for {symbol}, will retry batch"
                            )
                            # Wait and retry this batch on rate limit
                            await asyncio.sleep(5)
                            break
                        else:
                            logger.warning(
                                f"Failed to fetch info for {symbol} during screening: {info}"
                            )
                        continue  # Skip this symbol on error

                    # Check if the result is None (indicating failure)
                    if info is None:
                        logger.debug(f"No data available for {symbol} during screening")
                        continue  # Skip this symbol on error

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
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error screening stocks: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(
                    f"Rate limit exceeded during screening: {e}"
                ) from e
            else:
                raise YahooFinanceDataError(f"Yahoo Finance error screening stocks: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error screening stocks: {e}")
            raise YahooFinanceDataError(f"Unexpected error screening stocks: {e}") from e

    @retry(attempts=3, delay=5)
    async def _get_default_universe(self) -> List[str]:
        """
        Get a default universe of stocks.

        Returns:
            List of stock symbols
        """
        # Try to get S&P 500 components from Wikipedia
        try:
            sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
            symbols = sp500["Symbol"].tolist()

            # Clean symbols (remove dots, convert to uppercase)
            symbols = [s.replace(".", "-").upper() for s in symbols if isinstance(s, str)]

            # Cache the universe for future use
            redis_client.set("stocks:universe", symbols, expiry=86400)  # 24-hour expiry

            logger.info(f"Successfully fetched {len(symbols)} symbols for default universe")
            return symbols
        except Exception as e:  # Changed to generic Exception
            logger.error(f"Yahoo Finance error fetching default universe: {e}")
            # Attempt to classify specific Yahoo Finance errors
            if "rate limit" in str(e).lower():
                raise YahooFinanceRateLimitError(
                    f"Rate limit exceeded fetching default universe: {e}"
                ) from e
            else:
                raise YahooFinanceDataError(
                    f"Yahoo Finance error fetching default universe: {e}"
                ) from e
        except Exception as e:
            logger.error(f"Error fetching S&P 500 components from Wikipedia: {e}")

            # Fallback to a small list of major stocks
            logger.info("Using fallback stock universe")
            fallback_symbols = [
                "AAPL",
                "MSFT",
                "AMZN",
                "GOOGL",
                "META",
                "TSLA",
                "NVDA",
                "JPM",
                "JNJ",
                "V",
                "PG",
                "UNH",
                "HD",
                "BAC",
                "MA",
                "DIS",
                "ADBE",
                "CRM",
                "NFLX",
                "INTC",
                "VZ",
                "CSCO",
                "PFE",
                "KO",
                "PEP",
                "WMT",
                "MRK",
            ]
            return fallback_symbols

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
                if "min" in value and info[key] < value["min"]:
                    return False
                if "max" in value and info[key] > value["max"]:
                    return False
            elif info[key] != value:
                return False

        return True


# Create global Yahoo Finance client instance
yahoo_finance_client = YahooFinanceAPI()


# High-level function for compatibility with unified API
async def fetch_yahoo_finance_data(symbol: str, interval: str = "day", **kwargs) -> Dict[str, Any]:
    """
    Fetch data from Yahoo Finance API.

    Args:
        symbol: Stock symbol
        interval: Time interval ('1d', '1h', '15m', etc.)
        **kwargs: Additional parameters
            period: Optional time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd', 'max')

    Returns:
        Dict containing the fetched market data
    """
    try:
        # Convert interval format if needed (from 'day' to '1d' for example)
        yf_interval = interval
        if interval == "day":
            yf_interval = "1d"
        elif interval == "hour":
            yf_interval = "1h"
        elif interval == "minute":
            yf_interval = "1m"

        # Get period from kwargs or use a default
        period = kwargs.get("period", "1mo")

        # Fetch the data
        df = await yahoo_finance_client.get_historical_prices(
            symbol, period=period, interval=yf_interval
        )

        if df is None:
            logger.warning(f"No data returned from Yahoo Finance for {symbol}")
            return {}

        # Convert DataFrame to dict format for consistent return
        result = {
            "symbol": symbol,
            "interval": interval,
            "data": df.reset_index().to_dict("records"),
            "last_refreshed": datetime.now().isoformat(),
        }

        return result
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance: {e}")
        raise YahooFinanceDataError(f"Failed to fetch data: {str(e)}") from e
