"""
Data collection pipeline for the trading system.

Manages data collection, preprocessing, and caching from various sources:
- Polygon for market data
- Alpha Vantage for fundamentals and news
- Yahoo Finance for additional market data
"""

import asyncio
from datetime import datetime
from typing import Dict, List

import numpy as np

from src.config.settings import settings
from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.polygon import PolygonAPI
from src.data_sources.yahoo_finance import yahoo_finance_client
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("data_pipeline")


class DataPipeline:
    """
    Data collection and processing pipeline for the trading system.

    Responsible for:
    - Collecting market data from multiple sources
    - Preprocessing data for ML models
    - Caching data in Redis
    - Managing real-time data updates
    """

    def __init__(self):
        """Initialize the data pipeline."""
        self.polygon_client = PolygonAPI()

        # Initialize data fetch locks to prevent concurrent requests
        self._data_locks = {}

        logger.info("Data pipeline initialized")

    async def initialize(self):
        """Initialize the data pipeline and load initial data."""
        logger.info("Initializing data pipeline")

        # Initialize stock universe
        await self.initialize_stock_universe()

        # Initialize market data
        await self.initialize_market_data()

        # Connect to real-time data
        await self.connect_real_time_data()

        logger.info("Data pipeline initialization complete")

    async def initialize_stock_universe(self):
        """Initialize the stock universe for trading."""
        logger.info("Initializing stock universe")

        # Check if universe already exists in Redis
        universe = redis_client.get("stocks:universe")

        if not universe:
            logger.info("Fetching stock universe from Polygon")
            universe = await self.polygon_client.get_stock_universe(type="cs", active=True)

            if not universe:
                logger.error("Failed to fetch stock universe")
                return False

            logger.info(f"Fetched {len(universe)} symbols")

        # Filter universe based on trading criteria
        filtered_universe = await self.filter_stock_universe(universe)

        logger.info(f"Filtered universe contains {len(filtered_universe)} symbols")

        # Store filtered universe
        redis_client.set("stocks:filtered_universe", filtered_universe)

        return True

    async def filter_stock_universe(self, universe: List[Dict]):
        """
        Filter the stock universe based on trading criteria.

        Args:
            universe: Full stock universe

        Returns:
            Filtered universe
        """
        filtered = []
        count = 0

        # Process in batches
        batch_size = 100
        for i in range(0, min(len(universe), 1000), batch_size):
            batch = universe[i : i + batch_size]

            # Process each symbol in batch
            for stock in batch:
                symbol = stock["symbol"]

                try:
                    # Get basic ticker info from Yahoo Finance
                    info = await yahoo_finance_client.get_ticker_info(symbol)

                    if not info:
                        continue

                    # Apply filters
                    price = info.get("regularMarketPrice", 0)
                    volume = info.get("regularMarketVolume", 0)
                    market_cap = info.get("marketCap", 0)

                    # Filter criteria (customize as needed)
                    if price >= 5 and price <= 100 and volume >= 500000 and market_cap >= 500000000:
                        filtered.append(
                            {
                                "symbol": symbol,
                                "name": info.get("shortName", ""),
                                "price": price,
                                "volume": volume,
                                "market_cap": market_cap,
                                "sector": info.get("sector", ""),
                                "industry": info.get("industry", ""),
                            }
                        )

                        count += 1
                        if count % 10 == 0:
                            logger.info(
                                f"Processed {count} symbols, {len(filtered)} passed filters"
                            )

                    # Limit to the desired universe size
                    if len(filtered) >= settings.trading.initial_universe_size:
                        break
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

                # Add a small delay to avoid rate limits
                await asyncio.sleep(0.1)

            # Early exit if we have enough symbols
            if len(filtered) >= settings.trading.initial_universe_size:
                break

        return filtered

    async def initialize_market_data(self):
        """Initialize market data."""
        logger.info("Initializing market data")

        tasks = [
            self.update_market_status(),
            self.update_sector_performance(),
            self.update_economic_indicators(),
        ]

        # Run tasks concurrently
        await asyncio.gather(*tasks)

        logger.info("Market data initialization complete")

    async def update_market_status(self):
        """Update market status."""
        try:
            # Get market status
            market_status = await self.polygon_client.get_market_status()
            redis_client.set("market:status", market_status, expiry=300)  # 5 min expiry
            logger.info("Market status updated")
        except Exception as e:
            logger.error(f"Error updating market status: {e}")

    async def update_sector_performance(self):
        """Update sector performance."""
        try:
            # Get sector performance
            sector_performance = await alpha_vantage_client.get_sector_performance()
            if sector_performance:
                redis_client.set("market:sectors", sector_performance, expiry=3600)  # 1 hour expiry
                logger.info("Sector performance updated")
        except Exception as e:
            logger.error(f"Error updating sector performance: {e}")

    async def update_economic_indicators(self):
        """Update economic indicators."""
        try:
            # Get Treasury yields
            yield_10y = await alpha_vantage_client.get_treasury_yield(
                interval="daily", maturity="10year"
            )
            if yield_10y is not None:
                redis_client.set(
                    "economic:treasury_yield:10year", yield_10y, expiry=86400
                )  # 24 hour expiry

            # Get inflation data
            inflation = await alpha_vantage_client.get_inflation()
            if inflation is not None:
                redis_client.set("economic:inflation", inflation, expiry=86400)  # 24 hour expiry

            logger.info("Economic indicators updated")
        except Exception as e:
            logger.error(f"Error updating economic indicators: {e}")

    async def connect_real_time_data(self):
        """Connect to real-time market data."""
        logger.info("Connecting to real-time market data")

        # Connect to Polygon WebSocket
        ws_connected = await self.polygon_client.connect_websocket()

        if not ws_connected:
            logger.error("Failed to connect to Polygon WebSocket")
            return False

        logger.info("Connected to real-time market data")
        return True

    async def get_stock_data(
        self, symbol: str, data_type: str = "snapshot", force_refresh: bool = False
    ):
        """
        Get stock data from cache or fetch if not available.

        Args:
            symbol: Stock symbol
            data_type: Type of data to fetch ('snapshot', 'intraday', 'daily', 'fundamentals')
            force_refresh: Whether to force a refresh from the API

        Returns:
            Requested stock data
        """
        # Create lock key for this request
        lock_key = f"{symbol}_{data_type}"

        # Check if already fetching this data
        if lock_key in self._data_locks:
            # Wait for existing fetch to complete
            logger.debug(f"Waiting for existing {data_type} fetch for {symbol}")
            await self._data_locks[lock_key]
            return redis_client.get_stock_data(symbol, data_type)

        # Create a future to signal when fetch is complete
        self._data_locks[lock_key] = asyncio.Future()

        try:
            # Check cache first if not forcing refresh
            if not force_refresh:
                cached_data = redis_client.get_stock_data(symbol, data_type)
                if cached_data:
                    return cached_data

            # Data not in cache or force refresh, fetch from API
            if data_type == "snapshot":
                data = await self.polygon_client.get_stock_snapshot(symbol)
            elif data_type == "intraday":
                data = await self.polygon_client.get_intraday_bars(symbol, minutes=1, days=1)
            elif data_type == "daily":
                data = await self.polygon_client.get_daily_bars(symbol, days=30)
            elif data_type == "fundamentals":
                data = await yahoo_finance_client.get_ticker_info(symbol)
            elif data_type == "news":
                data = await alpha_vantage_client.get_symbol_news(symbol, limit=5)
            else:
                logger.error(f"Unknown data type: {data_type}")
                data = None

            # Cache data if fetch successful
            if data is not None:
                redis_client.set_stock_data(symbol, data, data_type)

            return data
        except Exception as e:
            logger.error(f"Error fetching {data_type} for {symbol}: {e}")
            return None
        finally:
            # Signal that fetch is complete
            self._data_locks[lock_key].set_result(True)
            del self._data_locks[lock_key]

    async def get_technical_indicators(self, symbol: str, timeframe: str = "intraday"):
        """
        Calculate technical indicators for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe to use ('intraday', 'daily')

        Returns:
            Dictionary with technical indicators
        """
        # Get price data
        if timeframe == "intraday":
            ohlc_data = await self.get_stock_data(symbol, "intraday")
        else:
            ohlc_data = await self.get_stock_data(symbol, "daily")

        if ohlc_data is None or isinstance(ohlc_data, dict) or ohlc_data.empty:
            logger.warning(f"No price data for {symbol}, cannot calculate indicators")
            return {}

        # Calculate technical indicators
        df = ohlc_data.copy()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))

        # Moving averages
        sma_20 = df["close"].rolling(20).mean()
        sma_50 = df["close"].rolling(50).mean()
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()

        # MACD
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        # Bollinger Bands
        bb_middle = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_middle

        # Current price position within Bollinger Bands
        current_price = df["close"].iloc[-1]
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

        # Volume indicators
        volume_sma_20 = df["volume"].rolling(20).mean()
        relative_volume = (
            df["volume"].iloc[-1] / volume_sma_20.iloc[-1] if volume_sma_20.iloc[-1] > 0 else 1.0
        )

        # Volatility
        returns = df["close"].pct_change()
        volatility_5d = returns.rolling(5).std().iloc[-1] * np.sqrt(252)  # Annualized

        # Trend detection
        price_above_sma20 = current_price > sma_20.iloc[-1]
        price_above_sma50 = current_price > sma_50.iloc[-1]
        sma20_above_sma50 = sma_20.iloc[-1] > sma_50.iloc[-1]

        # Momentum
        momentum_1d = (current_price / df["close"].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
        momentum_5d = (current_price / df["close"].iloc[-6] - 1) * 100 if len(df) >= 6 else 0

        # Compile indicators
        indicators = {
            "rsi_14": float(rsi_14.iloc[-1]),
            "sma_20": float(sma_20.iloc[-1]),
            "sma_50": float(sma_50.iloc[-1]),
            "macd": float(macd.iloc[-1]),
            "macd_signal": float(macd_signal.iloc[-1]),
            "macd_histogram": float(macd_hist.iloc[-1]),
            "bb_upper": float(bb_upper.iloc[-1]),
            "bb_middle": float(bb_middle.iloc[-1]),
            "bb_lower": float(bb_lower.iloc[-1]),
            "bb_width": float(bb_width.iloc[-1]),
            "bb_position": float(bb_position),
            "relative_volume": float(relative_volume),
            "volatility_5d": float(volatility_5d),
            "price_above_sma20": bool(price_above_sma20),
            "price_above_sma50": bool(price_above_sma50),
            "sma20_above_sma50": bool(sma20_above_sma50),
            "momentum_1d": float(momentum_1d),
            "momentum_5d": float(momentum_5d),
        }

        # Cache indicators
        redis_client.set_stock_data(symbol, indicators, "indicators")

        return indicators

    async def get_market_context(self):
        """
        Get current market context data.

        Returns:
            Dictionary with market context
        """
        # Get market status
        market_status = redis_client.get("market:status") or {}

        # Get sector performance
        sector_performance = redis_client.get("market:sectors") or {}

        # Get economic indicators
        treasury_yield = redis_client.get("economic:treasury_yield:10year") or {}
        inflation = redis_client.get("economic:inflation") or {}

        # Get VIX data
        vix_data = await self.get_stock_data("VIX", "snapshot")
        vix_value = vix_data.get("price", {}).get("last", 0) if vix_data else 0

        # Calculate market hours
        now = datetime.now()
        market_open_str = settings.timing.market_open
        market_close_str = settings.timing.market_close

        hours_open, minutes_open = map(int, market_open_str.split(":"))
        hours_close, minutes_close = map(int, market_close_str.split(":"))

        market_open = now.replace(hour=hours_open, minute=minutes_open, second=0, microsecond=0)
        market_close = now.replace(hour=hours_close, minute=minutes_close, second=0, microsecond=0)

        # Calculate time until close
        time_until_close = (market_close - now).total_seconds() / 3600 if now < market_close else 0

        # Determine market state
        if now < market_open:
            market_state = "pre_market"
        elif now > market_close:
            market_state = "after_hours"
        else:
            market_state = "open"

        # Create market context
        context = {
            "state": market_state,
            "official_status": market_status.get("market", "unknown"),
            "sector_performance": sector_performance.get("Rank 1 Day", {}).get(
                "Information Technology", 0
            ),
            "treasury_yield_10y": treasury_yield.get("value", 0),
            "inflation_rate": inflation.get("value", 0),
            "vix": vix_value,
            "time_until_close": time_until_close,
            "timestamp": now.isoformat(),
        }

        # Cache market context
        redis_client.set("market:context", context, expiry=300)  # 5 min expiry

        return context

    async def fetch_watchlist_data(self, watchlist: List[str]):
        """
        Fetch data for all symbols in the watchlist.

        Args:
            watchlist: List of stock symbols
        """
        logger.debug(f"Fetching data for {len(watchlist)} watchlist symbols")

        # Create tasks for each symbol
        tasks = []
        for symbol in watchlist:
            tasks.append(self.get_stock_data(symbol, "snapshot"))
            tasks.append(self.get_stock_data(symbol, "intraday"))
            tasks.append(self.get_technical_indicators(symbol))

        # Run all tasks concurrently
        await asyncio.gather(*tasks)

        logger.debug("Watchlist data fetch complete")

    async def update_pre_market_data(self):
        """Update pre-market data."""
        logger.info("Updating pre-market data")

        try:
            # Get pre-market movers
            pre_market_movers = await self.polygon_client.get_premarket_movers(limit=25)
            if pre_market_movers:
                redis_client.set(
                    "market:pre_market_movers", pre_market_movers, expiry=900
                )  # 15 min expiry

            # Get gainers and losers
            gainers_losers = await self.polygon_client.get_gainers_losers(limit=25)
            if gainers_losers:
                redis_client.set(
                    "market:gainers_losers", gainers_losers, expiry=900
                )  # 15 min expiry

            logger.info("Pre-market data updated")
        except Exception as e:
            logger.error(f"Error updating pre-market data: {e}")

    async def update_intraday_data(self):
        """Update intraday market data."""
        logger.info("Updating intraday market data")

        try:
            # Update market status
            await self.update_market_status()

            # Update market context
            await self.get_market_context()

            # Update gainers and losers
            gainers_losers = await self.polygon_client.get_gainers_losers(limit=20)
            if gainers_losers:
                redis_client.set(
                    "market:gainers_losers", gainers_losers, expiry=300
                )  # 5 min expiry

            # Update most active stocks
            most_active = await self.polygon_client.get_most_active(limit=20)
            if most_active:
                redis_client.set("market:most_active", most_active, expiry=300)  # 5 min expiry

            logger.info("Intraday market data updated")
        except Exception as e:
            logger.error(f"Error updating intraday market data: {e}")


# Create global instance
data_pipeline = DataPipeline()
