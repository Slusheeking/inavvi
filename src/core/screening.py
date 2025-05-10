"""
Stock screening logic for the trading system.

Identifies trading opportunities based on:
- Technical indicators
- Pattern recognition
- Sentiment analysis
- Ranking models
"""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from src.config.settings import settings
from src.core.data_pipeline import data_pipeline
from src.data_sources.polygon import PolygonAPI
from src.models.pattern_recognition import pattern_recognition_model
from src.models.ranking_model import ranking_model
from src.models.sentiment import sentiment_model
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("screening")


class StockScreener:
    """
    Stock screening module for identifying trading opportunities.

    Responsibilities:
    - Screen stocks for potential trading opportunities
    - Rank trading candidates
    - Manage watchlist
    - Apply technical, pattern, and sentiment filters
    """

    def __init__(self):
        """Initialize the stock screener."""
        self.polygon_client = PolygonAPI()

        # Screening parameters (default values, can be adjusted)
        self.min_price = 5.0
        self.max_price = 100.0
        self.min_volume = 500000
        self.min_relative_volume = 1.2
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        # Maximum number of candidates to track
        self.max_candidates = settings.trading.candidate_size

        logger.info("Stock screener initialized")

    async def initialize_watchlist(self):
        """Initialize the trading watchlist."""
        logger.info("Initializing watchlist")

        # Get current watchlist
        watchlist = redis_client.get_watchlist()

        # If watchlist is empty, create a new one
        if not watchlist:
            # Try to get pre-market movers first
            pre_market_movers = redis_client.get("market:pre_market_movers")

            if pre_market_movers:
                logger.info("Using pre-market movers for initial watchlist")
                candidates = [item["symbol"] for item in pre_market_movers[:30]]
            else:
                # Get gainers and losers as fallback
                gainers_losers = redis_client.get("market:gainers_losers")

                if gainers_losers:
                    logger.info("Using gainers/losers for initial watchlist")
                    candidates = []
                    candidates.extend(
                        [item["symbol"] for item in gainers_losers.get("gainers", [])[:15]]
                    )
                    candidates.extend(
                        [item["symbol"] for item in gainers_losers.get("losers", [])[:15]]
                    )
                else:
                    # Use static list of liquid stocks as last resort
                    logger.info("Using default stock list for initial watchlist")
                    candidates = [
                        "AAPL",
                        "MSFT",
                        "AMZN",
                        "GOOGL",
                        "META",
                        "TSLA",
                        "NVDA",
                        "AMD",
                        "NFLX",
                        "PYPL",
                        "DIS",
                        "BA",
                        "JPM",
                        "V",
                        "WMT",
                        "PFE",
                        "XOM",
                        "BAC",
                        "T",
                        "INTC",
                    ]

            # Deduplicate
            candidates = list(dict.fromkeys(candidates))

            # Limit to watchlist size
            watchlist = candidates[: settings.trading.watchlist_size]

            # Save watchlist
            redis_client.set_watchlist(watchlist)

        # Subscribe to real-time data for watchlist
        await self.polygon_client.subscribe_to_symbols(watchlist)

        logger.info(f"Watchlist initialized with {len(watchlist)} symbols")

        # Fetch initial data for watchlist
        await data_pipeline.fetch_watchlist_data(watchlist)

        return watchlist

    async def update_watchlist(self):
        """Update the trading watchlist based on rankings."""
        logger.debug("Updating watchlist")

        # Get current watchlist
        current_watchlist = redis_client.get_watchlist()

        # Get current candidates
        candidates = redis_client.get_ranked_candidates()

        # If no candidates, run ranking
        if not candidates:
            await self.rank_candidates()
            candidates = redis_client.get_ranked_candidates()

        if not candidates:
            logger.warning("No candidates available for watchlist update")
            return current_watchlist

        # Extract top symbols
        top_symbols = [item["symbol"] for item in candidates[: settings.trading.watchlist_size]]

        # Check if watchlist needs updating
        if set(top_symbols) != set(current_watchlist):
            # Unsubscribe from removed symbols
            removed_symbols = [s for s in current_watchlist if s not in top_symbols]
            if removed_symbols:
                await self.polygon_client.unsubscribe_from_symbols(removed_symbols)

            # Subscribe to new symbols
            new_symbols = [s for s in top_symbols if s not in current_watchlist]
            if new_symbols:
                await self.polygon_client.subscribe_to_symbols(new_symbols)

            # Update watchlist
            redis_client.set_watchlist(top_symbols)

            logger.info(
                f"Watchlist updated: added {len(new_symbols)}, removed {len(removed_symbols)}"
            )

            # Fetch data for new symbols
            if new_symbols:
                await data_pipeline.fetch_watchlist_data(new_symbols)

        return top_symbols

    async def scan_for_opportunities(self, symbols: Optional[List[str]] = None) -> List[Dict]:
        """
        Scan for trading opportunities.

        Args:
            symbols: List of symbols to scan, or None to use watchlist

        Returns:
            List of trading opportunities
        """
        logger.info("Scanning for trading opportunities")

        # Use watchlist if symbols not provided
        if symbols is None:
            symbols = redis_client.get_watchlist()

        if not symbols:
            logger.warning("No symbols to scan")
            return []

        # Track opportunities
        opportunities = []

        # Process each symbol
        for symbol in symbols:
            try:
                opportunity = await self.analyze_symbol(symbol)
                if opportunity:
                    opportunities.append(opportunity)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")

        # Sort by score (descending)
        opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.info(f"Found {len(opportunities)} potential trading opportunities")

        return opportunities

    async def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Analyze a symbol for trading opportunities.

        Args:
            symbol: Stock symbol

        Returns:
            Trading opportunity data if found, None otherwise
        """
        # Get price data
        price_data = await data_pipeline.get_stock_data(symbol, "snapshot")
        if not price_data:
            return None

        # Get OHLCV data
        intraday_data = await data_pipeline.get_stock_data(symbol, "intraday")
        if (
            not intraday_data is not None
            and isinstance(intraday_data, pd.DataFrame)
            and intraday_data.empty
        ):
            return None

        # Get technical indicators
        indicators = await data_pipeline.get_technical_indicators(symbol)
        if not indicators:
            return None

        # Run pattern recognition if we have valid DataFrame
        pattern = {"name": "unknown", "confidence": 0.0}
        if isinstance(intraday_data, pd.DataFrame) and not intraday_data.empty:
            pattern_name, pattern_confidence = pattern_recognition_model.predict_pattern(
                intraday_data
            )
            pattern = {"name": pattern_name, "confidence": pattern_confidence}

        # Get ranking score
        score = 0.0
        if isinstance(intraday_data, pd.DataFrame) and not intraday_data.empty:
            score = ranking_model.predict(intraday_data)

        # Get sentiment data
        news_items = await data_pipeline.get_stock_data(symbol, "news")
        sentiment_data = {"overall_score": 0.0}

        if news_items and sentiment_model:
            # Analyze sentiment
            analyzed_news = sentiment_model.analyze_news_items(news_items)
            overall_sentiment = sentiment_model.get_overall_sentiment(analyzed_news)
            sentiment_data = {
                "news": analyzed_news,
                "overall_score": overall_sentiment.get("overall_score", 0),
                "positive": overall_sentiment.get("positive", 0),
                "neutral": overall_sentiment.get("neutral", 0),
                "negative": overall_sentiment.get("negative", 0),
            }

        # Apply screening criteria
        current_price = price_data.get("price", {}).get("last", 0)
        volume = price_data.get("price", {}).get("volume", 0)
        indicators.get("rsi_14", 50)
        relative_volume = indicators.get("relative_volume", 1.0)

        # Basic price and volume filters
        if current_price < self.min_price or current_price > self.max_price:
            return None

        if volume < self.min_volume:
            return None

        if relative_volume < self.min_relative_volume:
            return None

        # Check if pattern confidence is high enough
        if pattern["name"] != "no_pattern" and pattern["confidence"] < 0.6:
            pattern["name"] = "no_pattern"

        # Create opportunity data
        opportunity = {
            "symbol": symbol,
            "price": price_data.get("price", {}),
            "pattern": pattern,
            "indicators": indicators,
            "sentiment": sentiment_data,
            "score": score,
            "timestamp": datetime.now().isoformat(),
        }

        # Store as candidate
        redis_client.add_candidate_score(
            symbol,
            score,
            {
                "price": current_price,
                "pattern": pattern["name"],
                "pattern_confidence": pattern["confidence"],
                "sentiment": sentiment_data.get("overall_score", 0),
                "rsi_14": indicators.get("rsi_14", 50),
                "macd_histogram": indicators.get("macd_histogram", 0),
                "bb_position": indicators.get("bb_position", 0.5),
                "timestamp": datetime.now().isoformat(),
            },
        )

        return opportunity

    async def rank_candidates(self):
        """Rank stocks in the watchlist based on trading signals."""
        logger.debug("Ranking watchlist candidates")

        # Get current watchlist
        watchlist = redis_client.get_watchlist()

        if not watchlist:
            logger.warning("Empty watchlist, cannot rank candidates")
            return

        # Scan for opportunities
        opportunities = await self.scan_for_opportunities(watchlist)

        # Update candidates in Redis
        candidates = []
        for opportunity in opportunities:
            candidates.append(
                {
                    "symbol": opportunity["symbol"],
                    "score": opportunity["score"],
                    "price": opportunity["price"].get("last", 0),
                    "pattern": opportunity["pattern"]["name"],
                    "pattern_confidence": opportunity["pattern"]["confidence"],
                    "sentiment": opportunity["sentiment"].get("overall_score", 0),
                    "rsi_14": opportunity["indicators"].get("rsi_14", 50),
                    "macd_histogram": opportunity["indicators"].get("macd_histogram", 0),
                    "bb_position": opportunity["indicators"].get("bb_position", 0.5),
                    "timestamp": opportunity["timestamp"],
                }
            )

        # Sort by score (descending)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Limit to max candidates
        candidates = candidates[: self.max_candidates]

        # Store ranked candidates
        redis_client.set_ranked_candidates(candidates)

        if candidates:
            logger.info(f"Top ranked candidates: {[c['symbol'] for c in candidates[:3]]}")
        else:
            logger.warning("No viable candidates found")

        return candidates

    async def get_pre_market_movers(self):
        """Get pre-market movers."""
        logger.info("Getting pre-market movers")

        # Get from cache or fetch
        pre_market_movers = redis_client.get("market:pre_market_movers")

        if not pre_market_movers:
            pre_market_movers = await self.polygon_client.get_premarket_movers(limit=25)
            if pre_market_movers:
                redis_client.set(
                    "market:pre_market_movers", pre_market_movers, expiry=900
                )  # 15 min expiry

        return pre_market_movers

    async def get_market_movers(self):
        """Get current market movers (gainers and losers)."""
        logger.info("Getting market movers")

        # Get from cache or fetch
        gainers_losers = redis_client.get("market:gainers_losers")

        if not gainers_losers:
            gainers_losers = await self.polygon_client.get_gainers_losers(limit=25)
            if gainers_losers:
                redis_client.set(
                    "market:gainers_losers", gainers_losers, expiry=300
                )  # 5 min expiry

        return gainers_losers

    async def get_most_active(self):
        """Get most active stocks by volume."""
        logger.info("Getting most active stocks")

        # Get from cache or fetch
        most_active = redis_client.get("market:most_active")

        if not most_active:
            most_active = await self.polygon_client.get_most_active(limit=25)
            if most_active:
                redis_client.set("market:most_active", most_active, expiry=300)  # 5 min expiry

        return most_active

    def filter_candidates_by_pattern(self, pattern_name: str) -> List[Dict]:
        """
        Filter candidates by pattern type.

        Args:
            pattern_name: Pattern name to filter by

        Returns:
            List of candidates with the specified pattern
        """
        candidates = redis_client.get_ranked_candidates()
        return [c for c in candidates if c.get("pattern") == pattern_name]

    def filter_candidates_by_indicator(
        self, indicator: str, min_value: float, max_value: float
    ) -> List[Dict]:
        """
        Filter candidates by indicator range.

        Args:
            indicator: Indicator name
            min_value: Minimum value
            max_value: Maximum value

        Returns:
            List of candidates within the specified range
        """
        candidates = redis_client.get_ranked_candidates()
        return [c for c in candidates if min_value <= c.get(indicator, 0) <= max_value]

    def get_bullish_candidates(self) -> List[Dict]:
        """
        Get bullish candidates based on technical indicators.

        Returns:
            List of bullish candidates
        """
        candidates = redis_client.get_ranked_candidates()

        # Define bullish criteria
        return [
            c
            for c in candidates
            if (
                c.get("rsi_14", 50) > 50
                and c.get("macd_histogram", 0) > 0
                and c.get("pattern") in ["breakout", "continuation", "flag"]
                and c.get("pattern_confidence", 0) >= 0.6
            )
        ]

    def get_bearish_candidates(self) -> List[Dict]:
        """
        Get bearish candidates based on technical indicators.

        Returns:
            List of bearish candidates
        """
        candidates = redis_client.get_ranked_candidates()

        # Define bearish criteria
        return [
            c
            for c in candidates
            if (
                c.get("rsi_14", 50) < 50
                and c.get("macd_histogram", 0) < 0
                and c.get("pattern") in ["reversal", "head_shoulders", "double_top"]
                and c.get("pattern_confidence", 0) >= 0.6
            )
        ]


# Create global instance
stock_screener = StockScreener()
