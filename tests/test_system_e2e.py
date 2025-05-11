"""
End-to-End Test for Trading System

This test runs the entire trading system from end to end using either:
1. Real historical data from the previous week, or
2. A pre-generated test dataset from a specific date

It tests:
1. Market scanning
2. Watchlist analysis
3. Trade decision making with LLM
4. Trade execution with Alpaca

The test uses real API keys for OpenRouter and Alpaca (paper trading).
"""

import asyncio
import os
import unittest
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from src.core.trade_execution import TradeExecutor
from src.llm.router import openrouter_client
from src.data_sources.polygon import PolygonAPI
from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.yahoo_finance import yahoo_finance_client
from src.utils.redis_client import redis_client
from src.models.pattern_recognition import pattern_recognition_model
from src.models.sentiment import sentiment_model
from src.models.ranking_model import ranking_model
from src.models.exit_optimization import exit_optimization_model
from src.training.dataset_generator import DatasetGenerator


class TestSystemE2E(unittest.TestCase):
    """End-to-End test for the trading system."""

    def setUp(self):
        """Set up test environment."""
        # Create test clients with real API keys from settings
        self.llm_client = openrouter_client
        self.polygon = PolygonAPI()
        self.alpha_vantage = alpha_vantage_client
        self.yahoo = yahoo_finance_client
        
        # Test symbols - one bullish, one bearish
        self.bullish_symbol = "AAPL"  # Expected to show bullish signals
        self.bearish_symbol = "MSFT"  # Expected to show bearish signals
        
        # Dataset configuration
        self.use_dataset = os.environ.get("USE_TEST_DATASET", "").lower() == "true"
        self.dataset_name = os.environ.get("TEST_DATASET_NAME", "dataset_last_week")
        self.dataset_date = os.environ.get("TEST_DATASET_DATE", (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
        
        # Initialize dataset generator
        self.dataset_generator = DatasetGenerator()
        
        # Ensure Redis is clean for the test
        self._clean_redis()
        
        # Fetch historical data for the test symbols (either from dataset or live)
        self.fetch_historical_data()

    def _clean_redis(self):
        """Clean Redis data for the test symbols."""
        # Clear any existing data for test symbols
        redis_client.delete_active_position(self.bullish_symbol)
        redis_client.delete_active_position(self.bearish_symbol)
        
        # Clear any existing signals
        redis_client.clear_trading_signal(self.bullish_symbol, "entry")
        redis_client.clear_trading_signal(self.bearish_symbol, "entry")
        
        # Clear candidate scores
        redis_client.delete(f"candidates:scores:{self.bullish_symbol}")
        redis_client.delete(f"candidates:scores:{self.bearish_symbol}")

    async def get_or_generate_dataset(self):
        """Get an existing dataset or generate a new one."""
        # Check if dataset exists
        dataset = self.dataset_generator.load_dataset(self.dataset_name)
        
        if dataset is not None:
            print(f"Using existing dataset: {self.dataset_name}")
            return dataset
        
        # Dataset doesn't exist, generate a new one
        print(f"Generating new dataset: {self.dataset_name} for date {self.dataset_date}")
        
        # Generate dataset
        dataset = await self.dataset_generator.generate_dataset(
            symbols=[self.bullish_symbol, self.bearish_symbol],
            start_date=self.dataset_date,
            end_date=None,  # Use start_date as end_date
            time_of_day="all",
            data_source="auto",
            include_news=True,
            include_market_data=True,
            dataset_name=self.dataset_name
        )
        
        return dataset

    def fetch_historical_data(self):
        """Fetch historical data for the test symbols (either from dataset or live)."""
        # Create a new event loop and set it as the current event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async fetch function
        loop.run_until_complete(self._fetch_historical_data_async())
        
    async def _fetch_historical_data_async(self):
        """Async implementation of fetch_historical_data."""
        if self.use_dataset:
            # Use dataset
            dataset = await self.get_or_generate_dataset()
            
            if dataset is None:
                print("Failed to load or generate dataset, falling back to live data")
                await self._fetch_live_data_async()
                return
            
            # Convert dataset price data to DataFrames
            self.bullish_data = self.dataset_generator.convert_to_dataframe(dataset, self.bullish_symbol)
            self.bearish_data = self.dataset_generator.convert_to_dataframe(dataset, self.bearish_symbol)
            
            # Store news data in Redis
            if "news_data" in dataset and dataset["news_data"]:
                redis_client.set("news:recent", dataset["news_data"])
                print(f"Stored {len(dataset['news_data'])} news items in Redis")
            
            # Store market data in Redis
            if "market_data" in dataset and dataset["market_data"]:
                if "market_status" in dataset["market_data"]:
                    redis_client.set("market:status", dataset["market_data"]["market_status"])
                
                if "sector_performance" in dataset["market_data"]:
                    redis_client.set("market:sectors", dataset["market_data"]["sector_performance"])
                
                print("Stored market data in Redis")
        else:
            # Use live data
            await self._fetch_live_data_async()
        
        # Store data in Redis for the system to use
        if self.bullish_data is not None and not self.bullish_data.empty:
            redis_client.set(f"stocks:intraday:{self.bullish_symbol}:1m", self.bullish_data)
            print(f"Stored {len(self.bullish_data)} data points for {self.bullish_symbol} in Redis")
        
        if self.bearish_data is not None and not self.bearish_data.empty:
            redis_client.set(f"stocks:intraday:{self.bearish_symbol}:1m", self.bearish_data)
            print(f"Stored {len(self.bearish_data)} data points for {self.bearish_symbol} in Redis")
        
        # Add to watchlist
        redis_client.set_watchlist([self.bullish_symbol, self.bearish_symbol])
        print(f"Added {self.bullish_symbol} and {self.bearish_symbol} to watchlist")
    
    async def _fetch_live_data_async(self):
        """Fetch real historical data from live APIs."""
        print("Fetching live data from APIs")
        # Get last week's date range
        end_date = datetime.now()
        # Calculate start date but we don't need to use it directly
        _ = end_date - timedelta(days=7)
        
        # Fetch data from Polygon
        self.bullish_data = await self.polygon.get_intraday_bars(self.bullish_symbol, minutes=1, days=7)
        self.bearish_data = await self.polygon.get_intraday_bars(self.bearish_symbol, minutes=1, days=7)
        
        # If data fetching failed, use Yahoo Finance as fallback
        if self.bullish_data is None or self.bullish_data.empty:
            print(f"Polygon data fetch failed for {self.bullish_symbol}, using Yahoo Finance as fallback")
            self.bullish_data = await self.yahoo.get_intraday_prices(self.bullish_symbol)
        
        if self.bearish_data is None or self.bearish_data.empty:
            print(f"Polygon data fetch failed for {self.bearish_symbol}, using Yahoo Finance as fallback")
            self.bearish_data = await self.yahoo.get_intraday_prices(self.bearish_symbol)

    def test_system_e2e(self):
        """Test the entire system end-to-end."""
        # Skip this test if we're in CI environment or if explicitly disabled
        if os.environ.get("SKIP_LIVE_TESTS", "").lower() == "true":
            self.skipTest("Skipping live API test")
        
        print(f"\n--- Running E2E Test with {'Dataset' if self.use_dataset else 'Live Data'} ---")
        
        # Create a new event loop and set it as the current event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async test
        loop.run_until_complete(self._run_e2e_test())

    async def _run_e2e_test(self):
        """Run the end-to-end test asynchronously."""
        # 1. Test market scanning
        await self._test_market_scanning()
        
        # 2. Test watchlist analysis
        await self._test_watchlist_analysis()
        
        # 3. Test trade decision making with LLM
        bullish_decision, bearish_decision = await self._test_trade_decisions()
        
        # 4. Test trade execution with Alpaca
        await self._test_trade_execution(bullish_decision, bearish_decision)

    async def _test_market_scanning(self):
        """Test the market scanning functionality."""
        print("\n--- Testing Market Scanning ---")
        
        # Get market status
        market_status = await self.polygon.get_market_status()
        self.assertIsNotNone(market_status, "Market status should not be None")
        print(f"Market status: {market_status.get('market', 'unknown')}")
        
        # Get sector performance
        sector_performance = await self.alpha_vantage.get_sector_performance()
        self.assertIsNotNone(sector_performance, "Sector performance should not be None")
        print("Sector performance retrieved successfully")
        
        # Get economic indicators
        treasury_yield = await self.alpha_vantage.get_treasury_yield(interval="daily", maturity="10year")
        self.assertIsNotNone(treasury_yield, "Treasury yield should not be None")
        print("Treasury yield retrieved successfully")
        
        # Store market context in Redis
        market_context = {
            "state": market_status.get("market", "unknown"),
            "sector_performance": 0.0,  # Default value
            "vix": 15.0,  # Default value
            "breadth": 0.65,  # Default value
            "time_until_close": 3.5,  # Default value
        }
        
        # Extract sector performance if available
        if sector_performance and "Rank A: Real-Time Performance" in sector_performance:
            tech_sector = sector_performance["Rank A: Real-Time Performance"].get("Information Technology", 0)
            if isinstance(tech_sector, str):
                tech_sector = float(tech_sector.strip("%"))
            market_context["sector_performance"] = tech_sector
        
        # Store in Redis
        redis_client.set("market:context", market_context)
        print(f"Market context stored in Redis: {market_context}")
        
        # Verify that the market scanning was successful
        self.assertIsNotNone(redis_client.get("market:context"), "Market context should be stored in Redis")

    async def _test_watchlist_analysis(self):
        """Test the watchlist analysis functionality."""
        print("\n--- Testing Watchlist Analysis ---")
        
        # Process each symbol in the watchlist
        for symbol in [self.bullish_symbol, self.bearish_symbol]:
            # Get stock data
            snapshot = await self.polygon.get_stock_snapshot(symbol)
            intraday_data = await self.polygon.get_intraday_bars(symbol, minutes=1, days=1)
            
            if not snapshot or not isinstance(intraday_data, pd.DataFrame) or intraday_data.empty:
                print(f"No data for {symbol}, using Yahoo Finance as fallback")
                # Try Yahoo Finance as fallback
                ticker_info = await self.yahoo.get_ticker_info(symbol)
                intraday_data = await self.yahoo.get_intraday_prices(symbol)
                
                if ticker_info:
                    snapshot = {
                        "symbol": symbol,
                        "price": {
                            "last": ticker_info.get("currentPrice", 0),
                            "open": ticker_info.get("open", 0),
                            "high": ticker_info.get("dayHigh", 0),
                            "low": ticker_info.get("dayLow", 0),
                            "close": ticker_info.get("previousClose", 0),
                            "volume": ticker_info.get("volume", 0),
                        }
                    }
            
            # If still no data, create synthetic data for testing
            if not isinstance(intraday_data, pd.DataFrame) or intraday_data.empty:
                print(f"Creating synthetic data for {symbol} as both Polygon and Yahoo Finance failed")
                # Create synthetic data with 100 rows of dummy OHLCV data
                dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1min')
                intraday_data = pd.DataFrame({
                    'open': [100 + i * 0.1 for i in range(100)],
                    'high': [101 + i * 0.1 for i in range(100)],
                    'low': [99 + i * 0.1 for i in range(100)],
                    'close': [100.5 + i * 0.1 for i in range(100)],
                    'volume': [1000 + i * 10 for i in range(100)],
                }, index=dates)
                
                # If no snapshot, create synthetic snapshot too
                if not snapshot:
                    snapshot = {
                        "symbol": symbol,
                        "price": {
                            "last": 100.5,
                            "open": 100.0,
                            "high": 101.0,
                            "low": 99.0,
                            "close": 100.5,
                            "volume": 10000,
                        }
                    }
            
            # Run pattern recognition
            pattern_name, pattern_confidence = pattern_recognition_model.predict_pattern(intraday_data)
            print(f"{symbol} pattern: {pattern_name} with confidence {pattern_confidence:.2f}")
            
            # Get ranking score
            ranking_score = ranking_model.predict(intraday_data)
            # Extract the first element of the array if it's an array
            if isinstance(ranking_score, np.ndarray) and ranking_score.size > 0:
                ranking_score_value = float(ranking_score[0])
            else:
                ranking_score_value = float(ranking_score)
            print(f"{symbol} ranking score: {ranking_score_value:.2f}")
            
            # Get sentiment
            news_items = await self.alpha_vantage.get_symbol_news(symbol, limit=5)
            sentiment_data = {}
            
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
                print(f"{symbol} sentiment score: {sentiment_data.get('overall_score', 0):.2f}")
            
            # Create candidate data
            candidate = {
                "symbol": symbol,
                "price": snapshot.get("price", {}),
                "pattern": {"name": pattern_name, "confidence": pattern_confidence},
                "ranking_score": ranking_score,
                "sentiment": sentiment_data,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add to Redis
            redis_client.add_candidate_score(
                symbol,
                ranking_score_value,
                {
                    "price": snapshot.get("price", {}).get("last", 0),
                    "pattern": pattern_name,
                    "pattern_confidence": pattern_confidence,
                    "sentiment": sentiment_data.get("overall_score", 0),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            
            # Store the candidate data
            redis_client.set(f"candidates:data:{symbol}", candidate)
            print(f"Stored candidate data for {symbol} in Redis")
        
        # Get top candidates
        top_candidates = redis_client.get_ranked_candidates()
        self.assertIsNotNone(top_candidates, "Top candidates should not be None")
        print(f"Top candidates: {[c['symbol'] for c in top_candidates]}")
        
        # Verify that the watchlist analysis was successful
        self.assertTrue(len(top_candidates) > 0, "Should have at least one candidate")

    async def _test_trade_decisions(self):
        """Test the trade decision making with LLM."""
        print("\n--- Testing Trade Decisions ---")
        
        # Create a TradeExecutor
        executor = TradeExecutor()
        executor.trading_mode = "paper"
        
        # Initialize the executor
        await executor.initialize()
        
        # Get top candidates
        candidates = redis_client.get_ranked_candidates()
        if not candidates:
            self.fail("No candidates available for trading")
        
        # Get bullish and bearish candidates
        bullish_candidate = next((c for c in candidates if c["symbol"] == self.bullish_symbol), None)
        bearish_candidate = next((c for c in candidates if c["symbol"] == self.bearish_symbol), None)
        
        # If candidates not found, use the first two candidates
        if not bullish_candidate and len(candidates) > 0:
            bullish_candidate = candidates[0]
        
        if not bearish_candidate and len(candidates) > 1:
            bearish_candidate = candidates[1]
        
        bullish_decision = None
        bearish_decision = None
        
        # Test bullish candidate
        if bullish_candidate:
            print(f"\nEvaluating bullish candidate: {bullish_candidate['symbol']}")
            # Get candidate data
            candidate_data = redis_client.get(f"candidates:data:{bullish_candidate['symbol']}")
            if not candidate_data:
                candidate_data = bullish_candidate
            
            # Make trade decision
            bullish_decision = await executor.make_trade_decision(candidate_data)
            print(f"Bullish decision: {bullish_decision}")
            
            # Verify decision
            self.assertIsNotNone(bullish_decision, "Bullish decision should not be None")
            self.assertIn("decision", bullish_decision, "Bullish decision should have 'decision' key")
            
            # We expect a "trade" decision for the bullish symbol
            if bullish_candidate["symbol"] == self.bullish_symbol:
                print(f"Expecting 'trade' decision for {self.bullish_symbol}")
                # Note: We don't assert this because the LLM might make a different decision based on the data
        
        # Test bearish candidate
        if bearish_candidate:
            print(f"\nEvaluating bearish candidate: {bearish_candidate['symbol']}")
            # Get candidate data
            candidate_data = redis_client.get(f"candidates:data:{bearish_candidate['symbol']}")
            if not candidate_data:
                candidate_data = bearish_candidate
            
            # Make trade decision
            bearish_decision = await executor.make_trade_decision(candidate_data)
            print(f"Bearish decision: {bearish_decision}")
            
            # Verify decision
            self.assertIsNotNone(bearish_decision, "Bearish decision should not be None")
            self.assertIn("decision", bearish_decision, "Bearish decision should have 'decision' key")
            
            # We expect a "no_trade" decision for the bearish symbol
            if bearish_candidate["symbol"] == self.bearish_symbol:
                print(f"Expecting 'no_trade' decision for {self.bearish_symbol}")
                # Note: We don't assert this because the LLM might make a different decision based on the data
        
        return bullish_decision, bearish_decision

    async def _test_trade_execution(self, bullish_decision, bearish_decision):
        """Test trade execution with Alpaca."""
        print("\n--- Testing Trade Execution ---")
        
        # Create a TradeExecutor
        executor = TradeExecutor()
        executor.trading_mode = "paper"
        
        # Test bullish trade execution
        if bullish_decision and bullish_decision.get("decision") == "trade":
            print("\nExecuting trade for bullish candidate")
            # Execute trade
            bullish_result = await executor.execute_trade(self.bullish_symbol, bullish_decision)
            print(f"Bullish trade result: {bullish_result}")
            
            # Verify trade execution
            self.assertIsNotNone(bullish_result, "Bullish trade result should not be None")
            
            # Check if position was created
            position = redis_client.get_active_position(self.bullish_symbol)
            if position:
                print(f"Position created for {self.bullish_symbol}: {position}")
        else:
            print("No trade decision for bullish candidate, skipping execution")
        
        # Test bearish trade execution (should not execute)
        if bearish_decision and bearish_decision.get("decision") == "trade":
            print("\nExecuting trade for bearish candidate")
            # Execute trade
            bearish_result = await executor.execute_trade(self.bearish_symbol, bearish_decision)
            print(f"Bearish trade result: {bearish_result}")
            
            # Verify trade execution
            self.assertIsNotNone(bearish_result, "Bearish trade result should not be None")
            
            # Check if position was created
            position = redis_client.get_active_position(self.bearish_symbol)
            if position:
                print(f"Position created for {self.bearish_symbol}: {position}")
        else:
            print("No trade decision for bearish candidate, skipping execution")
        
        # Test position monitoring
        active_positions = redis_client.get_all_active_positions()
        if active_positions:
            print(f"\nMonitoring {len(active_positions)} active positions")
            
            # Process each position
            for symbol, position_data in active_positions.items():
                print(f"Monitoring position for {symbol}")
                
                # Get current data
                snapshot = await executor.polygon_client.get_stock_snapshot(symbol)
                intraday_data = await executor.polygon_client.get_intraday_bars(symbol, minutes=1, days=1)
                
                if not snapshot or not isinstance(intraday_data, pd.DataFrame) or intraday_data.empty:
                    print(f"No data for {symbol}, skipping position monitoring")
                    continue
                
                # Update position P&L
                current_price = snapshot.get("price", {}).get("last", 0)
                redis_client.update_position_pnl(symbol, current_price)
                
                # Get updated position data
                position_data = redis_client.get_active_position(symbol)
                
                # Check exit signals from ML model
                exit_signals = exit_optimization_model.evaluate_exit_conditions(
                    intraday_data, position_data
                )
                
                print(f"Exit signals for {symbol}: {exit_signals}")
                
                # We don't actually execute the exit in the test to avoid closing positions
                # But we verify that the exit signals are generated
                self.assertIsNotNone(exit_signals, f"Exit signals for {symbol} should not be None")
        else:
            print("No active positions to monitor")


if __name__ == "__main__":
    unittest.main()
