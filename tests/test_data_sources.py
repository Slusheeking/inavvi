"""
Tests for data source modules.
"""

import asyncio
import os
import unittest
import pandas as pd

from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.polygon import PolygonAPI
from src.data_sources.yahoo_finance import yahoo_finance_client


class TestDataSources(unittest.TestCase):
    """Test the data source APIs."""

    def setUp(self):
        """Set up test case."""
        # Initialize API clients
        self.polygon = PolygonAPI()
        self.alpha_vantage = alpha_vantage_client
        self.yahoo = yahoo_finance_client
        
        # Test symbols
        self.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Check if API keys are available
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        
        # Skip tests if API keys are not available
        if not self.polygon_api_key:
            print("Polygon API key not available. Some tests will be skipped.")
        
        if not self.alpha_vantage_api_key:
            print("Alpha Vantage API key not available. Some tests will be skipped.")

    def test_polygon_api_key(self):
        """Test that Polygon API key is available."""
        self.assertIsNotNone(self.polygon_api_key, "Polygon API key is not available")
        
    def test_alpha_vantage_api_key(self):
        """Test that Alpha Vantage API key is available."""
        self.assertIsNotNone(self.alpha_vantage_api_key, "Alpha Vantage API key is not available")
    
    def test_polygon_client_initialization(self):
        """Test that Polygon client initializes correctly."""
        self.assertIsNotNone(self.polygon.rest_client, "Polygon REST client not initialized")
        self.assertIsNotNone(self.polygon.api_key, "Polygon API key not set in client")
    
    def test_alpha_vantage_client_initialization(self):
        """Test that Alpha Vantage client initializes correctly."""
        self.assertIsNotNone(self.alpha_vantage.api_key, "Alpha Vantage API key not set in client")
    
    def test_yahoo_finance_client_initialization(self):
        """Test that Yahoo Finance client initializes correctly."""
        self.assertIsNotNone(self.yahoo, "Yahoo Finance client not initialized")
        self.assertGreater(self.yahoo.rate_limit_delay, 0, "Yahoo Finance rate limit delay not set")
    
    @unittest.skipIf(os.getenv("POLYGON_API_KEY") is None, "Polygon API key not available")
    def test_polygon_market_status(self):
        """Test fetching market status from Polygon."""
        result = asyncio.run(self.polygon.get_market_status())
        self.assertIsNotNone(result, "Market status should not be None")
        self.assertIn("market", result, "Market status should contain 'market' key")
        self.assertIn("exchanges", result, "Market status should contain 'exchanges' key")
    
    @unittest.skipIf(os.getenv("POLYGON_API_KEY") is None, "Polygon API key not available")
    def test_polygon_stock_universe(self):
        """Test fetching stock universe from Polygon."""
        # Limit to just a few stocks to avoid rate limiting
        result = asyncio.run(self.polygon.get_stock_universe())
        self.assertIsNotNone(result, "Stock universe should not be None")
        self.assertIsInstance(result, list, "Stock universe should be a list")
        if len(result) > 0:
            self.assertIn("symbol", result[0], "Stock universe items should contain 'symbol' key")
    
    @unittest.skipIf(os.getenv("ALPHA_VANTAGE_API_KEY") is None, "Alpha Vantage API key not available")
    def test_alpha_vantage_sector_performance(self):
        """Test fetching sector performance from Alpha Vantage."""
        result = asyncio.run(self.alpha_vantage.get_sector_performance())
        self.assertIsNotNone(result, "Sector performance should not be None")
        
        # If we got data, check if it contains sector information
        if result:
            # Check if any sector data is returned
            has_rank_data = any(key.startswith("Rank") for key in result.keys())
            if not has_rank_data:
                print("Warning: No rank data found in sector performance")
    
    def test_yahoo_finance_ticker_info(self):
        """Test fetching ticker info from Yahoo Finance."""
        # Use a well-known ticker that's unlikely to disappear
        result = asyncio.run(self.yahoo.get_ticker_info("SPY"))
        # We don't assert on the exact content because it might change
        # Just check that we get some data back
        self.assertIsNotNone(result, "Ticker info should not be None")
        if result:
            self.assertIsInstance(result, dict, "Ticker info should be a dictionary")
    
    def test_data_source_fallback(self):
        """Test that data sources can fall back to synthetic data."""
        # This test doesn't actually call any APIs
        # It just verifies that the synthetic data generation works
        from src.training.data_fetcher import DataFetcher
        
        # Initialize with no real data sources
        fetcher = DataFetcher(use_polygon=False, use_alpha_vantage=False)
        
        # Fetch data for a symbol
        data = fetcher.fetch_historical_data(symbols=["TEST"])
        
        # Verify that we got synthetic data
        self.assertIn("TEST", data, "Synthetic data should be generated for TEST symbol")
        self.assertIsInstance(data["TEST"], pd.DataFrame, "Synthetic data should be a DataFrame")
        self.assertGreater(len(data["TEST"]), 0, "Synthetic data should not be empty")
    
    def test_redis_caching(self):
        """Test that Redis caching works for data sources."""
        from src.utils.redis_client import redis_client
        
        # Create a test key and data
        test_key = "test:data_sources:redis_test"
        test_data = {"test": "data"}
        
        # Store in Redis
        redis_client.set(test_key, test_data, ex=60)
        
        # Retrieve from Redis
        retrieved_data = redis_client.get(test_key)
        
        # Verify that data was stored and retrieved correctly
        self.assertEqual(retrieved_data, test_data, "Redis caching should store and retrieve data correctly")
        
        # Clean up
        redis_client._conn.delete(test_key)


if __name__ == "__main__":
    unittest.main()
