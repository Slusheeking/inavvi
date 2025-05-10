"""
Test suite for the trading system.

This module provides the testing framework for the trading system components:
- Data source tests
- LLM integration tests
- Model tests
- Trading logic tests

Tests can be run using pytest.
"""

import datetime
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# Add path for importing source code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import trading system components for test fixtures
from src.models import PatternRecognitionModel, RankingModel, FinancialSentimentModel
from src.utils import get_logger

# Configure test logger
logger = get_logger(__name__)

# Test data constants
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
TEST_TIMEFRAMES = ["1m", "5m", "15m", "1h", "1d"]

# Ensure test data directory exists
os.makedirs(TEST_DATA_DIR, exist_ok=True)


# Common test utilities
def get_test_data_path(filename: str) -> str:
    """
    Get the path to a test data file.

    Args:
        filename: Name of test file

    Returns:
        Full path to the test file
    """
    return os.path.join(TEST_DATA_DIR, filename)


@contextmanager
def does_not_raise():
    """Context manager for asserting no exceptions are raised"""
    yield


def load_test_data(filename: str) -> Dict[str, Any]:
    """
    Load test data from a JSON file.

    Args:
        filename: JSON file to load

    Returns:
        Dict containing test data
    """
    filepath = get_test_data_path(filename)
    with open(filepath, "r") as f:
        return json.load(f)


def generate_ohlcv_dataframe(
    symbol: str = "AAPL", days: int = 30, interval: str = "1d", add_indicators: bool = False
) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.

    Args:
        symbol: Stock symbol
        days: Number of days of data
        interval: Time interval
        add_indicators: Whether to add technical indicators

    Returns:
        DataFrame with sample price data
    """
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)

    # Generate dates
    if interval == "1d":
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq=interval)

    # Generate random price data
    base_price = np.random.uniform(50, 200)
    price_volatility = base_price * 0.01

    # Create random walk
    np.random.seed(42)  # For reproducibility
    price_changes = np.random.normal(0, price_volatility, size=len(dates))
    closes = base_price + np.cumsum(price_changes)

    # Generate OHLC from close prices
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": closes * np.random.uniform(0.99, 1.01, size=len(dates)),
            "high": closes * np.random.uniform(1.01, 1.03, size=len(dates)),
            "low": closes * np.random.uniform(0.97, 0.99, size=len(dates)),
            "close": closes,
            "volume": np.random.randint(100000, 10000000, size=len(dates)),
        }
    )

    # Set symbol and interval
    df["symbol"] = symbol
    df["interval"] = interval

    # Add indicators if requested
    if add_indicators:
        # Simple Moving Averages
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()

        # Relative Strength Index
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

    return df


# Mock clients
def mock_redis_client() -> mock.MagicMock:
    """
    Get a mock Redis client for testing.

    Returns:
        Mock Redis client with common methods configured
    """
    redis_mock = mock.MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = True
    redis_mock.exists.return_value = False
    redis_mock.expire.return_value = True
    return redis_mock


def mock_db_client() -> mock.MagicMock:
    """
    Get a mock database client for testing.

    Returns:
        Mock database client with common methods configured
    """
    db_mock = mock.MagicMock()
    db_mock.execute_query.return_value = []
    db_mock.insert_data.return_value = True
    db_mock.update_data.return_value = True
    db_mock.delete_data.return_value = True
    return db_mock


def mock_llm_client() -> mock.MagicMock:
    """
    Get a mock LLM client for testing.

    Returns:
        Mock LLM client for predictable responses
    """
    llm_mock = mock.MagicMock()

    # Configure generate method to return predictable responses
    def mock_generate(prompt):
        if "sentiment" in prompt.lower():
            return {"sentiment": 0.75, "confidence": 0.85}
        elif "pattern" in prompt.lower():
            return {"pattern": "bullish_flag", "confidence": 0.82}
        else:
            return {"response": "This is a mock LLM response"}

    llm_mock.generate.side_effect = mock_generate
    return llm_mock


# Pytest fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Fixture that provides sample OHLCV data for multiple symbols"""
    data = {}
    for symbol in TEST_SYMBOLS[:3]:  # Limit to 3 symbols for performance
        data[symbol] = generate_ohlcv_dataframe(symbol=symbol, days=30)
    return data


@pytest.fixture
def mock_data_sources():
    """Fixture that provides mocked data sources"""
    with (
        mock.patch("src.data_sources.alpha_vantage_client") as alpha_mock,
        mock.patch("src.data_sources.polygon_client") as polygon_mock,
        mock.patch("src.data_sources.yahoo_finance_client") as yahoo_mock,
    ):
        # Configure mocks to return sample data
        for mock_client in [alpha_mock, polygon_mock, yahoo_mock]:
            mock_client.get_historical_data.return_value = generate_ohlcv_dataframe()
            mock_client.get_quote.return_value = {"price": 150.25, "change": 2.5}

        yield {"alpha_vantage": alpha_mock, "polygon": polygon_mock, "yahoo_finance": yahoo_mock}


# Export utilities
__all__ = [
    # Paths and data access
    "get_test_data_path",
    "load_test_data",
    "TEST_DATA_DIR",
    "TEST_SYMBOLS",
    "TEST_TIMEFRAMES",
    # Data generation
    "generate_ohlcv_dataframe",
    # Mock clients
    "mock_redis_client",
    "mock_db_client",
    "mock_llm_client",
    # Testing helpers
    "does_not_raise",
    # Pytest fixtures will be auto-discovered
]
