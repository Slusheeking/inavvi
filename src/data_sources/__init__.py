"""
Data sources module for the trading system.

This module provides interfaces to various market data providers:
- Alpha Vantage
- Polygon
- Yahoo Finance
"""

import logging
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Union

# Import provider-specific modules
from .alpha_vantage import (
    AlphaVantageAPI,
    AlphaVantageDataError,
    AlphaVantageInvalidParameterError,
    AlphaVantageRateLimitError,
    AlphaVantageSymbolNotFoundError,
    alpha_vantage_client,
    fetch_alpha_vantage_data,
)
from .polygon import PolygonAPI, PolygonDataError, fetch_polygon_data, polygon_client
from .yahoo_finance import (
    YahooFinanceAPI,
    YahooFinanceDataError,
    YahooFinanceInvalidIntervalError,
    YahooFinanceRateLimitError,
    YahooFinanceSymbolNotFoundError,
    fetch_yahoo_finance_data,
    yahoo_finance_client,
)

# Configure module logger
logger = logging.getLogger(__name__)


class DataProvider(Enum):
    """Enumeration of available data providers"""

    ALPHA_VANTAGE = auto()
    POLYGON = auto()
    YAHOO_FINANCE = auto()


# Error class mapping for consistent error handling
ERROR_MAPPING = {
    DataProvider.ALPHA_VANTAGE: {
        "data": AlphaVantageDataError,
        "rate_limit": AlphaVantageRateLimitError,
        "symbol_not_found": AlphaVantageSymbolNotFoundError,
        "invalid_parameter": AlphaVantageInvalidParameterError,
    },
    DataProvider.POLYGON: {
        "data": PolygonDataError,
        # Polygon doesn't have specific error types yet, using generic for now
    },
    DataProvider.YAHOO_FINANCE: {
        "data": YahooFinanceDataError,
        "rate_limit": YahooFinanceRateLimitError,
        "symbol_not_found": YahooFinanceSymbolNotFoundError,
        "invalid_parameter": YahooFinanceInvalidIntervalError,
    },
}

# Provider function mapping
PROVIDER_FUNCTIONS = {
    DataProvider.ALPHA_VANTAGE: fetch_alpha_vantage_data,
    DataProvider.POLYGON: fetch_polygon_data,
    DataProvider.YAHOO_FINANCE: fetch_yahoo_finance_data,
}

# Client instance mapping
PROVIDER_CLIENTS = {
    DataProvider.ALPHA_VANTAGE: alpha_vantage_client,
    DataProvider.POLYGON: polygon_client,
    DataProvider.YAHOO_FINANCE: yahoo_finance_client,
}


def fetch_market_data(provider: Union[DataProvider, str], *args, **kwargs) -> Dict[str, Any]:
    """
    Unified function to fetch market data from any provider.

    Args:
        provider: The data provider to use (enum or string name)
        *args, **kwargs: Arguments to pass to the specific provider's fetch function

    Returns:
        Dict containing the fetched market data

    Raises:
        ValueError: If the provider is not supported
        Various provider-specific errors are propagated
    """
    # Convert string provider to enum if needed
    if isinstance(provider, str):
        try:
            provider = DataProvider[provider.upper()]
        except KeyError:
            raise ValueError(f"Unknown provider: {provider}")

    # Get the appropriate fetch function
    if provider not in PROVIDER_FUNCTIONS:
        raise ValueError(f"Provider not supported: {provider}")

    fetch_func = PROVIDER_FUNCTIONS[provider]

    try:
        return fetch_func(*args, **kwargs)
    except Exception as e:
        logger.exception(f"Error fetching data from {provider.name}: {e}")
        raise


def get_client(provider: Union[DataProvider, str]) -> Any:
    """Get a client instance for the specified provider"""
    # Convert string provider to enum if needed
    if isinstance(provider, str):
        try:
            provider = DataProvider[provider.upper()]
        except KeyError:
            raise ValueError(f"Unknown provider: {provider}")

    if provider not in PROVIDER_CLIENTS:
        raise ValueError(f"Provider not supported: {provider}")

    return PROVIDER_CLIENTS[provider]


__all__ = [
    # High-level unified functions
    "fetch_market_data",
    "get_client",
    "DataProvider",
    # Provider-specific high-level functions
    "fetch_alpha_vantage_data",
    # Client instances
    "alpha_vantage_client",
    "polygon_client",
    "yahoo_finance_client",
    # Client classes
    "AlphaVantageAPI",
    "PolygonAPI",
    "YahooFinanceAPI",
    # Exception classes
    "AlphaVantageDataError",
    "AlphaVantageRateLimitError",
    "AlphaVantageSymbolNotFoundError",
    "AlphaVantageInvalidParameterError",
    "PolygonDataError",
    "YahooFinanceDataError",
    "YahooFinanceRateLimitError",
    "YahooFinanceSymbolNotFoundError",
    "YahooFinanceInvalidIntervalError",
]
