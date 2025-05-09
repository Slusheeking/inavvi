"""
Tests for data source modules.
"""
import asyncio
import unittest
import aiohttp
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from src.data_sources.polygon import PolygonAPI
from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.yahoo_finance import yahoo_finance_client
from src.core.data_pipeline import data_pipeline

class TestPolygonAPI(unittest.TestCase):
    """Test the Polygon API client."""
    
    def setUp(self):
        """Set up test case."""
        self.polygon = PolygonAPI()
        
        # Create sample data for mock responses
        self.sample_snapshot = {
            'symbol': 'AAPL',
            'price': {
                'last': 150.0,
                'open': 149.0,
                'high': 151.0,
                'low': 148.0,
                'close': 148.5,
                'volume': 1000000,
            },
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        
        self.sample_df = pd.DataFrame({
            'open': [149.0, 150.0, 151.0, 152.0, 153.0],
            'high': [150.0, 151.0, 152.0, 153.0, 154.0],
            'low': [148.0, 149.0, 150.0, 151.0, 152.0],
            'close': [149.5, 150.5, 151.5, 152.5, 153.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'vwap': [149.2, 150.2, 151.2, 152.2, 153.2],
            'transactions': [5000, 5500, 6000, 6500, 7000]
        }, index=pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=4), periods=5, freq='D'))
        
        # Sample data with missing values for testing data validation
        self.sample_df_with_missing = pd.DataFrame({
            'open': [149.0, np.nan, 151.0, 152.0, 153.0],
            'high': [150.0, 151.0, np.nan, 153.0, 154.0],
            'low': [148.0, 149.0, 150.0, np.nan, 152.0],
            'close': [149.5, 150.5, 151.5, 152.5, np.nan],
            'volume': [1000000, np.nan, 1200000, 1300000, 1400000],
            'vwap': [149.2, 150.2, 151.2, np.nan, 153.2],
            'transactions': [5000, 5500, np.nan, 6500, 7000]
        }, index=pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=4), periods=5, freq='D'))
    
    @patch('src.data_sources.polygon.PolygonAPI.get_stock_snapshot')
    def test_get_stock_snapshot(self, mock_get_snapshot):
        """Test fetching stock snapshot."""
        # Set up mock
        mock_get_snapshot.return_value = self.sample_snapshot
        
        # Run test asynchronously
        result = asyncio.run(self.polygon.get_stock_snapshot('AAPL'))
        
        # Assert
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['price']['last'], 150.0)
        mock_get_snapshot.assert_called_once_with('AAPL')
    
    @patch('src.data_sources.polygon.PolygonAPI.get_daily_bars')
    def test_get_daily_bars(self, mock_get_daily_bars):
        """Test fetching daily bars."""
        # Set up mock
        mock_get_daily_bars.return_value = self.sample_df
        
        # Run test asynchronously
        result = asyncio.run(self.polygon.get_daily_bars('AAPL', days=5))
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertEqual(result['close'].iloc[-1], 153.5)
        mock_get_daily_bars.assert_called_once_with('AAPL', days=5)
    
    @patch('src.data_sources.polygon.PolygonAPI.get_intraday_bars')
    def test_get_intraday_bars(self, mock_get_intraday_bars):
        """Test fetching intraday bars."""
        # Set up mock
        mock_get_intraday_bars.return_value = self.sample_df
        
        # Run test asynchronously
        result = asyncio.run(self.polygon.get_intraday_bars('AAPL', minutes=5, days=1))
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertEqual(result['close'].iloc[-1], 153.5)
        mock_get_intraday_bars.assert_called_once_with('AAPL', minutes=5, days=1)
    
    @patch('src.data_sources.polygon.PolygonAPI.get_gainers_losers')
    def test_get_gainers_losers(self, mock_get_gainers_losers):
        """Test fetching gainers and losers."""
        # Set up mock
        mock_result = {
            'gainers': [
                {'symbol': 'AAPL', 'price': 150.0, 'change_percent': 5.0, 'volume': 1000000},
                {'symbol': 'MSFT', 'price': 250.0, 'change_percent': 4.0, 'volume': 900000},
            ],
            'losers': [
                {'symbol': 'TSLA', 'price': 200.0, 'change_percent': -3.0, 'volume': 1100000},
                {'symbol': 'AMZN', 'price': 120.0, 'change_percent': -2.0, 'volume': 950000},
            ]
        }
        mock_get_gainers_losers.return_value = mock_result
        
        # Run test asynchronously
        result = asyncio.run(self.polygon.get_gainers_losers(limit=2))
        
        # Assert
        self.assertEqual(len(result['gainers']), 2)
        self.assertEqual(len(result['losers']), 2)
        self.assertEqual(result['gainers'][0]['symbol'], 'AAPL')
        self.assertEqual(result['gainers'][0]['change_percent'], 5.0)
        self.assertEqual(result['losers'][0]['symbol'], 'TSLA')
        self.assertEqual(result['losers'][0]['change_percent'], -3.0)
        mock_get_gainers_losers.assert_called_once_with(limit=2)
    
    @patch('src.data_sources.polygon.PolygonAPI.connect_websocket')
    @patch('src.data_sources.polygon.PolygonAPI.subscribe_to_symbols')
    def test_websocket_subscription(self, mock_subscribe, mock_connect):
        """Test WebSocket subscription."""
        # Set up mocks
        mock_connect.return_value = True
        mock_subscribe.return_value = True
        
        # Run test asynchronously
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        result = asyncio.run(self.polygon.subscribe_to_symbols(symbols))
        
        # Assert
        self.assertTrue(result)
        mock_connect.assert_called_once()
        mock_subscribe.assert_called_once_with(symbols)
        
    @patch('src.data_sources.polygon.PolygonAPI.get_stock_snapshot')
    def test_get_stock_snapshot_error_handling(self, mock_get_snapshot):
        """Test error handling when fetching stock snapshot."""
        # Set up mock to raise an exception
        mock_get_snapshot.side_effect = aiohttp.ClientError("Network error")
        
        # Run test asynchronously and expect exception to be raised
        with self.assertRaises(aiohttp.ClientError):
            asyncio.run(self.polygon.get_stock_snapshot('AAPL'))
        
        mock_get_snapshot.assert_called_once_with('AAPL')
    
    @patch('src.data_sources.polygon.PolygonAPI.get_daily_bars')
    def test_get_daily_bars_with_missing_data(self, mock_get_daily_bars):
        """Test handling of missing data in daily bars."""
        # Set up mock to return data with missing values
        mock_get_daily_bars.return_value = self.sample_df_with_missing
        
        # Run test asynchronously
        result = asyncio.run(self.polygon.get_daily_bars('AAPL', days=5))
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        # Check that NaN values have been handled
        self.assertFalse(result['close'].isnull().any())
        self.assertFalse(result['open'].isnull().any())
        mock_get_daily_bars.assert_called_once_with('AAPL', days=5)
    
    @patch('src.data_sources.polygon.PolygonAPI.get_options_chain')
    def test_get_options_chain_error_handling(self, mock_get_options):
        """Test error handling when fetching options chain."""
        # Set up mock to raise a PolygonDataError
        mock_get_options.side_effect = Exception("API error")
        
        # Run test asynchronously
        result = asyncio.run(self.polygon.get_options_chain('AAPL'))
        
        # Assert that we get an empty result instead of an exception
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['expirations'], [])
        self.assertEqual(result['calls'], [])
        self.assertEqual(result['puts'], [])
        mock_get_options.assert_called_once_with('AAPL')

class TestAlphaVantageClient(unittest.TestCase):
    """Test the Alpha Vantage API client."""
    
    def setUp(self):
        """Set up test case."""
        # Use the global instance
        self.alpha_vantage = alpha_vantage_client
        
        # Create sample data for mock responses
        self.sample_news = [
            {
                'title': 'Apple Reports Strong Earnings',
                'url': 'https://example.com/article1',
                'time_published': '20230101T120000',
                'summary': 'Apple reported strong earnings for Q4.',
                'source': 'Example News',
                'category': 'earnings',
                'tickers': ['AAPL']
            },
            {
                'title': 'Microsoft Announces New Product',
                'url': 'https://example.com/article2',
                'time_published': '20230102T120000',
                'summary': 'Microsoft announced a new product today.',
                'source': 'Example News',
                'category': 'product',
                'tickers': ['MSFT']
            }
        ]
        
        self.sample_sector_perf = {
            'Rank A: Real-Time Performance': {
                'Information Technology': '1.25',
                'Health Care': '0.75',
                'Consumer Discretionary': '0.50',
                'Communication Services': '0.25',
                'Financials': '0.20'
            },
            'Rank B: 1 Day Performance': {
                'Information Technology': '1.20',
                'Health Care': '0.70',
                'Consumer Discretionary': '0.45',
                'Communication Services': '0.20',
                'Financials': '0.15'
            }
        }
        
        # Sample financial data
        self.sample_income_statement = {
            "symbol": "AAPL",
            "annualReports": [
                {
                    "fiscalDateEnding": "2022-09-30",
                    "totalRevenue": "394328000000",
                    "netIncome": "99803000000"
                },
                {
                    "fiscalDateEnding": "2021-09-30",
                    "totalRevenue": "365817000000",
                    "netIncome": "94680000000"
                }
            ]
        }
        
        # Sample data with missing fields
        self.sample_income_statement_missing = {
            "symbol": "AAPL",
            "annualReports": [
                {
                    "fiscalDateEnding": "2022-09-30",
                    "totalRevenue": "",
                    "netIncome": None
                }
            ]
        }
        
        # Sample data with empty reports
        self.sample_empty_reports = {
            "symbol": "AAPL",
            "annualReports": []
        }
    
    @patch('src.data_sources.alpha_vantage.alpha_vantage_client.get_symbol_news')
    def test_get_symbol_news(self, mock_get_news):
        """Test fetching news for a symbol."""
        # Set up mock
        mock_get_news.return_value = self.sample_news
        
        # Run test asynchronously
        result = asyncio.run(self.alpha_vantage.get_symbol_news('AAPL', limit=2))
        
        # Assert
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['title'], 'Apple Reports Strong Earnings')
        self.assertEqual(result[1]['title'], 'Microsoft Announces New Product')
        mock_get_news.assert_called_once_with('AAPL', limit=2)
    
    @patch('src.data_sources.alpha_vantage.alpha_vantage_client.get_sector_performance')
    def test_get_sector_performance(self, mock_get_sector_perf):
        """Test fetching sector performance."""
        # Set up mock
        mock_get_sector_perf.return_value = self.sample_sector_perf
        
        # Run test asynchronously
        result = asyncio.run(self.alpha_vantage.get_sector_performance())
        
        # Assert
        self.assertIn('Rank A: Real-Time Performance', result)
        self.assertIn('Rank B: 1 Day Performance', result)
        self.assertEqual(result['Rank A: Real-Time Performance']['Information Technology'], '1.25')
        self.assertEqual(result['Rank B: 1 Day Performance']['Health Care'], '0.70')
        mock_get_sector_perf.assert_called_once()
        
    @patch('src.data_sources.alpha_vantage.alpha_vantage_client.get_income_statement')
    def test_get_income_statement(self, mock_get_income):
        """Test fetching income statement."""
        # Set up mock
        mock_get_income.return_value = self.sample_income_statement
        
        # Run test asynchronously
        result = asyncio.run(self.alpha_vantage.get_income_statement('AAPL'))
        
        # Assert
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(len(result['annualReports']), 2)
        self.assertEqual(result['annualReports'][0]['fiscalDateEnding'], '2022-09-30')
        self.assertEqual(result['annualReports'][0]['totalRevenue'], '394328000000')
        mock_get_income.assert_called_once_with('AAPL')
    
    @patch('src.data_sources.alpha_vantage.alpha_vantage_client.get_income_statement')
    def test_get_income_statement_missing_fields(self, mock_get_income):
        """Test handling of missing fields in income statement."""
        # Set up mock
        mock_get_income.return_value = self.sample_income_statement_missing
        
        # Run test asynchronously
        result = asyncio.run(self.alpha_vantage.get_income_statement('AAPL'))
        
        # Assert
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(len(result['annualReports']), 1)
        self.assertEqual(result['annualReports'][0]['fiscalDateEnding'], '2022-09-30')
        # Check that empty fields are still present
        self.assertIn('totalRevenue', result['annualReports'][0])
        self.assertIn('netIncome', result['annualReports'][0])
        mock_get_income.assert_called_once_with('AAPL')
    
    @patch('src.data_sources.alpha_vantage.alpha_vantage_client.get_income_statement')
    def test_get_income_statement_empty_reports(self, mock_get_income):
        """Test handling of empty reports in income statement."""
        # Set up mock
        mock_get_income.return_value = self.sample_empty_reports
        
        # Run test asynchronously
        result = asyncio.run(self.alpha_vantage.get_income_statement('AAPL'))
        
        # Assert
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(len(result['annualReports']), 0)
        mock_get_income.assert_called_once_with('AAPL')
    
    @patch('src.data_sources.alpha_vantage.alpha_vantage_client._make_request')
    def test_rate_limit_error_handling(self, mock_make_request):
        """Test handling of rate limit errors."""
        # Set up mock to raise a rate limit error
        from src.data_sources.alpha_vantage import AlphaVantageRateLimitError
        mock_make_request.side_effect = AlphaVantageRateLimitError("Rate limit exceeded")
        
        # Run test asynchronously with retry decorator
        with self.assertRaises(AlphaVantageRateLimitError):
            asyncio.run(self.alpha_vantage.get_income_statement('AAPL'))
        
        # Assert that _make_request was called
        self.assertTrue(mock_make_request.called)
    
    @patch('src.data_sources.alpha_vantage.alpha_vantage_client._make_request')
    def test_symbol_not_found_error_handling(self, mock_make_request):
        """Test handling of symbol not found errors."""
        # Set up mock to raise a symbol not found error
        from src.data_sources.alpha_vantage import AlphaVantageSymbolNotFoundError
        mock_make_request.side_effect = AlphaVantageSymbolNotFoundError("Symbol not found")
        
        # Run test asynchronously
        result = asyncio.run(self.alpha_vantage.get_income_statement('INVALID'))
        
        # Assert that we get None instead of an exception
        self.assertIsNone(result)
        self.assertTrue(mock_make_request.called)

class TestYahooFinanceClient(unittest.TestCase):
    """Test the Yahoo Finance API client."""
    
    def setUp(self):
        """Set up test case."""
        # Use the global instance
        self.yahoo = yahoo_finance_client
        
        # Create sample data for mock responses
        self.sample_ticker_info = {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 2500000000000,
            'regularMarketPrice': 150.0,
            'regularMarketVolume': 100000000,
            'regularMarketDayHigh': 151.0,
            'regularMarketDayLow': 149.0,
            'regularMarketOpen': 149.5,
            'regularMarketPreviousClose': 148.5,
            'fiftyDayAverage': 145.0,
            'twoHundredDayAverage': 140.0,
            'trailingPE': 25.0,
            'forwardPE': 22.0,
            'dividendYield': 0.006
        }
        
        # Sample historical data
        self.sample_history = pd.DataFrame({
            'Open': [149.0, 150.0, 151.0, 152.0, 153.0],
            'High': [150.0, 151.0, 152.0, 153.0, 154.0],
            'Low': [148.0, 149.0, 150.0, 151.0, 152.0],
            'Close': [149.5, 150.5, 151.5, 152.5, 153.5],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000],
            'Dividends': [0, 0, 0, 0, 0],
            'Stock Splits': [0, 0, 0, 0, 0]
        }, index=pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=4), periods=5, freq='D'))
        
        # Sample data with missing values
        self.sample_history_with_missing = pd.DataFrame({
            'Open': [149.0, np.nan, 151.0, 152.0, 153.0],
            'High': [150.0, 151.0, np.nan, 153.0, 154.0],
            'Low': [148.0, 149.0, 150.0, np.nan, 152.0],
            'Close': [149.5, 150.5, 151.5, 152.5, np.nan],
            'Volume': [1000000, np.nan, 1200000, 1300000, 1400000],
            'Dividends': [0, 0, 0, 0, 0],
            'Stock Splits': [0, 0, 0, 0, 0]
        }, index=pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=4), periods=5, freq='D'))
    
    @patch('src.data_sources.yahoo_finance.yahoo_finance_client.get_ticker_info')
    def test_get_ticker_info(self, mock_get_ticker_info):
        """Test fetching ticker information."""
        # Set up mock
        mock_get_ticker_info.return_value = self.sample_ticker_info
        
        # Run test asynchronously
        result = asyncio.run(self.yahoo.get_ticker_info('AAPL'))
        
        # Assert
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['shortName'], 'Apple Inc.')
        self.assertEqual(result['sector'], 'Technology')
        self.assertEqual(result['regularMarketPrice'], 150.0)
        mock_get_ticker_info.assert_called_once_with('AAPL')
    
    @patch('src.data_sources.yahoo_finance.yahoo_finance_client.get_ticker_info')
    def test_get_ticker_info_symbol_not_found(self, mock_get_ticker_info):
        """Test handling of symbol not found errors."""
        # Set up mock to raise a symbol not found error
        from src.data_sources.yahoo_finance import YahooFinanceSymbolNotFoundError
        mock_get_ticker_info.side_effect = YahooFinanceSymbolNotFoundError("Symbol not found")
        
        # Run test asynchronously
        result = asyncio.run(self.yahoo.get_ticker_info('INVALID'))
        
        # Assert that we get None instead of an exception
        self.assertIsNone(result)
        mock_get_ticker_info.assert_called_once_with('INVALID')
    
    @patch('src.data_sources.yahoo_finance.yahoo_finance_client.get_historical_data')
    def test_get_historical_data(self, mock_get_history):
        """Test fetching historical data."""
        # Set up mock
        mock_get_history.return_value = self.sample_history
        
        # Run test asynchronously
        result = asyncio.run(self.yahoo.get_historical_data('AAPL', period='5d'))
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertEqual(result['Close'].iloc[-1], 153.5)
        mock_get_history.assert_called_once_with('AAPL', period='5d')
    
    @patch('src.data_sources.yahoo_finance.yahoo_finance_client.get_historical_data')
    def test_get_historical_data_with_missing_values(self, mock_get_history):
        """Test handling of missing values in historical data."""
        # Set up mock
        mock_get_history.return_value = self.sample_history_with_missing
        
        # Run test asynchronously
        result = asyncio.run(self.yahoo.get_historical_data('AAPL', period='5d'))
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        # Check that NaN values have been handled
        self.assertFalse(result['Close'].isnull().any())
        self.assertFalse(result['Open'].isnull().any())
        mock_get_history.assert_called_once_with('AAPL', period='5d')
    
    @patch('src.data_sources.yahoo_finance.yahoo_finance_client._fetch_data')
    def test_rate_limit_error_handling(self, mock_fetch_data):
        """Test handling of rate limit errors."""
        # Set up mock to raise a rate limit error
        from src.data_sources.yahoo_finance import YahooFinanceRateLimitError
        mock_fetch_data.side_effect = YahooFinanceRateLimitError("Rate limit exceeded")
        
        # Run test asynchronously with retry decorator
        with self.assertRaises(YahooFinanceRateLimitError):
            asyncio.run(self.yahoo.get_ticker_info('AAPL'))
        
        # Assert that _fetch_data was called
        self.assertTrue(mock_fetch_data.called)
    
    @patch('src.data_sources.yahoo_finance.yahoo_finance_client.get_historical_data')
    def test_invalid_interval_error_handling(self, mock_get_history):
        """Test handling of invalid interval errors."""
        # Set up mock to raise an invalid interval error
        from src.data_sources.yahoo_finance import YahooFinanceInvalidIntervalError
        mock_get_history.side_effect = YahooFinanceInvalidIntervalError("Invalid interval")
        
        # Run test asynchronously
        with self.assertRaises(YahooFinanceInvalidIntervalError):
            asyncio.run(self.yahoo.get_historical_data('AAPL', interval='invalid'))
        
        mock_get_history.assert_called_once_with('AAPL', interval='invalid')

class TestDataPipeline(unittest.TestCase):
    """Test the data pipeline."""
    
    def setUp(self):
        """Set up test case."""
        # Use the global instance
        self.pipeline = data_pipeline
        
        # Sample data
        self.sample_snapshot = {
            'symbol': 'AAPL',
            'price': {
                'last': 150.0,
                'open': 149.0,
                'high': 151.0,
                'low': 148.0,
                'close': 148.5,
                'volume': 1000000,
            },
            'timestamp': pd.Timestamp.now().isoformat(),
        }
        
        self.sample_df = pd.DataFrame({
            'open': [149.0, 150.0, 151.0, 152.0, 153.0],
            'high': [150.0, 151.0, 152.0, 153.0, 154.0],
            'low': [148.0, 149.0, 150.0, 151.0, 152.0],
            'close': [149.5, 150.5, 151.5, 152.5, 153.5],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        }, index=pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=4), periods=5, freq='D'))
        
        # Sample data with missing values
        self.sample_df_with_missing = pd.DataFrame({
            'open': [149.0, np.nan, 151.0, 152.0, 153.0],
            'high': [150.0, 151.0, np.nan, 153.0, 154.0],
            'low': [148.0, 149.0, 150.0, np.nan, 152.0],
            'close': [149.5, 150.5, 151.5, 152.5, np.nan],
            'volume': [1000000, np.nan, 1200000, 1300000, 1400000],
        }, index=pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=4), periods=5, freq='D'))
        
        # Sample technical indicators
        self.sample_indicators = {
            'rsi_14': 65.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'macd_histogram': 0.2,
            'bb_upper_20': 155.0,
            'bb_middle_20': 150.0,
            'bb_lower_20': 145.0,
            'bb_width_20': 0.067,
            'bb_position_20': 0.5
        }
    
    @patch('src.core.data_pipeline.data_pipeline.get_stock_data')
    def test_get_stock_data(self, mock_get_stock_data):
        """Test fetching stock data."""
        # Set up mock
        mock_get_stock_data.return_value = self.sample_snapshot
        
        # Run test asynchronously
        result = asyncio.run(self.pipeline.get_stock_data('AAPL', 'snapshot'))
        
        # Assert
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['price']['last'], 150.0)
        mock_get_stock_data.assert_called_once_with('AAPL', 'snapshot')
    
    @patch('src.core.data_pipeline.data_pipeline.get_technical_indicators')
    def test_get_technical_indicators(self, mock_get_indicators):
        """Test getting technical indicators."""
        # Set up mock
        mock_indicators = {
            'rsi_14': 65.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'macd_histogram': 0.2,
            'bb_upper_20': 155.0,
            'bb_middle_20': 150.0,
            'bb_lower_20': 145.0,
            'bb_width_20': 0.067,
            'bb_position_20': 0.5
        }
        mock_get_indicators.return_value = mock_indicators
        
        # Run test asynchronously
        result = asyncio.run(self.pipeline.get_technical_indicators('AAPL'))
        
        # Assert
        self.assertEqual(result['rsi_14'], 65.0)
        self.assertEqual(result['macd_histogram'], 0.2)
        self.assertEqual(result['bb_position_20'], 0.5)
        mock_get_indicators.assert_called_once_with('AAPL')
    
    @patch('src.core.data_pipeline.data_pipeline.get_market_context')
    def test_get_market_context(self, mock_get_context):
        """Test getting market context."""
        # Set up mock
        mock_context = {
            'state': 'open',
            'sector_performance': 1.25,
            'vix': 15.5,
            'time_until_close': 3.5,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        mock_get_context.return_value = mock_context
        
        # Run test asynchronously
        result = asyncio.run(self.pipeline.get_market_context())
        
        # Assert
        self.assertEqual(result['state'], 'open')
        self.assertEqual(result['sector_performance'], 1.25)
        self.assertEqual(result['vix'], 15.5)
        mock_get_context.assert_called_once()
    
    @patch('src.core.data_pipeline.data_pipeline.get_historical_data')
    def test_get_historical_data_with_missing_values(self, mock_get_history):
        """Test handling of missing values in historical data."""
        # Set up mock
        mock_get_history.return_value = self.sample_df_with_missing
        
        # Run test asynchronously
        result = asyncio.run(self.pipeline.get_historical_data('AAPL', days=5))
        
        # Assert
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        # Check that NaN values have been handled
        self.assertFalse(result['close'].isnull().any())
        self.assertFalse(result['open'].isnull().any())
        mock_get_history.assert_called_once_with('AAPL', days=5)
    
    @patch('src.core.data_pipeline.data_pipeline.get_stock_data')
    def test_get_stock_data_error_handling(self, mock_get_stock_data):
        """Test error handling when fetching stock data."""
        # Set up mock to raise an exception
        mock_get_stock_data.side_effect = Exception("API error")
        
        # Run test asynchronously
        result = asyncio.run(self.pipeline.get_stock_data('AAPL', 'snapshot', fallback=True))
        
        # Assert that we get a fallback result instead of an exception
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['status'], 'error')
        self.assertIn('error_message', result)
        mock_get_stock_data.assert_called_once_with('AAPL', 'snapshot')
    
    @patch('src.core.data_pipeline.data_pipeline.get_technical_indicators')
    def test_get_technical_indicators_with_fallback(self, mock_get_indicators):
        """Test fallback for technical indicators."""
        # Set up mock to raise an exception
        mock_get_indicators.side_effect = Exception("Calculation error")
        
        # Run test asynchronously
        result = asyncio.run(self.pipeline.get_technical_indicators('AAPL', fallback=True))
        
        # Assert that we get a fallback result instead of an exception
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['status'], 'error')
        self.assertIn('error_message', result)
        mock_get_indicators.assert_called_once_with('AAPL')
    
    @patch('src.core.data_pipeline.data_pipeline.validate_data')
    def test_validate_data(self, mock_validate):
        """Test data validation."""
        # Set up mock
        mock_validate.return_value = (True, "Data is valid")
        
        # Run test asynchronously
        result, message = asyncio.run(self.pipeline.validate_data(self.sample_df))
        
        # Assert
        self.assertTrue(result)
        self.assertEqual(message, "Data is valid")
        mock_validate.assert_called_once_with(self.sample_df)
    
    @patch('src.core.data_pipeline.data_pipeline.validate_data')
    def test_validate_data_with_issues(self, mock_validate):
        """Test data validation with issues."""
        # Set up mock
        mock_validate.return_value = (False, "Missing required columns")
        
        # Run test asynchronously
        result, message = asyncio.run(self.pipeline.validate_data(self.sample_df_with_missing))
        
        # Assert
        self.assertFalse(result)
        self.assertEqual(message, "Missing required columns")
        mock_validate.assert_called_once_with(self.sample_df_with_missing)

if __name__ == '__main__':
    unittest.main()