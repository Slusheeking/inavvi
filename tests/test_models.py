"""
Test module for the trading system models.

This test script ensures all models are working correctly:
- Tests model loading and initialization
- Tests prediction capabilities
- Checks integration between components
- Verifies data processing pipelines
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

# Try to import directly first
try:
    from src.models.sentiment import sentiment_model, analyze_sentiment
    from src.models.pattern_recognition import pattern_recognition_model, analyze_pattern
    from src.models.exit_optimization import exit_optimization_model, evaluate_exit_strategy
    from src.models.ranking_model import ranking_model
    from src.training.data_fetcher import DataFetcher
except ImportError:
    # If direct import fails, add project root to path and try again
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Now import should work
    from src.models.sentiment import sentiment_model, analyze_sentiment
    from src.models.pattern_recognition import pattern_recognition_model, analyze_pattern
    from src.models.exit_optimization import exit_optimization_model, evaluate_exit_strategy
    from src.models.ranking_model import ranking_model
    from src.training.data_fetcher import DataFetcher


class TestTradingSystem(unittest.TestCase):
    """Test suite for the trading system models."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test resources."""
        print("Setting up test resources...")
        
        # Initialize data fetcher for tests
        cls.data_fetcher = DataFetcher(
            data_days=30,  # Small window for testing
            use_polygon=False,  # No need for actual API calls
            use_alpha_vantage=False,
            use_redis_cache=False,
            data_dir="./test_data"
        )
        
        # Generate synthetic test data
        cls.test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
        cls.test_data = {}
        
        for symbol in cls.test_symbols:
            # Generate synthetic price data
            cls.test_data[symbol] = cls._generate_synthetic_data(symbol)
        
        # Generate a few news items for testing sentiment
        cls.test_news = [
            {
                "title": "Test company reports strong earnings",
                "summary": "The company exceeded analyst expectations with strong quarterly results.",
                "source": "Test Source",
                "published_at": datetime.now().isoformat()
            },
            {
                "title": "Test company faces challenges",
                "summary": "The company reported weaker than expected results due to market headwinds.",
                "source": "Test Source",
                "published_at": datetime.now().isoformat()
            },
            {
                "title": "Market remains stable amid uncertainty",
                "summary": "Despite global challenges, markets showed resilience in recent trading.",
                "source": "Test Source",
                "published_at": datetime.now().isoformat()
            }
        ]
        
        # Create test directory if it doesn't exist
        os.makedirs("./test_data", exist_ok=True)
        
        print("Test resources ready.")
    
    @staticmethod
    def _generate_synthetic_data(symbol: str) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Initialize with base price
        np.random.seed(hash(symbol) % 10000)
        base_price = np.random.uniform(50, 500)
        
        # Generate prices with random walk
        num_days = len(dates)
        returns = np.random.normal(0.0005, 0.015, num_days)
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV data
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.005, num_days))
        df.loc[df.index[0], 'open'] = prices[0] * 0.995
        
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.005, num_days)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.005, num_days)))
        df['volume'] = np.random.randint(50000, 5000000, num_days)
        
        # Fix any NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Add technical indicators
        # Moving averages
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # MACD
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Some additional features needed for ranking model
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        df['return_20d'] = df['close'].pct_change(20)
        df['return_60d'] = df['close'].pct_change(60)
        
        df['close_ma5_ratio'] = df['close'] / df['ma5'] - 1
        df['close_ma10_ratio'] = df['close'] / df['ma10'] - 1
        df['close_ma20_ratio'] = df['close'] / df['ma20'] - 1
        
        df['ma5_ma20_ratio'] = df['ma5'] / df['ma20'] - 1
        
        df['volatility_5d'] = df['close'].pct_change().rolling(5).std()
        df['volatility_10d'] = df['close'].pct_change().rolling(10).std()
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma10'] = df['volume'].rolling(10).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        
        df['volume_ratio_5d'] = df['volume'] / df['volume_ma5']
        df['volume_ratio_10d'] = df['volume'] / df['volume_ma10']
        df['volume_ratio_20d'] = df['volume'] / df['volume_ma20']
        
        # Additional features for models
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def test_sentiment_model(self):
        """Test the sentiment analysis model."""
        print("Testing sentiment model...")
        
        # Test initialization
        self.assertIsNotNone(sentiment_model, "Sentiment model should be initialized")
        
        # Test prediction
        test_text = "The company reported strong earnings that exceeded expectations."
        result = analyze_sentiment(test_text)
        
        # Check result structure
        self.assertIsInstance(result, dict)
        self.assertIn("sentiment", result)
        self.assertIn("entities", result)
        
        # Check sentiment scores
        sentiment = result["sentiment"]
        self.assertIn("sentiment_score", sentiment)
        
        # Test with news items
        for news_item in self.test_news:
            result = analyze_sentiment(news_item["title"] + ". " + news_item["summary"])
            self.assertIsInstance(result, dict)
            self.assertIn("sentiment", result)
        
        print("Sentiment model tests passed.")
    
    def test_pattern_recognition_model(self):
        """Test the pattern recognition model."""
        print("Testing pattern recognition model...")
        
        # Test initialization
        self.assertIsNotNone(pattern_recognition_model, "Pattern recognition model should be initialized")
        
        # Test on each test data
        for symbol, df in self.test_data.items():
            # Test pattern analysis
            result = analyze_pattern(df)
            
            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn("pattern", result)
            self.assertIn("signals", result)
            
            # Skip if not enough data
            if len(df) < pattern_recognition_model.window_size:
                continue
                
            # Check price levels
            self.assertIn("price_levels", result)
            levels = result["price_levels"]
            self.assertIn("current", levels)
            self.assertIn("support", levels)
            self.assertIn("resistance", levels)
            
            # Check technical indicators
            self.assertIn("technical_indicators", result)
            indicators = result["technical_indicators"]
            self.assertIn("rsi_14", indicators)
            self.assertIn("macd", indicators)
        
        print("Pattern recognition model tests passed.")
    
    def test_exit_optimization_model(self):
        """Test the exit optimization model."""
        print("Testing exit optimization model...")
        
        # Test initialization
        self.assertIsNotNone(exit_optimization_model, "Exit optimization model should be initialized")
        
        # Test on each test data
        for symbol, df in self.test_data.items():
            # Create test position data
            position_data = {
                "entry_price": df["close"].iloc[-20],
                "entry_time": df.index[-20],
                "position_size": 1.0,
                "stop_loss": df["close"].iloc[-20] * 0.95,
                "take_profit": df["close"].iloc[-20] * 1.05
            }
            
            # Test exit condition evaluation
            result = evaluate_exit_strategy(df, position_data)
            
            # Check result structure
            self.assertIsInstance(result, dict)
            self.assertIn("exit", result)
            self.assertIn("size", result)
            self.assertIn("reason", result)
            self.assertIn("confidence", result)
            
            # Check risk metrics
            self.assertIn("risk_metrics", result)
            metrics = result["risk_metrics"]
            self.assertIn("profit_pct", metrics)
            self.assertIn("max_drawdown", metrics)
        
        print("Exit optimization model tests passed.")
    
    def test_ranking_model(self):
        """Test the ranking model."""
        print("Testing ranking model...")
        
        # Test initialization
        self.assertIsNotNone(ranking_model, "Ranking model should be initialized")
        
        # Test with our test data
        ranking_data = {}
        for symbol, df in self.test_data.items():
            # Use just the latest row for ranking
            ranking_data[symbol] = df.iloc[-1:].copy()
        
        # Test the ranking
        scores = ranking_model.rank_stocks(ranking_data)
        
        # Check result structure
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0, "Should return scores for at least one stock")
        
        # Check if scores are between 0 and 1
        for symbol, score in scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Test explanation for one stock
        symbol = list(ranking_data.keys())[0]
        explanation = ranking_model.explain_ranking(ranking_data[symbol])
        
        # Check explanation structure
        self.assertIsInstance(explanation, dict)
        self.assertIn("score", explanation)
        self.assertIn("percentile", explanation)
        self.assertIn("factor_groups", explanation)
        
        print("Ranking model tests passed.")
    
    def test_model_integration(self):
        """Test integration between different models."""
        print("Testing model integration...")
        
        # Test a complete workflow
        # 1. Get price data
        symbol = self.test_symbols[0]
        df = self.test_data[symbol]
        
        # 2. Check for patterns
        pattern_result = analyze_pattern(df)
        
        # 3. If a signal exists, simulate a position
        if pattern_result["signals"]:
            # Create a position
            entry_price = df["close"].iloc[-20]
            position_data = {
                "entry_price": entry_price,
                "entry_time": df.index[-20],
                "position_size": 1.0,
                "stop_loss": entry_price * 0.95,
                "take_profit": entry_price * 1.05
            }
            
            # 4. Evaluate exit conditions
            exit_result = evaluate_exit_strategy(df, position_data)
            
            # Check that results are consistent
            if exit_result["exit"]:
                self.assertGreater(exit_result["size"], 0.0)
                self.assertGreater(exit_result["confidence"], 0.0)
        
        # 5. Use ranking model to select stocks
        # Get ranking scores
        ranking_model.rank_stocks({symbol: df.iloc[-1:].copy()})
        
        # 6. Get sentiment for news
        news_text = self.test_news[0]["title"] + ". " + self.test_news[0]["summary"]
        sentiment_result = analyze_sentiment(news_text)
        
        # Check sentiment score
        self.assertIn("sentiment", sentiment_result)
        self.assertIn("sentiment_score", sentiment_result["sentiment"])
        
        print("Model integration tests passed.")
    
    def test_torch_gpu_availability(self):
        """Test if PyTorch can use GPU if available."""
        print("Testing PyTorch GPU availability...")
        
        # This test checks if GPU is available and if models are using it
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            # Check if models are on GPU (if supported)
            if hasattr(sentiment_model, 'models') and sentiment_model.models and "primary" in sentiment_model.models:
                device = next(sentiment_model.models["primary"].parameters()).device
                self.assertEqual(device.type, "cuda", "Sentiment model should be on GPU")
        else:
            print("CUDA is not available, models should be on CPU")
            # Check if models are on CPU
            if hasattr(sentiment_model, 'models') and sentiment_model.models and "primary" in sentiment_model.models:
                device = next(sentiment_model.models["primary"].parameters()).device
                self.assertEqual(device.type, "cpu", "Sentiment model should be on CPU")
        
        print("PyTorch GPU tests completed.")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test resources."""
        print("Cleaning up test resources...")
        
        # Clean up test data directory
        if os.path.exists("./test_data"):
            for file in os.listdir("./test_data"):
                os.remove(os.path.join("./test_data", file))
            os.rmdir("./test_data")
        
        print("Test cleanup complete.")


if __name__ == "__main__":
    print("=" * 50)
    print("TRADING SYSTEM MODEL TESTS")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2)