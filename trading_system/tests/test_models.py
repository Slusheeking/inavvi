"""
Tests for ML models.
"""
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import torch

from src.models.pattern_recognition import PatternRecognitionModel, OHLCVDataset
from src.models.ranking_model import MultiFactorRankingModel
from src.models.sentiment import FinancialSentimentModel
from src.models.exit_optimization import ExitOptimizationModel

class TestPatternRecognitionModel(unittest.TestCase):
    """Test cases for PatternRecognitionModel."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test model without loading from file
        self.model = PatternRecognitionModel(model_path=None, lookback=20)
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=30)
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 30),
            'high': np.random.uniform(105, 115, 30),
            'low': np.random.uniform(95, 105, 30),
            'close': np.random.uniform(100, 110, 30),
            'volume': np.random.uniform(1000000, 2000000, 30)
        }, index=dates)
    
    def test_predict_pattern(self):
        """Test pattern prediction."""
        # Get prediction
        pattern_name, confidence = self.model.predict_pattern(self.sample_data)
        
        # Check that we get a string pattern name
        self.assertIsInstance(pattern_name, str)
        # Check that confidence is a float between 0 and 1
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_predict(self):
        """Test prediction method."""
        # Get prediction
        predictions = self.model.predict(self.sample_data)
        
        # Check that we get a dictionary of pattern probabilities
        self.assertIsInstance(predictions, dict)
        
        # Check that all values are floats between 0 and 1
        for pattern, prob in predictions.items():
            self.assertIsInstance(pattern, str)
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
        
        # Check that probabilities sum to approximately 1
        self.assertAlmostEqual(sum(predictions.values()), 1.0, places=5)
    
    def test_dataset(self):
        """Test OHLCV dataset."""
        # Create dataset
        dataset = OHLCVDataset([self.sample_data], [0], lookback=20)
        
        # Check length
        self.assertEqual(len(dataset), 1)
        
        # Get sample
        sample, label = dataset[0]
        
        # Check sample shape
        self.assertEqual(sample.shape, (5, 20))  # 5 channels, 20 bars
        
        # Check label
        self.assertEqual(label, 0)


class TestRankingModel(unittest.TestCase):
    """Test cases for MultiFactorRankingModel."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test model without loading from file
        self.model = MultiFactorRankingModel(model_path=None)
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=30)
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 30),
            'high': np.random.uniform(105, 115, 30),
            'low': np.random.uniform(95, 105, 30),
            'close': np.random.uniform(100, 110, 30),
            'volume': np.random.uniform(1000000, 2000000, 30)
        }, index=dates)
    
    def test_predict(self):
        """Test prediction method without a trained model."""
        # Since model is not trained, should return 0
        score = self.model.predict(self.sample_data)
        
        # Check that we get a float score
        self.assertIsInstance(score, float)
        # Without a trained model, score should be 0
        self.assertEqual(score, 0.0)
    
    def test_score_multiple(self):
        """Test scoring multiple stocks."""
        # Create multiple stocks
        stock_data = {
            'AAPL': self.sample_data.copy(),
            'MSFT': self.sample_data.copy(),
            'GOOGL': self.sample_data.copy()
        }
        
        # Get scores
        scores = self.model.score_multiple(stock_data)
        
        # Check that we get a dictionary of scores
        self.assertIsInstance(scores, dict)
        self.assertEqual(len(scores), 3)
        
        # Check that all values are floats
        for symbol, score in scores.items():
            self.assertIsInstance(symbol, str)
            self.assertIsInstance(score, float)


class TestSentimentModel(unittest.TestCase):
    """Test cases for FinancialSentimentModel."""
    
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def setUp(self, mock_tokenizer, mock_model):
        """Set up test environment with mocked transformer components."""
        # Mock tokenizer
        self.mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = self.mock_tokenizer
        
        # Mock model
        self.mock_model = MagicMock()
        mock_model.return_value = self.mock_model
        self.mock_model.config.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        # Create test model
        self.model = FinancialSentimentModel(model_path=None, model_name="test/model")
        
        # Sample news items
        self.sample_news = [
            {
                'title': 'Company reports strong earnings',
                'summary': 'Quarterly results exceed expectations.'
            },
            {
                'title': 'Market concerns grow',
                'summary': 'Investors worry about economic indicators.'
            }
        ]
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis."""
        # Mock tokenizer return value
        self.mock_tokenizer.return_value = {'input_ids': torch.zeros((1, 10)), 'attention_mask': torch.ones((1, 10))}
        
        # Mock model output
        class MockOutput:
            def __init__(self):
                self.logits = torch.tensor([[0.1, 0.3, 0.6]])
        
        self.mock_model.return_value = MockOutput()
        
        # Analyze sentiment
        sentiment = self.model.analyze_sentiment("Test text")
        
        # Check result
        self.assertIsInstance(sentiment, dict)
        self.assertEqual(len(sentiment), 3)
        self.assertIn('positive', sentiment)
        self.assertIn('neutral', sentiment)
        self.assertIn('negative', sentiment)
        
        # Positive should have highest score
        self.assertGreater(sentiment['positive'], sentiment['neutral'])
        self.assertGreater(sentiment['positive'], sentiment['negative'])
    
    def test_analyze_texts(self):
        """Test analyzing multiple texts."""
        # Mock tokenizer return value
        self.mock_tokenizer.return_value = {'input_ids': torch.zeros((2, 10)), 'attention_mask': torch.ones((2, 10))}
        
        # Mock model output
        class MockOutput:
            def __init__(self):
                self.logits = torch.tensor([
                    [0.1, 0.3, 0.6],  # First text: positive
                    [0.6, 0.3, 0.1]   # Second text: negative
                ])
        
        self.mock_model.return_value = MockOutput()
        
        # Analyze texts
        sentiments = self.model.analyze_texts(["First text", "Second text"])
        
        # Check results
        self.assertIsInstance(sentiments, list)
        self.assertEqual(len(sentiments), 2)
        self.assertGreater(sentiments[0]['positive'], sentiments[0]['negative'])
        self.assertGreater(sentiments[1]['negative'], sentiments[1]['positive'])


class TestExitOptimizationModel(unittest.TestCase):
    """Test cases for ExitOptimizationModel."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test model without loading from file
        self.model = ExitOptimizationModel(model_path=None)
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', periods=30)
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 30),
            'high': np.random.uniform(105, 115, 30),
            'low': np.random.uniform(95, 105, 30),
            'close': np.random.uniform(100, 110, 30),
            'volume': np.random.uniform(1000000, 2000000, 30)
        }, index=dates)
        
        # Sample position data
        self.position_data = {
            'symbol': 'AAPL',
            'entry_price': 100.0,
            'entry_time': dates[0],
            'quantity': 10,
            'stop_loss': 95.0,
            'take_profit': 110.0,
            'trailing_stop': 2.0,
            'max_time': 4.0  # 4 hours
        }
    
    def test_predict_exit_action(self):
        """Test exit action prediction."""
        # Get prediction
        action = self.model.predict_exit_action(self.sample_data, self.position_data)
        
        # Check result structure
        self.assertIsInstance(action, dict)
        self.assertIn('action', action)
        self.assertIn('confidence', action)
        self.assertIn('probabilities', action)
        
        # Check action is a valid action
        self.assertIn(action['action'], ['hold', 'exit_partial', 'exit_half', 'exit_full'])
        
        # Check confidence is a float between 0 and 1
        self.assertIsInstance(action['confidence'], float)
        self.assertGreaterEqual(action['confidence'], 0.0)
        self.assertLessEqual(action['confidence'], 1.0)
        
        # Check probabilities dict
        self.assertIsInstance(action['probabilities'], dict)
        self.assertEqual(len(action['probabilities']), 4)  # 4 possible actions
        
        # Check probabilities sum to approximately 1
        self.assertAlmostEqual(sum(action['probabilities'].values()), 1.0, places=5)
    
    def test_evaluate_exit_conditions(self):
        """Test exit condition evaluation."""
        # Get recommendation
        recommendation = self.model.evaluate_exit_conditions(
            self.sample_data, 
            self.position_data,
            confidence_threshold=0.5
        )
        
        # Check result structure
        self.assertIsInstance(recommendation, dict)
        self.assertIn('exit', recommendation)
        self.assertIn('size', recommendation)
        self.assertIn('reason', recommendation)
        self.assertIn('confidence', recommendation)
        self.assertIn('prediction', recommendation)
        
        # Check exit is boolean
        self.assertIsInstance(recommendation['exit'], bool)
        
        # Check size is a float between 0 and 1
        self.assertIsInstance(recommendation['size'], float)
        self.assertGreaterEqual(recommendation['size'], 0.0)
        self.assertLessEqual(recommendation['size'], 1.0)
        
        # Check reason is a string
        self.assertIsInstance(recommendation['reason'], str)
        
        # Check confidence is a float between 0 and 1
        self.assertIsInstance(recommendation['confidence'], float)
        self.assertGreaterEqual(recommendation['confidence'], 0.0)
        self.assertLessEqual(recommendation['confidence'], 1.0)
        
        # Check prediction is the same format as predict_exit_action result
        self.assertIsInstance(recommendation['prediction'], dict)
        self.assertIn('action', recommendation['prediction'])
        self.assertIn('confidence', recommendation['prediction'])
        self.assertIn('probabilities', recommendation['prediction'])
    
    def test_manual_exit_conditions(self):
        """Test manual exit conditions."""
        # Set up test cases
        test_cases = [
            # Stop loss test
            {
                'data': self.sample_data.copy(),
                'position': {**self.position_data, 'stop_loss': 105.0},  # Set stop above current price
                'expected': {'stop_loss_triggered': True}
            },
            # Take profit test
            {
                'data': self.sample_data.copy(),
                'position': {**self.position_data, 'take_profit': 105.0},  # Set target below current price
                'expected': {'take_profit_triggered': True}
            },
            # Trailing stop test
            {
                'data': pd.DataFrame({
                    'open': [100] * 30,
                    'high': [110] * 29 + [105],  # High point then drop
                    'low': [95] * 30,
                    'close': [100] * 28 + [108, 102],  # Drop more than 2% from high
                    'volume': [1000000] * 30
                }, index=pd.date_range(start='2023-01-01', periods=30)),
                'position': {**self.position_data, 'trailing_stop': 2.0},
                'expected': {'trailing_stop_triggered': True}
            }
        ]
        
        # Run test cases
        for case in test_cases:
            result = self.model._check_manual_exit_conditions(case['data'], case['position'])
            for key, value in case['expected'].items():
                self.assertEqual(result[key], value, f"Failed on condition: {key}")


if __name__ == '__main__':
    unittest.main()