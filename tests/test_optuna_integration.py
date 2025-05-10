"""
Tests for Optuna integration and performance improvements.
"""
import os
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import shutil

import pandas as pd
import numpy as np

from src.training.train_models import ModelTrainer
from src.training.data_fetcher import DataFetcher
from src.models.ranking_model import RankingModel


class TestOptunaIntegration(unittest.TestCase):
    """Test Optuna integration for hyperparameter optimization."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock Redis client
        self.redis_patcher = patch('src.utils.redis_client.redis_client')
        self.mock_redis = self.redis_patcher.start()
        
        # Create model trainer with mocked components
        self.model_trainer = ModelTrainer()
        self.model_trainer.data_dir = self.test_dir
        self.model_trainer.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.model_trainer.models_dir, exist_ok=True)
        
        # Mock data fetcher
        self.model_trainer.data_fetcher = MagicMock()
        
        # Create test data
        self._create_test_data()

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop Redis mock
        self.redis_patcher.stop()
        
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def _create_test_data(self):
        """Create test data for training."""
        # Create test features and targets for ranking model
        features = pd.DataFrame({
            'return_1d': np.random.randn(100),
            'return_5d': np.random.randn(100),
            'volume_1d': np.random.rand(100) * 1000000,
            'rsi_14': np.random.rand(100) * 100,
            'macd': np.random.randn(100),
            'bb_width': np.random.rand(100),
        })
        
        # Add all required features with random values
        for feature in RankingModel(None).feature_names:
            if feature not in features.columns:
                features[feature] = np.random.randn(100)
        
        targets = pd.Series(np.random.randint(0, 2, 100))
        
        # Set up data fetcher mock to return test data
        self.model_trainer.prepare_ranking_data = MagicMock(
            return_value=(features, targets, features.iloc[:20], targets.iloc[:20])
        )

    @patch('optuna.create_study')
    def test_optimize_hyperparameters(self, mock_create_study):
        """Test hyperparameter optimization with Optuna."""
        # Mock Optuna study
        mock_study = MagicMock()
        mock_study.best_params = {
            'learning_rate': 0.001,
            'hidden_dim_1': 64,
            'hidden_dim_2': 32,
            'hidden_dim_3': 16,
            'dropout': 0.3,
            'weight_decay': 0.0001,
            'batch_size': 32
        }
        mock_create_study.return_value = mock_study
        
        # Run optimization
        params = self.model_trainer.optimize_hyperparameters('ranking', n_trials=2)
        
        # Verify study was created and optimized
        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()
        
        # Verify parameters were returned
        self.assertEqual(params, mock_study.best_params)

    @patch('src.models.ranking_model.ranking_model.train')
    def test_train_with_optuna(self, mock_train):
        """Test training with Optuna hyperparameters."""
        # Mock train method
        mock_train.return_value = {'val_auc': [0.8]}
        
        # Mock optimize_hyperparameters
        self.model_trainer.optimize_hyperparameters = MagicMock(
            return_value={
                'learning_rate': 0.001,
                'hidden_dim_1': 64,
                'hidden_dim_2': 32,
                'hidden_dim_3': 16,
                'dropout': 0.3,
                'weight_decay': 0.0001,
                'batch_size': 32
            }
        )
        
        # Train model with Optuna
        result = self.model_trainer.train_ranking_model(use_optuna=True, n_trials=2)
        
        # Verify optimize_hyperparameters was called
        self.model_trainer.optimize_hyperparameters.assert_called_once_with('ranking', 2)
        
        # Verify train was called with optimized parameters
        mock_train.assert_called_once()
        
        # Verify parameters were included in result
        self.assertIn('params', result)
        self.assertEqual(result['params']['learning_rate'], 0.001)

    def test_model_compression(self):
        """Test model compression for faster inference."""
        # Create a ranking model
        model = RankingModel()
        
        # Mock save_model to avoid actual file operations
        model.save_model = MagicMock(return_value=True)
        
        # Compress model
        compressed_model = model.compress_model()
        
        # Verify model was compressed
        self.assertTrue(hasattr(compressed_model, '_is_compressed'))
        self.assertTrue(compressed_model._is_compressed)
        
        # Verify save_model was called
        model.save_model.assert_called_once()


class TestDataFetcherPerformance(unittest.TestCase):
    """Test performance improvements in DataFetcher."""

    def setUp(self):
        """Set up test fixtures."""
        # Create data fetcher
        self.data_fetcher = DataFetcher(use_redis_cache=True)
        
        # Mock Redis client
        self.redis_patcher = patch('src.utils.redis_client.redis_client')
        self.mock_redis = self.redis_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop Redis mock
        self.redis_patcher.stop()

    def test_get_cached_dataset(self):
        """Test get_cached_dataset method."""
        # Mock Redis get to return None (cache miss)
        self.mock_redis.get.return_value = None
        
        # Create generator function
        def generator_func():
            return {'data': 'test_data'}
        
        # Get dataset (should call generator_func)
        data = self.data_fetcher.get_cached_dataset('test_key', generator_func)
        
        # Verify Redis get was called
        self.mock_redis.get.assert_called_once_with('dataset:test_key')
        
        # Verify Redis set was called with generated data
        self.mock_redis.set.assert_called_once()
        
        # Verify correct data was returned
        self.assertEqual(data, {'data': 'test_data'})
        
        # Mock Redis get to return cached data
        self.mock_redis.get.reset_mock()
        self.mock_redis.set.reset_mock()
        self.mock_redis.get.return_value = {'data': 'cached_data'}
        
        # Get dataset again (should use cache)
        data = self.data_fetcher.get_cached_dataset('test_key', generator_func)
        
        # Verify Redis get was called
        self.mock_redis.get.assert_called_once_with('dataset:test_key')
        
        # Verify Redis set was not called
        self.mock_redis.set.assert_not_called()
        
        # Verify cached data was returned
        self.assertEqual(data, {'data': 'cached_data'})

    def test_batch_fetch_data(self):
        """Test batch_fetch_data method."""
        # Create mock fetch function
        fetch_func = MagicMock(return_value={'AAPL': 'data1', 'MSFT': 'data2'})
        
        # Fetch data in batches
        symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'FB']
        self.data_fetcher.batch_fetch_data(symbols, fetch_func, batch_size=2)
        
        # Verify fetch_func was called multiple times with batches
        self.assertEqual(fetch_func.call_count, 3)  # 5 symbols with batch size 2 = 3 calls
        
        # Verify all symbols were processed
        fetch_func.assert_any_call(['AAPL', 'MSFT'])
        fetch_func.assert_any_call(['GOOG', 'AMZN'])
        fetch_func.assert_any_call(['FB'])


if __name__ == '__main__':
    unittest.main()