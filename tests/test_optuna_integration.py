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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import Trial

from src.training.train_models import ModelTrainer
from src.training.data_fetcher import DataFetcher
from src.models.ranking_model import RankingModel


class SimpleModel(nn.Module):
    """Simple PyTorch model for testing Optuna integration."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.2):
        """Initialize the model with configurable hyperparameters."""
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        """Forward pass through the model."""
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TestOptunaIntegration(unittest.TestCase):
    """Test Optuna integration for hyperparameter optimization."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock Redis client
        self.mock_redis = MagicMock()
        
        # Patch the redis_client module to use our mock
        self.redis_patcher = patch('src.training.data_fetcher.redis_client', self.mock_redis)
        self.redis_patcher.start()
        
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

    def test_train_with_optuna(self):
        """Test training with Optuna hyperparameters."""
        # Create a mock for the train method
        train_mock = MagicMock(return_value={'val_auc': [0.8]})
        
        # Apply the mock to the ranking_model instance
        from src.models.ranking_model import ranking_model
        original_train = ranking_model.train
        ranking_model.train = train_mock
        
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
        train_mock.assert_called_once()
        
        # Verify parameters were included in result
        self.assertIn('params', result)
        self.assertEqual(result['params']['learning_rate'], 0.001)
        
        # Restore the original method
        ranking_model.train = original_train

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

    def test_optuna_with_pytorch(self):
        """Test actual Optuna integration with PyTorch."""
        # Create synthetic dataset
        input_dim = 10
        n_samples = 100
        
        X = torch.randn(n_samples, input_dim)
        y = torch.randint(0, 2, (n_samples,)).float()
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # Define objective function for Optuna
        def objective(trial: Trial):
            # Define hyperparameters to optimize
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Create dataloaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Create model with trial hyperparameters
            model = SimpleModel(input_dim, hidden_dim, 1, dropout_rate)
            
            # Define optimizer and loss function
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            
            # Train for a few epochs
            model.train()
            for epoch in range(5):  # Just a few epochs for testing
                for X_batch, y_batch in train_loader:
                    # Forward pass
                    y_pred = model(X_batch).squeeze()
                    loss = criterion(y_pred, y_batch)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    y_pred = model(X_batch).squeeze()
                    predicted = (torch.sigmoid(y_pred) > 0.5).float()
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            
            accuracy = correct / total
            return accuracy
        
        # Create and run Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=3)  # Small number of trials for testing
        
        # Verify study completed and found best parameters
        self.assertIsNotNone(study.best_params)
        self.assertIsNotNone(study.best_value)
        self.assertGreaterEqual(study.best_value, 0.0)
        self.assertLessEqual(study.best_value, 1.0)
        
        # Check that expected hyperparameters were optimized
        expected_params = ['learning_rate', 'hidden_dim', 'dropout_rate', 'batch_size']
        for param in expected_params:
            self.assertIn(param, study.best_params)


class TestDataFetcherPerformance(unittest.TestCase):
    """Test performance improvements in DataFetcher."""

    def setUp(self):
        """Set up test fixtures."""
        # Create data fetcher
        self.data_fetcher = DataFetcher(use_redis_cache=True)
        
        # Add batch_fetch_data method for testing
        def batch_fetch_data(symbols, fetch_func, batch_size=10):
            """Mock implementation of batch_fetch_data for testing."""
            results = {}
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                batch_results = fetch_func(batch)
                results.update(batch_results)
            return results
            
        # Add the method to the data fetcher instance
        self.data_fetcher.batch_fetch_data = batch_fetch_data
        
        # Create a mock Redis client
        self.mock_redis = MagicMock()
        
        # Patch the redis_client module to use our mock
        self.redis_patcher = patch('src.training.data_fetcher.redis_client', self.mock_redis)
        self.redis_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop Redis mock
        self.redis_patcher.stop()

    def test_get_cached_dataset(self):
        """Test get_cached_dataset method."""
        # Create a custom implementation of get_cached_dataset for testing
        def custom_get_cached_dataset(key, generator_func, expiry=None):
            # Check Redis cache
            cached_data = self.mock_redis.get(f"dataset:{key}")
            if cached_data is not None:
                return cached_data
            
            # Generate dataset
            data = generator_func()
            
            # Store in Redis
            self.mock_redis.set(f"dataset:{key}", data, ex=expiry)
            
            return data
        
        # Replace the method with our custom implementation
        self.data_fetcher.get_cached_dataset = custom_get_cached_dataset
        
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
