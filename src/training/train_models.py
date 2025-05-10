"""
Automated training system for all ML models in the trading system.

This script handles:
- Data downloading and preprocessing from real APIs
- Hyperparameter optimization with Optuna
- Training and validation of all models
- Experiment tracking with MLflow
- GPU acceleration for training
- Model versioning and deployment
- Scheduled off-hours training

Run this script with:
`python -m src.training.train_models --model all --optimize True --gpu True`
"""

import optuna
import mlflow
import numpy as np
import pandas as pd
import os
import time
import logging
import warnings
import json
import yaml
import shutil
import hashlib
import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, List, Union, Any, Callable, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("model_trainer")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Check GPU availability before imports to avoid errors
try:
    # Try PyTorch first
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    GPU_COUNT = torch.cuda.device_count() if GPU_AVAILABLE else 0
    logger.info(f"PyTorch detected {GPU_COUNT} GPUs")
except ImportError:
    logger.warning("Could not check GPU availability with PyTorch")
    GPU_AVAILABLE = False
    GPU_COUNT = 0

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client
from src.training.data_fetcher import DataFetcher

# Import model modules
from src.models.pattern_recognition import PatternRecognitionModel
from src.models.ranking_model import RankingModel
from src.models.sentiment import FinancialSentimentModel
from src.models.exit_optimization import ExitOptimizationModel

# Import GPU-capable libraries if GPU is available
try:
    if GPU_AVAILABLE:
        # Import PyTorch for GPU
        import torch.cuda
        
        # Import XGBoost with GPU support
        import xgboost as xgb
        
        # Import other ML libraries with GPU support
        import lightgbm as lgb
        import catboost as cb
        
        logger.info("Successfully imported GPU-capable libraries")
        USE_GPU = True
    else:
        USE_GPU = False
except Exception as e:
    logger.warning(f"Could not import GPU libraries: {e}")
    USE_GPU = False

# Initialize MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


class ModelTrainer:
    """
    Centralized system for training all ML models in the trading system.
    
    Features:
    - Automated data processing and feature engineering
    - Hyperparameter optimization
    - GPU acceleration
    - Experiment tracking
    - Model versioning and deployment
    """
    
    def __init__(
        self,
        models_to_train: List[str] = None,
        data_days: int = 365,
        use_gpu: bool = False,
        optimize_hyperparams: bool = True,
        optimization_trials: int = 50,
        use_mlflow: bool = True,
        mlflow_experiment_name: str = "trading_system_training"
    ):
        """
        Initialize the model trainer.
        
        Args:
            models_to_train: List of models to train ['pattern', 'ranking', 'sentiment', 'exit', 'all']
            data_days: Number of days of historical data to use
            use_gpu: Whether to use GPU acceleration if available
            optimize_hyperparams: Whether to run hyperparameter optimization
            optimization_trials: Number of trials for hyperparameter optimization
            use_mlflow: Whether to use MLflow for experiment tracking
            mlflow_experiment_name: Name of the MLflow experiment
        """
        # Configuration
        self.models_to_train = models_to_train or ["all"]
        if "all" in self.models_to_train:
            self.models_to_train = ["pattern", "ranking", "sentiment", "exit"]
        
        self.data_days = data_days
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.optimize_hyperparams = optimize_hyperparams
        self.optimization_trials = optimization_trials
        self.use_mlflow = use_mlflow
        
        # Initialize data fetcher
        self.data_fetcher = DataFetcher(
            data_days=data_days,
            use_polygon=True,
            use_alpha_vantage=True,
            use_redis_cache=True
        )
        
        # Data storage
        self.data = {}
        self.features = {}
        self.training_datasets = {}
        self.validation_datasets = {}
        
        # Model storage
        self.models = {}
        self.hyperparameters = {}
        self.model_metrics = {}
        self.best_params = {}
        
        # Performance tracking
        self.training_times = {}
        self.optimization_times = {}
        
        # Versioning
        self.model_versions = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # MLflow setup
        if self.use_mlflow:
            self.mlflow_experiment_name = mlflow_experiment_name
            mlflow.set_experiment(mlflow_experiment_name)
            logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            logger.info(f"MLflow experiment name: {mlflow_experiment_name}")
        
        # Initialize GPU if requested
        if self.use_gpu:
            self._setup_gpu()
            logger.info(f"Initialized with GPU acceleration enabled")
        else:
            logger.info(f"Initialized with CPU processing")
        
        # Ensure directories exist
        os.makedirs(settings.data_dir, exist_ok=True)
        os.makedirs(settings.models_dir, exist_ok=True)
        os.makedirs(Path(settings.models_dir) / "pattern", exist_ok=True)
        os.makedirs(Path(settings.models_dir) / "ranking", exist_ok=True)
        os.makedirs(Path(settings.models_dir) / "sentiment", exist_ok=True)
        os.makedirs(Path(settings.models_dir) / "exit", exist_ok=True)
        os.makedirs(Path(settings.models_dir) / "archive", exist_ok=True)
        
        logger.info(f"ModelTrainer initialized to train: {', '.join(self.models_to_train)}")
    
    def _setup_gpu(self):
        """Setup GPU acceleration for training."""
        if not GPU_AVAILABLE:
            logger.warning("GPU requested but not available")
            self.use_gpu = False
            return
        
        try:
            # Set PyTorch to use GPU
            torch.cuda.set_device(0)
            
            # Configure XGBoost for GPU
            self.xgb_params = {"tree_method": "gpu_hist", "device": "cuda"}
            
            # Test GPU memory
            test_tensor = torch.zeros((1000, 1000), device='cuda')
            del test_tensor
            torch.cuda.empty_cache()
            
            logger.info(f"GPU setup completed successfully")
        except Exception as e:
            logger.error(f"Error setting up GPU: {e}")
            self.use_gpu = False
            logger.warning("Falling back to CPU processing")
    
    def download_data(self):
        """
        Download and prepare data for all models using DataFetcher.
        """
        logger.info(f"Downloading data for {self.data_days} days")
        
        try:
            # Get universe of symbols
            universe = self.data_fetcher.get_universe()
            
            # Fetch historical price data
            historical_data = self.data_fetcher.fetch_historical_data(
                symbols=universe,
                timeframe="1d"
            )
            
            # Check if we got any data
            if not historical_data:
                logger.warning("No historical data fetched")
                return False
            
            # Store the data - we'll use pandas dataframes and leverage PyTorch/GPU directly
            # instead of using RAPIDS/cuDF
            self.data = historical_data
            logger.info(f"Downloaded data for {len(self.data)} symbols")
            
            logger.info(f"Downloaded data for {len(self.data)} symbols")
            
            # Download news data for sentiment analysis
            if "sentiment" in self.models_to_train:
                self._download_news_data()
            
            return True
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            return False
    
    def _download_news_data(self):
        """Download news data for sentiment analysis."""
        logger.info("Downloading news data for sentiment analysis")
        
        try:
            # Fetch news data from data fetcher
            self.news_data = self.data_fetcher.fetch_news_data(days=30)
            
            if not self.news_data:
                logger.warning("No news data fetched")
                return False
            
            logger.info(f"Downloaded {len(self.news_data)} news items")
            return True
        except Exception as e:
            logger.error(f"Error downloading news data: {e}")
            self.news_data = []
            return False
    
    def prepare_features(self):
        """
        Prepare features for all models.
        """
        logger.info("Preparing features for training")
        
        try:
            # Create features for each model type
            if "pattern" in self.models_to_train:
                self._prepare_pattern_features()
            
            if "ranking" in self.models_to_train:
                self._prepare_ranking_features()
            
            if "sentiment" in self.models_to_train:
                self._prepare_sentiment_features()
            
            if "exit" in self.models_to_train:
                self._prepare_exit_features()
            
            logger.info("Feature preparation completed")
            return True
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return False
    
    def _prepare_pattern_features(self):
        """Prepare features for pattern recognition model."""
        logger.info("Preparing pattern recognition features")
        
        # Initialize pattern recognition model for feature extraction
        pattern_model = PatternRecognitionModel(lookback=settings.model.lookback_period)
        
        # Extract pattern features from price data
        pattern_samples = []
        pattern_labels = []
        
        symbols = list(self.data.keys())
        
        # Set number of pattern examples to generate per symbol
        patterns_per_symbol = 5
        
        # Process each symbol
        for symbol in symbols:
            df = self.data[symbol]
            
            if len(df) < pattern_model.lookback + 10:
                logger.warning(f"Not enough data for {symbol}, skipping pattern extraction")
                continue
            
            # Generate pattern examples and labels
            try:
                # Note: With RAPIDS removed, we're always working with pandas DataFrames now
                
                # Number of potential windows
                num_windows = len(df) - pattern_model.lookback
                
                # Randomly select windows
                for _ in range(patterns_per_symbol):
                    # Random start index
                    start_idx = np.random.randint(0, num_windows)
                    
                    # Extract window
                    window = df.iloc[start_idx:start_idx + pattern_model.lookback].copy()
                    
                    # For demonstration, assign random pattern labels
                    # In practice, these would be generated by a labeling function
                    pattern_idx = np.random.randint(0, len(pattern_model.PATTERN_CLASSES))
                    
                    # Add to results
                    pattern_samples.append(window)
                    pattern_labels.append(pattern_idx)
            except Exception as e:
                logger.error(f"Error processing {symbol} for pattern features: {e}")
        
        # Store the pattern features
        self.features["pattern"] = {
            "samples": pattern_samples,
            "labels": pattern_labels
        }
        
        logger.info(f"Generated {len(pattern_samples)} pattern samples with {len(pattern_model.PATTERN_CLASSES)} classes")
    
    def _prepare_ranking_features(self):
        """Prepare features for ranking model."""
        logger.info("Preparing ranking features")
        
        # Initialize ranking model for feature extraction
        ranking_model = RankingModel()
        
        # Generate training data with labels
        target_threshold = 1.5  # 1.5% threshold for positive examples
        
        # Extract features using the ranking model
        try:
            training_data = self._generate_ranking_training_data(
                self.data, 
                target_threshold=target_threshold
            )
            
            if training_data.empty:
                logger.warning("No training data generated for ranking model")
                return
            
            # Store the features
            self.features["ranking"] = {
                "training_data": training_data,
            }
            
            # Log feature stats
            num_positive = (training_data["target"] == 1).sum()
            num_negative = (training_data["target"] == 0).sum()
            logger.info(f"Generated {len(training_data)} ranking samples with {num_positive} positive and {num_negative} negative examples")
        
        except Exception as e:
            logger.error(f"Error generating ranking features: {e}")
    
    def _generate_ranking_training_data(self, historical_data, target_threshold=1.5, lookback_days=10, forward_days=5):
        """
        Generate training data for the ranking model.
        
        Args:
            historical_data: Dictionary of DataFrames with historical price data
            target_threshold: Percentage threshold for positive examples
            lookback_days: Number of days to look back for features
            forward_days: Number of days to look forward for target
            
        Returns:
            DataFrame with features and target labels
        """
        all_features = []
        
        for symbol, df in historical_data.items():
            # Using standard pandas DataFrames
            
            if len(df) < lookback_days + forward_days + 5:  # Need enough data
                continue
            
            # Create features for each valid start point
            for i in range(lookback_days, len(df) - forward_days):
                try:
                    # Extract lookback window for feature calculation
                    window = df.iloc[i-lookback_days:i]
                    
                    # Calculate target (forward return)
                    current_price = df["close"].iloc[i]
                    future_price = df["close"].iloc[i+forward_days]
                    forward_return = (future_price / current_price - 1) * 100
                    
                    # Binary classification target
                    target = 1 if forward_return >= target_threshold else 0
                    
                    # Extract features
                    feature_dict = self._extract_ranking_features(window, symbol)
                    
                    # Add metadata and target
                    feature_dict["symbol"] = symbol
                    feature_dict["date"] = df.index[i]
                    feature_dict["target"] = target
                    feature_dict["forward_return"] = forward_return
                    
                    all_features.append(feature_dict)
                except Exception as e:
                    logger.debug(f"Error creating features for {symbol} at index {i}: {e}")
                    continue
        
        if not all_features:
            return pd.DataFrame()
        
        # Combine all features into a DataFrame
        features_df = pd.DataFrame(all_features)
        
        return features_df
    
    def _extract_ranking_features(self, df, symbol):
        """Extract features for ranking model from OHLCV data."""
        features = {}
        
        # Price momentum features
        for period in [1, 3, 5, 10]:
            if len(df) > period:
                features[f"return_{period}d"] = (df["close"].iloc[-1] / df["close"].iloc[-period-1] - 1) * 100
        
        # Volatility features
        if len(df) > 5:
            features["volatility_5d"] = df["close"].pct_change().std() * 100
        
        # Volume features
        if "volume" in df.columns and len(df) > 5:
            features["volume_ratio_5d"] = df["volume"].iloc[-1] / df["volume"].iloc[-6:-1].mean()
        
        # Technical indicators
        if len(df) > 14:
            # RSI
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            features["rsi_14"] = 100 - (100 / (1 + rs.iloc[-1])) if not pd.isna(rs.iloc[-1]) else 50
        
        if len(df) > 20:
            # Bollinger Bands
            ma20 = df["close"].rolling(20).mean().iloc[-1]
            std20 = df["close"].rolling(20).std().iloc[-1]
            features["bb_width"] = (2 * std20) / ma20 if ma20 > 0 else 0
            features["price_to_upper_band"] = (df["close"].iloc[-1] - (ma20 + 2 * std20)) / ma20 if ma20 > 0 else 0
            features["price_to_lower_band"] = (df["close"].iloc[-1] - (ma20 - 2 * std20)) / ma20 if ma20 > 0 else 0
        
        # Add symbol-specific features (e.g., sector)
        features["is_tech"] = 1 if symbol in ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA"] else 0
        features["is_financial"] = 1 if symbol in ["JPM", "V", "BAC", "WFC", "GS"] else 0
        features["is_index"] = 1 if symbol in ["SPY", "QQQ", "IWM", "DIA"] else 0
        
        # Fill any missing values
        for key in features:
            if pd.isna(features[key]):
                features[key] = 0
        
        return features
    
    def _prepare_sentiment_features(self):
        """Prepare features for sentiment analysis model."""
        logger.info("Preparing sentiment features")
        
        if not hasattr(self, 'news_data') or not self.news_data:
            logger.warning("No news data available for sentiment analysis")
            return
        
        try:
            # Extract text and sentiment from news data
            texts = []
            labels = []
            
            for item in self.news_data:
                # Extract text
                title = item.get("title", "")
                summary = item.get("summary", "")
                text = f"{title} {summary}".strip()
                
                if not text:
                    continue
                
                # Extract sentiment score
                sentiment_score = item.get("sentiment_score", 0)
                
                # Convert to label (0=negative, 1=neutral, 2=positive)
                if sentiment_score > 0.2:
                    label = 2  # positive
                elif sentiment_score < -0.2:
                    label = 0  # negative
                else:
                    label = 1  # neutral
                
                # Add to results
                texts.append(text)
                labels.append(label)
            
            # Split into training and validation sets
            if len(texts) > 10:
                # Split 80/20
                split_idx = int(len(texts) * 0.8)
                train_texts = texts[:split_idx]
                train_labels = labels[:split_idx]
                val_texts = texts[split_idx:]
                val_labels = labels[split_idx:]
                
                # Store the features
                self.features["sentiment"] = {
                    "texts": train_texts,
                    "labels": train_labels,
                    "eval_texts": val_texts,
                    "eval_labels": val_labels
                }
                
                # Log feature stats
                label_counts = np.bincount(train_labels)
                logger.info(f"Generated {len(train_texts)} sentiment training samples "
                           f"with {label_counts[0]} negative, {label_counts[1]} neutral, "
                           f"and {label_counts[2]} positive examples")
            else:
                logger.warning("Not enough news data for sentiment analysis")
        
        except Exception as e:
            logger.error(f"Error generating sentiment features: {e}")
    
    def _prepare_exit_features(self):
        """Prepare features for exit optimization model."""
        logger.info("Preparing exit optimization features")
        
        # Initialize exit optimization model
        exit_model = ExitOptimizationModel(use_sac=True)
        
        # Initialize episode data
        episodes = []
        
        try:
            # Generate simulated trade episodes
            episodes_per_symbol = 3
            max_steps = 30
            
            symbols = list(self.data.keys())
            
            # Process each symbol
            for symbol in symbols:
                df = self.data[symbol]
                
                if len(df) < 50:  # Need enough data
                    logger.warning(f"Not enough data for {symbol}, skipping exit optimization")
                    continue
                
                # We're always working with pandas DataFrames now
                
                # Generate episodes
                for _ in range(episodes_per_symbol):
                    try:
                        # Random entry point
                        entry_idx = np.random.randint(20, len(df) - max_steps - 1)
                        entry_price = df['close'].iloc[entry_idx]
                        
                        # Initialize episode data
                        states = []
                        actions = []
                        rewards = []
                        
                        # Initialize position
                        position_size = 1.0  # Start with full position
                        highest_price = entry_price
                        
                        # Simulate episode
                        for step in range(max_steps):
                            # Current index
                            current_idx = entry_idx + step
                            
                            # Create state features
                            current_price = df['close'].iloc[current_idx]
                            profit_pct = (current_price / entry_price - 1) * 100
                            time_in_trade = step / max_steps  # Normalize
                            price_to_high = current_price / highest_price - 1
                            
                            # Extract features for state
                            lookback_data = df.iloc[:current_idx+1].copy()
                            
                            # Create position data for feature extraction
                            position_data = {
                                "symbol": symbol,
                                "entry_price": entry_price,
                                "entry_time": df.index[entry_idx],
                                "position_size": position_size,
                                "stop_loss": entry_price * 0.95,  # 5% stop loss
                                "take_profit": entry_price * 1.1,  # 10% take profit
                                "trailing_stop": 2.0  # 2% trailing stop
                            }
                            
                            # Extract features using exit model
                            state = exit_model._extract_features(lookback_data, position_data)
                            
                            # Decide action (for simulation)
                            # Simple rules for generating action labels:
                            # - Hold if small profit/loss
                            # - Partial exit if decent profit
                            # - Full exit if large profit or loss
                            if profit_pct > 5.0 or profit_pct < -3.0:
                                action = 4  # exit_full 
                                exit_size = 1.0
                            elif profit_pct > 3.0:
                                action = 3  # exit_half
                                exit_size = 0.5
                            elif profit_pct > 1.5:
                                action = 2  # exit_third
                                exit_size = 0.33
                            elif profit_pct > 0.5:
                                action = 1  # exit_quarter
                                exit_size = 0.25
                            else:
                                action = 0  # hold
                                exit_size = 0.0
                            
                            # Calculate reward (risk-adjusted)
                            # Update highest price
                            highest_price = max(highest_price, current_price)
                            
                            # Next price for reward calculation
                            next_idx = min(current_idx + 1, len(df) - 1)
                            next_price = df['close'].iloc[next_idx]
                            
                            # Calculate reward
                            if action == 0:  # hold
                                # Reward based on price change
                                price_change = (next_price - current_price) / current_price * 100
                                reward = price_change * 0.1  # Scale down for holds
                            else:
                                # Reward based on profit for exits
                                reward = profit_pct * exit_size
                            
                            # Update position
                            new_position_size = position_size - position_size * exit_size
                            position_size = new_position_size
                            
                            # Store transition
                            states.append(state)
                            actions.append(action)
                            rewards.append(reward)
                            
                            # End episode if position closed
                            if position_size <= 0:
                                break
                        
                        # Add episode to results
                        episode = {
                            "symbol": symbol,
                            "entry_idx": entry_idx,
                            "entry_price": entry_price,
                            "states": states,
                            "actions": actions,
                            "rewards": rewards
                        }
                        
                        episodes.append(episode)
                    
                    except Exception as e:
                        logger.error(f"Error generating episode for {symbol}: {e}")
            
            # Store the episodes
            self.features["exit"] = {
                "episodes": episodes
            }
            
            logger.info(f"Generated {len(episodes)} exit optimization episodes")
        
        except Exception as e:
            logger.error(f"Error generating exit optimization features: {e}")
    
    def split_data(self):
        """
        Split data into training and validation sets.
        """
        logger.info("Splitting data into training and validation sets")
        
        try:
            # Split pattern recognition data
            if "pattern" in self.features:
                self._split_pattern_data()
            
            # Split ranking data
            if "ranking" in self.features:
                self._split_ranking_data()
            
            # Sentiment data is already split in prepare_sentiment_features
            
            # Exit optimization data doesn't need traditional splitting
            
            logger.info("Data splitting completed")
            return True
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return False
    
    def _split_pattern_data(self):
        """Split pattern recognition data into training and validation sets."""
        pattern_samples = self.features["pattern"]["samples"]
        pattern_labels = self.features["pattern"]["labels"]
        
        if not pattern_samples:
            logger.warning("No pattern samples to split")
            return
        
        # Use sklearn's train_test_split
        from sklearn.model_selection import train_test_split
        
        try:
            # Split 80/20
            train_samples, val_samples, train_labels, val_labels = train_test_split(
                pattern_samples, pattern_labels, test_size=0.2, random_state=42, stratify=pattern_labels
            )
            
            # Store the split data
            self.training_datasets["pattern"] = {
                "samples": train_samples,
                "labels": train_labels
            }
            
            self.validation_datasets["pattern"] = {
                "samples": val_samples,
                "labels": val_labels
            }
            
            logger.info(f"Split pattern data: {len(train_samples)} training, {len(val_samples)} validation")
        
        except Exception as e:
            logger.error(f"Error splitting pattern data: {e}")
    
    def _split_ranking_data(self):
        """Split ranking data into training and validation sets."""
        if "ranking" not in self.features or "training_data" not in self.features["ranking"]:
            logger.warning("No ranking data to split")
            return
        
        training_data = self.features["ranking"]["training_data"]
        
        if training_data.empty:
            logger.warning("Empty ranking training data")
            return
        
        try:
            # For ranking data, we'll use time-based split if 'date' is available,
            # otherwise fall back to random split
            if 'date' in training_data.columns:
                # Sort by date
                training_data = training_data.sort_values('date')
                
                # Use the last 20% for validation
                split_idx = int(len(training_data) * 0.8)
                train_data = training_data.iloc[:split_idx]
                val_data = training_data.iloc[split_idx:]
            else:
                # Random split
                from sklearn.model_selection import train_test_split
                
                # Save target column
                target_col = "target"
                
                # Split features and target
                X = training_data.drop(columns=[target_col])
                y = training_data[target_col]
                
                # Split 80/20
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                
                # Recombine features and target
                train_data = pd.concat([X_train, y_train], axis=1)
                val_data = pd.concat([X_val, y_val], axis=1)
            
            # Store the split data
            self.training_datasets["ranking"] = train_data
            self.validation_datasets["ranking"] = val_data
