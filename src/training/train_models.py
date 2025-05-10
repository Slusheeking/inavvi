"""
Automated training system for all ML models in the trading system.

This script handles:
- Data downloading and preprocessing
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
except:
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

# Import model modules
from src.models.pattern_recognition import PatternRecognitionModel
from src.models.ranking_model import RankingModel
from src.models.sentiment import FinancialSentimentModel
from src.models.exit_optimization import ExitOptimizationModel

# Try to import GPU acceleration libraries
try:
    if GPU_AVAILABLE:
        # For RAPIDS ecosystem (GPU accelerated pandas-like operations)
        import cudf
        import cuml
        import cupy as cp

        # For distributed GPU computing
        import dask_cudf
        import dask
        from dask.distributed import Client, LocalCluster
        
        try:
            from dask_cuda import LocalCUDACluster
            CUDA_CLUSTER_AVAILABLE = True
        except:
            CUDA_CLUSTER_AVAILABLE = False
            logger.warning("dask_cuda not available, falling back to CPU dask.distributed")
        
        # GPU-accelerated ML tools
        from cuml.preprocessing import StandardScaler as CuStandardScaler
        from cuml.model_selection import train_test_split as cu_train_test_split
        
        # XGBoost with GPU support
        import xgboost as xgb
        
        logger.info("Successfully imported GPU acceleration libraries")
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
            
            # Initialize dask client for parallel GPU processing
            if CUDA_CLUSTER_AVAILABLE and GPU_COUNT > 0:
                logger.info(f"Starting Dask CUDA cluster with {GPU_COUNT} GPUs")
                cluster = LocalCUDACluster(n_workers=GPU_COUNT)
                self.client = Client(cluster)
            else:
                # Fall back to CPU dask
                logger.info("Starting Dask CPU cluster")
                cluster = LocalCluster(n_workers=os.cpu_count())
                self.client = Client(cluster)
            
            logger.info(f"Dask dashboard link: {self.client.dashboard_link}")
            
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
        Download and prepare data for all models.
        """
        logger.info(f"Downloading data for {self.data_days} days")
        
        # Get universe of symbols from Redis
        symbols = redis_client.get_watchlist() or []
        
        # Add major indices and ETFs
        indices = ["SPY", "QQQ", "IWM", "VIX", "XLF", "XLE", "XLI", "XLK", "XLV", "XLP", "XLY", "XLB", "XLU"]
        for index in indices:
            if index not in symbols:
                symbols.append(index)
        
        if not symbols:
            # Fallback symbols if no watchlist is available
            symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "V", "PG", "JNJ", "SPY", "QQQ"]
        
        logger.info(f"Downloading data for {len(symbols)} symbols")
        
        try:
            # Download data for each symbol (actually retrieve from Redis in this implementation)
            historical_data = {}
            
            for symbol in symbols:
                # Check if data exists in Redis
                redis_key = f"stocks:history:{symbol}:{self.data_days}d:1d"
                df = redis_client.get(redis_key)
                
                if df is not None and not df.empty:
                    historical_data[symbol] = df
            
            if not historical_data:
                logger.warning("No historical data found in Redis")
                
                # In a real implementation, we would download data here
                # For now, we'll use synthetic data for demonstration
                historical_data = self._generate_synthetic_data(symbols)
            
            # Convert to GPU DataFrames if using GPU
            if self.use_gpu:
                try:
                    gpu_data = {}
                    for symbol, df in historical_data.items():
                        gpu_df = cudf.DataFrame.from_pandas(df)
                        gpu_data[symbol] = gpu_df
                    
                    self.data = gpu_data
                    logger.info(f"Converted {len(gpu_data)} symbols to GPU DataFrames")
                except Exception as e:
                    logger.error(f"Error converting to GPU DataFrames: {e}")
                    self.data = historical_data
            else:
                self.data = historical_data
            
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
            # Check if news data exists in Redis
            news_data = redis_client.get("news:recent")
            
            if news_data and isinstance(news_data, list) and len(news_data) > 0:
                logger.info(f"Found {len(news_data)} news items in Redis")
                self.news_data = news_data
            else:
                logger.warning("No news data found in Redis")
                
                # Generate synthetic news data for demonstration
                self.news_data = self._generate_synthetic_news_data()
            
            return True
        except Exception as e:
            logger.error(f"Error downloading news data: {e}")
            self.news_data = []
            return False
    
    def _generate_synthetic_data(self, symbols):
        """Generate synthetic data for demonstration."""
        logger.info("Generating synthetic price data for training")
        
        synthetic_data = {}
        
        for symbol in symbols:
            # Create a date range
            days = self.data_days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')
            
            # Generate synthetic OHLCV data
            n = len(date_range)
            base_price = np.random.uniform(50, 500)
            
            # Generate price series with realistic properties
            returns = np.random.normal(0.0005, 0.015, n)  # Daily returns
            returns[0] = 0
            log_prices = np.cumsum(returns) + np.log(base_price)
            prices = np.exp(log_prices)
            
            # Generate OHLCV data
            df = pd.DataFrame({
                'open': prices * (1 + np.random.normal(0, 0.005, n)),
                'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
                'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
                'close': prices,
                'volume': np.random.lognormal(mean=np.log(1000000), sigma=0.5, size=n)
            }, index=date_range)
            
            # Fix any inconsistencies (high > open, high > close, etc.)
            for i in range(n):
                high = max(df.iloc[i]['open'], df.iloc[i]['close'], df.iloc[i]['high'])
                low = min(df.iloc[i]['open'], df.iloc[i]['close'], df.iloc[i]['low'])
                df.iloc[i, df.columns.get_loc('high')] = high
                df.iloc[i, df.columns.get_loc('low')] = low
            
            synthetic_data[symbol] = df
        
        logger.info(f"Generated synthetic data for {len(synthetic_data)} symbols")
        return synthetic_data
    
    def _generate_synthetic_news_data(self, count=200):
        """Generate synthetic news data for sentiment analysis training."""
        logger.info(f"Generating {count} synthetic news items for sentiment training")
        
        # Define templates for positive, negative, and neutral news
        positive_templates = [
            "{company} reports record profits in quarterly earnings",
            "{company} announces new product launch that exceeds expectations",
            "{company} stock surges after beating analyst estimates",
            "{company} expands into new markets with promising growth potential",
            "{company} announced increased dividend payout to shareholders",
            "Analysts upgrade {company} citing strong growth prospects",
            "{company} signs major new partnership with {company2}",
            "{company} completes successful acquisition of {company2}"
        ]
        
        negative_templates = [
            "{company} misses earnings expectations, shares plummet",
            "{company} announces layoffs amid restructuring efforts",
            "{company} faces regulatory investigation over business practices",
            "{company} issues profit warning for upcoming quarter",
            "Analysts downgrade {company} citing competitive pressures",
            "{company} reports significant drop in market share",
            "{company} delayed product launch raises investor concerns",
            "{company} faces lawsuit from {company2} over patent infringement"
        ]
        
        neutral_templates = [
            "{company} reports earnings in line with analyst expectations",
            "{company} maintains current outlook for fiscal year",
            "{company} appoints new board member",
            "{company} to present at upcoming industry conference",
            "{company} announces regular quarterly dividend",
            "Analysts maintain neutral rating on {company} stock",
            "{company} relocates headquarters to new office space",
            "{company} holds annual shareholder meeting"
        ]
        
        # List of company names (use symbols from data)
        companies = list(self.data.keys()) if hasattr(self, 'data') and self.data else [
            "Apple", "Microsoft", "Amazon", "Google", "Meta", "Nvidia", "Tesla", 
            "JPMorgan", "Visa", "Procter & Gamble", "Johnson & Johnson", "Walmart", "Exxon"
        ]
        
        # Generate news items
        news_items = []
        now = datetime.now()
        
        for i in range(count):
            # Select a company
            company = np.random.choice(companies)
            company2 = np.random.choice([c for c in companies if c != company])
            
            # Determine sentiment (slight bias toward neutral)
            sentiment_type = np.random.choice(["positive", "negative", "neutral"], p=[0.3, 0.3, 0.4])
            
            # Select a template
            if sentiment_type == "positive":
                template = np.random.choice(positive_templates)
                score = np.random.uniform(0.6, 1.0)
            elif sentiment_type == "negative":
                template = np.random.choice(negative_templates)
                score = np.random.uniform(-1.0, -0.6)
            else:
                template = np.random.choice(neutral_templates)
                score = np.random.uniform(-0.3, 0.3)
            
            # Fill in the template
            title = template.format(company=company, company2=company2)
            
            # Generate a longer summary
            summary = f"In a {sentiment_type} development for investors, {title.lower()}. "
            summary += f"This news could impact the company's financial performance in the coming quarter."
            
            # Generate a timestamp (within the past 30 days)
            days_ago = np.random.randint(0, 30)
            timestamp = (now - timedelta(days=days_ago)).isoformat()
            
            # Create the news item
            news_item = {
                "title": title,
                "summary": summary,
                "source": np.random.choice(["Bloomberg", "Reuters", "CNBC", "WSJ", "Financial Times"]),
                "url": f"https://example.com/news/{i}",
                "published_at": timestamp,
                "sentiment_score": score,
                "symbols": [company],
                "relevance_score": np.random.uniform(0.7, 1.0)
            }
            
            news_items.append(news_item)
        
        return news_items
    
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
                # Convert to pandas DataFrame if using GPU
                if self.use_gpu and isinstance(df, cudf.DataFrame):
                    df = df.to_pandas()
                
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
            training_data = ranking_model.generate_training_data(
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
                
                # Convert to pandas DataFrame if using GPU
                if self.use_gpu and isinstance(df, cudf.DataFrame):
                    df = df.to_pandas()
                
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
            
            logger.info(f"Split ranking data: {len(train_data)} training, {len(val_data)} validation")
        
        except Exception as e:
            logger.error(f"Error splitting ranking data: {e}")
    
    def optimize_hyperparameters(self):
        """
        Optimize hyperparameters for all models using Optuna.
        """
        if not self.optimize_hyperparams:
            logger.info("Hyperparameter optimization disabled, skipping")
            return True
        
        logger.info("Starting hyperparameter optimization")
        
        try:
            # Create Optuna study for each model
            if "pattern" in self.models_to_train and "pattern" in self.training_datasets:
                self._optimize_pattern_hyperparams()
            
            if "ranking" in self.models_to_train and "ranking" in self.training_datasets:
                self._optimize_ranking_hyperparams()
            
            if "sentiment" in self.models_to_train and "sentiment" in self.features:
                self._optimize_sentiment_hyperparams()
            
            if "exit" in self.models_to_train and "exit" in self.features:
                self._optimize_exit_hyperparams()
            
            logger.info("Hyperparameter optimization completed")
            return True
        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {e}")
            return False
    
    def _optimize_pattern_hyperparams(self):
        """Optimize hyperparameters for pattern recognition model."""
        logger.info("Optimizing pattern recognition model hyperparameters")
        
        # Start timing
        start_time = time.time()
        
        # Define the objective function for Optuna
        def objective(trial):
            # Sample hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            
            # Create model with sampled hyperparameters
            model = PatternRecognitionModel(lookback=settings.model.lookback_period)
            
            # Train model with validation
            train_samples = self.training_datasets["pattern"]["samples"]
            train_labels = self.training_datasets["pattern"]["labels"]
            val_samples = self.validation_datasets["pattern"]["samples"]
            val_labels = self.validation_datasets["pattern"]["labels"]
            
            # Train for a limited number of epochs
            history = model.train(
                train_data=train_samples,
                train_labels=train_labels,
                val_data=val_samples,
                val_labels=val_labels,
                epochs=5,  # Reduced epochs for optimization
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Return validation accuracy as the objective value
            val_acc = max(history.get('val_acc', [0]))
            
            return val_acc
        
        # Create an Optuna study and optimize
        study_name = f"pattern_recognition_{self.timestamp}"
        
        if self.use_mlflow:
            # Set up MLflow for logging
            mlflow.start_run(run_name=study_name)
            mlflow.log_params({
                "model_type": "pattern_recognition",
                "optimization_trials": self.optimization_trials,
                "timestamp": self.timestamp
            })
        
        try:
            # Create and run the study
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner()
            )
            
            study.optimize(objective, n_trials=self.optimization_trials, timeout=3600)  # 1 hour timeout
            
            # Get the best hyperparameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Store the best hyperparameters
            self.best_params["pattern"] = best_params
            
            # Log the results
            logger.info(f"Best pattern recognition hyperparameters: {best_params}, " 
                       f"Validation accuracy: {best_value:.4f}")
            
            if self.use_mlflow:
                # Log best parameters and metrics to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric("best_val_accuracy", best_value)
                
                # Log optimization history
                for step, trial in enumerate(study.trials):
                    mlflow.log_metric("val_accuracy", trial.value, step=step)
        
        except Exception as e:
            logger.error(f"Error in pattern hyperparameter optimization: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record optimization time
            end_time = time.time()
            self.optimization_times["pattern"] = end_time - start_time
            logger.info(f"Pattern optimization took {end_time - start_time:.2f} seconds")
    
    def _optimize_ranking_hyperparams(self):
        """Optimize hyperparameters for ranking model."""
        logger.info("Optimizing ranking model hyperparameters")
        
        # Start timing
        start_time = time.time()
        
        # Define the objective function for Optuna
        def objective(trial):
            # Sample XGBoost hyperparameters
            xgb_params = {
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10)
            }
            
            # Sample LightGBM hyperparameters
            lgb_params = {
                "num_leaves": trial.suggest_int("lgb_num_leaves", 10, 100),
                "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("lgb_feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("lgb_bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("lgb_bagging_freq", 1, 10)
            }
            
            # Sample ensemble weights
            xgb_weight = trial.suggest_float("xgb_weight", 0.1, 1.0)
            lgb_weight = trial.suggest_float("lgb_weight", 0.1, 1.0)
            cb_weight = trial.suggest_float("cb_weight", 0.1, 1.0)
            
            # Normalize weights
            total = xgb_weight + lgb_weight + cb_weight
            xgb_weight /= total
            lgb_weight /= total
            cb_weight /= total
            
            # Create model with sampled hyperparameters
            model = RankingModel()
            
            # Update model hyperparameters
            model.hyperparams['xgboost'].update(xgb_params)
            model.hyperparams['lightgbm'].update(lgb_params)
            model.hyperparams['meta']['weights'] = {
                'xgboost': xgb_weight,
                'lightgbm': lgb_weight,
                'catboost': cb_weight
            }
            
            # Train model
            train_data = self.training_datasets["ranking"]
            val_data = self.validation_datasets["ranking"]
            
            # Train the model
            model.train(
                training_data=train_data,
                optimize_hyperparams=False,  # We're already optimizing here
                use_time_series_cv=True
            )
            
            # Evaluate on validation data
            val_metrics = model.model_metrics.get('ensemble', {})
            val_auc = val_metrics.get('auc', 0)
            
            return val_auc
        
        # Create an Optuna study and optimize
        study_name = f"ranking_model_{self.timestamp}"
        
        if self.use_mlflow:
            # Set up MLflow for logging
            mlflow.start_run(run_name=study_name)
            mlflow.log_params({
                "model_type": "ranking",
                "optimization_trials": self.optimization_trials,
                "timestamp": self.timestamp
            })
        
        try:
            # Create and run the study
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner()
            )
            
            study.optimize(objective, n_trials=self.optimization_trials, timeout=3600)  # 1 hour timeout
            
            # Get the best hyperparameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Store the best hyperparameters
            self.best_params["ranking"] = best_params
            
            # Log the results
            logger.info(f"Best ranking hyperparameters: {best_params}, " 
                       f"Validation AUC: {best_value:.4f}")
            
            if self.use_mlflow:
                # Log best parameters and metrics to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric("best_val_auc", best_value)
                
                # Log optimization history
                for step, trial in enumerate(study.trials):
                    mlflow.log_metric("val_auc", trial.value, step=step)
        
        except Exception as e:
            logger.error(f"Error in ranking hyperparameter optimization: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record optimization time
            end_time = time.time()
            self.optimization_times["ranking"] = end_time - start_time
            logger.info(f"Ranking optimization took {end_time - start_time:.2f} seconds")
    
    def _optimize_sentiment_hyperparams(self):
        """Optimize hyperparameters for sentiment analysis model."""
        logger.info("Optimizing sentiment analysis model hyperparameters")
        
        # Start timing
        start_time = time.time()
        
        # Check if we have enough data
        if ("sentiment" not in self.features or 
            "texts" not in self.features["sentiment"] or 
            len(self.features["sentiment"]["texts"]) < 50):
            logger.warning("Not enough sentiment data for hyperparameter optimization")
            return
        
        # Define the objective function for Optuna
        def objective(trial):
            # Sample hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
            weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
            warmup_steps = trial.suggest_int("warmup_steps", 100, 1000)
            
            # Create model with default parameters
            model = FinancialSentimentModel(
                model_name="finbert",
                use_ensemble=False  # Disable ensemble for faster optimization
            )
            
            # Train model with sampled hyperparameters
            texts = self.features["sentiment"]["texts"]
            labels = self.features["sentiment"]["labels"]
            eval_texts = self.features["sentiment"]["eval_texts"]
            eval_labels = self.features["sentiment"]["eval_labels"]
            
            try:
                # Fine-tune for a limited number of epochs
                history = model.fine_tune(
                    texts=texts,
                    labels=labels,
                    eval_texts=eval_texts,
                    eval_labels=eval_labels,
                    epochs=1,  # Just one epoch for optimization
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    warmup_steps=warmup_steps
                )
                
                # Return validation accuracy or F1 score
                val_metric = model.metrics.get('f1', 0)
                
                return val_metric
            except Exception as e:
                logger.error(f"Error in sentiment trial: {e}")
                return 0.0  # Return a bad score for failed trials
        
        # Create an Optuna study and optimize
        study_name = f"sentiment_model_{self.timestamp}"
        
        if self.use_mlflow:
            # Set up MLflow for logging
            mlflow.start_run(run_name=study_name)
            mlflow.log_params({
                "model_type": "sentiment",
                "optimization_trials": min(10, self.optimization_trials),  # Reduce trials for sentiment
                "timestamp": self.timestamp
            })
        
        try:
            # Create and run the study
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Limit trials for sentiment since training is expensive
            actual_trials = min(10, self.optimization_trials)
            
            study.optimize(objective, n_trials=actual_trials, timeout=3600)  # 1 hour timeout
            
            # Get the best hyperparameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Store the best hyperparameters
            self.best_params["sentiment"] = best_params
            
            # Log the results
            logger.info(f"Best sentiment hyperparameters: {best_params}, " 
                       f"Validation F1: {best_value:.4f}")
            
            if self.use_mlflow:
                # Log best parameters and metrics to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric("best_val_f1", best_value)
                
                # Log optimization history
                for step, trial in enumerate(study.trials):
                    mlflow.log_metric("val_f1", trial.value, step=step)
        
        except Exception as e:
            logger.error(f"Error in sentiment hyperparameter optimization: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record optimization time
            end_time = time.time()
            self.optimization_times["sentiment"] = end_time - start_time
            logger.info(f"Sentiment optimization took {end_time - start_time:.2f} seconds")
    
    def _optimize_exit_hyperparams(self):
        """Optimize hyperparameters for exit optimization model."""
        logger.info("Optimizing exit optimization model hyperparameters")
        
        # Start timing
        start_time = time.time()
        
        # Define the objective function for Optuna
        def objective(trial):
            # Sample hyperparameters
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
            gamma = trial.suggest_float("gamma", 0.95, 0.99)
            tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
            alpha = trial.suggest_float("alpha", 0.1, 0.5)
            hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
            
            # Create model with SAC
            model = ExitOptimizationModel(use_sac=True)
            
            # Initialize the SAC agent with the sampled hyperparameters
            if hasattr(model, 'agent'):
                model.agent.gamma = gamma
                model.agent.tau = tau
                model.agent.alpha = alpha
                
                # We can't easily change the hidden_dim without recreating the model
                # This is a simplification for the objective function
            
            # Get the episodes from features
            if "exit" not in self.features or "episodes" not in self.features["exit"]:
                logger.warning("No exit optimization episodes for training")
                return 0.0
            
            episodes = self.features["exit"]["episodes"]
            
            # Train the model with a limited number of episodes
            # Take a sample of episodes to speed up optimization
            sample_size = min(20, len(episodes))
            episode_sample = np.random.choice(episodes, sample_size, replace=False)
            
            # Train for a reduced number of epochs
            try:
                metrics = model.train_sac(
                    training_data=episode_sample,
                    epochs=2,  # Reduced epochs for optimization
                    batch_size=64,
                    lr=learning_rate,
                    gamma=gamma,
                    tau=tau,
                    alpha=alpha,
                    updates_per_step=1
                )
                
                # Return the average reward as the objective
                avg_reward = np.mean(metrics.get('epoch_rewards', [0]))
                
                return avg_reward
            except Exception as e:
                logger.error(f"Error in exit trial: {e}")
                return -100.0  # Return a bad score for failed trials
        
        # Create an Optuna study and optimize
        study_name = f"exit_model_{self.timestamp}"
        
        if self.use_mlflow:
            # Set up MLflow for logging
            mlflow.start_run(run_name=study_name)
            mlflow.log_params({
                "model_type": "exit_optimization",
                "optimization_trials": min(20, self.optimization_trials),  # Reduce trials for RL
                "timestamp": self.timestamp
            })
        
        try:
            # Create and run the study
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner()
            )
            
            # Limit trials for RL since training is expensive
            actual_trials = min(20, self.optimization_trials)
            
            study.optimize(objective, n_trials=actual_trials, timeout=3600)  # 1 hour timeout
            
            # Get the best hyperparameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Store the best hyperparameters
            self.best_params["exit"] = best_params
            
            # Log the results
            logger.info(f"Best exit optimization hyperparameters: {best_params}, " 
                       f"Average Reward: {best_value:.4f}")
            
            if self.use_mlflow:
                # Log best parameters and metrics to MLflow
                mlflow.log_params(best_params)
                mlflow.log_metric("best_avg_reward", best_value)
                
                # Log optimization history
                for step, trial in enumerate(study.trials):
                    mlflow.log_metric("avg_reward", trial.value, step=step)
        
        except Exception as e:
            logger.error(f"Error in exit hyperparameter optimization: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record optimization time
            end_time = time.time()
            self.optimization_times["exit"] = end_time - start_time
            logger.info(f"Exit optimization took {end_time - start_time:.2f} seconds")
    
    def train_models(self):
        """
        Train all models with the optimized hyperparameters.
        """
        logger.info("Starting model training")
        
        try:
            # Train each model
            if "pattern" in self.models_to_train:
                self._train_pattern_model()
            
            if "ranking" in self.models_to_train:
                self._train_ranking_model()
            
            if "sentiment" in self.models_to_train:
                self._train_sentiment_model()
            
            if "exit" in self.models_to_train:
                self._train_exit_model()
            
            logger.info("Model training completed")
            return True
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    
    def _train_pattern_model(self):
        """Train pattern recognition model."""
        logger.info("Training pattern recognition model")
        
        # Start timing
        start_time = time.time()
        
        # MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=f"pattern_model_training_{self.timestamp}")
        
        try:
            # Get training and validation data
            if "pattern" not in self.training_datasets:
                logger.warning("No pattern data for training")
                return
            
            train_samples = self.training_datasets["pattern"]["samples"]
            train_labels = self.training_datasets["pattern"]["labels"]
            val_samples = self.validation_datasets["pattern"]["samples"]
            val_labels = self.validation_datasets["pattern"]["labels"]
            
            # Create model
            lookback = settings.model.lookback_period
            
            # Use optimized hyperparameters if available
            if "pattern" in self.best_params:
                hp = self.best_params["pattern"]
                learning_rate = hp.get("learning_rate", 0.001)
                batch_size = hp.get("batch_size", 32)
                hidden_dim = hp.get("hidden_dim", 128)
                model = PatternRecognitionModel(lookback=lookback, hidden_dim=hidden_dim)
            else:
                # Use default hyperparameters
                learning_rate = 0.001
                batch_size = 32
                model = PatternRecognitionModel(lookback=lookback)
            
            # Train model with full epochs
            epochs = settings.model.epochs
            
            if self.use_mlflow:
                # Log parameters
                mlflow.log_params({
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "lookback": lookback,
                    "training_samples": len(train_samples),
                    "validation_samples": len(val_samples)
                })
            
            # Train the model
            history = model.train(
                train_data=train_samples,
                train_labels=train_labels,
                val_data=val_samples,
                val_labels=val_labels,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Store the trained model
            self.models["pattern"] = model
            
            # Log the training history
            logger.info(f"Pattern model training completed with {len(history.get('train_loss', []))} epochs")
            
            if history.get('val_acc'):
                final_val_acc = history['val_acc'][-1]
                logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
                
                # Store metrics
                self.model_metrics["pattern"] = {
                    "val_accuracy": final_val_acc,
                    "train_accuracy": history['train_acc'][-1] if 'train_acc' in history else 0.0,
                    "val_loss": history['val_loss'][-1] if 'val_loss' in history else 0.0,
                    "train_loss": history['train_loss'][-1]
                }
                
                if self.use_mlflow:
                    # Log metrics
                    mlflow.log_metric("val_accuracy", final_val_acc)
                    mlflow.log_metric("train_accuracy", history['train_acc'][-1] if 'train_acc' in history else 0.0)
                    mlflow.log_metric("val_loss", history['val_loss'][-1] if 'val_loss' in history else 0.0)
                    mlflow.log_metric("train_loss", history['train_loss'][-1])
                    
                    # Log training curves
                    for epoch, (train_loss, val_loss, train_acc, val_acc) in enumerate(zip(
                        history['train_loss'],
                        history.get('val_loss', [0] * len(history['train_loss'])),
                        history.get('train_acc', [0] * len(history['train_loss'])),
                        history.get('val_acc', [0] * len(history['train_loss']))
                    )):
                        mlflow.log_metrics({
                            "epoch_train_loss": train_loss,
                            "epoch_val_loss": val_loss,
                            "epoch_train_acc": train_acc,
                            "epoch_val_acc": val_acc
                        }, step=epoch)
            
            # Save model
            model_path = os.path.join(settings.models_dir, "pattern", f"pattern_model_{self.timestamp}.pt")
            model.save_model(model_path)
            
            # Also save to default path
            model.save_model(settings.model.pattern_model_path)
            
            logger.info(f"Pattern model saved to {model_path}")
            
            # Store model version info
            self.model_versions["pattern"] = {
                "path": model_path,
                "timestamp": self.timestamp,
                "metrics": self.model_metrics.get("pattern", {})
            }
        
        except Exception as e:
            logger.error(f"Error training pattern model: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record training time
            end_time = time.time()
            self.training_times["pattern"] = end_time - start_time
            logger.info(f"Pattern training took {end_time - start_time:.2f} seconds")
    
    def _train_ranking_model(self):
        """Train ranking model."""
        logger.info("Training ranking model")
        
        # Start timing
        start_time = time.time()
        
        # MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=f"ranking_model_training_{self.timestamp}")
        
        try:
            # Get training data
            if "ranking" not in self.training_datasets:
                logger.warning("No ranking data for training")
                return
            
            training_data = self.training_datasets["ranking"]
            
            # Create model
            model = RankingModel()
            
            # Apply optimized hyperparameters if available
            if "ranking" in self.best_params:
                hp = self.best_params["ranking"]
                
                # Update XGBoost hyperparameters
                model.hyperparams['xgboost'].update({
                    "max_depth": hp.get("xgb_max_depth", 6),
                    "learning_rate": hp.get("xgb_learning_rate", 0.05),
                    "subsample": hp.get("xgb_subsample", 0.8),
                    "colsample_bytree": hp.get("xgb_colsample_bytree", 0.8),
                    "min_child_weight": hp.get("xgb_min_child_weight", 1)
                })
                
                # Update LightGBM hyperparameters
                model.hyperparams['lightgbm'].update({
                    "num_leaves": hp.get("lgb_num_leaves", 31),
                    "learning_rate": hp.get("lgb_learning_rate", 0.05),
                    "feature_fraction": hp.get("lgb_feature_fraction", 0.9),
                    "bagging_fraction": hp.get("lgb_bagging_fraction", 0.8),
                    "bagging_freq": hp.get("lgb_bagging_freq", 5)
                })
                
                # Update ensemble weights
                xgb_weight = hp.get("xgb_weight",
                                    # Update ensemble weights
                xgb_weight = hp.get("xgb_weight", 0.4)
                lgb_weight = hp.get("lgb_weight", 0.3)
                cb_weight = hp.get("cb_weight", 0.3)
                
                # Normalize weights
                total = xgb_weight + lgb_weight + cb_weight
                xgb_weight /= total
                lgb_weight /= total
                cb_weight /= total
                
                model.hyperparams['meta']['weights'] = {
                    'xgboost': xgb_weight,
                    'lightgbm': lgb_weight,
                    'catboost': cb_weight
                }
            
            # Log hyperparameters
            if self.use_mlflow:
                mlflow.log_params({
                    "xgb_max_depth": model.hyperparams['xgboost']["max_depth"],
                    "xgb_learning_rate": model.hyperparams['xgboost']["learning_rate"],
                    "lgb_num_leaves": model.hyperparams['lightgbm']["num_leaves"],
                    "lgb_learning_rate": model.hyperparams['lightgbm']["learning_rate"],
                    "ensemble_xgb_weight": model.hyperparams['meta']['weights']['xgboost'],
                    "ensemble_lgb_weight": model.hyperparams['meta']['weights']['lightgbm'],
                    "ensemble_cb_weight": model.hyperparams['meta']['weights']['catboost'],
                    "training_samples": len(training_data)
                })
            
            # Train the model
            success = model.train(
                training_data=training_data,
                target_col='target',
                optimize_hyperparams=False,  # Already optimized
                use_time_series_cv=True
            )
            
            if success:
                # Store the trained model
                self.models["ranking"] = model
                
                # Store metrics
                if model.model_metrics:
                    self.model_metrics["ranking"] = model.model_metrics
                    
                    if self.use_mlflow:
                        # Log metrics
                        for model_name, metrics in model.model_metrics.items():
                            for metric_name, value in metrics.items():
                                mlflow.log_metric(f"{model_name}_{metric_name}", value)
                
                # Log feature importance
                feature_importance = model.feature_importance.get('combined', {})
                if feature_importance and self.use_mlflow:
                    # Log top 20 features
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
                    for feature, importance in top_features:
                        mlflow.log_metric(f"feature_importance_{feature}", importance)
                
                # Save model
                model.save_model()
                
                logger.info(f"Ranking model saved successfully")
                
                # Store model version info
                self.model_versions["ranking"] = {
                    "path": settings.model.ranking_model_path,
                    "timestamp": self.timestamp,
                    "metrics": self.model_metrics.get("ranking", {})
                }
            else:
                logger.error("Ranking model training failed")
        
        except Exception as e:
            logger.error(f"Error training ranking model: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record training time
            end_time = time.time()
            self.training_times["ranking"] = end_time - start_time
            logger.info(f"Ranking training took {end_time - start_time:.2f} seconds")
    
    def _train_sentiment_model(self):
        """Train sentiment analysis model."""
        logger.info("Training sentiment analysis model")
        
        # Start timing
        start_time = time.time()
        
        # MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=f"sentiment_model_training_{self.timestamp}")
        
        try:
            # Get sentiment data
            if "sentiment" not in self.features:
                logger.warning("No sentiment data for training")
                return
            
            # Extract training and validation data
            texts = self.features["sentiment"]["texts"]
            labels = self.features["sentiment"]["labels"]
            eval_texts = self.features["sentiment"].get("eval_texts", None)
            eval_labels = self.features["sentiment"].get("eval_labels", None)
            
            # Create model
            if "sentiment" in self.best_params:
                # Use optimized hyperparameters
                hp = self.best_params["sentiment"]
                learning_rate = hp.get("learning_rate", 2e-5)
                batch_size = hp.get("batch_size", 16)
                weight_decay = hp.get("weight_decay", 0.01)
                warmup_steps = hp.get("warmup_steps", 500)
            else:
                # Use default hyperparameters
                learning_rate = 2e-5
                batch_size = 16
                weight_decay = 0.01
                warmup_steps = 500
            
            # Create the model
            model = FinancialSentimentModel(
                model_name="finbert",
                use_ensemble=True,
                entity_extraction=True,
                temporal_tracking=True
            )
            
            if self.use_mlflow:
                mlflow.log_params({
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "training_samples": len(texts),
                    "validation_samples": len(eval_texts) if eval_texts else 0,
                    "model_name": "finbert",
                    "use_ensemble": True
                })
            
            # Train model with full epochs
            epochs = min(settings.model.epochs, 3)  # Limit epochs for sentiment models
            
            # Fine-tune the model
            history = model.fine_tune(
                texts=texts,
                labels=labels,
                eval_texts=eval_texts,
                eval_labels=eval_labels,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_steps=warmup_steps
            )
            
            # Store the trained model
            self.models["sentiment"] = model
            
            # Store metrics
            self.model_metrics["sentiment"] = model.metrics
            
            if self.use_mlflow:
                # Log metrics
                for metric_name, value in model.metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"final_{metric_name}", value)
                
                # Log training history
                if history:
                    for epoch, (train_loss, eval_loss) in enumerate(zip(
                        history.get('train_loss', []),
                        history.get('eval_loss', [0] * len(history.get('train_loss', [])))
                    )):
                        mlflow.log_metrics({
                            "epoch_train_loss": train_loss,
                            "epoch_eval_loss": eval_loss
                        }, step=epoch)
                    
                    # Log validation metrics for each epoch
                    for epoch, (acc, precision, recall, f1) in enumerate(zip(
                        history.get('accuracy', []),
                        history.get('precision', []),
                        history.get('recall', []),
                        history.get('f1', [])
                    )):
                        mlflow.log_metrics({
                            "epoch_accuracy": acc,
                            "epoch_precision": precision,
                            "epoch_recall": recall,
                            "epoch_f1": f1
                        }, step=epoch)
            
            # Save model
            timestamp_dir = os.path.join(settings.models_dir, "sentiment", f"sentiment_model_{self.timestamp}")
            model.save_model(timestamp_dir)
            
            # Also save to default path
            model.save_model(settings.model.sentiment_model_path)
            
            logger.info(f"Sentiment model saved to {timestamp_dir}")
            
            # Store model version info
            self.model_versions["sentiment"] = {
                "path": timestamp_dir,
                "timestamp": self.timestamp,
                "metrics": self.model_metrics.get("sentiment", {})
            }
        
        except Exception as e:
            logger.error(f"Error training sentiment model: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record training time
            end_time = time.time()
            self.training_times["sentiment"] = end_time - start_time
            logger.info(f"Sentiment training took {end_time - start_time:.2f} seconds")
    
    def _train_exit_model(self):
        """Train exit optimization model."""
        logger.info("Training exit optimization model")
        
        # Start timing
        start_time = time.time()
        
        # MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=f"exit_model_training_{self.timestamp}")
        
        try:
            # Get training data
            if "exit" not in self.features or "episodes" not in self.features["exit"]:
                logger.warning("No exit optimization episodes for training")
                return
            
            episodes = self.features["exit"]["episodes"]
            
            # Use optimized hyperparameters if available
            if "exit" in self.best_params:
                hp = self.best_params["exit"]
                learning_rate = hp.get("learning_rate", 3e-4)
                gamma = hp.get("gamma", 0.99)
                tau = hp.get("tau", 0.005)
                alpha = hp.get("alpha", 0.2)
                hidden_dim = hp.get("hidden_dim", 256)
            else:
                # Use default hyperparameters
                learning_rate = 3e-4
                gamma = 0.99
                tau = 0.005
                alpha = 0.2
                hidden_dim = 256
            
            # Create model with SAC
            model = ExitOptimizationModel(use_sac=True)
            
            # If we could configure hidden_dim, we would do it here
            # Since it requires recreating the model, we'll leave it for now
            
            if self.use_mlflow:
                mlflow.log_params({
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "tau": tau,
                    "alpha": alpha,
                    "hidden_dim": hidden_dim,
                    "episodes": len(episodes),
                    "use_sac": True
                })
            
            # Train the model
            epochs = settings.model.epochs
            
            metrics = model.train_sac(
                training_data=episodes,
                epochs=epochs,
                batch_size=64,
                lr=learning_rate,
                gamma=gamma,
                tau=tau,
                alpha=alpha,
                updates_per_step=1
            )
            
            # Store the trained model
            self.models["exit"] = model
            
            # Store metrics
            self.model_metrics["exit"] = {
                "avg_reward": np.mean(metrics.get('epoch_rewards', [0])),
                "final_reward": metrics.get('epoch_rewards', [0])[-1] if metrics.get('epoch_rewards') else 0,
                "avg_actor_loss": np.mean(metrics.get('epoch_actor_losses', [0])),
                "avg_critic_loss": np.mean(metrics.get('epoch_critic_losses', [0])),
                "avg_value_loss": np.mean(metrics.get('epoch_value_losses', [0])),
                "avg_alpha_loss": np.mean(metrics.get('epoch_alpha_losses', [0])),
                "avg_entropy": np.mean(metrics.get('epoch_entropies', [0]))
            }
            
            if self.use_mlflow:
                # Log final metrics
                for metric_name, value in self.model_metrics["exit"].items():
                    mlflow.log_metric(metric_name, value)
                
                # Log training curves
                for epoch, (reward, actor_loss, critic_loss, value_loss, alpha_loss, entropy) in enumerate(zip(
                    metrics.get('epoch_rewards', []),
                    metrics.get('epoch_actor_losses', []),
                    metrics.get('epoch_critic_losses', []),
                    metrics.get('epoch_value_losses', []),
                    metrics.get('epoch_alpha_losses', []),
                    metrics.get('epoch_entropies', [])
                )):
                    mlflow.log_metrics({
                        "epoch_reward": reward,
                        "epoch_actor_loss": actor_loss,
                        "epoch_critic_loss": critic_loss,
                        "epoch_value_loss": value_loss,
                        "epoch_alpha_loss": alpha_loss,
                        "epoch_entropy": entropy
                    }, step=epoch)
            
            # Save model
            model_path = os.path.join(settings.models_dir, "exit", f"exit_model_{self.timestamp}.pt")
            model.save_model(model_path)
            
            # Also save to default path
            model.save_model(settings.model.exit_model_path)
            
            logger.info(f"Exit optimization model saved to {model_path}")
            
            # Store model version info
            self.model_versions["exit"] = {
                "path": model_path,
                "timestamp": self.timestamp,
                "metrics": self.model_metrics.get("exit", {})
            }
        
        except Exception as e:
            logger.error(f"Error training exit model: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record training time
            end_time = time.time()
            self.training_times["exit"] = end_time - start_time
            logger.info(f"Exit training took {end_time - start_time:.2f} seconds")
    
    def evaluate_models(self):
        """
        Evaluate all trained models and log performance metrics.
        """
        logger.info("Evaluating models")
        
        try:
            # Evaluate each model
            if "pattern" in self.models:
                self._evaluate_pattern_model()
            
            if "ranking" in self.models:
                self._evaluate_ranking_model()
            
            if "sentiment" in self.models:
                self._evaluate_sentiment_model()
            
            if "exit" in self.models:
                self._evaluate_exit_model()
            
            logger.info("Model evaluation completed")
            return True
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            return False
    
    def _evaluate_pattern_model(self):
        """Evaluate pattern recognition model."""
        logger.info("Evaluating pattern recognition model")
        
        # Start timing
        start_time = time.time()
        
        # MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=f"pattern_model_evaluation_{self.timestamp}")
        
        try:
            # Get validation data
            if "pattern" not in self.validation_datasets:
                logger.warning("No pattern validation data for evaluation")
                return
            
            val_samples = self.validation_datasets["pattern"]["samples"]
            val_labels = self.validation_datasets["pattern"]["labels"]
            
            # Get model
            if "pattern" not in self.models:
                logger.warning("No pattern model to evaluate")
                return
            
            model = self.models["pattern"]
            
            # Evaluate model
            val_metrics = model.evaluate(val_samples, val_labels)
            
            # Log metrics
            logger.info(f"Pattern model evaluation metrics: {val_metrics}")
            
            if self.use_mlflow:
                for metric_name, value in val_metrics.items():
                    mlflow.log_metric(f"eval_{metric_name}", value)
            
            # Update model metrics
            self.model_metrics["pattern"].update({
                f"eval_{k}": v for k, v in val_metrics.items()
            })
            
            # Generate confusion matrix
            if hasattr(model, 'generate_confusion_matrix'):
                cm = model.generate_confusion_matrix(val_samples, val_labels)
                
                # Log confusion matrix if available
                if cm is not None and self.use_mlflow:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    # Plot confusion matrix
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=model.PATTERN_CLASSES,
                              yticklabels=model.PATTERN_CLASSES)
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title('Confusion Matrix')
                    
                    # Save figure
                    cm_path = os.path.join(settings.models_dir, "pattern", f"confusion_matrix_{self.timestamp}.png")
                    plt.savefig(cm_path)
                    plt.close()
                    
                    # Log figure to MLflow
                    mlflow.log_artifact(cm_path)
            
            # Evaluate on a few example patterns if available
            if val_samples and len(val_samples) > 0:
                # Select a few samples
                sample_indices = np.random.choice(len(val_samples), min(5, len(val_samples)), replace=False)
                
                for i in sample_indices:
                    sample = val_samples[i]
                    true_label = val_labels[i]
                    
                    # Make prediction
                    prediction = model.predict_pattern(sample)
                    
                    logger.info(f"Sample {i}: True: {model.PATTERN_CLASSES[true_label]}, "
                               f"Predicted: {prediction[0]} with confidence {prediction[1]:.4f}")
        
        except Exception as e:
            logger.error(f"Error evaluating pattern model: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record evaluation time
            end_time = time.time()
            logger.info(f"Pattern evaluation took {end_time - start_time:.2f} seconds")
    
    def _evaluate_ranking_model(self):
        """Evaluate ranking model."""
        logger.info("Evaluating ranking model")
        
        # Start timing
        start_time = time.time()
        
        # MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=f"ranking_model_evaluation_{self.timestamp}")
        
        try:
            # Get validation data
            if "ranking" not in self.validation_datasets:
                logger.warning("No ranking validation data for evaluation")
                return
            
            val_data = self.validation_datasets["ranking"]
            
            # Get model
            if "ranking" not in self.models:
                logger.warning("No ranking model to evaluate")
                return
            
            model = self.models["ranking"]
            
            # Evaluate model performance on validation data
            if hasattr(model, 'analyze_model_performance'):
                # Convert validation DataFrame to dictionary of DataFrames
                symbols = val_data['symbol'].unique() if 'symbol' in val_data.columns else []
                
                if not symbols:
                    logger.warning("No symbols found in validation data for performance analysis")
                else:
                    # Create a dictionary of historical data for each symbol
                    historical_data = {}
                    
                    for symbol in symbols:
                        if symbol in self.data:
                            historical_data[symbol] = self.data[symbol]
                    
                    # Analyze model performance
                    lookback_days = 30
                    performance = model.analyze_model_performance(historical_data, lookback_days)
                    
                    # Log performance metrics
                    logger.info(f"Ranking model performance: " 
                               f"Accuracy: {performance.get('accuracy', 0):.4f}, "
                               f"Total predictions: {performance.get('total_predictions', 0)}")
                    
                    if self.use_mlflow:
                        for metric_name, value in performance.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"performance_{metric_name}", value)
                    
                    # Update model metrics
                    self.model_metrics["ranking"]["performance"] = performance
            
            # Generate feature importance visualization
            if hasattr(model, 'analyze_feature_importance'):
                feature_analysis = model.analyze_feature_importance()
                
                if feature_analysis and self.use_mlflow:
                    # Log top features
                    if 'top_features' in feature_analysis:
                        top_features = list(feature_analysis['top_features'].items())
                        
                        # Generate feature importance plot
                        import matplotlib.pyplot as plt
                        import seaborn as sns
                        
                        # Create plot for top 20 features
                        top_n = min(20, len(top_features))
                        top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:top_n]
                        
                        plt.figure(figsize=(12, 8))
                        sns.barplot(x=[x[1] for x in top_features], 
                                   y=[x[0] for x in top_features])
                        plt.title('Top Feature Importance')
                        plt.xlabel('Importance')
                        plt.ylabel('Feature')
                        plt.tight_layout()
                        
                        # Save figure
                        fi_path = os.path.join(settings.models_dir, "ranking", 
                                             f"feature_importance_{self.timestamp}.png")
                        plt.savefig(fi_path)
                        plt.close()
                        
                        # Log figure to MLflow
                        mlflow.log_artifact(fi_path)
                    
                    # Log category importance
                    if 'category_importance' in feature_analysis:
                        for category, info in feature_analysis['category_importance'].items():
                            mlflow.log_metric(f"category_{category}_importance", 
                                            info.get('total_importance', 0))
            
            # Test model on a few examples
            # Generate ranking for a subset of symbols
            test_symbols = list(self.data.keys())[:10]
            test_data = {symbol: self.data[symbol] for symbol in test_symbols}
            
            ranked_stocks = model.rank_stocks(test_data)
            
            if ranked_stocks:
                # Log top ranked stocks
                logger.info("Top ranked stocks:")
                for i, stock in enumerate(ranked_stocks[:5]):
                    logger.info(f"{i+1}. {stock['symbol']}: Score {stock['score']:.4f}")
        
        except Exception as e:
            logger.error(f"Error evaluating ranking model: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record evaluation time
            end_time = time.time()
            logger.info(f"Ranking evaluation took {end_time - start_time:.2f} seconds")
    
    def _evaluate_sentiment_model(self):
        """Evaluate sentiment analysis model."""
        logger.info("Evaluating sentiment analysis model")
        
        # Start timing
        start_time = time.time()
        
        # MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=f"sentiment_model_evaluation_{self.timestamp}")
        
        try:
            # Get evaluation data
            if "sentiment" not in self.features or "eval_texts" not in self.features["sentiment"]:
                logger.warning("No sentiment evaluation data")
                return
            
            eval_texts = self.features["sentiment"]["eval_texts"]
            eval_labels = self.features["sentiment"]["eval_labels"]
            
            # Get model
            if "sentiment" not in self.models:
                logger.warning("No sentiment model to evaluate")
                return
            
            model = self.models["sentiment"]
            
            # Evaluate model on test data
            predictions = []
            true_labels = eval_labels
            
            for text in eval_texts:
                # Get sentiment
                result = model.analyze_sentiment(text, extract_entities=False)
                sentiment = result["sentiment"]
                
                # Get predicted label
                scores = {
                    "negative": sentiment.get("negative", 0),
                    "neutral": sentiment.get("neutral", 0),
                    "positive": sentiment.get("positive", 0)
                }
                
                pred_label = max(scores.items(), key=lambda x: x[1])[0]
                
                # Convert to numeric label (0=negative, 1=neutral, 2=positive)
                label_map = {"negative": 0, "neutral": 1, "positive": 2}
                pred_numeric = label_map.get(pred_label, 1)
                
                predictions.append(pred_numeric)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='weighted'
            )
            cm = confusion_matrix(true_labels, predictions)
            
            # Log metrics
            logger.info(f"Sentiment evaluation metrics: "
                       f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                       f"Recall: {recall:.4f}, F1: {f1:.4f}")
            
            if self.use_mlflow:
                mlflow.log_metrics({
                    "eval_accuracy": accuracy,
                    "eval_precision": precision,
                    "eval_recall": recall,
                    "eval_f1": f1
                })
            
            # Update model metrics
            self.model_metrics["sentiment"].update({
                "eval_accuracy": accuracy,
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_f1": f1
            })
            
            # Generate confusion matrix visualization
            if self.use_mlflow:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=["Negative", "Neutral", "Positive"],
                           yticklabels=["Negative", "Neutral", "Positive"])
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title('Sentiment Confusion Matrix')
                
                # Save figure
                cm_path = os.path.join(settings.models_dir, "sentiment", 
                                     f"confusion_matrix_{self.timestamp}.png")
                plt.savefig(cm_path)
                plt.close()
                
                # Log figure to MLflow
                mlflow.log_artifact(cm_path)
            
            # Test entity extraction on a few examples
            if hasattr(model, 'extract_entities'):
                # Sample a few financial news texts
                sample_news = [
                    "Apple reported better than expected earnings, with iPhone sales beating forecasts.",
                    "Tesla shares plummeted after Elon Musk announced production delays for the Cybertruck.",
                    "The Federal Reserve decided to maintain interest rates at the current level, citing inflation concerns."
                ]
                
                for text in sample_news:
                    # Extract entities
                    entities = model.extract_entities(text)
                    
                    # Log entities
                    entity_str = ", ".join([f"{e['text']} ({e['type']})" for e in entities])
                    logger.info(f"Entities in '{text[:50]}...': {entity_str}")
            
            # Generate a sentiment report if news data is available
            if hasattr(model, 'generate_sentiment_report') and hasattr(self, 'news_data'):
                # Generate report
                report = model.generate_sentiment_report(
                    self.news_data[:100],  # Use a subset of news data
                    include_entities=True,
                    include_network=True
                )
                
                # Log report summary
                logger.info(f"Sentiment report: "
                           f"Overall sentiment: {report.get('overall_sentiment', {}).get('overall_score', 0):.4f}, "
                           f"Positive items: {report.get('positive_count', 0)}, "
                           f"Negative items: {report.get('negative_count', 0)}, "
                           f"Neutral items: {report.get('neutral_count', 0)}")
                
                # Save report to file
                report_path = os.path.join(settings.models_dir, "sentiment", 
                                         f"sentiment_report_{self.timestamp}.json")
                
                with open(report_path, 'w') as f:
                    import json
                    json.dump(report, f, indent=2)
                
                if self.use_mlflow:
                    # Log report to MLflow
                    mlflow.log_artifact(report_path)
                    
                    # Log summary metrics
                    mlflow.log_metrics({
                        "report_overall_sentiment": report.get('overall_sentiment', {}).get('overall_score', 0),
                        "report_positive_count": report.get('positive_count', 0),
                        "report_negative_count": report.get('negative_count', 0),
                        "report_neutral_count": report.get('neutral_count', 0)
                    })
        
        except Exception as e:
            logger.error(f"Error evaluating sentiment model: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record evaluation time
            end_time = time.time()
            logger.info(f"Sentiment evaluation took {end_time - start_time:.2f} seconds")
    
    def _evaluate_exit_model(self):
        """Evaluate exit optimization model."""
        logger.info("Evaluating exit optimization model")
        
        # Start timing
        start_time = time.time()
        
        # MLflow tracking
        if self.use_mlflow:
            mlflow.start_run(run_name=f"exit_model_evaluation_{self.timestamp}")
        
        try:
            # Get model
            if "exit" not in self.models:
                logger.warning("No exit model to evaluate")
                return
            
            model = self.models["exit"]
            
            # Backtest model if historical data is available
            if hasattr(model, 'backtest') and self.data:
                # Select a subset of symbols for backtesting
                test_symbols = list(self.data.keys())[:5]
                test_data = {symbol: self.data[symbol] for symbol in test_symbols}
                
                # Run backtest with different strategies
                logger.info("Running exit strategy backtest...")
                
                backtest_results = model.backtest(
                    historical_data=test_data,
                    exit_strategy="model",
                    initial_capital=10000.0,
                    position_size_pct=0.2
                )
                
                # Run simple strategy for comparison
                simple_results = model.backtest(
                    historical_data=test_data,
                    exit_strategy="simple",
                    initial_capital=10000.0,
                    position_size_pct=0.2
                )
                
                # Run hold strategy for comparison
                hold_results = model.backtest(
                    historical_data=test_data,
                    exit_strategy="hold",
                    initial_capital=10000.0,
                    position_size_pct=0.2
                )
                
                # Log backtest results
                logger.info(f"Exit model backtest results: "
                           f"Model strategy: {backtest_results.get('total_return', 0):.2f}% return, "
                           f"Simple strategy: {simple_results.get('total_return', 0):.2f}% return, "
                           f"Hold strategy: {hold_results.get('total_return', 0):.2f}% return")
                
                if self.use_mlflow:
                    # Log backtest metrics
                    for strategy, results in [
                        ("model", backtest_results),
                        ("simple", simple_results),
                        ("hold", hold_results)
                    ]:
                        for metric, value in results.items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"backtest_{strategy}_{metric}", value)
                
                # Update model metrics
                self.model_metrics["exit"]["backtest"] = {
                    "model_return": backtest_results.get('total_return', 0),
                    "model_sharpe": backtest_results.get('sharpe_ratio', 0),
                    "model_win_rate": backtest_results.get('win_rate', 0),
                    "simple_return": simple_results.get('total_return', 0),
                    "simple_sharpe": simple_results.get('sharpe_ratio', 0),
                    "simple_win_rate": simple_results.get('win_rate', 0),
                    "hold_return": hold_results.get('total_return', 0),
                    "hold_sharpe": hold_results.get('sharpe_ratio', 0),
                    "hold_win_rate": hold_results.get('win_rate', 0)
                }
                
                # Generate equity curve visualization
                if self.use_mlflow and 'equity_curve' in backtest_results:
                    import matplotlib.pyplot as plt
                    
                    # Plot equity curves
                    plt.figure(figsize=(12, 6))
                    
                    # Model equity curve
                    if 'equity_curve' in backtest_results:
                        plt.plot(backtest_results['equity_curve'], label='Model Strategy')
                    
                    # Simple equity curve
                    if 'equity_curve' in simple_results:
                        plt.plot(simple_results['equity_curve'], label='Simple Strategy')
                    
                    # Hold equity curve
                    if 'equity_curve' in hold_results:
                        plt.plot(hold_results['equity_curve'], label='Hold Strategy')
                    
                    plt.title('Equity Curves')
                    plt.xlabel('Trades')
                    plt.ylabel('Capital ($)')
                    plt.legend()
                    plt.grid(True)
                    
                    # Save figure
                    equity_path = os.path.join(settings.models_dir, "exit", 
                                            f"equity_curve_{self.timestamp}.png")
                    plt.savefig(equity_path)
                    plt.close()
                    
                    # Log figure to MLflow
                    mlflow.log_artifact(equity_path)
            
            # Test model on a few examples
            # Generate exit recommendations for a few positions
            for symbol, df in list(self.data.items())[:3]:
                if len(df) < 50:
                    continue
                
                # Convert to pandas DataFrame if using GPU
                if self.use_gpu and isinstance(df, cudf.DataFrame):
                    df_pandas = df.to_pandas()
                else:
                    df_pandas = df
                
                # Create a sample position
                entry_idx = len(df_pandas) - 20
                entry_price = df_pandas['close'].iloc[entry_idx]
                
                position_data = {
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "entry_time": df_pandas.index[entry_idx],
                    "position_size": 1.0,
                    "stop_loss": entry_price * 0.95,  # 5% stop loss
                    "take_profit": entry_price * 1.1,  # 10% take profit
                    "trailing_stop": 2.0  # 2% trailing stop
                }
                
                # Get exit recommendation
                current_data = df_pandas.iloc[:len(df_pandas)-5]  # Leave some room at the end
                
                recommendation = model.evaluate_exit_conditions(current_data, position_data)
                
                # Log recommendation
                logger.info(f"Exit recommendation for {symbol}: "
                           f"{'Exit' if recommendation.get('exit', False) else 'Hold'} "
                           f"with size {recommendation.get('size', 0):.2f}, "
                           f"Reason: {recommendation.get('reason', 'unknown')}, "
                           f"Confidence: {recommendation.get('confidence', 0):.4f}")
        
        except Exception as e:
            logger.error(f"Error evaluating exit model: {e}")
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Record evaluation time
            end_time = time.time()
            logger.info(f"Exit evaluation took {end_time - start_time:.2f} seconds")
    
    def save_results(self):
        """
        Save training results and metrics.
        """
        logger.info("Saving training results")
        
        try:
            # Create results directory
            results_dir = os.path.join(settings.models_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Create timestamp directory
            timestamp_dir = os.path.join(results_dir, self.timestamp)
            os.makedirs(timestamp_dir, exist_ok=True)
            
            # Save metrics
            metrics_path = os.path.join(timestamp_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f, indent=2)
            
            # Save training times
            times_path = os.path.join(timestamp_dir, "training_times.json")
            with open(times_path, 'w') as f:
                json.dump({
                    "training_times": self.training_times,
                    "optimization_times": self.optimization_times,
                    "total_time": sum(self.training_times.values()) + sum(self.optimization_times.values())
                }, f, indent=2)
            
            # Save model versions
            versions_path = os.path.join(timestamp_dir, "model_versions.json")
            with open(versions_path, 'w') as f:
                json.dump(self.model_versions, f, indent=2)
            
            # Save hyperparameters
            params_path = os.path.join(timestamp_dir, "hyperparameters.json")
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            
            # Generate a summary
            summary = {
                "timestamp": self.timestamp,
                "models_trained": self.models_to_train,
                "use_gpu": self.use_gpu,
                "optimize_hyperparams": self.optimize_hyperparams,
                "data_days": self.data_days,
                "training_summary": {}
            }
            
            # Add summary for each model
            for model_name in self.models_to_train:
                if model_name in self.model_metrics:
                    metrics = self.model_metrics[model_name]
                    
                    # Extract key metrics
                    key_metrics = {}
                    
                    if model_name == "pattern":
                        key_metrics["accuracy"] = metrics.get("val_accuracy", 0)
                    elif model_name == "ranking":
                        if "ensemble" in metrics:
                            key_metrics["auc"] = metrics["ensemble"].get("auc", 0)
                            key_metrics["accuracy"] = metrics["ensemble"].get("accuracy", 0)
                    elif model_name == "sentiment":
                        key_metrics["accuracy"] = metrics.get("accuracy", 0)
                        key_metrics["f1"] = metrics.get("f1", 0)
                    elif model_name == "exit":
                        key_metrics["avg_reward"] = metrics.get("avg_reward", 0)
                        if "backtest" in metrics:
                            key_metrics["model_return"] = metrics["backtest"].get("model_return", 0)
                            key_metrics["model_sharpe"] = metrics["backtest"].get("model_sharpe", 0)
                    
                    # Add training time
                    if model_name in self.training_times:
                        key_metrics["training_time"] = f"{self.training_times[model_name]:.2f}s"
                    
                    summary["training_summary"][model_name] = key_metrics
            
            # Save summary
            summary_path = os.path.join(timestamp_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Generate README.md
            readme_content = f"""# Model Training Results {self.timestamp}

## Summary
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Models Trained**: {', '.join(self.models_to_train)}
- **GPU Used**: {'Yes' if self.use_gpu else 'No'}
- **Hyperparameter Optimization**: {'Yes' if self.optimize_hyperparams else 'No'}
- **Data Period**: {self.data_days} days

## Performance Metrics
"""
            
            for model_name, metrics in summary["training_summary"].items():
                readme_content += f"### {model_name.title()} Model\n"
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        readme_content += f"- **{metric_name}**: {value:.4f}\n"
                    else:
                        readme_content += f"- **{metric_name}**: {value}\n"
                readme_content += "\n"
            
            readme_content += "## Training Details\n"
            for model_name in self.models_to_train:
                if model_name in self.training_times:
                    optimization_time = self.optimization_times.get(model_name, 0)
                    training_time = self.training_times[model_name]
                    
                    readme_content += f"- **{model_name.title()}**: "
                    readme_content += f"Optimization: {optimization_time:.2f}s, "
                    readme_content += f"Training: {training_time:.2f}s, "
                    readme_content += f"Total: {optimization_time + training_time:.2f}s\n"
            
            # Save README
            readme_path = os.path.join(timestamp_dir, "README.md")
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            logger.info(f"Saved training results to {timestamp_dir}")
            
            # Update Redis with training results
            redis_client.set("model_training:latest", {
                "timestamp": self.timestamp,
                "summary": summary
            })
            
            return True
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def run_training_pipeline(self):
        """
        Run the full training pipeline.
        """
        logger.info("Starting training pipeline")
        
        # Overall timing
        start_time = time.time()
        
        # MLflow experiment for the entire pipeline
        if self.use_mlflow:
            mlflow.start_run(run_name=f"training_pipeline_{self.timestamp}")
            mlflow.log_params({
                "models_to_train": self.models_to_train,
                "data_days": self.data_days,
                "use_gpu": self.use_gpu,
                "optimize_hyperparams": self.optimize_hyperparams,
                "optimization_trials": self.optimization_trials,
                "timestamp": self.timestamp
            })
        
        try:
            # Step 1: Download data
            logger.info("Step 1: Downloading data")
            if not self.download_data():
                raise Exception("Data download failed")
            
            # Step 2: Prepare features
            logger.info("Step 2: Preparing features")
            if not self.prepare_features():
                raise Exception("Feature preparation failed")
            
            # Step 3: Split data
            logger.info("Step 3: Splitting data")
            if not self.split_data():
                raise Exception("Data splitting failed")
            
            # Step 4: Optimize hyperparameters
            if self.optimize_hyperparams:
                logger.info("Step 4: Optimizing hyperparameters")
                if not self.optimize_hyperparameters():
                    logger.warning("Hyperparameter optimization encountered issues - continuing with defaults")
            else:
                logger.info("Step 4: Skipping hyperparameter optimization (disabled)")
            
            # Step 5: Train models
            logger.info("Step 5: Training models")
            if not self.train_models():
                raise Exception("Model training failed")
            
            # Step 6: Evaluate models
            logger.info("Step 6: Evaluating models")
            if not self.evaluate_models():
                logger.warning("Model evaluation encountered issues")
            
            # Step 7: Save results
            logger.info("Step 7: Saving results")
            if not self.save_results():
                logger.warning("Saving results encountered issues")
            
            # Calculate total time
            end_time = time.time()
            total_time = end_time - start_time
            
            logger.info(f"Training pipeline completed successfully in {total_time:.2f} seconds")
            
            if self.use_mlflow:
                mlflow.log_metric("total_duration_seconds", total_time)
                
                # Log summary metrics for each model
                for model_name, metrics in self.model_metrics.items():
                    if model_name == "pattern" and "val_accuracy" in metrics:
                        mlflow.log_metric(f"{model_name}_accuracy", metrics["val_accuracy"])
                    elif model_name == "ranking" and "ensemble" in metrics:
                        mlflow.log_metric(f"{model_name}_auc", metrics["ensemble"].get("auc", 0))
                    elif model_name == "sentiment":
                        mlflow.log_metric(f"{model_name}_accuracy", metrics.get("accuracy", 0))
                        mlflow.log_metric(f"{model_name}_f1", metrics.get("f1", 0))
                    elif model_name == "exit":
                        mlflow.log_metric(f"{model_name}_avg_reward", metrics.get("avg_reward", 0))
                        if "backtest" in metrics:
                            mlflow.log_metric(f"{model_name}_model_return", metrics["backtest"].get("model_return", 0))
            
            return True
        
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            
            if self.use_mlflow:
                mlflow.log_param("pipeline_status", "failed")
                mlflow.log_param("failure_reason", str(e))
            
            return False
        
        finally:
            # End MLflow run if active
            if self.use_mlflow:
                mlflow.end_run()
            
            # Clean up GPU memory if used
            if self.use_gpu:
                try:
                    if hasattr(self, 'client'):
                        self.client.close()
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass

# Command-line interface
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train ML models for trading system")
    
    parser.add_argument("--model", type=str, default="all",
                       help="Model to train (pattern, ranking, sentiment, exit, all)")
    
    parser.add_argument("--days", type=int, default=365,
                       help="Number of days of historical data to use")
    
    parser.add_argument("--gpu", type=str, default="auto",
                       help="Whether to use GPU acceleration (true, false, auto)")
    
    parser.add_argument("--optimize", type=str, default="true",
                       help="Whether to run hyperparameter optimization (true, false)")
    
    parser.add_argument("--trials", type=int, default=50,
                       help="Number of trials for hyperparameter optimization")
    
    parser.add_argument("--mlflow", type=str, default="true",
                       help="Whether to use MLflow for experiment tracking (true, false)")
    
    parser.add_argument("--experiment", type=str, default="trading_system_training",
                       help="MLflow experiment name")
    
    parser.add_argument("--mode", type=str, default="pipeline",
                       help="Mode (pipeline, optimize, train, evaluate)")
    
    return parser.parse_args()

# Scheduled training function for off-hours training
async def schedule_training():
    """Schedule model training to run during off-hours."""
    try:
        # Get current hour
        current_hour = datetime.now().hour
        
        # Define off-hours (typically outside of market hours)
        # US market hours are 9:30 AM - 4:00 PM Eastern Time
        is_off_hours = current_hour < 9 or current_hour > 16
        
        # Check if it's a weekend
        is_weekend = datetime.now().weekday() >= 5  # 5=Saturday, 6=Sunday
        
        if is_off_hours or is_weekend:
            logger.info("Starting scheduled model training (off-hours or weekend)")
            
            # Determine models to train
            # On weekends, train all models
            # On weekdays, rotate through models (based on day of week)
            if is_weekend:
                models_to_train = ["all"]
                days = 365  # Use more data on weekends
                optimize = True
            else:
                # Rotate through models on weekdays
                weekday = datetime.now().weekday()
                if weekday == 0:  # Monday
                    models_to_train = ["pattern"]
                elif weekday == 1:  # Tuesday
                    models_to_train = ["ranking"]
                elif weekday == 2:  # Wednesday
                    models_to_train = ["sentiment"]
                elif weekday == 3:  # Thursday
                    models_to_train = ["exit"]
                else:  # Friday
                    models_to_train = ["all"]
                
                days = 180  # Use less data on weekdays
                optimize = weekday == 4  # Only optimize on Fridays
            
            # Initialize trainer
            trainer = ModelTrainer(
                models_to_train=models_to_train,
                data_days=days,
                use_gpu=GPU_AVAILABLE,
                optimize_hyperparams=optimize,
                optimization_trials=30 if optimize else 0,
                use_mlflow=True
            )
            
            # Run training pipeline
            success = trainer.run_training_pipeline()
            
            logger.info(f"Scheduled training {'completed successfully' if success else 'failed'}")
            
            # Update Redis with training status
            redis_client.set("model_training:scheduled", {
                "timestamp": datetime.now().isoformat(),
                "models": models_to_train,
                "success": success
            })
            
            return success
        else:
            logger.info("Not in off-hours, skipping scheduled training")
            return False
    
    except Exception as e:
        logger.error(f"Error in scheduled training: {e}")
        return False

# Main function
def main():
    # Parse arguments
    args = parse_args()
    
    # Parse boolean arguments
    use_gpu = args.gpu.lower() == "true" if args.gpu.lower() != "auto" else GPU_AVAILABLE
    optimize = args.optimize.lower() == "true"
    use_mlflow = args.mlflow.lower() == "true"
    
    # Parse models to train
    if args.model == "all":
        models_to_train = ["all"]
    else:
        models_to_train = args.model.split(",")
    
    # Create trainer
    trainer = ModelTrainer(
        models_to_train=models_to_train,
        data_days=args.days,
        use_gpu=use_gpu,
        optimize_hyperparams=optimize,
        optimization_trials=args.trials,
        use_mlflow=use_mlflow,
        mlflow_experiment_name=args.experiment
    )
    
    # Run based on mode
    if args.mode == "pipeline":
        # Run full pipeline
        success = trainer.run_training_pipeline()
        
        if success:
            logger.info("Training pipeline completed successfully")
            return 0
        else:
            logger.error("Training pipeline failed")
            return 1
    
    elif args.mode == "optimize":
        # Run optimization only
        trainer.download_data()
        trainer.prepare_features()
        trainer.split_data()
        
        success = trainer.optimize_hyperparameters()
        
        if success:
            logger.info("Hyperparameter optimization completed successfully")
            
            # Save optimization results
            results_dir = os.path.join(settings.models_dir, "optimization")
            os.makedirs(results_dir, exist_ok=True)
            
            results_path = os.path.join(results_dir, f"optimization_{trainer.timestamp}.json")
            with open(results_path, 'w') as f:
                json.dump(trainer.best_params, f, indent=2)
            
            logger.info(f"Optimization results saved to {results_path}")
            
            return 0
        else:
            logger.error("Hyperparameter optimization failed")
            return 1
    
    elif args.mode == "train":
        # Run training only
        trainer.download_data()
        trainer.prepare_features()
        trainer.split_data()
        
        success = trainer.train_models()
        
        if success:
            logger.info("Model training completed successfully")
            return 0
        else:
            logger.error("Model training failed")
            return 1
    
    elif args.mode == "evaluate":
        # Run evaluation only
        trainer.download_data()
        
        success = trainer.evaluate_models()
        
        if success:
            logger.info("Model evaluation completed successfully")
            return 0
        else:
            logger.error("Model evaluation failed")
            return 1
    
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

# Entry point for scheduling
async def run_scheduled_training():
    """Run scheduled training as a standalone task."""
    return await schedule_training()

if __name__ == "__main__":
    # Run main function
    exit_code = main()
    sys.exit(exit_code)