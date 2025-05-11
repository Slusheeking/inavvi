"""
Training module for the trading system.

This module provides functionality for training ML models:
- Data preparation pipelines
- Model training workflows
- Cross-validation utilities
- Hyperparameter optimization
- Scheduled off-hours training
- Model evaluation and metrics tracking
"""
import argparse
import os
# import sys  # Unused import
# import logging  # Unused import
import json
import time
from datetime import datetime  # timedelta is unused
# from pathlib import Path  # Unused import
from typing import Dict, List, Tuple, Any, Callable  # Union is unused
from concurrent.futures import ThreadPoolExecutor
import asyncio
# Import schedule package
try:
    import schedule
except ImportError:
    # Handle the case where schedule is not installed
    schedule = None

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split  # StratifiedKFold is unused
# from sklearn.preprocessing import StandardScaler  # Unused import
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # roc_auc_score is unused
import optuna

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client
from src.models.pattern_recognition import pattern_recognition_model
from src.models.exit_optimization import exit_optimization_model
from src.models.ranking_model import ranking_model
from src.training.data_fetcher import DataFetcher

# Set up logger
logger = setup_logger("model_training")

# Import models
try:
    from src.models.sentiment import sentiment_model, train_sentiment_model
except ImportError:
    logger.error("Could not import sentiment_model. Check the module path.")
    sentiment_model = None
    train_sentiment_model = None

# Import exit optimization model
try:
    from src.models.exit_optimization import train_exit_model
except ImportError:
    logger.error("Could not import exit_optimization model. Check the module path.")
    train_exit_model = None

# Log schedule import status
if schedule is None:
    logger.warning("Schedule package not installed. Scheduled training will not be available.")

# Logger is already set up above

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class ModelTrainer:
    """
    Class for training and evaluating trading models.
    Handles data preparation, training workflows, and model evaluation.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.data_fetcher = DataFetcher(
            data_days=365,
            use_polygon=True,
            use_alpha_vantage=True,
            use_redis_cache=True,
            use_gpu=torch.cuda.is_available()
        )
        
        # Directories for training data and model outputs
        self.data_dir = settings.data_dir
        self.models_dir = settings.models_dir
        
        # Training parameters
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.val_size = 0.2
        self.random_state = 42
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {}
        
        # Optuna storage
        self.optuna_storage = "sqlite:///models/optuna.db"
        
        logger.info("ModelTrainer initialized")
    
    def prepare_sentiment_data(self) -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        Prepare data for sentiment model training.
        
        Returns:
            Tuple of (train_texts, train_labels, val_texts, val_labels)
        """
        logger.info("Preparing sentiment training data")
        
        # Fetch financial news data
        news_items = self.data_fetcher.fetch_news_data(days=90)
        
        if not news_items:
            logger.warning("No news data found, using synthetic data")
            
            # Generate synthetic data if no real data is available
            texts = [
                "The company reported record profits for the quarter.",
                "Shares plummeted after disappointing earnings.",
                "Market remained stable during trading.",
                "Investors are concerned about high debt levels.",
                "New product launch exceeded expectations.",
                "Expansion into new markets announced.",
                "Revenue growth slowed unexpectedly.",
                "Board approved a share buyback program.",
                "Economic indicators suggest a recession.",
                "Merger deal approved by regulators.",
                "CEO announces retirement, shares drop on the news.",
                "Analysts upgrade stock rating to 'buy' with higher price target.",
                "Company misses quarterly revenue expectations but beats on earnings.",
                "Regulatory investigation causes investor concern.",
                "Strong holiday sales boost retail sector outlook."
            ]
            
            labels = [2, 0, 1, 0, 2, 2, 0, 2, 0, 1, 0, 2, 1, 0, 2]  # 0=negative, 1=neutral, 2=positive
            
            # Split into train/validation
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=self.val_size, random_state=self.random_state, stratify=labels
            )
            
            return train_texts, train_labels, val_texts, val_labels
        
        # Process news items to extract text and create sentiment labels
        texts = []
        labels = []
        
        for item in news_items:
            # Combine title and summary
            title = item.get("title", "")
            summary = item.get("summary", "")
            
            if title and summary:
                text = f"{title}. {summary}"
            elif title:
                text = title
            elif summary:
                text = summary
            else:
                continue
            
            texts.append(text)
            
            # Assign sentiment label based on sentiment score
            sentiment_score = item.get("sentiment_score", 0)
            
            if sentiment_score > 0.2:
                label = 2  # Positive
            elif sentiment_score < -0.2:
                label = 0  # Negative
            else:
                label = 1  # Neutral
                
            labels.append(label)
        
        # Split into train/validation
        if len(texts) > 10:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=self.val_size, random_state=self.random_state, stratify=labels
            )
            
            logger.info(f"Prepared {len(train_texts)} training and {len(val_texts)} validation examples")
            return train_texts, train_labels, val_texts, val_labels
        else:
            logger.warning("Not enough data for validation split, using all for training")
            return texts, labels, [], []
    
    def prepare_pattern_data(self) -> Tuple[List[pd.DataFrame], List[int], List[pd.DataFrame], List[int]]:
        """
        Prepare data for pattern recognition model training.
        
        Returns:
            Tuple of (train_data, train_labels, val_data, val_labels)
        """
        logger.info("Preparing pattern recognition training data")
        
        # Fetch historical price data for a set of liquid stocks
        stock_universe = self.data_fetcher.get_universe(size=100)
        historical_data = self.data_fetcher.fetch_historical_data(symbols=stock_universe, timeframe="1d")
        
        # Check if data was fetched successfully
        if not historical_data:
            logger.warning("No historical data found, using synthetic data")
            
            # Generate synthetic data with patterns
            return self._generate_synthetic_pattern_data()
        
        # Process historical data to identify patterns
        # This would typically involve labeling the data with pattern types
        # For now, we'll use a simple approach to generate labels
        
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        
        pattern_window = 50  # Window size for pattern recognition
        
        for symbol, df in historical_data.items():
            # Skip if not enough data
            if len(df) < pattern_window * 2:
                continue
                
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Generate pattern labels using rule-based approach
            for i in range(pattern_window, len(df) - pattern_window):
                window = df.iloc[i-pattern_window:i].copy()
                
                # Determine pattern type using rules
                pattern_type = self._detect_pattern_type(window)
                
                if np.random.random() < 0.8:  # 80% to training, 20% to validation
                    train_data.append(window)
                    train_labels.append(pattern_type)
                else:
                    val_data.append(window)
                    val_labels.append(pattern_type)
        
        logger.info(f"Prepared {len(train_data)} training and {len(val_data)} validation examples")
        return train_data, train_labels, val_data, val_labels
    
    def _generate_synthetic_pattern_data(self) -> Tuple[List[pd.DataFrame], List[int], List[pd.DataFrame], List[int]]:
        """
        Generate synthetic data with patterns for training.
        
        Returns:
            Tuple of (train_data, train_labels, val_data, val_labels)
        """
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        
        # Pattern types to generate
        pattern_types = [0, 1, 2, 3, 4, 5]  # 0=none, 1=double_top, 2=double_bottom, etc.
        
        # Generate 100 examples of each pattern type
        for pattern_type in pattern_types:
            for _ in range(100):
                # Generate synthetic OHLCV data with the pattern
                df = self._generate_pattern(pattern_type)
                
                if np.random.random() < 0.8:  # 80% to training, 20% to validation
                    train_data.append(df)
                    train_labels.append(pattern_type)
                else:
                    val_data.append(df)
                    val_labels.append(pattern_type)
        
        logger.info(f"Generated {len(train_data)} training and {len(val_data)} validation synthetic examples")
        return train_data, train_labels, val_data, val_labels
    
    def _generate_pattern(self, pattern_type: int) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data with a specific pattern.
        
        Args:
            pattern_type: Type of pattern to generate
            
        Returns:
            DataFrame with OHLCV data containing the pattern
        """
        # Base price and time parameters
        days = 50
        base_price = 100.0
        volatility = 0.01
        
        # Generate random walk for close prices
        close_prices = [base_price]
        for _ in range(days - 1):
            price_change = np.random.normal(0, volatility)
            new_price = close_prices[-1] * (1 + price_change)
            close_prices.append(new_price)
        
        # Generate OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=days)
        df = pd.DataFrame(index=dates)
        df['close'] = close_prices
        
        # Calculate open, high, low based on close prices
        df['open'] = df['close'].shift(1)
        df.loc[df.index[0], 'open'] = df['close'].iloc[0] * (1 + np.random.normal(0, volatility))
        
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, volatility/2, days)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, volatility/2, days)))
        
        # Generate random volume
        df['volume'] = np.random.randint(10000, 100000, size=days)
        
        # Modify data to create the pattern
        if pattern_type == 1:  # Double top
            peak1_idx = days // 3
            peak2_idx = (2 * days) // 3
            
            # Create two peaks
            df.loc[df.index[peak1_idx], 'high'] = base_price * 1.15
            df.loc[df.index[peak1_idx], 'close'] = base_price * 1.10
            
            df.loc[df.index[peak2_idx], 'high'] = base_price * 1.15
            df.loc[df.index[peak2_idx], 'close'] = base_price * 1.08
            
            # Lower prices after second peak
            for i in range(peak2_idx + 1, days):
                factor = 1.0 - 0.01 * (i - peak2_idx)
                df.loc[df.index[i], 'close'] *= factor
                df.loc[df.index[i], 'high'] *= factor
                df.loc[df.index[i], 'low'] *= factor
                df.loc[df.index[i], 'open'] *= factor
        
        elif pattern_type == 2:  # Double bottom
            trough1_idx = days // 3
            trough2_idx = (2 * days) // 3
            
            # Create two troughs
            df.loc[df.index[trough1_idx], 'low'] = base_price * 0.85
            df.loc[df.index[trough1_idx], 'close'] = base_price * 0.90
            
            df.loc[df.index[trough2_idx], 'low'] = base_price * 0.85
            df.loc[df.index[trough2_idx], 'close'] = base_price * 0.92
            
            # Higher prices after second trough
            for i in range(trough2_idx + 1, days):
                factor = 1.0 + 0.01 * (i - trough2_idx)
                df.loc[df.index[i], 'close'] *= factor
                df.loc[df.index[i], 'high'] *= factor
                df.loc[df.index[i], 'low'] *= factor
                df.loc[df.index[i], 'open'] *= factor
                
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Moving averages
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        
        # Relative Strength Index (RSI)
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
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        return df
    
    def _detect_pattern_type(self, window: pd.DataFrame) -> int:
        """
        Detect pattern type using rule-based approach.
        
        Args:
            window: DataFrame window to analyze
            
        Returns:
            Pattern type index
        """
        # Simple rules for pattern detection
        # In a real implementation, this would be much more sophisticated
        
        # Get key statistics
        close = window['close'].values
        high = window['high'].values
        low = window['low'].values
        
        # Check for double top
        if self._is_double_top(high, close):
            return 1
        
        # Check for double bottom
        elif self._is_double_bottom(low, close):
            return 2
        
        # Check for head and shoulders
        elif self._is_head_shoulders(high, close):
            return 3
        
        # Check for inverse head and shoulders
        elif self._is_inv_head_shoulders(low, close):
            return 4
        
        # Check for ascending triangle
        elif self._is_ascending_triangle(high, low, close):
            return 5
        
        # No pattern detected
        else:
            return 0
    
    def _is_double_top(self, high: np.ndarray, close: np.ndarray) -> bool:
        """Simple double top detection."""
        # Find peaks in the high prices
        peaks = []
        for i in range(1, len(high) - 1):
            if high[i] > high[i-1] and high[i] > high[i+1]:
                peaks.append((i, high[i]))
        
        # Need at least two peaks
        if len(peaks) < 2:
            return False
        
        # Find two highest peaks
        peaks.sort(key=lambda x: x[1], reverse=True)
        peak1_idx, peak1_val = peaks[0]
        peak2_idx, peak2_val = peaks[1]
        
        # Peaks should be separated
        if abs(peak1_idx - peak2_idx) < 5:
            return False
        
        # Peaks should be roughly at the same level
        if abs(peak1_val - peak2_val) / peak1_val > 0.05:
            return False
        
        # There should be a trough between the peaks
        min_between = min(close[min(peak1_idx, peak2_idx):max(peak1_idx, peak2_idx)])
        
        # Minimum between peaks should be significantly lower
        if (peak1_val - min_between) / peak1_val < 0.03:
            return False
        
        return True
    
    def _is_double_bottom(self, low: np.ndarray, close: np.ndarray) -> bool:
        """Simple double bottom detection."""
        # Find troughs in the low prices
        troughs = []
        for i in range(1, len(low) - 1):
            if low[i] < low[i-1] and low[i] < low[i+1]:
                troughs.append((i, low[i]))
        
        # Need at least two troughs
        if len(troughs) < 2:
            return False
        
        # Find two lowest troughs
        troughs.sort(key=lambda x: x[1])
        trough1_idx, trough1_val = troughs[0]
        trough2_idx, trough2_val = troughs[1]
        
        # Troughs should be separated
        if abs(trough1_idx - trough2_idx) < 5:
            return False
        
        # Troughs should be roughly at the same level
        if abs(trough1_val - trough2_val) / trough1_val > 0.05:
            return False
        
        # There should be a peak between the troughs
        max_between = max(close[min(trough1_idx, trough2_idx):max(trough1_idx, trough2_idx)])
        
        # Maximum between troughs should be significantly higher
        if (max_between - trough1_val) / trough1_val < 0.03:
            return False
        
        return True
    
    def _is_head_shoulders(self, high: np.ndarray, close: np.ndarray) -> bool:
        """Simple head and shoulders detection."""
        # Find peaks in the high prices
        peaks = []
        for i in range(1, len(high) - 1):
            if high[i] > high[i-1] and high[i] > high[i+1]:
                peaks.append((i, high[i]))
        
        # Need at least three peaks
        if len(peaks) < 3:
            return False
        
        # Sort peaks by value
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Get the highest peak (head)
        head_idx, head_val = peaks[0]
        
        # Find potential shoulders
        left_shoulders = [(i, val) for i, val in peaks[1:] if i < head_idx]
        right_shoulders = [(i, val) for i, val in peaks[1:] if i > head_idx]
        
        # Need at least one peak on each side
        if not left_shoulders or not right_shoulders:
            return False
        
        # Get the highest peak on each side (shoulders)
        left_shoulder = max(left_shoulders, key=lambda x: x[1])
        right_shoulder = max(right_shoulders, key=lambda x: x[1])
        
        # Shoulders should be at similar heights
        if abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] > 0.1:
            return False
        
        # Head should be higher than shoulders
        if head_val < left_shoulder[1] or head_val < right_shoulder[1]:
            return False
        
        return True
    
    def _is_inv_head_shoulders(self, low: np.ndarray, close: np.ndarray) -> bool:
        """Simple inverse head and shoulders detection."""
        # Find troughs in the low prices
        troughs = []
        for i in range(1, len(low) - 1):
            if low[i] < low[i-1] and low[i] < low[i+1]:
                troughs.append((i, low[i]))
        
        # Need at least three troughs
        if len(troughs) < 3:
            return False
        
        # Sort troughs by value
        troughs.sort(key=lambda x: x[1])
        
        # Get the lowest trough (head)
        head_idx, head_val = troughs[0]
        
        # Find potential shoulders
        left_shoulders = [(i, val) for i, val in troughs[1:] if i < head_idx]
        right_shoulders = [(i, val) for i, val in troughs[1:] if i > head_idx]
        
        # Need at least one trough on each side
        if not left_shoulders or not right_shoulders:
            return False
        
        # Get the lowest trough on each side (shoulders)
        left_shoulder = min(left_shoulders, key=lambda x: x[1])
        right_shoulder = min(right_shoulders, key=lambda x: x[1])
        
        # Shoulders should be at similar heights
        if abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1] > 0.1:
            return False
        
        # Head should be lower than shoulders
        if head_val > left_shoulder[1] or head_val > right_shoulder[1]:
            return False
        
        return True
    
    def _is_ascending_triangle(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> bool:
        """Simple ascending triangle detection."""
        # Find a horizontal resistance line (tops are at similar levels)
        tops = []
        for i in range(1, len(high) - 1):
            if high[i] > high[i-1] and high[i] > high[i+1]:
                tops.append((i, high[i]))
        
        if len(tops) < 2:
            return False
        
        # Calculate resistance level as average of top values
        resistance = sum(val for _, val in tops) / len(tops)
        
        # Check if tops are roughly at the same level
        if any(abs(val - resistance) / resistance > 0.02 for _, val in tops):
            return False
        
        # Find bottoms
        bottoms = []
        for i in range(1, len(low) - 1):
            if low[i] < low[i-1] and low[i] < low[i+1]:
                bottoms.append((i, low[i]))
        
        if len(bottoms) < 2:
            return False
        
        # Check if bottoms are ascending
        bottoms.sort(key=lambda x: x[0])  # Sort by index
        is_ascending = all(bottoms[i][1] > bottoms[i-1][1] for i in range(1, len(bottoms)))
        
        return is_ascending
    
    def prepare_exit_optimization_data(self) -> List[Dict]:
        """
        Prepare data for exit optimization model training.
        
        Returns:
            List of training episodes
        """
        logger.info("Preparing exit optimization training data")
        
        # Fetch historical price data
        stock_universe = self.data_fetcher.get_universe(size=50)
        historical_data = self.data_fetcher.fetch_historical_data(symbols=stock_universe, timeframe="1d")
        
        # Check if data was fetched successfully
        if not historical_data:
            logger.warning("No historical data found, using synthetic data")
            return self._generate_synthetic_exit_data()
        
        # Process historical data to create exit optimization episodes
        training_episodes = []
        
        for symbol, df in historical_data.items():
            # Skip if not enough data
            if len(df) < 200:
                continue
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Generate training episodes
            episodes = self._generate_exit_episodes(df, symbol)
            training_episodes.extend(episodes)
        
        logger.info(f"Prepared {len(training_episodes)} exit optimization training episodes")
        return training_episodes
    
    def _generate_synthetic_exit_data(self) -> List[Dict]:
        """
        Generate synthetic data for exit optimization training.
        
        Returns:
            List of training episodes
        """
        training_episodes = []
        
        # Generate 500 synthetic episodes
        for _ in range(500):
            # Generate synthetic price path
            steps = 20
            entry_price = np.random.uniform(50, 200)
            prices = [entry_price]
            
            for _ in range(steps):
                price_change = np.random.normal(0, 0.01)
                prices.append(prices[-1] * (1 + price_change))
            
            # Generate states
            states = []
            for i in range(steps):
                current_price = prices[i]
                price_to_entry = current_price / entry_price - 1
                profit_pct = price_to_entry * 100
                
                # Generate features
                state = np.zeros(16)  # 16 features
                state[0] = profit_pct / 20
                state[1] = i / steps
                state[2] = 1.0  # Full position
                state[3] = price_to_entry
                state[4] = np.random.uniform(-0.05, 0)  # Price to high
                state[5] = np.random.uniform(0, 0.05)  # Price to low
                state[6] = np.random.uniform(0, 1)  # RSI
                state[7] = np.random.uniform(0, 1)  # BB position
                state[8] = np.random.uniform(-1, 1)  # MACD histogram
                state[9] = np.random.uniform(0, 1)  # Volatility
                state[10] = np.random.uniform(0, 1)  # ATR
                state[11] = np.random.uniform(0, 1)  # Volume ratio
                state[12] = np.random.uniform(-1, 1)  # Market trend
                state[13] = np.random.uniform(-1, 1)  # Sector trend
                state[14] = np.random.uniform(-1, 1)  # Sharpe ratio
                state[15] = np.random.uniform(0, 1)  # Max drawdown
                
                states.append(state)
            
            training_episodes.append({
                "states": states,
                "entry_price": entry_price,
                "prices": prices
            })
        
        logger.info(f"Generated {len(training_episodes)} synthetic exit optimization episodes")
        return training_episodes
    
    def _generate_exit_episodes(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """
        Generate exit optimization episodes from historical data.
        
        Args:
            df: DataFrame with price data and indicators
            symbol: Stock symbol
            
        Returns:
            List of training episodes
        """
        episodes = []
        
        # Parameters
        episode_length = 20
        min_gain = 0.01  # Minimum gain for entry
        
        # Iterate through potential entry points
        for i in range(0, len(df) - episode_length - 1, 10):  # Step by 10 for more varied entries
            # Define entry point
            entry_idx = i
            entry_price = df['close'].iloc[entry_idx]
            
            # Skip if price doesn't move enough
            max_future_price = df['high'].iloc[entry_idx:entry_idx+episode_length].max()
            min_future_price = df['low'].iloc[entry_idx:entry_idx+episode_length].min()
            
            price_range = max(max_future_price / entry_price - 1, 1 - min_future_price / entry_price)
            
            if price_range < min_gain:
                continue
            
            # Generate states for this episode
            states = []
            prices = [entry_price]
            
            for j in range(episode_length):
                current_idx = entry_idx + j
                
                if current_idx >= len(df):
                    break
                
                current_price = df['close'].iloc[current_idx]
                prices.append(current_price)
                
                # Calculate state features
                state = np.zeros(16)  # 16 features
                
                # Basic price and position features
                price_to_entry = current_price / entry_price - 1
                profit_pct = price_to_entry * 100
                
                high_since_entry = df['high'].iloc[entry_idx:current_idx+1].max()
                low_since_entry = df['low'].iloc[entry_idx:current_idx+1].min()
                
                price_to_high = current_price / high_since_entry - 1
                price_to_low = current_price / low_since_entry - 1
                
                # Technical indicators
                rsi = df['rsi'].iloc[current_idx] / 100
                
                # Get Bollinger Bands values
                bb_upper = df['bb_upper'].iloc[current_idx]
                bb_lower = df['bb_lower'].iloc[current_idx]
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
                
                macd_hist = df['macd_hist'].iloc[current_idx]
                vol_5d = df['close'].pct_change(5).iloc[current_idx:].std() * 100
                
                # Fill state vector
                state[0] = profit_pct / 20
                state[1] = j / episode_length
                state[2] = 1.0  # Full position
                state[3] = price_to_entry
                state[4] = price_to_high
                state[5] = price_to_low
                state[6] = rsi
                state[7] = bb_position
                state[8] = np.clip(macd_hist, -1, 1)
                state[9] = np.clip(vol_5d / 5, 0, 1)
                
                # Additional features (simplified here)
                state[10] = np.random.uniform(0, 1)  # ATR
                state[11] = np.random.uniform(0, 1)  # Volume ratio
                state[12] = np.random.uniform(-1, 1)  # Market trend
                state[13] = np.random.uniform(-1, 1)  # Sector trend
                state[14] = np.random.uniform(-1, 1)  # Sharpe ratio
                state[15] = np.random.uniform(0, 1)  # Max drawdown
                
                states.append(state)
            
            if states:
                episodes.append({
                    "symbol": symbol,
                    "states": states,
                    "entry_price": entry_price,
                    "prices": prices
                })
        
        return episodes
    
    def prepare_ranking_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for ranking model training.
        
        Returns:
            Tuple of (train_features, train_targets, val_features, val_targets)
        """
        logger.info("Preparing ranking model training data")
        
        # Fetch historical price data
        stock_universe = self.data_fetcher.get_universe(size=200)
        historical_data = self.data_fetcher.fetch_historical_data(symbols=stock_universe, timeframe="1d")
        
        # Check if data was fetched successfully
        if not historical_data:
            logger.warning("No historical data found, using synthetic data")
            return self._generate_synthetic_ranking_data()
        
        # Process historical data to extract features and targets
        all_features = []
        all_targets = []
        
        for symbol, df in historical_data.items():
            # Skip if not enough data
            if len(df) < 60:
                continue
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            # Extract features and targets
            features, targets = self._extract_ranking_features(df, symbol)
            
            all_features.append(features)
            all_targets.append(targets)
        
        # Combine data
        if all_features and all_targets:
            features_df = pd.concat(all_features, ignore_index=True)
            targets_series = pd.concat(all_targets, ignore_index=True)
            
            # Split into train/validation
            train_features, val_features, train_targets, val_targets = train_test_split(
                features_df, targets_series, test_size=self.val_size, random_state=self.random_state
            )
            
            logger.info(f"Prepared {len(train_features)} training and {len(val_features)} validation examples")
            return train_features, train_targets, val_features, val_targets
        else:
            logger.warning("Failed to extract features from historical data, using synthetic data")
            return self._generate_synthetic_ranking_data()
    
    def _generate_synthetic_ranking_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Generate synthetic data for ranking model training.
        
        Returns:
            Tuple of (train_features, train_targets, val_features, val_targets)
        """
        # Generate 1000 synthetic examples
        num_samples = 1000
        
        # Define feature names (same as in RankingModel)
        feature_names = [
            "return_1d", "return_5d", "return_10d", "return_20d", "return_60d",
            "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio", "close_ma50_ratio", 
            "ma5_ma20_ratio", "ma20_ma50_ratio", 
            "volatility_5d", "volatility_10d", "volatility_20d", "volatility_60d",
            "volume_1d", "volume_ma5", "volume_ma10", "volume_ma20",
            "volume_ratio_5d", "volume_ratio_10d", "volume_ratio_20d",
            "rsi_14", "rsi_divergence", "macd", "macd_signal", "macd_histogram",
            "bb_width", "bb_percent", "atr_14", "atr_percent",
            "range_breakout", "support_test", "resistance_test", "gap_up", "gap_down",
            "momentum_5d", "momentum_10d", "momentum_20d", "momentum_divergence",
            "sentiment_score", "news_volume", "social_sentiment",
            "market_return_1d", "sector_return_1d", "beta", "relative_strength"
        ]
        
        # Generate random features
        data = np.random.randn(num_samples, len(feature_names))
        features_df = pd.DataFrame(data, columns=feature_names)
        
        # Generate targets (future returns)
        # Positively correlated with momentum and sentiment, negatively with volatility
        momentum_cols = [col for col in feature_names if 'momentum' in col or 'return' in col]
        sentiment_cols = [col for col in feature_names if 'sentiment' in col]
        volatility_cols = [col for col in feature_names if 'volatility' in col]
        
        momentum_factor = features_df[momentum_cols].mean(axis=1)
        sentiment_factor = features_df[sentiment_cols].mean(axis=1)
        volatility_factor = features_df[volatility_cols].mean(axis=1)
        
        # Target is future return (binary classification: outperform/underperform)
        raw_target = 0.6 * momentum_factor + 0.3 * sentiment_factor - 0.2 * volatility_factor + np.random.randn(num_samples) * 0.5
        targets_series = pd.Series((raw_target > 0).astype(int))
        
        # Split into train/validation
        train_features, val_features, train_targets, val_targets = train_test_split(
            features_df, targets_series, test_size=self.val_size, random_state=self.random_state
        )
        
        logger.info(f"Generated {len(train_features)} training and {len(val_features)} validation synthetic examples")
        return train_features, train_targets, val_features, val_targets
    
    def _extract_ranking_features(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and targets for ranking model from price data.
        
        Args:
            df: DataFrame with price data and indicators
            symbol: Stock symbol
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        # We'll extract features at multiple points in time
        features_list = []
        targets_list = []
        
        # Use every 5 days as a new sample point
        for i in range(60, len(df) - 20, 5):
            # Current data point
            current_idx = i
            
            # Extract features
            feature_dict = {}
            
            # Returns
            feature_dict["return_1d"] = df['close'].pct_change(1).iloc[current_idx]
            feature_dict["return_5d"] = df['close'].pct_change(5).iloc[current_idx]
            feature_dict["return_10d"] = df['close'].pct_change(10).iloc[current_idx]
            feature_dict["return_20d"] = df['close'].pct_change(20).iloc[current_idx]
            feature_dict["return_60d"] = df['close'].pct_change(60).iloc[current_idx]
            
            # Moving average ratios
            feature_dict["close_ma5_ratio"] = df['close'].iloc[current_idx] / df['ma5'].iloc[current_idx] - 1
            feature_dict["close_ma10_ratio"] = df['close'].iloc[current_idx] / df['ma10'].iloc[current_idx] - 1
            feature_dict["close_ma20_ratio"] = df['close'].iloc[current_idx] / df['ma20'].iloc[current_idx] - 1
            
            if 'ma50' in df.columns:
                feature_dict["close_ma50_ratio"] = df['close'].iloc[current_idx] / df['ma50'].iloc[current_idx] - 1
            else:
                feature_dict["close_ma50_ratio"] = 0.0
                
            feature_dict["ma5_ma20_ratio"] = df['ma5'].iloc[current_idx] / df['ma20'].iloc[current_idx] - 1
            
            if 'ma50' in df.columns:
                feature_dict["ma20_ma50_ratio"] = df['ma20'].iloc[current_idx] / df['ma50'].iloc[current_idx] - 1
            else:
                feature_dict["ma20_ma50_ratio"] = 0.0
            
            # Volatility
            feature_dict["volatility_5d"] = df['close'].pct_change().rolling(5).std().iloc[current_idx]
            feature_dict["volatility_10d"] = df['close'].pct_change().rolling(10).std().iloc[current_idx]
            feature_dict["volatility_20d"] = df['close'].pct_change().rolling(20).std().iloc[current_idx]
            
            if current_idx >= 60:
                feature_dict["volatility_60d"] = df['close'].pct_change().rolling(60).std().iloc[current_idx]
            else:
                feature_dict["volatility_60d"] = 0.0
            
            # Volume
            feature_dict["volume_1d"] = df['volume'].iloc[current_idx] / df['volume'].rolling(20).mean().iloc[current_idx]
            feature_dict["volume_ma5"] = df['volume'].rolling(5).mean().iloc[current_idx] / df['volume'].rolling(20).mean().iloc[current_idx]
            feature_dict["volume_ma10"] = df['volume'].rolling(10).mean().iloc[current_idx] / df['volume'].rolling(20).mean().iloc[current_idx]
            feature_dict["volume_ma20"] = 1.0  # Normalized to itself
            
            feature_dict["volume_ratio_5d"] = df['volume'].iloc[current_idx] / df['volume'].rolling(5).mean().iloc[current_idx]
            feature_dict["volume_ratio_10d"] = df['volume'].iloc[current_idx] / df['volume'].rolling(10).mean().iloc[current_idx]
            feature_dict["volume_ratio_20d"] = df['volume'].iloc[current_idx] / df['volume'].rolling(20).mean().iloc[current_idx]
            
            # Technical indicators
            feature_dict["rsi_14"] = df['rsi'].iloc[current_idx] / 100
            
            # RSI divergence (simplified)
            rsi_5d_ago = df['rsi'].iloc[current_idx - 5] if current_idx >= 5 else 50
            price_5d_ago = df['close'].iloc[current_idx - 5] if current_idx >= 5 else df['close'].iloc[current_idx]
            price_change = df['close'].iloc[current_idx] / price_5d_ago - 1
            rsi_change = (df['rsi'].iloc[current_idx] - rsi_5d_ago) / 100
            feature_dict["rsi_divergence"] = -1 if price_change > 0 and rsi_change < 0 else (1 if price_change < 0 and rsi_change > 0 else 0)
            
            # MACD
            feature_dict["macd"] = df['macd'].iloc[current_idx]
            feature_dict["macd_signal"] = df['macd_signal'].iloc[current_idx]
            feature_dict["macd_histogram"] = df['macd_hist'].iloc[current_idx]
            
            # Bollinger Bands
            bb_width = (df['bb_upper'].iloc[current_idx] - df['bb_lower'].iloc[current_idx]) / df['bb_middle'].iloc[current_idx]
            feature_dict["bb_width"] = bb_width
            
            bb_percent = (df['close'].iloc[current_idx] - df['bb_lower'].iloc[current_idx]) / (df['bb_upper'].iloc[current_idx] - df['bb_lower'].iloc[current_idx])
            feature_dict["bb_percent"] = bb_percent
            
            # ATR (simplified)
            feature_dict["atr_14"] = (df['high'].iloc[current_idx] - df['low'].iloc[current_idx]) / df['close'].iloc[current_idx]
            feature_dict["atr_percent"] = feature_dict["atr_14"] / df['close'].iloc[current_idx]
            
            # Price patterns (simplified binary indicators)
            feature_dict["range_breakout"] = 1 if df['high'].iloc[current_idx] > df['high'].rolling(20).max().iloc[current_idx-1] else 0
            feature_dict["support_test"] = 1 if df['low'].iloc[current_idx] < df['low'].rolling(20).min().iloc[current_idx-1] * 1.02 else 0
            feature_dict["resistance_test"] = 1 if df['high'].iloc[current_idx] > df['high'].rolling(20).max().iloc[current_idx-1] * 0.98 else 0
            feature_dict["gap_up"] = 1 if df['open'].iloc[current_idx] > df['close'].iloc[current_idx-1] * 1.01 else 0
            feature_dict["gap_down"] = 1 if df['open'].iloc[current_idx] < df['close'].iloc[current_idx-1] * 0.99 else 0
            
            # Momentum
            feature_dict["momentum_5d"] = df['close'].pct_change(5).iloc[current_idx]
            feature_dict["momentum_10d"] = df['close'].pct_change(10).iloc[current_idx]
            feature_dict["momentum_20d"] = df['close'].pct_change(20).iloc[current_idx]
            
            # Momentum divergence (simplified)
            price_trend_20d = df['close'].pct_change(20).iloc[current_idx]
            price_trend_5d = df['close'].pct_change(5).iloc[current_idx]
            feature_dict["momentum_divergence"] = -1 if price_trend_20d > 0 and price_trend_5d < 0 else (1 if price_trend_20d < 0 and price_trend_5d > 0 else 0)
            
            # Sentiment (synthetic for now)
            feature_dict["sentiment_score"] = np.random.uniform(-1, 1)
            feature_dict["news_volume"] = np.random.uniform(0, 1)
            feature_dict["social_sentiment"] = np.random.uniform(-1, 1)
            
            # Market factors (synthetic for now)
            feature_dict["market_return_1d"] = np.random.normal(0, 0.01)
            feature_dict["sector_return_1d"] = np.random.normal(0, 0.015)
            feature_dict["beta"] = np.random.uniform(0.5, 1.5)
            feature_dict["relative_strength"] = np.random.uniform(-1, 1)
            
            # Target: 10-day forward return
            if current_idx + 10 < len(df):
                target = 1 if df['close'].iloc[current_idx + 10] > df['close'].iloc[current_idx] else 0
            else:
                # Skip this sample if we don't have enough future data
                continue
            
            features_list.append(feature_dict)
            targets_list.append(target)
        
        if features_list and targets_list:
            features_df = pd.DataFrame(features_list)
            targets_series = pd.Series(targets_list)
            return features_df, targets_series
        else:
            # Return empty dataframes if no features extracted
            return pd.DataFrame(), pd.Series()
    
    def train_sentiment_model(self, use_optuna: bool = True, n_trials: int = 30) -> Dict[str, Any]:
        """
        Train the sentiment analysis model with optional hyperparameter optimization.
        
        Args:
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials
            
        Returns:
            Training metrics and history
        """
        logger.info(f"Training sentiment analysis model (use_optuna={use_optuna})")
        
        # Prepare data
        train_texts, train_labels, val_texts, val_labels = self.prepare_sentiment_data()
        
        # Skip training if no data
        if not train_texts:
            logger.warning("No training data available for sentiment model")
            return {"error": "No training data available"}
        
        # Get hyperparameters
        params = {}
        if use_optuna and train_sentiment_model:
            try:
                params = self.optimize_hyperparameters("sentiment", n_trials)
                logger.info(f"Using optimized parameters: {params}")
            except Exception as e:
                logger.error(f"Error during hyperparameter optimization: {e}")
                logger.info("Falling back to default parameters")
        
        # Train the model
        if train_sentiment_model:
            # Prepare training data
            training_args = {
                "texts": train_texts,
                "labels": train_labels,
                "eval_texts": val_texts,
                "eval_labels": val_labels,
                "use_default_data": False
            }
            
            # Add optimized parameters if available
            if params:
                if "batch_size" in params:
                    training_args["batch_size"] = params["batch_size"]
                if "learning_rate" in params:
                    training_args["learning_rate"] = params["learning_rate"]
                if "max_length" in params:
                    training_args["max_length"] = params["max_length"]
            
            result = train_sentiment_model(**training_args)
            result["params"] = params
            
            logger.info("Sentiment model training completed")
            return result
        else:
            logger.warning("Sentiment model training function not available")
            return {"error": "Training function not available"}
    
    def train_pattern_recognition_model(self, use_optuna: bool = True, n_trials: int = 30) -> Dict[str, Any]:
        """
        Train the pattern recognition model with optional hyperparameter optimization.
        
        Args:
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials
            
        Returns:
            Training metrics and history
        """
        logger.info(f"Training pattern recognition model (use_optuna={use_optuna})")
        
        # Prepare data
        train_data, train_labels, val_data, val_labels = self.prepare_pattern_data()
        
        # Skip training if no data
        if not train_data:
            logger.warning("No training data available for pattern recognition model")
            return {"error": "No training data available"}
        
        # Get hyperparameters
        params = {}
        if use_optuna:
            try:
                params = self.optimize_hyperparameters("pattern", n_trials)
                logger.info(f"Using optimized parameters: {params}")
                
                # Update model architecture if parameters are available
                if "hidden_size" in params:
                    pattern_recognition_model.hidden_size = params["hidden_size"]
                if "num_layers" in params:
                    pattern_recognition_model.num_layers = params["num_layers"]
                if "dropout" in params:
                    pattern_recognition_model.dropout = params["dropout"]
            except Exception as e:
                logger.error(f"Error during hyperparameter optimization: {e}")
                logger.info("Falling back to default parameters")
        
        # Train the model
        history = pattern_recognition_model.train(
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            batch_size=params.get("batch_size", self.batch_size),
            learning_rate=params.get("learning_rate", self.learning_rate),
            num_epochs=self.num_epochs
        )
        
        logger.info("Pattern recognition model training completed")
        return {"history": history, "metrics": pattern_recognition_model.metrics, "params": params}
    
    def train_exit_optimization_model(self, use_optuna: bool = True, n_trials: int = 30) -> Dict[str, Any]:
        """
        Train the exit optimization model with optional hyperparameter optimization.
        
        Args:
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials
            
        Returns:
            Training metrics and history
        """
        logger.info(f"Training exit optimization model (use_optuna={use_optuna})")
        
        # Prepare data
        training_episodes = self.prepare_exit_optimization_data()
        
        # Skip training if no data
        if not training_episodes:
            logger.warning("No training data available for exit optimization model")
            return {"error": "No training data available"}
        
        # Get hyperparameters
        params = {}
        if use_optuna:
            try:
                params = self.optimize_hyperparameters("exit", n_trials)
                logger.info(f"Using optimized parameters: {params}")
                
                # Update model architecture if parameters are available
                if "hidden_size" in params:
                    exit_optimization_model.hidden_size = params["hidden_size"]
                if "num_layers" in params:
                    exit_optimization_model.num_layers = params["num_layers"]
                if "gamma" in params:
                    exit_optimization_model.gamma = params["gamma"]
            except Exception as e:
                logger.error(f"Error during hyperparameter optimization: {e}")
                logger.info("Falling back to default parameters")
        
        # Train the model
        # Import the train_exit_model function from exit_optimization
        from src.models.exit_optimization import train_exit_model
        
        metrics, model = train_exit_model(
            training_data=training_episodes,
            epochs=self.num_epochs,
            batch_size=params.get("batch_size", 64),
            learning_rate=params.get("learning_rate", self.learning_rate)
        )
        
        logger.info("Exit optimization model training completed")
        return {"metrics": metrics, "model": "exit_optimization_model", "params": params}
    
    def optimize_hyperparameters(self, model_name: str, n_trials: int = 30) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            model_name: Name of model to optimize
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters
        """
        logger.info(f"Optimizing hyperparameters for {model_name} model with {n_trials} trials")
        
        # Create Optuna study
        study = optuna.create_study(
            study_name=f"{model_name}_optimization",
            direction="maximize",
            storage=self.optuna_storage,
            load_if_exists=True
        )
        
        # Define objective function based on model type
        if model_name == "ranking":
            objective = self._create_ranking_objective()
        elif model_name == "pattern":
            objective = self._create_pattern_objective()
        elif model_name == "exit":
            objective = self._create_exit_objective()
        elif model_name == "sentiment":
            objective = self._create_sentiment_objective()
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
        
        # Log best parameters
        logger.info(f"Best parameters for {model_name}: {study.best_params}")
        
        # Return best parameters
        return study.best_params
    
    def _create_ranking_objective(self) -> Callable[[optuna.Trial], float]:
        """Create objective function for ranking model."""
        def objective(trial: optuna.Trial) -> float:
            # Define hyperparameters to optimize
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "hidden_dims": [
                    trial.suggest_int("hidden_dim_1", 32, 256),
                    trial.suggest_int("hidden_dim_2", 16, 128),
                    trial.suggest_int("hidden_dim_3", 8, 64)
                ],
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128])
            }
            
            # Prepare data
            train_features, train_targets, val_features, val_targets = self.prepare_ranking_data()
            
            # Skip if no data
            if train_features.empty:
                return 0.0
            
            # Train model with these parameters
            history = ranking_model.train(
                train_features=train_features,
                train_targets=train_targets,
                val_features=val_features,
                val_targets=val_targets,
                epochs=self.num_epochs,
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"]
            )
            
            # Return validation metric
            val_auc = history.get("val_auc", [0.0])[-1]  # Use final validation AUC
            return val_auc
        
        return objective
    
    def _create_pattern_objective(self) -> Callable[[optuna.Trial], float]:
        """Create objective function for pattern recognition model."""
        def objective(trial: optuna.Trial) -> float:
            # Define hyperparameters to optimize
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64])
            }
            
            # Prepare data
            train_data, train_labels, val_data, val_labels = self.prepare_pattern_data()
            
            # Skip if no data
            if not train_data:
                return 0.0
            
            # Set model parameters
            pattern_recognition_model.hidden_size = params["hidden_size"]
            pattern_recognition_model.num_layers = params["num_layers"]
            pattern_recognition_model.dropout = params["dropout"]
            
            # Train model with these parameters
            metrics = pattern_recognition_model.train(
                train_data=train_data,
                train_labels=train_labels,
                val_data=val_data,
                val_labels=val_labels,
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                epochs=self.num_epochs
            )
            
            # Return validation metric
            val_f1 = metrics.get("val_f1", 0.0)
            return val_f1
        
        return objective
    
    def _create_exit_objective(self) -> Callable[[optuna.Trial], float]:
        """Create objective function for exit optimization model."""
        def objective(trial: optuna.Trial) -> float:
            # Define hyperparameters to optimize
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                "hidden_size": trial.suggest_int("hidden_size", 32, 256),
                "num_layers": trial.suggest_int("num_layers", 1, 3),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64])
            }
            
            # Prepare data
            episodes = self.prepare_exit_data()
            
            # Skip if no data
            if not episodes:
                return 0.0
            
            # Set model parameters
            exit_optimization_model.hidden_size = params["hidden_size"]
            exit_optimization_model.num_layers = params["num_layers"]
            exit_optimization_model.gamma = params["gamma"]
            
            # Train model with these parameters
            metrics = exit_optimization_model.train(
                episodes=episodes,
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                epochs=self.num_epochs
            )
            
            # Return validation metric
            avg_reward = metrics.get("avg_reward", 0.0)
            return avg_reward
        
        return objective
    
    def _create_sentiment_objective(self) -> Callable[[optuna.Trial], float]:
        """Create objective function for sentiment model."""
        def objective(trial: optuna.Trial) -> float:
            # Define hyperparameters to optimize
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                "max_length": trial.suggest_categorical("max_length", [128, 256, 512])
            }
            
            # Prepare data
            train_texts, train_labels, val_texts, val_labels = self.prepare_sentiment_data()
            
            # Skip if no validation data
            if not val_texts or not val_labels:
                return 0.0
            
            # Train model with these parameters
            if train_sentiment_model:
                metrics = train_sentiment_model(
                    train_texts=train_texts,
                    train_labels=train_labels,
                    val_texts=val_texts,
                    val_labels=val_labels,
                    learning_rate=params["learning_rate"],
                    batch_size=params["batch_size"],
                    max_length=params["max_length"],
                    epochs=self.num_epochs
                )
                
                # Return validation metric
                val_accuracy = metrics.get("val_accuracy", 0.0)
                return val_accuracy
            else:
                return 0.0
        
        return objective
    
    def train_ranking_model(self, use_optuna: bool = True, n_trials: int = 30) -> Dict[str, Any]:
        """
        Train the ranking model with optional hyperparameter optimization.
        
        Args:
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials
            
        Returns:
            Training metrics and history
        """
        logger.info(f"Training ranking model (use_optuna={use_optuna})")
        
        # Prepare data
        train_features, train_targets, val_features, val_targets = self.prepare_ranking_data()
        
        # Skip training if no data
        if train_features.empty:
            logger.warning("No training data available for ranking model")
            return {"error": "No training data available"}
        
        # Get hyperparameters
        params = {}
        if use_optuna:
            try:
                params = self.optimize_hyperparameters("ranking", n_trials)
                logger.info(f"Using optimized parameters: {params}")
                
                # Extract hidden_dims from params
                if "hidden_dim_1" in params and "hidden_dim_2" in params and "hidden_dim_3" in params:
                    params["hidden_dims"] = [
                        params.pop("hidden_dim_1"),
                        params.pop("hidden_dim_2"),
                        params.pop("hidden_dim_3")
                    ]
            except Exception as e:
                logger.error(f"Error during hyperparameter optimization: {e}")
                logger.info("Falling back to default parameters")
        
        # Train the model
        history = ranking_model.train(
            train_features=train_features,
            train_targets=train_targets,
            val_features=val_features,
            val_targets=val_targets,
            epochs=self.num_epochs,
            batch_size=params.get("batch_size", self.batch_size),
            learning_rate=params.get("learning_rate", self.learning_rate),
            weight_decay=params.get("weight_decay", 1e-5)
        )
        
        logger.info("Ranking model training completed")
        return {"history": history, "metrics": ranking_model.metrics, "params": params}
    
    def train_all_models(self, parallel: bool = False, use_optuna: bool = True, n_trials: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Train all models.
        
        Args:
            parallel: Whether to train models in parallel
            use_optuna: Whether to use Optuna for hyperparameter optimization
            n_trials: Number of Optuna trials
            
        Returns:
            Dictionary of training results for each model
        """
        logger.info(f"Training all models (parallel={parallel}, use_optuna={use_optuna}, n_trials={n_trials})")
        
        results = {}
        
        if parallel:
            # Train models in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all training tasks
                sentiment_future = executor.submit(self.train_sentiment_model, use_optuna, n_trials)
                pattern_future = executor.submit(self.train_pattern_recognition_model, use_optuna, n_trials)
                exit_future = executor.submit(self.train_exit_optimization_model, use_optuna, n_trials)
                ranking_future = executor.submit(self.train_ranking_model, use_optuna, n_trials)
                
                # Get results
                results["sentiment"] = sentiment_future.result()
                results["pattern"] = pattern_future.result()
                results["exit"] = exit_future.result()
                results["ranking"] = ranking_future.result()
        else:
            # Train models sequentially
            results["sentiment"] = self.train_sentiment_model(use_optuna, n_trials)
            results["pattern"] = self.train_pattern_recognition_model(use_optuna, n_trials)
            results["exit"] = self.train_exit_optimization_model(use_optuna, n_trials)
            results["ranking"] = self.train_ranking_model(use_optuna, n_trials)
        
        logger.info("All model training completed")
        return results
    
    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all models on validation data.
        
        Returns:
            Dictionary of evaluation metrics for each model
        """
        logger.info("Evaluating all models")
        
        metrics = {}
        
        # Evaluate sentiment model
        try:
            # Get validation data
            train_texts, train_labels, val_texts, val_labels = self.prepare_sentiment_data()
            
            if val_texts and val_labels:
                # Use validation data for evaluation
                predictions = []
                for text in val_texts:
                    result = sentiment_model.analyze_sentiment(text)
                    sentiment_scores = result.get("sentiment", {})
                    probs = [
                        sentiment_scores.get("negative", 0),
                        sentiment_scores.get("neutral", 0),
                        sentiment_scores.get("positive", 0)
                    ]
                    predictions.append(np.argmax(probs))
                
                # Calculate metrics
                accuracy = accuracy_score(val_labels, predictions)
                precision = precision_score(val_labels, predictions, average="weighted", zero_division=0)
                recall = recall_score(val_labels, predictions, average="weighted", zero_division=0)
                f1 = f1_score(val_labels, predictions, average="weighted", zero_division=0)
                
                metrics["sentiment"] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
            else:
                metrics["sentiment"] = sentiment_model.metrics
        except Exception as e:
            logger.error(f"Error evaluating sentiment model: {e}")
            metrics["sentiment"] = {"error": str(e)}
        
        # Evaluate pattern recognition model
        try:
            # Use metrics from training
            metrics["pattern"] = pattern_recognition_model.metrics
        except Exception as e:
            logger.error(f"Error evaluating pattern recognition model: {e}")
            metrics["pattern"] = {"error": str(e)}
        
        # Evaluate exit optimization model
        try:
            # Use metrics from training
            metrics["exit"] = exit_optimization_model.performance_metrics
        except Exception as e:
            logger.error(f"Error evaluating exit optimization model: {e}")
            metrics["exit"] = {"error": str(e)}
        
        # Evaluate ranking model
        try:
            # Use metrics from training
            metrics["ranking"] = ranking_model.metrics
        except Exception as e:
            logger.error(f"Error evaluating ranking model: {e}")
            metrics["ranking"] = {"error": str(e)}
        
        logger.info("Model evaluation completed")
        return metrics


async def run_scheduled_training():
    """Run scheduled model training tasks."""
    logger.info("Starting scheduled model training")
    
    # Check if training should run now
    current_hour = datetime.now().hour
    is_weekend = datetime.now().weekday() >= 5
    
    # Run full training on weekends during off-hours
    if is_weekend and (current_hour < 9 or current_hour > 16):
        logger.info("Running full model training (weekend schedule)")
        trainer = ModelTrainer()
        
        # Train all models
        results = trainer.train_all_models(parallel=True)
        
        # Store results in Redis for monitoring
        if settings.use_redis_cache:
            redis_client.set("training:last_results", json.dumps(results), ex=86400)
            redis_client.set("training:last_full_run", datetime.now().isoformat(), ex=86400 * 7)
    
    # Run incremental training on weekdays during off-hours
    elif current_hour < 7 or current_hour > 20:
        logger.info("Running incremental model training (weekday schedule)")
        trainer = ModelTrainer()
        
        # Train only selected models with reduced epochs
        trainer.num_epochs = 10
        
        # Determine which model to train based on day of week
        day_of_week = datetime.now().weekday()
        
        if day_of_week == 0:  # Monday
            logger.info("Training sentiment model (incremental)")
            result = trainer.train_sentiment_model()
        elif day_of_week == 1:  # Tuesday
            logger.info("Training pattern recognition model (incremental)")
            result = trainer.train_pattern_recognition_model()
        elif day_of_week == 2:  # Wednesday
            logger.info("Training exit optimization model (incremental)")
            result = trainer.train_exit_optimization_model()
        elif day_of_week == 3:  # Thursday
            logger.info("Training ranking model (incremental)")
            result = trainer.train_ranking_model()
        else:  # Friday
            logger.info("Evaluating all models")
            result = trainer.evaluate_models()
        
        # Store results in Redis for monitoring
        if settings.use_redis_cache:
            redis_client.set("training:last_incremental_run", datetime.now().isoformat(), ex=86400 * 7)
            redis_client.set("training:last_incremental_result", json.dumps(result), ex=86400)
    
    else:
        logger.info("Not in training window, skipping scheduled training")


def schedule_training():
    """Schedule regular model training."""
    logger.info("Setting up training schedule")
    
    # Schedule full training on weekends
    schedule.every().saturday.at("02:00").do(lambda: asyncio.run(run_scheduled_training()))
    schedule.every().sunday.at("02:00").do(lambda: asyncio.run(run_scheduled_training()))
    
    # Schedule incremental training on weekdays
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        schedule.every().__getattribute__(day).at("03:00").do(lambda: asyncio.run(run_scheduled_training()))
    
    # Run scheduler in a loop
    while True:
        schedule.run_pending()
        time.sleep(60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train trading models")
    
    parser.add_argument("--model", type=str, choices=["sentiment", "pattern", "exit", "ranking", "all"],
                        default="all", help="Model to train")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for training")
    
    parser.add_argument("--parallel", action="store_true",
                        help="Train models in parallel (only for 'all')")
    
    parser.add_argument("--schedule", action="store_true",
                        help="Run the training scheduler")
    
    return parser.parse_args()


def main():
    """Main function for model training."""
    args = parse_args()
    
    # If schedule mode, run the scheduler
    if args.schedule:
        schedule_training()
        return
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Update training parameters
    trainer.num_epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.learning_rate
    
    # Train selected model
    if args.model == "sentiment":
        result = trainer.train_sentiment_model()
    elif args.model == "pattern":
        result = trainer.train_pattern_recognition_model()
    elif args.model == "exit":
        result = trainer.train_exit_optimization_model()
    elif args.model == "ranking":
        result = trainer.train_ranking_model()
    elif args.model == "all":
        result = trainer.train_all_models(parallel=args.parallel)
    
    # Print results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
