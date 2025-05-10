"""
Multi-factor ranking model for stock screening and selection.

This model ranks stocks based on technical, volume, and momentum features
to identify the best trading opportunities.
"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("ranking_model")

class RankingModel:
    """
    Multi-factor ranking model for stock selection.
    
    Uses a combination of XGBoost and feature engineering to rank stocks
    based on their probability of making profitable intraday moves.
    """
    
    def __init__(self):
        """Initialize the ranking model."""
        self.model = None
        self.feature_names = []
        self.model_path = settings.model.ranking_model_path
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.last_trained = None
        
        # Default hyperparameters for XGBoost model
        self.hyperparams = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'n_estimators': 100,
            'tree_method': 'gpu_hist',  # Use GPU acceleration
            'predictor': 'gpu_predictor',  # Use GPU for prediction
            'random_state': 42
        }
        
        # Load model if it exists
        self._load_model()
        
    def _load_model(self) -> bool:
        """
        Load model from disk if it exists.
        
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        try:
            model_path = Path(self.model_path)
            if model_path.exists():
                try:
                    # Load model with version compatibility mode
                    self.model = xgb.Booster()
                    self.model.load_model(str(model_path))
                    
                    # Load model metadata if it exists
                    metadata_path = model_path.with_suffix('.json')
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        self.feature_names = metadata.get('feature_names', [])
                        self.feature_importance = metadata.get('feature_importance', {})
                        self.last_trained = metadata.get('last_trained')
                    
                    logger.info(f"Model loaded from {model_path}")
                    return True
                except xgb.core.XGBoostError as xgb_error:
                    # Handle XGBoost specific errors
                    logger.error(f"XGBoost error loading model: {xgb_error}")
                    logger.warning("Model format incompatible with current XGBoost version. Using fallback ranking method.")
                    
                    # Backup the incompatible model file
                    backup_path = str(model_path) + ".backup"
                    import shutil
                    shutil.copy2(str(model_path), backup_path)
                    logger.info(f"Backed up incompatible model to {backup_path}")
                    
                    # Set model to None to use simple ranking
                    self.model = None
                    return False
            else:
                logger.warning(f"Model file not found at {model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Set model to None to use simple ranking
            self.model = None
            return False
        
    def save_model(self) -> bool:
        """
        Save model to disk.
        
        Returns:
            bool: True if model was saved successfully, False otherwise
        """
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False
            
            # Create directory if it doesn't exist
            model_dir = os.path.dirname(self.model_path)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            self.model.save_model(self.model_path)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'last_trained': datetime.now().isoformat(),
                'hyperparams': self.hyperparams
            }
            
            metadata_path = Path(self.model_path).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @log_execution_time(logger)
    def train(self, training_data: pd.DataFrame, target_col: str = 'target') -> bool:
        """
        Train the ranking model.
        
        Args:
            training_data: DataFrame with features and target
            target_col: Name of the target column
            
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            if training_data.empty:
                logger.error("Empty training data")
                return False
            
            # Separate features and target
            if target_col not in training_data.columns:
                logger.error(f"Target column '{target_col}' not found in training data")
                return False
            
            X = training_data.drop(columns=[target_col])
            y = training_data[target_col]
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            
            # Train model
            self.model = xgb.train(
                self.hyperparams,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=10,
                verbose_eval=10
            )
            
            # Get feature importance
            importance = self.model.get_score(importance_type='gain')
            total = sum(importance.values())
            self.feature_importance = {k: v/total for k, v in importance.items()}
            
            # Save model
            self.last_trained = datetime.now().isoformat()
            self.save_model()
            
            # Log training results
            y_pred = self.model.predict(dval)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(y_val, y_pred_binary)
            precision = precision_score(y_val, y_pred_binary, zero_division=0)
            recall = recall_score(y_val, y_pred_binary, zero_division=0)
            f1 = f1_score(y_val, y_pred_binary, zero_division=0)
            
            logger.info(f"Model training completed with:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    @log_execution_time(logger)
    def rank_stocks(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Rank stocks based on the model predictions.
        
        Args:
            stock_data: Dictionary of stock DataFrames with features
            
        Returns:
            List of ranked stocks with scores
        """
        try:
            if not stock_data:
                logger.warning("Empty stock data for ranking")
                return []
            
            # If model is not trained, use simple ranking formula
            if self.model is None:
                logger.warning("Model not trained, using simple ranking formula")
                return self._simple_ranking(stock_data)
            
            # Process each stock
            ranked_stocks = []
            
            for symbol, df in stock_data.items():
                try:
                    # Extract features
                    features = self._extract_features(df)
                    
                    if features is None:
                        continue
                    
                    # Scale features
                    features_array = np.array([list(features.values())])
                    features_scaled = self.scaler.transform(features_array)
                    
                    # Create DMatrix
                    dfeatures = xgb.DMatrix(features_scaled, feature_names=list(features.keys()))
                    
                    # Get prediction
                    score = float(self.model.predict(dfeatures)[0])
                    
                    # Get price information
                    current_price = df['close'].iloc[-1] if 'close' in df.columns else 0
                    
                    # Add to ranked list
                    ranked_stocks.append({
                        'symbol': symbol,
                        'score': score,
                        'price': current_price,
                        'features': features,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error ranking stock {symbol}: {e}")
                    continue
            
            # Sort by score (descending)
            ranked_stocks.sort(key=lambda x: x['score'], reverse=True)
            
            return ranked_stocks
        except Exception as e:
            logger.error(f"Error ranking stocks: {e}")
            return []
    
    def _simple_ranking(self, stock_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Simple ranking method when no model is available.
        
        Args:
            stock_data: Dictionary of stock DataFrames with features
            
        Returns:
            List of ranked stocks with scores
        """
        try:
            ranked_stocks = []
            
            for symbol, df in stock_data.items():
                try:
                    # Ensure minimum data requirements
                    if len(df) < 20:
                        continue
                    
                    # Extract base metrics
                    if 'close' not in df.columns or 'volume' not in df.columns:
                        continue
                    
                    current_price = df['close'].iloc[-1]
                    previous_price = df['close'].iloc[-2]
                    price_change = current_price - previous_price
                    price_change_pct = (price_change / previous_price) * 100
                    
                    # Calculate daily range
                    if 'high' in df.columns and 'low' in df.columns:
                        daily_high = df['high'].iloc[-1]
                        daily_low = df['low'].iloc[-1]
                        daily_range_pct = ((daily_high - daily_low) / daily_low) * 100
                    else:
                        daily_range_pct = 0
                    
                    # Volume metrics
                    current_volume = df['volume'].iloc[-1]
                    avg_volume = df['volume'].iloc[-20:].mean()
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
                    
                    # Calculate volatility
                    volatility = df['close'].pct_change().iloc[-20:].std() * 100
                    
                    # Simple scoring formula
                    # Higher weight to price momentum and volume spike
                    momentum_score = abs(price_change_pct) * 0.4
                    volatility_score = volatility * 0.2
                    volume_score = min(volume_ratio, 5) * 0.3
                    range_score = daily_range_pct * 0.1
                    
                    # Combine scores
                    score = momentum_score + volatility_score + volume_score + range_score
                    
                    # Direction adjustment - prefer upward movement
                    if price_change_pct > 0:
                        score *= 1.2
                    
                    # Extract features for transparency
                    features = {
                        'price_change_pct': float(price_change_pct),
                        'daily_range_pct': float(daily_range_pct),
                        'volume_ratio': float(volume_ratio),
                        'volatility': float(volatility)
                    }
                    
                    # Add to ranked list
                    ranked_stocks.append({
                        'symbol': symbol,
                        'score': float(score),
                        'price': float(current_price),
                        'features': features,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"Error in simple ranking for stock {symbol}: {e}")
                    continue
            
            # Sort by score (descending)
            ranked_stocks.sort(key=lambda x: x['score'], reverse=True)
            
            return ranked_stocks
        except Exception as e:
            logger.error(f"Error in simple ranking: {e}")
            return []
    
    def _extract_features(self, df: pd.DataFrame) -> Optional[Dict[str, float]]:
        """
        Extract features from stock data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of features
        """
        try:
            # Check minimum data requirements
            if df.empty or len(df) < 20:
                return None
            
            features = {}
            
            # Close price features
            if 'close' in df.columns:
                # Calculate price changes
                df['returns'] = df['close'].pct_change()
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                
                # Short-term momentum
                features['return_1d'] = df['returns'].iloc[-1]
                features['return_5d'] = df['close'].iloc[-1] / df['close'].iloc[-6] - 1 if len(df) >= 6 else 0
                features['return_10d'] = df['close'].iloc[-1] / df['close'].iloc[-11] - 1 if len(df) >= 11 else 0
                
                # Moving averages
                df['ma5'] = df['close'].rolling(5).mean()
                df['ma10'] = df['close'].rolling(10).mean()
                df['ma20'] = df['close'].rolling(20).mean()
                
                # MA ratios
                features['close_ma5_ratio'] = df['close'].iloc[-1] / df['ma5'].iloc[-1] if df['ma5'].iloc[-1] > 0 else 0
                features['close_ma10_ratio'] = df['close'].iloc[-1] / df['ma10'].iloc[-1] if df['ma10'].iloc[-1] > 0 else 0
                features['close_ma20_ratio'] = df['close'].iloc[-1] / df['ma20'].iloc[-1] if df['ma20'].iloc[-1] > 0 else 0
                features['ma5_ma20_ratio'] = df['ma5'].iloc[-1] / df['ma20'].iloc[-1] if df['ma20'].iloc[-1] > 0 else 0
                
                # Volatility
                features['volatility_5d'] = df['returns'].iloc[-5:].std() if len(df) >= 5 else 0
                features['volatility_10d'] = df['returns'].iloc[-10:].std() if len(df) >= 10 else 0
                features['volatility_20d'] = df['returns'].iloc[-20:].std() if len(df) >= 20 else 0
            
            # Volume features
            if 'volume' in df.columns:
                # Volume ratios
                features['volume_1d'] = df['volume'].iloc[-1]
                features['volume_ma5'] = df['volume'].iloc[-5:].mean() if len(df) >= 5 else df['volume'].iloc[-1]
                features['volume_ma10'] = df['volume'].iloc[-10:].mean() if len(df) >= 10 else features['volume_ma5']
                
                features['volume_ratio_5d'] = df['volume'].iloc[-1] / features['volume_ma5'] if features['volume_ma5'] > 0 else 0
                features['volume_ratio_10d'] = df['volume'].iloc[-1] / features['volume_ma10'] if features['volume_ma10'] > 0 else 0
                
                # Volume trend
                volume_trend = df['volume'].iloc[-5:].pct_change().mean() if len(df) >= 5 else 0
                features['volume_trend_5d'] = volume_trend
            
            # Range and candle features
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Daily range
                df['daily_range'] = (df['high'] - df['low']) / df['low'] * 100
                features['daily_range'] = df['daily_range'].iloc[-1]
                features['daily_range_avg_5d'] = df['daily_range'].iloc[-5:].mean() if len(df) >= 5 else features['daily_range']
                
                # Candle body size
                df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
                features['body_size'] = df['body_size'].iloc[-1]
                features['body_size_avg_5d'] = df['body_size'].iloc[-5:].mean() if len(df) >= 5 else features['body_size']
                
                # Upper and lower shadows
                df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df[['open', 'close']].max(axis=1) * 100
                df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df[['open', 'close']].min(axis=1) * 100
                
                features['upper_shadow'] = df['upper_shadow'].iloc[-1]
                features['lower_shadow'] = df['lower_shadow'].iloc[-1]
                
                # Bullish/bearish
                features['is_bullish'] = 1.0 if df['close'].iloc[-1] > df['open'].iloc[-1] else 0.0
                
                # Gap
                if len(df) > 1:
                    prev_close = df['close'].iloc[-2]
                    curr_open = df['open'].iloc[-1]
                    features['gap_pct'] = (curr_open - prev_close) / prev_close * 100
                else:
                    features['gap_pct'] = 0.0
            
            # Technical indicators
            if 'close' in df.columns:
                # RSI (14)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                features['rsi'] = df['rsi'].iloc[-1] if not pd.isna(df['rsi'].iloc[-1]) else 50.0
                
                # MACD
                if len(df) >= 26:
                    ema12 = df['close'].ewm(span=12, adjust=False).mean()
                    ema26 = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = ema12 - ema26
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
                    
                    features['macd'] = df['macd'].iloc[-1]
                    features['macd_signal'] = df['macd_signal'].iloc[-1]
                    features['macd_hist'] = df['macd_hist'].iloc[-1]
                
                # Bollinger Bands
                if len(df) >= 20:
                    df['bb_middle'] = df['close'].rolling(20).mean()
                    df['bb_std'] = df['close'].rolling(20).std()
                    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
                    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
                    
                    # Percent B indicator (position within the bands)
                    df['percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                    
                    features['bb_width'] = (df['bb_upper'].iloc[-1] - df['bb_lower'].iloc[-1]) / df['bb_middle'].iloc[-1]
                    features['percent_b'] = df['percent_b'].iloc[-1]
            
            # Remove any NaN values
            features = {k: float(v) if not pd.isna(v) else 0.0 for k, v in features.items()}
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def generate_training_data(
        self, 
        historical_data: Dict[str, pd.DataFrame], 
        target_threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        Generate training data from historical data.
        
        Args:
            historical_data: Dictionary of stock DataFrames with OHLCV data
            target_threshold: Percentage threshold for target classification
            
        Returns:
            DataFrame with features and target
        """
        try:
            all_features = []
            
            for symbol, df in historical_data.items():
                try:
                    # Minimum data check
                    if len(df) < 20:
                        continue
                    
                    # Generate features for each day
                    for i in range(20, len(df) - 1):
                        # Extract features using data up to day i
                        day_df = df.iloc[:i+1]
                        features = self._extract_features(day_df)
                        
                        if features is None:
                            continue
                        
                        # Calculate target (next day performance)
                        next_day = df.iloc[i+1]
                        if 'close' in df.columns and 'open' in df.columns:
                            entry_price = day_df['close'].iloc[-1]
                            max_price = next_day['high'] if 'high' in df.columns else next_day['close']
                            
                            # Target: 1 if next day high is at least target_threshold% above entry
                            max_return = (max_price - entry_price) / entry_price * 100
                            target = 1 if max_return >= target_threshold else 0
                            
                            # Add target to features
                            features['target'] = target
                            features['symbol'] = symbol
                            features['date'] = day_df.index[-1].strftime('%Y-%m-%d') if isinstance(day_df.index[-1], pd.Timestamp) else str(day_df.index[-1])
                            
                            all_features.append(features)
                except Exception as e:
                    logger.error(f"Error generating training data for {symbol}: {e}")
                    continue
            
            if not all_features:
                logger.warning("No training data generated")
                return pd.DataFrame()
            
            # Convert to DataFrame
            training_df = pd.DataFrame(all_features)
            
            # Remove non-feature columns for model training
            meta_columns = ['symbol', 'date', 'target']
            feature_columns = [col for col in training_df.columns if col not in meta_columns]
            
            # Balance classes (optional)
            logger.info(f"Training data class distribution: {training_df['target'].value_counts().to_dict()}")
            
            # Log training data info
            logger.info(f"Generated {len(training_df)} training samples with {len(feature_columns)} features")
            
            return training_df
        except Exception as e:
            logger.error(f"Error generating training data: {e}")
            return pd.DataFrame()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        Returns:
            Dictionary of feature importance values
        """
        return self.feature_importance
    
    def predict_single_stock(self, features: Dict[str, float]) -> float:
        """
        Make prediction for a single stock.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Prediction score
        """
        try:
            if self.model is None:
                logger.warning("Model not trained, cannot predict")
                return 0.0
            
            # Prepare features
            feature_array = []
            for feature in self.feature_names:
                feature_array.append(features.get(feature, 0.0))
            
            # Scale features
            features_array = np.array([feature_array])
            features_scaled = self.scaler.transform(features_array)
            
            # Create DMatrix
            dfeatures = xgb.DMatrix(features_scaled, feature_names=self.feature_names)
            
            # Get prediction
            score = float(self.model.predict(dfeatures)[0])
            
            return score
        except Exception as e:
            logger.error(f"Error predicting single stock: {e}")
            return 0.0

# Create global ranking model instance
ranking_model = RankingModel()

# Function to run training in background
async def train_model_from_history(days: int = 60, target_threshold: float = 1.5):
    """
    Train the ranking model using historical data.
    
    Args:
        days: Number of days of historical data to use
        target_threshold: Percentage threshold for target classification
    """
    try:
        logger.info(f"Starting model training with {days} days of historical data")
        
        # Get watchlist symbols or top stocks
        symbols = redis_client.get_watchlist()
        
        if not symbols:
            logger.warning("No symbols in watchlist for training")
            return False
        
        logger.info(f"Collecting historical data for {len(symbols)} symbols")
        
        # Placeholder for collecting historical data
        # In a real implementation, this would fetch data from Polygon/Alpha Vantage/etc.
        historical_data = {}
        
        # For demonstration, we'll generate some synthetic data
        for symbol in symbols:
            # Check if historical data exists in Redis
            redis_key = f"stocks:history:{symbol}:60d:1d"
            df = redis_client.get(redis_key)
            
            if df is not None and not df.empty:
                historical_data[symbol] = df
        
        if not historical_data:
            logger.warning("No historical data found for training")
            return False
        
        logger.info(f"Collected historical data for {len(historical_data)} symbols")
        
        # Generate training data
        training_data = ranking_model.generate_training_data(
            historical_data, 
            target_threshold=target_threshold
        )
        
        if training_data.empty:
            logger.warning("Generated training data is empty")
            return False
        
        # Train model
        success = ranking_model.train(training_data, target_col='target')
        
        if success:
            logger.info("Model training completed successfully")
            
            # Update Redis with model metadata
            metadata = {
                'last_trained': ranking_model.last_trained,
                'feature_count': len(ranking_model.feature_names),
                'feature_importance': ranking_model.feature_importance
            }
            redis_client.set("model:ranking:metadata", metadata)
            
            return True
        else:
            logger.error("Model training failed")
            return False
    except Exception as e:
        logger.error(f"Error in train_model_from_history: {e}")
        return False
