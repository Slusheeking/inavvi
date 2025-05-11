"""
Multi-factor ranking model for stock screening and selection.

This model ranks stocks based on technical, volume, and momentum features
using an ensemble of machine learning models with deep learning components.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("ranking_model")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


class FactorDataset(Dataset):
    """Dataset for financial factor data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        """
        Initialize factor dataset.
        
        Args:
            features: Feature matrix
            targets: Target values
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class DeepFactorModel(nn.Module):
    """Deep neural network for factor modeling."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32], output_dim: int = 1, dropout: float = 0.3):
        """
        Initialize deep factor model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super(DeepFactorModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout)]
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Sequential model
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RankingModel:
    """
    Multi-factor ranking model for stock selection.

    Uses a deep learning model to rank stocks based on technical, fundamental, and
    alternative data factors.
    """
    
    def __init__(self, factor_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the ranking model.
        
        Args:
            factor_weights: Optional dictionary of factor weights
        """
        self.models = {}
        self.feature_names = [
            # Price and return factors
            "return_1d", "return_5d", "return_10d", "return_20d", "return_60d",
            "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio", "close_ma50_ratio", 
            "ma5_ma20_ratio", "ma20_ma50_ratio", 
            
            # Volatility factors
            "volatility_5d", "volatility_10d", "volatility_20d", "volatility_60d",
            
            # Volume factors
            "volume_1d", "volume_ma5", "volume_ma10", "volume_ma20",
            "volume_ratio_5d", "volume_ratio_10d", "volume_ratio_20d",
            
            # Technical indicators
            "rsi_14", "rsi_divergence", "macd", "macd_signal", "macd_histogram",
            "bb_width", "bb_percent", "atr_14", "atr_percent",
            
            # Price patterns
            "range_breakout", "support_test", "resistance_test", "gap_up", "gap_down",
            
            # Momentum factors
            "momentum_5d", "momentum_10d", "momentum_20d", "momentum_divergence",
            
            # Sentiment and news
            "sentiment_score", "news_volume", "social_sentiment",
            
            # Market factors
            "market_return_1d", "sector_return_1d", "beta", "relative_strength"
        ]
        
        # Model architecture parameters
        self.input_dim = len(self.feature_names)
        self.hidden_dims = [128, 64, 32]
        self.output_dim = 1
        
        # Initialize deep factor model
        self.deep_model = DeepFactorModel(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim
        ).to(device)
        
        # Preprocessing
        self.scaler = StandardScaler()
        
        # Factor weights (can be updated during training)
        self.factor_weights = factor_weights or {
            "momentum": 0.3,
            "volume": 0.2,
            "volatility": 0.2,
            "trend": 0.2,
            "value": 0.1,
        }
        
        # Performance metrics
        self.metrics = {
            "auc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "information_ratio": 0.0
        }
        
        # Model directory and path
        self.model_dir = os.path.join(settings.models_dir, "ranking")
        self.model_path = os.path.join(self.model_dir, "model.pt")
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Feature importance tracking
        self.feature_importance = {}
        self.feature_correlations = {}
        
        # Load model if available
        self._load_model()
        
    def _load_model(self) -> bool:
        """
        Load model from disk if it exists.
        
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            ensemble_path = os.path.join(self.model_dir, "ensemble")
            os.makedirs(ensemble_path, exist_ok=True)
            
            model_file = os.path.join(ensemble_path, "deep_model.pt")
            scaler_file = os.path.join(ensemble_path, "scaler.pt")
            metadata_file = os.path.join(ensemble_path, "metadata.json")
            
            if os.path.exists(model_file) and os.path.exists(metadata_file):
                # Load model
                self.deep_model.load_state_dict(torch.load(model_file, map_location=device))
                
                # Load scaler if it exists
                if os.path.exists(scaler_file):
                    self.scaler = torch.load(scaler_file)
                
                # Load metadata
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    self.feature_names = metadata.get("feature_names", self.feature_names)
                    self.feature_importance = metadata.get("feature_importance", {})
                    self.feature_correlations = metadata.get("feature_correlations", {})
                    self.metrics = metadata.get("metrics", self.metrics)
                    self.factor_weights = metadata.get("factor_weights", self.factor_weights)
                
                logger.info(f"Model loaded from {ensemble_path}")
                return True
            elif os.path.exists(self.model_path):
                # Legacy model loading
                checkpoint = torch.load(self.model_path, map_location=device)
                self.deep_model.load_state_dict(checkpoint["model_state_dict"])
                self.scaler = checkpoint.get("scaler", StandardScaler())
                
                # Load metadata if available
                metadata_path = Path(self.model_path).with_suffix(".json")
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        self.feature_names = metadata.get("feature_names", self.feature_names)
                        self.feature_importance = metadata.get("feature_importance", {})
                        self.metrics = metadata.get("metrics", self.metrics)
                
                logger.info(f"Legacy model loaded from {self.model_path}")
                # Save in new format for future use
                self.save_model()
                return True
                
            logger.warning(f"No models found at {ensemble_path} or {self.model_path}")
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
        
    def save_model(self) -> bool:
        """
        Save model to disk.
        
        Returns:
            True if model was saved successfully, False otherwise
        """
        try:
            ensemble_path = os.path.join(self.model_dir, "ensemble")
            os.makedirs(ensemble_path, exist_ok=True)
            
            # Save deep model
            model_file = os.path.join(ensemble_path, "deep_model.pt")
            torch.save(self.deep_model.state_dict(), model_file)
            
            # Save scaler
            scaler_file = os.path.join(ensemble_path, "scaler.pt")
            torch.save(self.scaler, scaler_file)
            
            # Save metadata
            metadata = {
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance,
                "feature_correlations": self.feature_correlations,
                "metrics": self.metrics,
                "factor_weights": self.factor_weights,
                "last_updated": datetime.now().isoformat(),
                "model_info": {
                    "input_dim": self.input_dim,
                    "hidden_dims": self.hidden_dims,
                    "output_dim": self.output_dim
                }
            }
            
            metadata_file = os.path.join(ensemble_path, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Also save a checkpoint for compatibility
            checkpoint = {
                "model_state_dict": self.deep_model.state_dict(),
                "scaler": self.scaler,
                "metadata": metadata
            }
            
            torch.save(checkpoint, self.model_path)
            
            logger.info(f"Model saved to {ensemble_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def preprocess_features(self, features: pd.DataFrame) -> np.ndarray:
        """
        Preprocess feature data.
        
        Args:
            features: DataFrame containing feature data
            
        Returns:
            Preprocessed numpy array
        """
        # Ensure all feature columns exist
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Add missing features with zero values
            for feature in missing_features:
                features[feature] = 0.0
        
        # Select and order features
        X = features[self.feature_names].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Scale features if scaler is fitted
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def _calculate_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calculate feature importance using permutation method.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of feature importance
        """
        # Convert data to torch tensors
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Get baseline predictions
        self.deep_model.eval()
        with torch.no_grad():
            baseline_preds = self.deep_model(X_tensor).cpu().numpy().flatten()
        
        baseline_auc = roc_auc_score(y, baseline_preds)
        
        # Calculate importance for each feature
        importance = {}
        for i, feature_name in enumerate(self.feature_names):
            # Create a copy of X with the feature shuffled
            X_shuffled = X.copy()
            np.random.shuffle(X_shuffled[:, i])
            
            # Get predictions with shuffled feature
            X_shuffled_tensor = torch.FloatTensor(X_shuffled).to(device)
            with torch.no_grad():
                shuffled_preds = self.deep_model(X_shuffled_tensor).cpu().numpy().flatten()
            
            # Calculate AUC drop
            shuffled_auc = roc_auc_score(y, shuffled_preds)
            importance_score = baseline_auc - shuffled_auc
            
            importance[feature_name] = float(importance_score)
        
        # Normalize importance
        total_importance = sum(max(0, imp) for imp in importance.values())
        if total_importance > 0:
            importance = {f: max(0, imp) / total_importance for f, imp in importance.items()}
        
        return importance
    
    def train(
        self, 
        train_features: pd.DataFrame, 
        train_targets: pd.Series,
        val_features: Optional[pd.DataFrame] = None,
        val_targets: Optional[pd.Series] = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        Train the ranking model.
        
        Args:
            train_features: Training feature data
            train_targets: Training target data
            val_features: Validation feature data
            val_targets: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Preprocess features
        X_train = train_features[self.feature_names].values
        
        # Fit scaler on training data
        self.scaler.fit(X_train)
        
        # Scale training data
        X_train_scaled = self.scaler.transform(X_train)
        y_train = train_targets.values
        
        # Preprocess validation data if provided
        if val_features is not None and val_targets is not None:
            X_val = val_features[self.feature_names].values
            X_val_scaled = self.scaler.transform(X_val)
            y_val = val_targets.values
            has_validation = True
        else:
            # Split training data for validation
            dataset = FactorDataset(X_train_scaled, y_train)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            has_validation = False
        
        if has_validation:
            # Create datasets and data loaders
            train_dataset = FactorDataset(X_train_scaled, y_train)
            val_dataset = FactorDataset(X_val_scaled, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Define optimizer and loss function
        optimizer = Adam(self.deep_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_auc": [],
            "val_auc": []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.deep_model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = self.deep_model(inputs).view(-1)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * inputs.size(0)
                train_preds.extend(torch.sigmoid(outputs).cpu().detach().numpy())
                train_targets.extend(targets.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader.dataset)
            train_auc = roc_auc_score(train_targets, train_preds)
            
            # Validation phase
            self.deep_model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    # Forward pass
                    outputs = self.deep_model(inputs).view(-1)
                    loss = criterion(outputs, targets)
                    
                    # Track metrics
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_auc = roc_auc_score(val_targets, val_preds)
            
            # Update history
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            history["train_auc"].append(train_auc)
            history["val_auc"].append(val_auc)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                best_model_state = self.deep_model.state_dict().copy()
            elif epoch - best_epoch >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train AUC: {train_auc:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val AUC: {val_auc:.4f}"
            )
        
        # Restore best model
        if best_model_state:
            self.deep_model.load_state_dict(best_model_state)
        
        # Calculate feature importance
        if has_validation:
            self.feature_importance = self._calculate_feature_importance(X_val_scaled, y_val)
        else:
            # Use a subset of training data for feature importance calculation
            subset_size = min(5000, len(X_train_scaled))
            subset_indices = np.random.choice(len(X_train_scaled), subset_size, replace=False)
            X_subset = X_train_scaled[subset_indices]
            y_subset = y_train[subset_indices]
            self.feature_importance = self._calculate_feature_importance(X_subset, y_subset)
        
        # Update metrics
        self.metrics = {
            "auc": val_auc,
            "precision": 0.0,  # Would calculate with thresholded predictions
            "recall": 0.0,     # Would calculate with thresholded predictions
            "sharpe": 0.0,     # Would calculate with backtest results
            "sortino": 0.0,    # Would calculate with backtest results
            "information_ratio": 0.0  # Would calculate with backtest results
        }
        
        # Save model
        self.save_model()
        
        return history
        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict stock scores.
        
        Args:
            features: DataFrame containing feature data
            
        Returns:
            Array of stock scores
        """
        # Check for cached predictions
        cache_key = None
        if settings.advanced.use_redis_cache:
            cache_key = f"ranking:predictions:{hash(features.to_json())}"
            cached_predictions = redis_client.get(cache_key)
            if cached_predictions is not None:
                logger.debug("Using cached predictions")
                return cached_predictions
        
        # Preprocess features
        X = self.preprocess_features(features)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # Make predictions
        self.deep_model.eval()
        with torch.no_grad():
            predictions = torch.sigmoid(self.deep_model(X_tensor)).cpu().numpy().flatten()
        
        # Cache predictions
        if settings.advanced.use_redis_cache and cache_key:
            redis_client.set(cache_key, predictions, ex=3600)  # Cache for 1 hour
        
        return predictions
    
    def rank_stocks(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Rank stocks based on their features.
        
        Args:
            stock_data: Dictionary of stock DataFrames with features
            
        Returns:
            Dictionary of stock scores
        """
        # Extract features for each stock
        features_list = []
        symbols = []
        
        for symbol, data in stock_data.items():
            # Skip if data is empty
            if data.empty:
                continue
                
            # Use the most recent row of data
            latest_data = data.iloc[-1:].copy()
            
            # Store symbol
            symbols.append(symbol)
            
            # Store features
            features_list.append(latest_data)
        
        # Combine features
        if not features_list:
            logger.warning("No valid stock data provided")
            return {}
            
        combined_features = pd.concat(features_list, ignore_index=True)
        
        # Make predictions
        scores = self.predict(combined_features)
        
        # Create score dictionary
        score_dict = {symbol: float(score) for symbol, score in zip(symbols, scores)}
        
        # Sort by score (descending)
        sorted_scores = {k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1], reverse=True)}
        
        return sorted_scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance.
        
        Returns:
            Dictionary of feature importance
        """
        return self.feature_importance
        
    def compress_model(self) -> 'RankingModel':
        """
        Compress model for faster inference using quantization.
        
        Returns:
            Self with compressed model
        """
        logger.info("Compressing ranking model for faster inference")
        
        try:
            # Check if model is already compressed
            if hasattr(self, '_is_compressed') and self._is_compressed:
                logger.info("Model is already compressed")
                return self
                
            # Prepare model for quantization
            self.deep_model.eval()
            
            # Quantize model to int8
            quantized_model = torch.quantization.quantize_dynamic(
                self.deep_model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            # Replace model with quantized version
            self.deep_model = quantized_model
            self._is_compressed = True
            
            # Save compressed model
            self.save_model()
            
            logger.info("Model compression completed")
            return self
        except Exception as e:
            logger.error(f"Error compressing model: {e}")
            logger.info("Continuing with uncompressed model")
            return self
    
    def update_factor_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Update factor weights.
        
        Args:
            new_weights: Dictionary of new factor weights
        """
        self.factor_weights.update(new_weights)
        self.save_model()
    
    def explain_ranking(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain the ranking of a stock.
        
        Args:
            features: DataFrame containing feature data for a single stock
            
        Returns:
            Dictionary with ranking explanation
        """
        # Check input
        if len(features) != 1:
            logger.warning(f"Expected features for 1 stock, got {len(features)}")
            if len(features) > 1:
                features = features.iloc[0:1]
        
        # Get prediction
        score = float(self.predict(features)[0])
        
        # Get feature values and importance
        feature_values = features[self.feature_names].iloc[0].to_dict()
        
        # Calculate contribution of each feature
        contributions = {}
        for feature, value in feature_values.items():
            importance = self.feature_importance.get(feature, 0.0)
            # Normalize value using mean and std from scaler
            feature_idx = self.feature_names.index(feature)
            mean = self.scaler.mean_[feature_idx] if hasattr(self.scaler, 'mean_') else 0
            std = self.scaler.scale_[feature_idx] if hasattr(self.scaler, 'scale_') else 1
            normalized_value = (value - mean) / std if std != 0 else 0
            
            # Calculate contribution (simplified)
            contribution = normalized_value * importance
            contributions[feature] = contribution
        
        # Sort contributions by absolute value
        sorted_contributions = {
            k: v for k, v in sorted(
                contributions.items(), 
                key=lambda item: abs(item[1]), 
                reverse=True
            )
        }
        
        # Group features by factor type
        factor_groups = {
            "momentum": [f for f in self.feature_names if "momentum" in f or "return" in f],
            "volume": [f for f in self.feature_names if "volume" in f],
            "volatility": [f for f in self.feature_names if "volatility" in f or "atr" in f],
            "trend": [f for f in self.feature_names if "ma" in f or "rsi" in f or "macd" in f],
            "sentiment": [f for f in self.feature_names if "sentiment" in f or "news" in f],
            "market": [f for f in self.feature_names if "market" in f or "sector" in f or "beta" in f]
        }
        
        # Calculate factor group contributions
        group_contributions = {}
        for group, features in factor_groups.items():
            group_contrib = sum(contributions.get(f, 0) for f in features)
            group_contributions[group] = group_contrib
        
        # Create explanation
        # Get top positive and negative factors (limited to 5 each)
        positive_factors = {k: v for k, v in sorted_contributions.items() if v > 0}
        negative_factors = {k: v for k, v in sorted_contributions.items() if v < 0}
        
        # Take only the top 5 items
        top_positive = dict(list(positive_factors.items())[:5])
        top_negative = dict(list(negative_factors.items())[:5])
        
        explanation = {
            "score": score,
            "percentile": int(score * 100),  # Simple percentile approximation
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "factor_groups": group_contributions
        }
        
        return explanation


# Initialize global instance
ranking_model = RankingModel(
    factor_weights=getattr(settings.model, "factor_weights", None)
)