"""
Pattern recognition model for financial time series data.

Uses deep learning to identify chart patterns and technical setups
in price data for trading signals.
"""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client
from src.metrics.ml_metrics import get_collector, MetricsTimer

# Set up logger
logger = setup_logger("pattern_recognition")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Patterns to recognize
PATTERNS = {
    0: "none",
    1: "double_top",
    2: "double_bottom",
    3: "head_shoulders",
    4: "inv_head_shoulders",
    5: "triangle_ascending",
    6: "triangle_descending",
    7: "triangle_symmetrical",
    8: "flag_bullish",
    9: "flag_bearish",
    10: "wedge_rising",
    11: "wedge_falling",
    12: "channel_up",
    13: "channel_down",
    14: "cup_with_handle",
    15: "breakout"
}

# Pattern metadata
PATTERN_METADATA = {
    "double_top": {"bullish": False, "min_bars": 20, "reliability": 0.7},
    "double_bottom": {"bullish": True, "min_bars": 20, "reliability": 0.7},
    "head_shoulders": {"bullish": False, "min_bars": 30, "reliability": 0.75},
    "inv_head_shoulders": {"bullish": True, "min_bars": 30, "reliability": 0.75},
    "triangle_ascending": {"bullish": True, "min_bars": 15, "reliability": 0.65},
    "triangle_descending": {"bullish": False, "min_bars": 15, "reliability": 0.65},
    "triangle_symmetrical": {"bullish": None, "min_bars": 15, "reliability": 0.6},
    "flag_bullish": {"bullish": True, "min_bars": 10, "reliability": 0.6},
    "flag_bearish": {"bullish": False, "min_bars": 10, "reliability": 0.6},
    "wedge_rising": {"bullish": False, "min_bars": 20, "reliability": 0.65},
    "wedge_falling": {"bullish": True, "min_bars": 20, "reliability": 0.65},
    "channel_up": {"bullish": True, "min_bars": 20, "reliability": 0.7},
    "channel_down": {"bullish": False, "min_bars": 20, "reliability": 0.7},
    "cup_with_handle": {"bullish": True, "min_bars": 40, "reliability": 0.8},
    "breakout": {"bullish": True, "min_bars": 5, "reliability": 0.6}
}


class TimeSeriesDataset(Dataset):
    """Dataset for time series data with sliding windows."""
    
    def __init__(self, data: np.ndarray, window_size: int = 50, stride: int = 1, labels: Optional[np.ndarray] = None):
        """
        Initialize dataset with time series data.
        
        Args:
            data: Input time series data of shape (n_samples, n_features)
            window_size: Size of sliding window
            stride: Stride for sliding window
            labels: Optional labels for each window
        """
        self.data = data
        self.window_size = window_size
        self.stride = stride
        self.labels = labels
        
        # Create sliding windows
        self.windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            self.windows.append(i)
            
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window_start = self.windows[idx]
        window_end = window_start + self.window_size
        
        window_data = self.data[window_start:window_end]
        window_tensor = torch.tensor(window_data, dtype=torch.float32)
        
        item = {"features": window_tensor}
        
        if self.labels is not None:
            if self.labels.ndim == 1:
                # Single label for each window
                label = self.labels[window_end - 1]
                item["label"] = torch.tensor(label, dtype=torch.long)
            else:
                # Multiple labels or features for each window
                label = self.labels[window_end - 1]
                item["label"] = torch.tensor(label, dtype=torch.float32)
                
        return item


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for pattern recognition.
    
    Combines 1D convolutions for feature extraction with LSTM for sequential pattern recognition.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        num_classes: int = len(PATTERNS),
        dropout: float = 0.2
    ):
        """
        Initialize the CNN-LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout)
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
        
        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Input shape: (batch_size, seq_len, input_dim)
        # Reshape for CNN: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply CNN
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Reshape for LSTM: (batch_size, seq_len_after_pool, cnn_output_channels)
        x = x.permute(0, 2, 1)
        
        # Apply LSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Apply final dropout and classification
        x = self.dropout2(context)
        x = self.fc(x)
        
        return x


class PatternRecognitionModel:
    """
    Pattern recognition model for financial time series data.
    
    Uses deep learning to identify chart patterns and provide trading signals.
    """
    
    def __init__(self, model_path: Optional[str] = None, window_size: int = 40):
        """
        Initialize the pattern recognition model.
        
        Args:
            model_path: Path to a trained model file
            window_size: Window size for pattern recognition
        """
        self.window_size = window_size
        self.input_dim = 5  # OHLCV
        self.hidden_dim = 128
        self.num_layers = 2
        self.num_classes = len(PATTERNS)
        
        # Create model
        self.model = CNNLSTMModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes
        ).to(device)
        
        # Training parameters
        self.batch_size = 64
        self.learning_rate = 1e-3
        self.num_epochs = 50
        
        # Performance metrics
        self.metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "val_accuracy": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_f1": 0.0
        }
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
            
        # Feature normalization parameters
        self.feature_means = None
        self.feature_stds = None
        
        # Metadata
        self.metadata = {
            "window_size": window_size,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "metrics": self.metrics,
            "patterns": PATTERNS,
            "pattern_metadata": PATTERN_METADATA,
            "last_updated": datetime.now().isoformat()
        }
        
    def _preprocess_data(self, ohlcv_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess OHLCV data for model input.
        
        Args:
            ohlcv_df: DataFrame with OHLCV data
            
        Returns:
            Preprocessed numpy array
        """
        if ohlcv_df.shape[0] < self.window_size:
            logger.warning(f"Data too short: {ohlcv_df.shape[0]} < {self.window_size}")
            # Pad with zeros if too short
            pad_rows = self.window_size - ohlcv_df.shape[0]
            pad_df = pd.DataFrame(np.zeros((pad_rows, ohlcv_df.shape[1])), columns=ohlcv_df.columns)
            ohlcv_df = pd.concat([pad_df, ohlcv_df], ignore_index=True)
        
        # Extract OHLCV features
        features = ohlcv_df[['open', 'high', 'low', 'close', 'volume']].values
        
        # Normalize features
        if features.shape[0] > 0:
            # Calculate means and stds on this data if not already set
            if self.feature_means is None or self.feature_stds is None:
                self.feature_means = np.mean(features, axis=0)
                self.feature_stds = np.std(features, axis=0)
                self.feature_stds[self.feature_stds == 0] = 1.0  # Avoid division by zero
                
            # Normalize
            features = (features - self.feature_means) / self.feature_stds
        
        return features
    
    def train(
        self, 
        train_data: List[pd.DataFrame], 
        train_labels: List[int],
        val_data: Optional[List[pd.DataFrame]] = None,
        val_labels: Optional[List[int]] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train the pattern recognition model.
        
        Args:
            train_data: List of DataFrames with OHLCV data
            train_labels: List of pattern labels
            val_data: Optional validation data
            val_labels: Optional validation labels
            batch_size: Batch size for training
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            
        Returns:
            Training history
        """
        # Set training parameters
        batch_size = batch_size or self.batch_size
        learning_rate = learning_rate or self.learning_rate
        num_epochs = num_epochs or self.num_epochs
        
        # Preprocess data
        processed_data = []
        for df in train_data:
            processed_data.append(self._preprocess_data(df))
        
        # Combine into dataset
        X_train = np.concatenate(processed_data, axis=0)
        y_train = np.array(train_labels)
        
        # Create validation set if provided
        if val_data is not None and val_labels is not None:
            processed_val_data = []
            for df in val_data:
                processed_val_data.append(self._preprocess_data(df))
            
            X_val = np.concatenate(processed_val_data, axis=0)
            y_val = np.array(val_labels)
        else:
            # Split training data for validation
            dataset = TimeSeriesDataset(X_train, self.window_size, stride=1, labels=y_train)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        if val_data is not None and val_labels is not None:
            train_dataset = TimeSeriesDataset(X_train, self.window_size, stride=1, labels=y_train)
            val_dataset = TimeSeriesDataset(X_val, self.window_size, stride=1, labels=y_val)
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Define optimizer and loss function
        optimizer = Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                features = batch["features"].to(device)
                labels = batch["label"].to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            y_true = []
            y_pred = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].to(device)
                    labels = batch["label"].to(device)
                    
                    # Forward pass
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    # Track metrics
                    val_loss += loss.item() * features.size(0)
                    _, predicted = torch.max(outputs, 1)
                    
                    # Collect predictions and true labels
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            
            # Calculate validation metrics
            avg_val_loss = val_loss / len(val_dataset)
            val_acc = accuracy_score(y_true, y_pred)
            val_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            history["val_loss"].append(avg_val_loss)
            history["val_acc"].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.metrics = {
                    "accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1
                }
                self.metadata["metrics"] = self.metrics
                self.metadata["last_updated"] = datetime.now().isoformat()
                
                # Record metrics in the ML metrics system
                metrics_collector = get_collector("pattern_recognition")
                metrics_collector.record_accuracy(
                    accuracy=train_acc,
                    precision=val_precision,
                    recall=val_recall,
                    f1=val_f1
                )
                
                # Save model
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(settings.models_dir, f"pattern_model_{timestamp}.pt")
                self.save_model(save_path)
                
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f}"
            )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(settings.models_dir, f"pattern_model_final_{timestamp}.pt")
        self.save_model(save_path)
        
        # Also save to default model path
        if hasattr(settings.model, "pattern_model_path"):
            self.save_model(settings.model.pattern_model_path)
        
        return history
    
    def predict(self, ohlcv_data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Predict patterns in OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            threshold: Confidence threshold for predictions
            
        Returns:
            Dictionary with pattern predictions
        """
        # Get metrics collector
        metrics_collector = get_collector("pattern_recognition")
        
        # Check if data is sufficient
        if len(ohlcv_data) < self.window_size:
            logger.warning(f"Data too short for prediction: {len(ohlcv_data)} < {self.window_size}")
            
            # Record error in metrics
            metrics_collector.record_error(
                error_type="InsufficientData",
                error_message=f"Data too short for prediction: {len(ohlcv_data)} < {self.window_size}"
            )
            
            return {
                "pattern": "none",
                "confidence": 0.0,
                "pattern_idx": 0,
                "bullish": None,
                "reliability": 0.0
            }
        
        # Check if we should use cached result
        if settings.advanced.use_redis_cache:
            cache_key = f"pattern:{hash(ohlcv_data.iloc[-self.window_size:].to_json())}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.debug("Using cached pattern prediction")
                return json.loads(cached_result)
        
        # Use metrics timer to measure inference latency
        with MetricsTimer("pattern_recognition", "inference"):
            # Preprocess data
            features = self._preprocess_data(ohlcv_data)
            
            # Create sliding windows if needed
            if len(features) > self.window_size:
                # Use the most recent window
                features = features[-self.window_size:]
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
                
                pattern_idx = predicted.item()
                pattern_name = PATTERNS.get(pattern_idx, "none")
                pattern_conf = confidence.item()
                
                # Check against threshold
                if pattern_conf < threshold and pattern_idx != 0:
                    pattern_idx = 0
                    pattern_name = "none"
                    pattern_conf = 1.0 - pattern_conf
                
                # Get additional pattern metadata
                pattern_info = PATTERN_METADATA.get(pattern_name, {"bullish": None, "reliability": 0.0})
                
                result = {
                    "pattern": pattern_name,
                    "confidence": pattern_conf,
                    "pattern_idx": pattern_idx,
                    "bullish": pattern_info.get("bullish"),
                    "reliability": pattern_info.get("reliability", 0.0),
                    "probabilities": {PATTERNS[i]: float(prob) for i, prob in enumerate(probabilities.cpu().numpy())}
                }
        
        # Record confidence metric
        # Note: We don't have ground truth for online predictions, so we don't record accuracy
        metrics_collector.record_confidence(
            confidence=pattern_conf,
            correct=True,  # We don't know if it's correct in real-time prediction
            prediction_type="pattern_recognition"
        )
        
        # Cache the result
        if settings.advanced.use_redis_cache:
            redis_client.set(cache_key, json.dumps(result), ex=3600)  # Cache for 1 hour
            
        return result
    
    def predict_pattern(self, ohlcv_data: pd.DataFrame, threshold: float = 0.5) -> Tuple[str, float]:
        """
        Predict pattern in OHLCV data and return pattern name and confidence as a tuple.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            threshold: Confidence threshold for predictions
            
        Returns:
            Tuple of (pattern_name, confidence)
        """
        result = self.predict(ohlcv_data, threshold)
        return result["pattern"], result["confidence"]
    
    def save_model(self, path: str) -> bool:
        """
        Save model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model state and metadata
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "metadata": self.metadata,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if path exists
            if not os.path.exists(path):
                logger.error(f"Model file not found: {path}")
                return False
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            
            # Load model state
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Initialize with default weights if model_state_dict is not found
                logger.warning(f"model_state_dict not found in {path}, using default weights")
                # Model will use the default initialized weights
            
            # Load metadata
            if "metadata" in checkpoint:
                self.metadata = checkpoint["metadata"]
                self.window_size = self.metadata.get("window_size", self.window_size)
                self.metrics = self.metadata.get("metrics", self.metrics)
            
            # Load normalization parameters
            if "feature_means" in checkpoint:
                self.feature_means = checkpoint["feature_means"]
            if "feature_stds" in checkpoint:
                self.feature_stds = checkpoint["feature_stds"]
                
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def analyze_chart(self, ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of a price chart.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with analysis results
        """
        # Ensure we have enough data
        if len(ohlcv_data) < self.window_size:
            logger.warning(f"Not enough data for analysis: {len(ohlcv_data)} < {self.window_size}")
            return {
                "pattern": "none",
                "signals": [],
                "additional_info": "Insufficient data for analysis"
            }
        
        # Get pattern prediction
        pattern_prediction = self.predict(ohlcv_data)
        
        # Generate trading signals
        signals = self._generate_signals(ohlcv_data, pattern_prediction)
        
        # Calculate key price levels
        price_levels = self._calculate_price_levels(ohlcv_data, pattern_prediction)
        
        # Additional technical analysis
        technical_indicators = self._calculate_technical_indicators(ohlcv_data)
        
        # Combine results
        analysis = {
            "pattern": pattern_prediction["pattern"],
            "confidence": pattern_prediction["confidence"],
            "signals": signals,
            "price_levels": price_levels,
            "technical_indicators": technical_indicators,
            "pattern_details": {
                "is_bullish": pattern_prediction["bullish"],
                "reliability": pattern_prediction["reliability"],
                "all_probabilities": pattern_prediction["probabilities"]
            }
        }
        
        return analysis
    
    def _generate_signals(self, ohlcv_data: pd.DataFrame, pattern_prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on pattern recognition.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            pattern_prediction: Pattern prediction results
            
        Returns:
            List of trading signals
        """
        signals = []
        pattern = pattern_prediction["pattern"]
        confidence = pattern_prediction["confidence"]
        is_bullish = pattern_prediction["bullish"]
        
        # Skip if no pattern or low confidence
        if pattern == "none" or confidence < 0.5:
            return signals
        
        current_price = ohlcv_data["close"].iloc[-1]
        current_volume = ohlcv_data["volume"].iloc[-1]
        avg_volume = ohlcv_data["volume"].rolling(20).mean().iloc[-1]
        
        # Generate signal based on pattern
        if is_bullish:
            signal_type = "buy"
            reason = f"{pattern} pattern detected"
            
            # Add volume confirmation
            if current_volume > 1.5 * avg_volume:
                confidence_modifier = 0.1
                reason += " with strong volume confirmation"
            else:
                confidence_modifier = 0.0
                
            signals.append({
                "type": signal_type,
                "strength": confidence + confidence_modifier,
                "reason": reason,
                "price": current_price,
                "timestamp": datetime.now().isoformat()
            })
        elif is_bullish is False:  # Explicitly bearish (not None)
            signal_type = "sell"
            reason = f"{pattern} pattern detected"
            
            # Add volume confirmation
            if current_volume > 1.5 * avg_volume:
                confidence_modifier = 0.1
                reason += " with strong volume confirmation"
            else:
                confidence_modifier = 0.0
                
            signals.append({
                "type": signal_type,
                "strength": confidence + confidence_modifier,
                "reason": reason,
                "price": current_price,
                "timestamp": datetime.now().isoformat()
            })
        
        return signals
    
    def _calculate_price_levels(self, ohlcv_data: pd.DataFrame, pattern_prediction: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate important price levels based on the detected pattern.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            pattern_prediction: Pattern prediction results
            
        Returns:
            Dictionary with key price levels
        """
        price_levels = {}
        pattern = pattern_prediction["pattern"]
        
        # Calculate basic support and resistance
        close = ohlcv_data["close"]
        high = ohlcv_data["high"]
        low = ohlcv_data["low"]
        
        # Current price and recent min/max
        current_price = close.iloc[-1]
        recent_high = high.iloc[-20:].max()
        recent_low = low.iloc[-20:].min()
        
        price_levels["current"] = current_price
        price_levels["recent_high"] = recent_high
        price_levels["recent_low"] = recent_low
        
        # Add pattern-specific levels
        if pattern == "double_top":
            # Find the two tops
            price_levels["resistance"] = recent_high
            price_levels["neckline"] = low.iloc[-20:].nlargest(2).iloc[-1]
            price_levels["target"] = price_levels["neckline"] - (recent_high - price_levels["neckline"])
            
        elif pattern == "double_bottom":
            # Find the two bottoms
            price_levels["support"] = recent_low
            price_levels["neckline"] = high.iloc[-20:].nsmallest(2).iloc[-1]
            price_levels["target"] = price_levels["neckline"] + (price_levels["neckline"] - recent_low)
            
        elif pattern in ["head_shoulders", "inv_head_shoulders"]:
            if pattern == "head_shoulders":
                # For head and shoulders, neckline is support
                peaks = high.iloc[-30:].nlargest(3)
                head = peaks.iloc[0]
                shoulders = peaks.iloc[1:3].mean()
                price_levels["head"] = head
                price_levels["shoulders"] = shoulders
                price_levels["neckline"] = low.iloc[-30:].nlargest(2).mean()
                price_levels["target"] = price_levels["neckline"] - (head - price_levels["neckline"])
            else:
                # For inverse head and shoulders, neckline is resistance
                troughs = low.iloc[-30:].nsmallest(3)
                head = troughs.iloc[0]
                shoulders = troughs.iloc[1:3].mean()
                price_levels["head"] = head
                price_levels["shoulders"] = shoulders
                price_levels["neckline"] = high.iloc[-30:].nsmallest(2).mean()
                price_levels["target"] = price_levels["neckline"] + (price_levels["neckline"] - head)
        
        # Add common levels
        price_levels["support"] = price_levels.get("support", recent_low)
        price_levels["resistance"] = price_levels.get("resistance", recent_high)
        
        # Add moving averages
        price_levels["ma_20"] = close.rolling(20).mean().iloc[-1]
        price_levels["ma_50"] = close.rolling(50).mean().iloc[-1]
        price_levels["ma_200"] = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
        
        return price_levels
    
    def _calculate_technical_indicators(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate additional technical indicators.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with technical indicators
        """
        indicators = {}
        
        close = ohlcv_data["close"]
        high = ohlcv_data["high"]
        low = ohlcv_data["low"]
        
        # RSI (14-period)
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        indicators["rsi_14"] = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Bollinger Bands (20-period, 2 standard deviations)
        ma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        indicators["bb_upper"] = (ma_20 + 2 * std_20).iloc[-1]
        indicators["bb_middle"] = ma_20.iloc[-1]
        indicators["bb_lower"] = (ma_20 - 2 * std_20).iloc[-1]
        
        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        indicators["macd"] = macd.iloc[-1]
        indicators["macd_signal"] = signal.iloc[-1]
        indicators["macd_histogram"] = (macd - signal).iloc[-1]
        
        # ATR (14-period)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators["atr_14"] = tr.rolling(14).mean().iloc[-1]
        
        # Stochastic Oscillator (14, 3, 3)
        low_14 = low.rolling(14).min()
        high_14 = high.rolling(14).max()
        k = 100 * ((close - low_14) / (high_14 - low_14))
        indicators["stoch_k"] = k.iloc[-1]
        indicators["stoch_d"] = k.rolling(3).mean().iloc[-1]
        
        return indicators


# Create global instance
pattern_recognition_model = PatternRecognitionModel(
    model_path=getattr(settings.model, "pattern_model_path", None),
    window_size=50
)


def analyze_pattern(ohlcv_data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze chart pattern in OHLCV data."""
    return pattern_recognition_model.analyze_chart(ohlcv_data)


def get_patterns(ohlcv_data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
    """Get pattern predictions for OHLCV data."""
    return pattern_recognition_model.predict(ohlcv_data, threshold=threshold)


def generate_signals(ohlcv_data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate trading signals based on pattern recognition."""
    analysis = pattern_recognition_model.analyze_chart(ohlcv_data)
    return analysis["signals"]