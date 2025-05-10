"""
Enhanced pattern recognition model for identifying chart patterns.

Uses a Transformer-based architecture for better pattern recognition,
with support for pretrained models and improved feature extraction.
"""
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import optuna
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time

# Set up logger
logger = setup_logger("pattern_recognition")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Pattern definitions
PATTERN_CLASSES = [
    "no_pattern",       # 0: No specific pattern
    "breakout",         # 1: Price breaking out of a consolidation
    "reversal",         # 2: Price reversing its trend
    "continuation",     # 3: Continuation of existing trend
    "flag",             # 4: Bull/bear flag pattern
    "triangle",         # 5: Triangle pattern
    "head_shoulders",   # 6: Head and shoulders pattern
    "double_top",       # 7: Double top pattern
    "double_bottom",    # 8: Double bottom pattern
    "cup_handle",       # 9: Cup and handle pattern
    "ascending_channel",# 10: Ascending channel
    "descending_channel",# 11: Descending channel
    "wedge",            # 12: Wedge pattern
    "rounding_bottom",  # 13: Rounding bottom pattern
]

class OHLCVDataset(Dataset):
    """Enhanced dataset for OHLCV data with advanced preprocessing."""
    
    def __init__(self, data: List[pd.DataFrame], labels: List[int], lookback: int = 20, 
                 transform=None, add_technical_indicators: bool = True,
                 normalize_method: str = "minmax"):
        """
        Initialize the dataset.
        
        Args:
            data: List of DataFrames with OHLCV data
            labels: List of pattern class labels
            lookback: Number of bars to include in each sample
            transform: Optional transform to apply to the data
            add_technical_indicators: Whether to add technical indicators
            normalize_method: Method for normalizing data ('minmax', 'zscore', or 'percentchange')
        """
        self.data = data
        self.labels = labels
        self.lookback = lookback
        self.transform = transform
        self.add_technical_indicators = add_technical_indicators
        self.normalize_method = normalize_method
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (sample, label)
        """
        # Get the OHLCV data
        df = self.data[idx].copy()
        
        # Ensure we have enough data
        if len(df) < self.lookback:
            # Pad with the first value repeated if needed
            if not df.empty:
                first_row = df.iloc[0:1]
                padding = pd.concat([first_row] * (self.lookback - len(df)))
                df = pd.concat([padding, df])
            else:
                # Create empty DataFrame with zeros if df is empty
                padding = pd.DataFrame(
                    0, 
                    index=range(self.lookback),
                    columns=['open', 'high', 'low', 'close', 'volume']
                )
                df = padding
        
        # Add technical indicators if requested
        if self.add_technical_indicators:
            df = self._add_technical_indicators(df)
        
        # Take the last `lookback` bars
        df = df.iloc[-self.lookback:]
        
        # Normalize the data
        sample = self._normalize_sample(df)
        
        # Convert to tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        
        # Get the label
        label = self.labels[idx]
        
        # Apply transform if specified
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Ensure we have the required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Moving averages
        for period in [5, 10, 20]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            # Moving average position (close price relative to MA)
            df[f'ma_pos_{period}'] = (df['close'] - df[f'ma_{period}']) / df[f'ma_{period}']
        
        # Bollinger Bands
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['stddev'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['ma_20'] + 2 * df['stddev']
        df['bollinger_lower'] = df['ma_20'] - 2 * df['stddev']
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['ma_20']
        df['bollinger_pos'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volatility
        df['atr'] = df['high'] - df['low']
        df['atr_pct'] = df['atr'] / df['close']
        
        # Volume indicators
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_10']
        
        # Candle patterns (simplified)
        df['body'] = abs(df['close'] - df['open'])
        df['body_pct'] = df['body'] / df['open']
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_bullish'] = (df['close'] > df['open']).astype(float)
        
        # Fill NaN values with 0
        df.fillna(0, inplace=True)
        
        return df
    
    def _normalize_sample(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normalize the OHLCV data to create a tensor representation.
        
        Args:
            df: DataFrame with OHLCV and technical indicator data
            
        Returns:
            Normalized data as numpy array
        """
        # Determine which columns to use
        price_cols = ['open', 'high', 'low', 'close']
        derived_cols = [col for col in df.columns if col not in price_cols + ['volume']]
        
        # Extract data
        price_data = df[price_cols].values
        volume_data = df[['volume']].values if 'volume' in df.columns else np.zeros((len(df), 1))
        derived_data = df[derived_cols].values if derived_cols else np.zeros((len(df), 0))
        
        # Normalize price data based on specified method
        if self.normalize_method == 'minmax':
            # Min-max normalization within the sequence
            price_min = np.min(price_data[:, 2])  # Minimum of lows
            price_max = np.max(price_data[:, 1])  # Maximum of highs
            price_range = max(price_max - price_min, 1e-8)  # Avoid division by zero
            norm_price_data = (price_data - price_min) / price_range
        
        elif self.normalize_method == 'zscore':
            # Z-score normalization
            price_mean = np.mean(price_data[:, 3])  # Mean of close prices
            price_std = np.std(price_data[:, 3]) or 1.0  # Standard deviation of close prices (avoid div by zero)
            norm_price_data = (price_data - price_mean) / price_std
        
        elif self.normalize_method == 'percentchange':
            # Normalize relative to the first price in the sequence
            base_price = price_data[0, 3]  # First close price
            if base_price == 0:
                base_price = 1.0  # Avoid division by zero
            norm_price_data = price_data / base_price - 1.0
        
        else:
            # Default to min-max normalization
            price_min = np.min(price_data[:, 2])  # Minimum of lows
            price_max = np.max(price_data[:, 1])  # Maximum of highs
            price_range = max(price_max - price_min, 1e-8)  # Avoid division by zero
            norm_price_data = (price_data - price_min) / price_range
        
        # Normalize volume data
        vol_max = np.max(volume_data) if np.max(volume_data) > 0 else 1.0
        norm_volume_data = volume_data / vol_max
        
        # For derived indicators, we leave them as is since many are already normalized
        
        # Combine all data
        # Final shape: [channels, sequence_length]
        # Channels are: [open, high, low, close, volume, derived1, derived2, ...]
        combined_data = np.vstack([
            norm_price_data.T,          # Transpose to get [4, sequence_length]
            norm_volume_data.T,         # Transpose to get [1, sequence_length]
            derived_data.T              # Transpose to get [n_derived, sequence_length]
        ])
        
        return combined_data

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, embedding_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :]

class TransformerPatternModel(nn.Module):
    """Transformer-based model for pattern recognition."""
    
    def __init__(self, 
                 input_channels: int, 
                 d_model: int = 64,
                 nhead: int = 4, 
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 num_classes: int = len(PATTERN_CLASSES),
                 lookback: int = 20):
        """
        Initialize the transformer model.
        
        Args:
            input_channels: Number of input channels
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            num_classes: Number of pattern classes
            lookback: Sequence length
        """
        super(TransformerPatternModel, self).__init__()
        
        # Project input channels to model dimension
        self.input_projection = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, lookback)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.fc1 = nn.Linear(d_model * lookback, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, channels, sequence_length]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Transpose to [batch_size, sequence_length, channels]
        x = x.transpose(1, 2)
        
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Pass through output layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ResidualCNN(nn.Module):
    """Residual CNN model for pattern recognition."""
    
    def __init__(self, 
                 input_channels: int,
                 num_classes: int = len(PATTERN_CLASSES),
                 lookback: int = 20,
                 base_filters: int = 32,
                 n_blocks: int = 4,
                 dropout: float = 0.2):
        """
        Initialize the residual CNN model.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of pattern classes
            lookback: Length of the input sequence
            base_filters: Number of base filters (doubled in each block)
            n_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super(ResidualCNN, self).__init__()
        
        # Initial convolution
        self.conv_initial = nn.Conv1d(
            input_channels, base_filters, kernel_size=3, padding=1
        )
        self.bn_initial = nn.BatchNorm1d(base_filters)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        
        current_filters = base_filters
        for i in range(n_blocks):
            # Double filters every 2 blocks
            out_filters = current_filters * 2 if i % 2 == 1 else current_filters
            
            block = nn.Sequential(
                # First conv layer in block
                nn.Conv1d(current_filters, out_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_filters),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                # Second conv layer in block
                nn.Conv1d(out_filters, out_filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_filters)
            )
            
            self.residual_blocks.append(block)
            
            # Projection shortcut if dimensions change
            if current_filters != out_filters:
                shortcut = nn.Sequential(
                    nn.Conv1d(current_filters, out_filters, kernel_size=1),
                    nn.BatchNorm1d(out_filters)
                )
                self.residual_blocks.append(shortcut)
            else:
                self.residual_blocks.append(nn.Identity())
            
            current_filters = out_filters
        
        # Calculate final feature map size
        final_size = lookback
        n_pooling = n_blocks // 2
        for _ in range(n_pooling):
            final_size = final_size // 2
            
        # Global pooling and output layers
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Final dense layers
        self.fc = nn.Sequential(
            nn.Linear(current_filters * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, channels, sequence_length]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Initial convolution
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        
        # Residual blocks
        for i in range(0, len(self.residual_blocks), 2):
            block = self.residual_blocks[i]
            shortcut = self.residual_blocks[i+1]
            
            # Apply block
            residual = block(x)
            
            # Apply shortcut
            shortcut_x = shortcut(x)
            
            # Add residual connection
            x = F.relu(residual + shortcut_x)
            
            # Apply pooling every 2 blocks
            if i > 0 and i % 4 == 0:
                x = F.max_pool1d(x, kernel_size=2)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).squeeze(-1)
        max_pool = self.global_max_pool(x).squeeze(-1)
        
        # Concatenate pooling results
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Final dense layers
        x = self.fc(x)
        
        return x

pattern_model = PatternRecognitionModel  # Alias for easier import

class PatternRecognitionModel:
    """
    Enhanced pattern recognition model for identifying chart patterns.
    
    Features:
    - Multiple model architectures: CNN, Transformer, or ResNet
    - Hyperparameter optimization with Optuna
    - Proper model versioning and metadata tracking
    - Advanced data preprocessing and normalization options
    - Automatic selection of best model architecture
    """
    
    def __init__(self, model_path: Optional[str] = None, lookback: int = None,
                 model_type: str = "auto", pretrained: bool = True):
        """
        Initialize the pattern recognition model.
        
        Args:
            model_path: Path to the model file
            lookback: Number of bars to include in each sample
            model_type: Model architecture to use ("transformer", "resnet", "cnn", or "auto")
            pretrained: Whether to load pretrained weights (if available)
        """
        # Configure parameters
        self.lookback = lookback or settings.model.lookback_period
        self.model_type = model_type
        self.models_dir = os.path.join(settings.models_dir, "pattern_recognition")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Metadata for tracking model info
        self.metadata = {
            "model_type": model_type,
            "lookback": self.lookback,
            "classes": PATTERN_CLASSES,
            "created_at": datetime.now().isoformat(),
            "training_metrics": {},
            "version": "1.0.0"
        }
        
        # Model selection and instantiation
        self.model = self._create_model(model_type)
        self.model.to(device)
        
        # Load pretrained model if requested
        if pretrained:
            if model_path:
                # Use provided model path
                self.load_model(model_path)
            elif model_type != "auto":
                # Try to load the latest model of the specified type
                latest_model = self._find_latest_model(model_type)
                if latest_model:
                    self.load_model(latest_model)
            else:
                # Try each model type to find a pretrained model
                for mt in ["transformer", "resnet", "cnn"]:
                    latest_model = self._find_latest_model(mt)
                    if latest_model:
                        self.model_type = mt
                        self.model = self._create_model(mt)
                        self.model.to(device)
                        self.load_model(latest_model)
                        break
    
    def _create_model(self, model_type: str):
        """
        Create a model of the specified type.
        
        Args:
            model_type: Model architecture to use
            
        Returns:
            Neural network model
        """
        # Default parameters
        input_channels = 5  # OHLCV base
        additional_channels = 10  # Technical indicators
        total_channels = input_channels + additional_channels
        
        if model_type == "transformer":
            return TransformerPatternModel(
                input_channels=total_channels,
                d_model=64,
                nhead=4,
                num_encoder_layers=3,
                dim_feedforward=256,
                dropout=0.1,
                num_classes=len(PATTERN_CLASSES),
                lookback=self.lookback
            )
        elif model_type == "resnet":
            return ResidualCNN(
                input_channels=total_channels,
                num_classes=len(PATTERN_CLASSES),
                lookback=self.lookback,
                base_filters=32,
                n_blocks=4,
                dropout=0.2
            )
        elif model_type == "cnn" or model_type == "auto":
            # Default to the original CNN architecture
            return nn.Sequential(
                # Convolutional layers
                nn.Conv1d(total_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.MaxPool1d(2),
                
                # Flatten
                nn.Flatten(),
                
                # Fully connected layers
                nn.Linear((self.lookback // 8) * 128, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, len(PATTERN_CLASSES))
            )
        else:
            logger.error(f"Unknown model type: {model_type}")
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _find_latest_model(self, model_type: str) -> Optional[str]:
        """
        Find the latest model of the specified type.
        
        Args:
            model_type: Model type to search for
            
        Returns:
            Path to the latest model, or None if no model is found
        """
        model_pattern = f"pattern_{model_type}_*.pt"
        model_paths = list(Path(self.models_dir).glob(model_pattern))
        
        if not model_paths:
            return None
        
        # Sort by modification time (latest first)
        model_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return str(model_paths[0])
    
    def load_model(self, model_path: str):
        """
        Load model from file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            # Normalize path if needed
            if not os.path.isabs(model_path):
                # Check if the path is relative to models_dir
                if os.path.exists(os.path.join(self.models_dir, os.path.basename(model_path))):
                    model_path = os.path.join(self.models_dir, os.path.basename(model_path))
                # Otherwise check if it's relative to the default models path
                elif os.path.exists(os.path.join(settings.models_dir, os.path.basename(model_path))):
                    model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
            
            # Check if the file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Load model state dict
            state_dict = torch.load(model_path, map_location=device)
            
            # If it's a full model save with state_dict under 'model_state_dict' key
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                self.model.load_state_dict(state_dict['model_state_dict'])
                
                # Also load metadata if available
                if 'metadata' in state_dict:
                    self.metadata = state_dict['metadata']
                    
                    # Set model type based on metadata
                    if 'model_type' in self.metadata and self.metadata['model_type'] != self.model_type:
                        # Need to create a new model of the correct type
                        self.model_type = self.metadata['model_type']
                        self.model = self._create_model(self.model_type)
                        self.model.to(device)
                        self.model.load_state_dict(state_dict['model_state_dict'])
            else:
                # Try to load directly as state dict
                self.model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"Successfully loaded model from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return False
    
    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save model to file.
        
        Args:
            model_path: Path to save the model (optional)
            
        Returns:
            Path where the model was saved
        """
        try:
            # Generate model path if not provided
            if model_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_filename = f"pattern_{self.model_type}_{timestamp}.pt"
                model_path = os.path.join(self.models_dir, model_filename)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Update metadata
            self.metadata["saved_at"] = datetime.now().isoformat()
            
            # Save model with metadata
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'metadata': self.metadata
            }, model_path)
            
            logger.info(f"Model saved to {model_path}")
            return model_path
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return ""
    
    @log_execution_time(logger)
    def train(
        self, 
        train_data: List[pd.DataFrame], 
        train_labels: List[int],
        val_data: Optional[List[pd.DataFrame]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        opt_trials: int = 10,
        early_stopping_patience: int = 5,
        use_optuna: bool = True
    ):
        """
        Train the model with optional hyperparameter optimization.
        
        Args:
            train_data: List of DataFrames with OHLCV data for training
            train_labels: List of pattern class labels for training
            val_data: Optional list of DataFrames with OHLCV data for validation
            val_labels: Optional list of pattern class labels for validation
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            opt_trials: Number of Optuna optimization trials
            early_stopping_patience: Number of epochs to wait for improvement before early stopping
            use_optuna: Whether to use Optuna for hyperparameter optimization
        
        Returns:
            Training history
        """
        # Use values from config if not provided
        epochs = epochs or settings.model.epochs
        batch_size = batch_size or settings.model.batch_size
        learning_rate = learning_rate or settings.model.learning_rate
        
        # Create dataset
        add_technical_indicators = True
        normalize_method = "minmax"
        
        train_dataset = OHLCVDataset(
            train_data, train_labels, self.lookback,
            add_technical_indicators=add_technical_indicators,
            normalize_method=normalize_method
        )
        
        # Create validation dataset if provided
        if val_data and val_labels:
            val_dataset = OHLCVDataset(
                val_data, val_labels, self.lookback,
                add_technical_indicators=add_technical_indicators,
                normalize_method=normalize_method
            )
        else:
            # Split training data for validation if not provided
            val_size = int(0.2 * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # If using Optuna, optimize hyperparameters
        if use_optuna and opt_trials > 0:
            best_params, best_model_type = self._optimize_hyperparameters(
                train_dataset, val_dataset, opt_trials
            )
            
            # Update model with the best hyperparameters
            if best_model_type != self.model_type:
                logger.info(f"Changing model type from {self.model_type} to {best_model_type} based on optimization results")
                self.model_type = best_model_type
                self.model = self._create_model(best_model_type)
                self.model.to(device)
            
            # Extract hyperparameters
            learning_rate = best_params.get('learning_rate', learning_rate)
            batch_size = best_params.get('batch_size', batch_size)
            
            # Recreate data loaders with the new batch size
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Set model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Initialize training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_epoch': 0,
            'best_val_acc': 0
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        early_stopping_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Initialize metrics
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Train on batches
            for i, (inputs, labels) in enumerate(train_loader):
                # Move tensors to device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            
            # Validate the model
            val_loss, val_acc, val_metrics = self._validate(val_loader, criterion)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check if this is the best model so far
            if val_acc > history['best_val_acc']:
                history['best_val_acc'] = val_acc
                history['best_epoch'] = epoch
                
                # Save best model state
                best_model_state = self.model.state_dict().copy()
                
                # Reset early stopping counter
                early_stopping_counter = 0
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load the best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Update metadata with training info
        self.metadata["training_metrics"] = {
            "best_val_acc": float(history['best_val_acc']),
            "best_epoch": history['best_epoch'],
            "final_train_acc": float(train_acc),
            "final_val_acc": float(val_acc),
            "num_samples": len(train_dataset) + len(val_dataset),
            "trained_at": datetime.now().isoformat()
        }
        
        # Save the trained model
        model_path = self.save_model()
        
        # Also save to the default model path
        default_model_path = settings.model.pattern_model_path
        if default_model_path:
            # Convert to absolute path if it's a relative path
            if not os.path.isabs(default_model_path):
                default_model_path = os.path.join(settings.models_dir, os.path.basename(default_model_path))
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(default_model_path), exist_ok=True)
            
            # Copy the model to the default path
            import shutil
            shutil.copy2(model_path, default_model_path)
            logger.info(f"Copied model to default path: {default_model_path}")
        
        return history
    
    def _optimize_hyperparameters(
        self, 
        train_dataset: Dataset,
        val_dataset: Dataset,
        n_trials: int = 10
    ) -> Tuple[Dict[str, Any], str]:
        """
        Use Optuna to find the best hyperparameters.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            n_trials: Number of Optuna trials
            
        Returns:
            Tuple of (best hyperparameters dict, best model type)
        """
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        def objective(trial):
            # Sample hyperparameters
            model_type = trial.suggest_categorical("model_type", ["transformer", "resnet", "cnn"])
            
            # Basic hyperparameters
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            
            # Model-specific hyperparameters
            if model_type == "transformer":
                d_model = trial.suggest_categorical("d_model", [32, 64, 128])
                nhead = trial.suggest_categorical("nhead", [2, 4, 8])
                num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 4)
                
                # Create model
                model = TransformerPatternModel(
                    input_channels=15,  # OHLCV + technical indicators
                    d_model=d_model,
                    nhead=nhead,
                    num_encoder_layers=num_encoder_layers,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    num_classes=len(PATTERN_CLASSES),
                    lookback=self.lookback
                )
            elif model_type == "resnet":
                base_filters = trial.suggest_categorical("base_filters", [16, 32, 64])
                n_blocks = trial.suggest_int("n_blocks", 2, 6)
                
                # Create model
                model = ResidualCNN(
                    input_channels=15,  # OHLCV + technical indicators
                    num_classes=len(PATTERN_CLASSES),
                    lookback=self.lookback,
                    base_filters=base_filters,
                    n_blocks=n_blocks,
                    dropout=dropout
                )
            else:  # CNN
                # Create model using the _create_model method
                model = self._create_model("cnn")
                
                # Note: Hyperparameters for the basic CNN are not optimized via Optuna in this setup.
                # If needed, the _create_model method or this section would need to be updated
                # to accept and use hyperparameters suggested by Optuna for the CNN.
                # For now, it uses the default CNN architecture defined in _create_model.
            
            # Move model to device
            model.to(device)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train for a few epochs
            model.train()
            max_epochs = 10
            best_val_loss = float('inf')
            
            for epoch in range(max_epochs):
                # Train one epoch
                train_loss = 0.0
                for inputs, labels in train_loader:
                    # Move tensors to device
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # Zero the gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update weights
                    optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item() * inputs.size(0)
                
                # Validate
                model.eval()
                val_loss = 0.0
                val_correct = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        # Move tensors to device
                        inputs, labels = inputs.to(device), labels.to(device)
                        
                        # Forward pass
                        outputs = model(inputs)
                        
                        # Calculate loss
                        loss = criterion(outputs, labels)
                        
                        # Update metrics
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(val_loader.dataset)
                val_acc = val_correct / len(val_loader.dataset)
                
                # Check for improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                # Early stopping
                if epoch >= 5 and val_loss > best_val_loss * 1.1:
                    break
                
                # Report intermediate result
                trial.report(val_loss, epoch)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            return val_loss
        
        # Create Optuna study
        study = optuna.create_study(direction="minimize")
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials)
        
        # Get best hyperparameters
        best_params = study.best_params
        best_model_type = best_params.pop("model_type", self.model_type)
        
        logger.info(f"Best model type: {best_model_type}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params, best_model_type
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float, Dict]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (validation loss, validation accuracy, additional metrics)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # For more detailed metrics
        all_preds = []
        all_labels = []
        
        # No gradient calculation needed for validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move tensors to device
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Update metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                # Store for detailed metrics
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        additional_metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
        # Set model back to training mode
        self.model.train()
        
        return val_loss, val_acc, additional_metrics
    
    def predict(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """
        Predict the pattern from OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary of pattern probabilities
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Create a dataset with a single sample
        dataset = OHLCVDataset(
            [ohlcv_data], [0], self.lookback,  # Label doesn't matter for prediction
            add_technical_indicators=True,
            normalize_method="minmax"
        )
        
        # Get the sample
        sample, _ = dataset[0]
        
        # Add batch dimension
        sample = sample.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(sample)
            probs = F.softmax(outputs, dim=1)[0]
        
        # Convert to dictionary
        result = {
            pattern: float(prob) for pattern, prob in zip(PATTERN_CLASSES, probs.cpu().numpy())
        }
        
        return result
    
    def predict_pattern(self, ohlcv_data: pd.DataFrame, confidence_threshold: float = None) -> Tuple[str, float]:
        """
        Predict the most likely pattern from OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            confidence_threshold: Minimum confidence threshold (if None, uses settings)
            
        Returns:
            Tuple of (pattern name, confidence)
        """
        # Use threshold from settings if not provided
        if confidence_threshold is None:
            confidence_threshold = settings.model.confidence_threshold
        
        # Get all pattern probabilities
        pattern_probs = self.predict(ohlcv_data)
        
        # Find the most likely pattern
        pattern_name = max(pattern_probs.keys(), key=lambda k: pattern_probs[k])
        confidence = pattern_probs[pattern_name]
        
        # If below threshold, return "no_pattern"
        if confidence < confidence_threshold and pattern_name != "no_pattern":
            return "no_pattern", confidence
        
        return pattern_name, confidence
    
    def batch_predict(self, ohlcv_data_list: List[pd.DataFrame]) -> List[Dict[str, float]]:
        """
        Make predictions for multiple OHLCV sequences.
        
        Args:
            ohlcv_data_list: List of DataFrames with OHLCV data
            
        Returns:
            List of dictionaries with pattern probabilities
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Create a dataset
        dataset = OHLCVDataset(
            ohlcv_data_list, [0] * len(ohlcv_data_list), self.lookback,  # Labels don't matter for prediction
            add_technical_indicators=True,
            normalize_method="minmax"
        )
        
        # Create a data loader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Make predictions
        all_probs = []
        
        with torch.no_grad():
            for inputs, _ in loader:
                # Move tensors to device
                inputs = inputs.to(device)
                
                # Forward pass
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                
                # Add to list
                all_probs.append(probs.cpu().numpy())
        
        # Concatenate all batches
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Convert to list of dictionaries
        results = []
        for probs in all_probs:
            results.append({
                pattern: float(prob) for pattern, prob in zip(PATTERN_CLASSES, probs)
            })
        
        return results

# Create a global instance of the model
pattern_recognition_model = PatternRecognitionModel(
    model_path=settings.model.pattern_model_path,
    lookback=settings.model.lookback_period,
    model_type="auto",
    pretrained=True
)
