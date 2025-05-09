"""
Pattern recognition CNN model for identifying chart patterns.

This model converts OHLCV data into image-like representations
and identifies common chart patterns like breakouts, reversals, etc.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from src.config.settings import settings
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("pattern_recognition")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Pattern definitions
PATTERN_CLASSES = [
    "no_pattern",  # 0: No specific pattern
    "breakout",    # 1: Price breaking out of a consolidation
    "reversal",    # 2: Price reversing its trend
    "continuation",# 3: Continuation of existing trend
    "flag",        # 4: Bull/bear flag pattern
    "triangle",    # 5: Triangle pattern
    "head_shoulders",  # 6: Head and shoulders pattern
    "double_top",  # 7: Double top pattern
    "double_bottom",  # 8: Double bottom pattern
]

class OHLCVDataset(Dataset):
    """Dataset for OHLCV data."""
    
    def __init__(self, data: List[pd.DataFrame], labels: List[int], lookback: int = 20, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data: List of DataFrames with OHLCV data
            labels: List of pattern class labels
            lookback: Number of bars to include in each sample
            transform: Optional transform to apply to the data
        """
        self.data = data
        self.labels = labels
        self.lookback = lookback
        self.transform = transform
    
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
        df = self.data[idx]
        
        # Ensure we have enough data
        if len(df) < self.lookback:
            # Pad with zeros if needed
            padding = pd.DataFrame(
                0, 
                index=range(self.lookback - len(df)),
                columns=df.columns
            )
            df = pd.concat([padding, df])
        
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
    
    def _normalize_sample(self, df: pd.DataFrame) -> np.ndarray:
        """
        Normalize the OHLCV data to create an image-like representation.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Normalized data as numpy array
        """
        # Extract OHLCV data
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        
        # Calculate price range for normalization
        price_min = np.min(lows)
        price_max = np.max(highs)
        price_range = max(price_max - price_min, 1e-8)  # Avoid division by zero
        
        # Normalize prices to [0, 1]
        norm_opens = (opens - price_min) / price_range
        norm_highs = (highs - price_min) / price_range
        norm_lows = (lows - price_min) / price_range
        norm_closes = (closes - price_min) / price_range
        
        # Normalize volumes to [0, 1]
        vol_max = np.max(volumes)
        norm_volumes = volumes / vol_max if vol_max > 0 else volumes
        
        # Combine into 5-channel image-like representation
        # Shape: [5, lookback] where channels are [open, high, low, close, volume]
        norm_data = np.stack([
            norm_opens, norm_highs, norm_lows, norm_closes, norm_volumes
        ])
        
        return norm_data

class PatternRecognitionCNN(nn.Module):
    """CNN model for pattern recognition."""
    
    def __init__(self, num_classes: int = len(PATTERN_CLASSES), lookback: int = 20, channels: int = 5):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of pattern classes
            lookback: Number of bars in each sample
            channels: Number of input channels (OHLCV = 5)
        """
        super(PatternRecognitionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Pooling layer
        self.pool = nn.MaxPool1d(2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
        # Calculate the size of the flattened features
        # After 3 pooling layers with stride 2, the size is reduced by 2^3 = 8
        flat_size = (lookback // 8) * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, channels, lookback]
            
        Returns:
            Output tensor of shape [batch_size, num_classes]
        """
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class PatternRecognitionModel:
    """
    Pattern recognition model for identifying chart patterns.
    """
    
    def __init__(self, model_path: Optional[str] = None, lookback: int = 20):
        """
        Initialize the pattern recognition model.
        
        Args:
            model_path: Path to the model file
            lookback: Number of bars to include in each sample
        """
        self.lookback = lookback
        self.model = PatternRecognitionCNN(lookback=lookback)
        self.model.to(device)
        
        # Load model if path is provided
        if model_path:
            # Convert to absolute path if it's a relative path
            if not os.path.isabs(model_path):
                model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
            
            if os.path.exists(model_path):
                self.load_model(model_path)
                logger.info(f"Loaded pattern recognition model from {model_path}")
        else:
            logger.warning("No model file found, using untrained model")
    
    def load_model(self, model_path: str):
        """
        Load model from file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            # Try to load the model state dict
            state_dict = torch.load(model_path, map_location=device)
            
            # Check if the state dict is compatible with the model
            try:
                # Check if we need to modify the state dict to match our model
                if "fc3.weight" in state_dict:
                    # The model was created with a different architecture
                    # Create a new state dict with the correct keys
                    logger.info("Converting model state dict to match current architecture")
                    
                    # Create a new state dict with only the keys we need
                    new_state_dict = {}
                    
                    # Copy the convolutional and batch norm layers if they exist
                    for key in ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", 
                               "conv3.weight", "conv3.bias", "bn1.weight", "bn1.bias", 
                               "bn1.running_mean", "bn1.running_var", "bn2.weight", "bn2.bias", 
                               "bn2.running_mean", "bn2.running_var", "bn3.weight", "bn3.bias", 
                               "bn3.running_mean", "bn3.running_var"]:
                        if key in state_dict:
                            new_state_dict[key] = state_dict[key]
                    
                    # Initialize the model with the partial state dict
                    self.model.load_state_dict(new_state_dict, strict=False)
                else:
                    # Try to load the full state dict
                    self.model.load_state_dict(state_dict)
                
                self.model.eval()
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                logger.warning("Using untrained model due to architecture mismatch")
        except Exception as e:
            logger.error(f"Error loading model file: {e}")
    
    def save_model(self, model_path: str):
        """
        Save model to file.
        
        Args:
            model_path: Path to save the model
        """
        try:
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def train(
        self, 
        train_data: List[pd.DataFrame], 
        train_labels: List[int],
        val_data: Optional[List[pd.DataFrame]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Train the model.
        
        Args:
            train_data: List of DataFrames with OHLCV data for training
            train_labels: List of pattern class labels for training
            val_data: Optional list of DataFrames with OHLCV data for validation
            val_labels: Optional list of pattern class labels for validation
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        
        Returns:
            Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Create dataset
        train_dataset = OHLCVDataset(train_data, train_labels, self.lookback)
        
        # Create validation dataset if provided
        if val_data and val_labels:
            val_dataset = OHLCVDataset(val_data, val_labels, self.lookback)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            # Split training data for validation if not provided
            val_size = int(0.2 * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Initialize metrics
            train_loss = 0.0
            train_correct = 0
            
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
                
                # Update weights
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate training metrics
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / len(train_loader.dataset)
            
            # Validate the model
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save model after training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(
            settings.models_dir,
            f"pattern_recognition_{timestamp}.pt"
        )
        self.save_model(model_save_path)
        
        return history
    
    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        val_loss = 0.0
        val_correct = 0
        
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
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        # Set model back to training mode
        self.model.train()
        
        return val_loss, val_acc
    
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
        dataset = OHLCVDataset([ohlcv_data], [0], self.lookback)  # Label doesn't matter for prediction
        
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
    
    def predict_pattern(self, ohlcv_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict the most likely pattern from OHLCV data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (pattern name, confidence)
        """
        # Get all pattern probabilities
        pattern_probs = self.predict(ohlcv_data)
        
        # Find the most likely pattern
        pattern_name = max(pattern_probs.keys(), key=pattern_probs.get)
        confidence = pattern_probs[pattern_name]
        
        return pattern_name, confidence

# Create a global instance of the model
pattern_recognition_model = PatternRecognitionModel(
    model_path=settings.model.pattern_model_path,
    lookback=20
)
