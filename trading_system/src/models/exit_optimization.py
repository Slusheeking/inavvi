"""
Exit optimization model for determining optimal exit points.

This model uses reinforcement learning to optimize trade exit decisions
based on price action, indicators, and position state.
"""
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.config.settings import settings
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("exit_optimization")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define actions
ACTIONS = {
    0: "hold",        # No action, continue holding position
    1: "exit_partial", # Exit 1/3 of the position
    2: "exit_half",   # Exit 1/2 of the position
    3: "exit_full"    # Exit the entire position
}

class PolicyNetwork(nn.Module):
    """
    Policy network for the reinforcement learning agent.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_actions: int = len(ACTIONS)):
        """
        Initialize the policy network.
        
        Args:
            input_dim: Input dimension (number of features)
            hidden_dim: Hidden layer dimension
            num_actions: Number of possible actions
        """
        super(PolicyNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Action probabilities of shape [batch_size, num_actions]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class ExitOptimizationModel:
    """
    Reinforcement learning model for optimizing trade exits.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the exit optimization model.
        
        Args:
            model_path: Path to the model file
        """
        # Define state features
        self.feature_names = [
            # Position features
            'profit_pct',        # Current profit percentage
            'time_in_trade',     # Time in trade (normalized)
            
            # Price features
            'price_to_entry',    # Current price relative to entry
            'price_to_high',     # Current price relative to high since entry
            'price_to_low',      # Current price relative to low since entry
            
            # Technical indicators
            'rsi_14',            # RSI (14)
            'bb_position',       # Position within Bollinger Bands
            'macd_histogram',    # MACD histogram
            
            # Volatility
            'volatility_5d',     # 5-day volatility
            
            # Volume
            'volume_ratio_5',    # Volume relative to 5-day average
        ]
        
        # Initialize policy network
        self.policy_network = PolicyNetwork(input_dim=len(self.feature_names))
        self.policy_network.to(device)
        
        # Load model if path is provided
        if model_path:
            # Convert to absolute path if it's a relative path
            if not os.path.isabs(model_path):
                model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
            
            if os.path.exists(model_path):
                self.load_model(model_path)
                logger.info(f"Loaded exit optimization model from {model_path}")
        else:
            logger.warning("No model file found, using untrained model")
        
        # Set to evaluation mode
        self.policy_network.eval()
    
    def load_model(self, model_path: str):
        """
        Load model from file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            # Load model state
            self.policy_network.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def save_model(self, model_path: str):
        """
        Save model to file.
        
        Args:
            model_path: Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model state
            torch.save(self.policy_network.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def train_ppo(
        self,
        training_data: List[Dict],
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.0005,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4
    ):
        """
        Train the model using Proximal Policy Optimization (PPO).
        
        Args:
            training_data: List of training episodes
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            gamma: Discount factor
            eps_clip: PPO clipping parameter
            k_epochs: Number of epochs per batch
            
        Returns:
            Training metrics
        """
        # Set model to training mode
        self.policy_network.train()
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # Initialize training metrics
        metrics = {
            'epoch_losses': [],
            'epoch_rewards': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Initialize epoch metrics
            epoch_loss = 0
            epoch_reward = 0
            
            # Process training data in batches
            for i in range(0, len(training_data), batch_size):
                # Get batch
                batch = training_data[i:i+batch_size]
                
                # Process each episode in the batch
                for episode in batch:
                    # Get episode data
                    states = episode['states']
                    actions = episode['actions']
                    rewards = episode['rewards']
                    
                    # Skip invalid episodes
                    if len(states) == 0 or len(actions) == 0 or len(rewards) == 0:
                        continue
                    
                    # Convert to tensors
                    states = torch.FloatTensor(states).to(device)
                    actions = torch.LongTensor(actions).to(device)
                    
                    # Calculate discounted rewards
                    discounted_rewards = []
                    R = 0
                    for r in reversed(rewards):
                        R = r + gamma * R
                        discounted_rewards.insert(0, R)
                    
                    # Normalize rewards
                    discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
                    if len(discounted_rewards) > 1:
                        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
                    
                    # Get old action probabilities
                    old_probs = []
                    for s, a in zip(states, actions):
                        # Get action probabilities
                        with torch.no_grad():
                            action_probs = self.policy_network(s.unsqueeze(0))
                            old_probs.append(action_probs[0][a].item())
                    
                    old_probs = torch.FloatTensor(old_probs).to(device)
                    
                    # PPO update
                    for _ in range(k_epochs):
                        # Get current action probabilities
                        action_probs = self.policy_network(states)
                        dist = Categorical(action_probs)
                        
                        # Get current log probabilities
                        curr_log_probs = dist.log_prob(actions)
                        
                        # Get entropy
                        entropy = dist.entropy().mean()
                        
                        # Calculate ratio
                        ratios = torch.exp(curr_log_probs - torch.log(old_probs + 1e-10))
                        
                        # Calculate surrogate loss
                        surr1 = ratios * discounted_rewards
                        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * discounted_rewards
                        
                        # Calculate total loss
                        loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                        
                        # Backward pass
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Update metrics
                        epoch_loss += loss.item()
                    
                    # Update reward metric
                    epoch_reward += sum(rewards)
            
            # Calculate average metrics
            avg_loss = epoch_loss / len(training_data)
            avg_reward = epoch_reward / len(training_data)
            
            # Update metrics
            metrics['epoch_losses'].append(avg_loss)
            metrics['epoch_rewards'].append(avg_reward)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
        
        # Save model after training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(
            settings.models_dir,
            f"exit_optimization_{timestamp}.pt"
        )
        self.save_model(model_save_path)
        
        # Also save as default model
        self.save_model(settings.model.exit_model_path)
        
        # Set back to evaluation mode
        self.policy_network.eval()
        
        return metrics
    
    def _extract_features(self, ohlcv_data: pd.DataFrame, position_data: Dict) -> np.ndarray:
        """
        Extract features for the model.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            position_data: Dictionary with position information
            
        Returns:
            Feature array
        """
        # Check that data contains required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in ohlcv_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return np.zeros(len(self.feature_names))
        
        # Check that we have enough data
        if len(ohlcv_data) < 20:
            logger.error(f"Not enough data: {len(ohlcv_data)} < 20")
            return np.zeros(len(self.feature_names))
        
        # Get the most recent data
        df = ohlcv_data.copy()
        
        # Get current price
        current_price = df['close'].iloc[-1]
        
        # Get entry information
        entry_price = position_data.get('entry_price', current_price)
        entry_time = position_data.get('entry_time', df.index[0])
        position_size = position_data.get('position_size', 1.0)
        
        # Get time since entry
        if isinstance(entry_time, str):
            try:
                entry_time = pd.Timestamp(entry_time)
            except:
                entry_time = df.index[0]
        
        time_in_trade = (df.index[-1] - entry_time).total_seconds() / 86400  # Convert to days
        time_in_trade = min(time_in_trade, 1.0)  # Cap at 1 day
        
        # Calculate profit percentage
        profit_pct = (current_price / entry_price - 1) * 100
        
        # Calculate price relatives
        high_since_entry = df['high'].max()
        low_since_entry = df['low'].min()
        
        price_to_entry = current_price / entry_price - 1
        price_to_high = current_price / high_since_entry - 1
        price_to_low = current_price / low_since_entry - 1
        
        # Calculate technical indicators
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        
        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-8)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        # Volatility
        df['returns'] = df['close'].pct_change()
        volatility_5d = df['returns'].rolling(5).std().iloc[-1] * 100  # Convert to percentage
        
        # Volume
        volume_sma_5 = df['volume'].rolling(5).mean()
        volume_ratio_5 = df['volume'].iloc[-1] / volume_sma_5.iloc[-1] if volume_sma_5.iloc[-1] else 1.0
        
        # Collect features
        features = [
            profit_pct,
            time_in_trade,
            price_to_entry,
            price_to_high,
            price_to_low,
            rsi_14.iloc[-1],
            bb_position,
            macd_histogram.iloc[-1],
            volatility_5d,
            volume_ratio_5
        ]
        
        # Normalize and clip features
        
        # Clip profit percentage to reasonable range
        profit_pct = np.clip(profit_pct, -20, 20)
        
        # Normalize RSI to [0, 1]
        rsi_14 = np.clip(rsi_14.iloc[-1] / 100, 0, 1)
        
        # Clip BB position to [0, 1]
        bb_position = np.clip(bb_position, 0, 1)
        
        # Clip MACD histogram
        macd_hist = np.clip(macd_histogram.iloc[-1], -1, 1)
        
        # Clip volume ratio
        volume_ratio_5 = np.clip(volume_ratio_5, 0, 5)
        
        # Replace with normalized values
        features = [
            profit_pct / 20,  # Scale to [-1, 1]
            time_in_trade,    # Already [0, 1]
            np.clip(price_to_entry, -1, 1),
            np.clip(price_to_high, -1, 0),
            np.clip(price_to_low, 0, 1),
            rsi_14,           # Already [0, 1]
            bb_position,      # Already [0, 1]
            macd_hist,        # Already [-1, 1]
            np.clip(volatility_5d / 5, 0, 1),  # Scale to [0, 1]
            volume_ratio_5 / 5  # Scale to [0, 1]
        ]
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        return features
    
    def predict_exit_action(
        self, 
        ohlcv_data: pd.DataFrame, 
        position_data: Dict
    ) -> Dict[str, Union[str, float]]:
        """
        Predict the optimal exit action.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            position_data: Dictionary with position information
            
        Returns:
            Dictionary with action and probabilities
        """
        # Extract features
        features = self._extract_features(ohlcv_data, position_data)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = self.policy_network(features_tensor)
        
        # Get action with highest probability
        action_idx = torch.argmax(action_probs, dim=1).item()
        action_name = ACTIONS[action_idx]
        
        # Convert probabilities to dictionary
        probs_dict = {ACTIONS[i]: float(prob) for i, prob in enumerate(action_probs[0].cpu().numpy())}
        
        # Create result
        result = {
            'action': action_name,
            'confidence': float(action_probs[0][action_idx].cpu().numpy()),
            'probabilities': probs_dict
        }
        
        return result
    
    def evaluate_exit_conditions(
        self, 
        ohlcv_data: pd.DataFrame, 
        position_data: Dict,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Evaluate exit conditions for a position.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            position_data: Dictionary with position information
            confidence_threshold: Minimum confidence for taking action
            
        Returns:
            Dictionary with exit recommendation
        """
        # Get model prediction
        prediction = self.predict_exit_action(ohlcv_data, position_data)
        
        # Get exit recommendation
        action = prediction['action']
        confidence = prediction['confidence']
        
        # Determine recommendation
        if action == "hold" or confidence < confidence_threshold:
            recommendation = {
                "exit": False,
                "size": 0.0,
                "reason": "hold_position",
                "confidence": confidence
            }
        elif action == "exit_partial":
            recommendation = {
                "exit": True,
                "size": 0.33,  # Exit 1/3 of position
                "reason": "partial_exit",
                "confidence": confidence
            }
        elif action == "exit_half":
            recommendation = {
                "exit": True,
                "size": 0.5,  # Exit half of position
                "reason": "half_exit",
                "confidence": confidence
            }
        elif action == "exit_full":
            recommendation = {
                "exit": True,
                "size": 1.0,  # Exit full position
                "reason": "full_exit",
                "confidence": confidence
            }
        else:
            recommendation = {
                "exit": False,
                "size": 0.0,
                "reason": "unknown_action",
                "confidence": confidence
            }
        
        # Add prediction details
        recommendation["prediction"] = prediction
        
        # Check manual exit conditions
        recommendation.update(self._check_manual_exit_conditions(ohlcv_data, position_data))
        
        return recommendation
    
    def _check_manual_exit_conditions(
        self, 
        ohlcv_data: pd.DataFrame, 
        position_data: Dict
    ) -> Dict[str, bool]:
        """
        Check manual exit conditions.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            position_data: Dictionary with position information
            
        Returns:
            Dictionary with manual exit flags
        """
        # Initialize results
        results = {
            "stop_loss_triggered": False,
            "take_profit_triggered": False,
            "trailing_stop_triggered": False,
            "time_stop_triggered": False
        }
        
        # Get current price
        current_price = ohlcv_data['close'].iloc[-1]
        
        # Get position details
        entry_price = position_data.get('entry_price', current_price)
        entry_time = position_data.get('entry_time', ohlcv_data.index[0])
        stop_loss = position_data.get('stop_loss', 0.0)
        take_profit = position_data.get('take_profit', 0.0)
        trailing_stop = position_data.get('trailing_stop', 0.0)
        max_time = position_data.get('max_time', 0.0)  # In hours
        
        # Check stop loss
        if stop_loss > 0 and current_price <= stop_loss:
            results["stop_loss_triggered"] = True
        
        # Check take profit
        if take_profit > 0 and current_price >= take_profit:
            results["take_profit_triggered"] = True
        
        # Check trailing stop
        if trailing_stop > 0:
            # Calculate highest price since entry
            high_since_entry = ohlcv_data['high'].max()
            
            # Calculate trailing stop price
            trail_price = high_since_entry * (1 - trailing_stop / 100)
            
            if current_price <= trail_price:
                results["trailing_stop_triggered"] = True
        
        # Check time stop
        if max_time > 0:
            # Get time since entry
            if isinstance(entry_time, str):
                try:
                    entry_time = pd.Timestamp(entry_time)
                except:
                    entry_time = ohlcv_data.index[0]
            
            time_diff = (ohlcv_data.index[-1] - entry_time).total_seconds() / 3600  # Convert to hours
            
            if time_diff >= max_time:
                results["time_stop_triggered"] = True
        
        return results

# Create a global instance of the model
exit_optimization_model = ExitOptimizationModel(
    model_path=settings.model.exit_model_path
)
