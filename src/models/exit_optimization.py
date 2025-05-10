"""
Exit optimization model for determining optimal exit points.

This model uses advanced reinforcement learning (SAC) to optimize trade exit decisions
based on price action, indicators, position state, and market context.
"""
import json
import os
import pickle
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.distributions import Normal
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler # Import StandardScaler

# Import RAPIDS libraries
try:
    import cudf
    import cupy as cp
    from cuml.preprocessing import StandardScaler as CumlStandardScaler
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

from src.config.settings import settings
from src.models.pattern_recognition import pattern_recognition_model
from src.utils.logging import setup_logger, log_execution_time

# Set up logger
logger = setup_logger("exit_optimization")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() and settings.advanced.use_gpu else "cpu")
logger.info(f"Using device: {device}")

# Enable RAPIDS acceleration for pandas operations where possible
if RAPIDS_AVAILABLE:
    try:
        cudf.pandas_accelerator()
        logger.info("RAPIDS pandas accelerator enabled for exit_optimization")
    except Exception as e:
        logger.warning(f"Could not enable RAPIDS pandas accelerator: {e}")

# Legacy model for compatibility with older saved models
class LegacyModel(nn.Module):
    """
    Legacy model architecture to support loading older model formats.
    This recreates the structure that was used when the model was saved.
    """
    def __init__(self, state_dim=10, action_dim=4, hidden_dim=128):
        super(LegacyModel, self).__init__()
        # Recreating the architecture of the saved model with fc1, fc2, fc3 layers
        # Using fixed dimensions based on the checkpoint: [128, 10], [128, 128], [4, 128]
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define actions
ACTIONS = {
    0: "hold",
    1: "exit_quarter",
    2: "exit_third",
    3: "exit_half",
    4: "exit_full",
}

# Continuous action space scaling
ACTION_SCALE = torch.tensor(1.0).to(device)
ACTION_BIAS = torch.tensor(0.5).to(device)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for SAC algorithm.
    Uses both a stochastic policy (the actor) and a value function (the critic).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.actor(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def get_action(self, state: torch.Tensor, evaluation: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        if evaluation:
            action = torch.tanh(mean) * ACTION_SCALE + ACTION_BIAS
            return action, None
        normal = Normal(mean, std)
        x = normal.rsample()
        log_prob = normal.log_prob(x)
        action = torch.tanh(x) * ACTION_SCALE + ACTION_BIAS
        log_prob -= torch.log(ACTION_SCALE * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def get_critic_value(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_action = torch.cat([state, action], 1)
        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)
        return q1, q2

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.value(state)


class ReplayBuffer:
    """Experience replay buffer for reinforcement learning."""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), np.array(action), np.array(reward), np.concatenate(next_state), np.array(done)

    def __len__(self) -> int:
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent for reinforcement learning.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 1e-4, # Reduced learning rate
        batch_size: int = 256,
        automatic_entropy_tuning: bool = True,
    ):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.target_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        for target_param, param in zip(self.target_value.parameters(), self.actor_critic.value.parameters()):
            target_param.data.copy_(param.data)
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(
            list(self.actor_critic.critic1.parameters()) + list(self.actor_critic.critic2.parameters()), lr=lr
        )
        self.value_optimizer = Adam(self.actor_critic.value.parameters(), lr=lr)
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.tensor(action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.training_info = {
            "actor_losses": [],
            "critic_losses": [],
            "value_losses": [],
            "alpha_losses": [],
            "entropies": [],
            "rewards": [],
        }

    def select_action(self, state: np.ndarray, evaluate: bool = False, scaler: Optional[StandardScaler] = None) -> Union[int, np.ndarray]:
        # Apply scaler if provided
        if scaler:
            state = scaler.transform(state.reshape(1, -1)).flatten()

        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor_critic.get_action(state, evaluation=evaluate)
        if evaluate:
            # Ensure action is within the expected range [0, 1] before mapping to discrete actions
            action_value = torch.clamp(action.item(), 0.0, 1.0)
            discrete_action = torch.round(action_value * 4).int().item()
            discrete_action = min(max(discrete_action, 0), 4)
            return discrete_action
        return action.cpu().data.numpy().flatten()

    def update(self, batch_size: Optional[int] = None) -> Tuple[float, float, float, float]:
        if batch_size is None:
            batch_size = self.batch_size
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)
        with torch.no_grad():
            new_actions, log_probs = self.actor_critic.get_action(state)
            q1, q2 = self.actor_critic.get_critic_value(state, new_actions)
            target_q = torch.min(q1, q2) - self.alpha * log_probs
        value = self.actor_critic.get_value(state)
        value_loss = F.mse_loss(value, target_q)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        with torch.no_grad():
            target_value = self.target_value(next_state)
            target_q = reward + (1 - done) * self.gamma * target_value
        current_q1, current_q2 = self.actor_critic.get_critic_value(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        new_actions, log_probs = self.actor_critic.get_action(state)
        q1, q2 = self.actor_critic.get_critic_value(state, new_actions)
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - min_q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        alpha_loss = torch.tensor(0.0)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        for target_param, param in zip(self.target_value.parameters(), self.actor_critic.value.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        self.training_info["actor_losses"].append(actor_loss.item())
        self.training_info["critic_losses"].append(critic_loss.item())
        self.training_info["value_losses"].append(value_loss.item())
        self.training_info["alpha_losses"].append(alpha_loss.item() if self.automatic_entropy_tuning else 0.0)
        return value_loss.item(), critic_loss.item(), actor_loss.item(), alpha_loss.item() if self.automatic_entropy_tuning else 0.0

    def save(self, path: str) -> bool:
        """
        Save model to disk with explicit versioning.
        
        Args:
            path: Path to save the model
            
        Returns:
            bool: True if saving was successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save with version information to help with future loading
            state_dim = self.actor_critic.critic1[0].in_features - self.actor_critic.mean.out_features
            action_dim = self.actor_critic.mean.out_features
            
            checkpoint = {
                'model_version': 'v2',  # Increment this when architecture changes
                'architecture': 'ActorCritic',  # Name of the architecture
                'state_dim': state_dim,
                'action_dim': action_dim,
                'hidden_dim': 256,  # Default hidden dimension
                'actor_critic_state_dict': self.actor_critic.state_dict(),
                'target_value_state_dict': self.target_value.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                'alpha': self.alpha,
                'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
                'training_info': self.training_info,
                'datetime': datetime.now().isoformat(),
            }
            
            # Remove None values
            checkpoint = {k: v for k, v in checkpoint.items() if v is not None}
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path} with version information")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: str) -> bool:
        """
        Load model from disk with support for different architectures.
        
        Args:
            path: Path to the saved model
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found at {path}")
                logger.info("Initializing with default values")
                return False
            
            # Load the checkpoint to examine its structure
            checkpoint = torch.load(path, map_location=device)
            
            # Check if it's the legacy format with fc1, fc2, fc3
            has_legacy_format = any('fc1' in key for key in checkpoint.keys())
            
            if has_legacy_format:
                logger.info("Detected legacy model format. Using compatibility loader.")
                
                # Extract state_dim and action_dim from the model architecture
                # For the actor_critic model, we can determine these from the layer dimensions
                state_dim = self.actor_critic.actor[0].in_features
                action_dim = self.actor_critic.mean.out_features
                
                # Option 1: Load the legacy model, then transfer its weights to the new model
                try:
                    # Create a legacy model instance with dimensions from our current model
                    legacy_model = LegacyModel(state_dim, action_dim)
                    legacy_model = legacy_model.to(device)
                    
                    # Direct load into legacy model
                    legacy_model.load_state_dict(checkpoint)
                    logger.info(f"Successfully loaded legacy model with state_dim={state_dim}, action_dim={action_dim}")
                    
                    # Now transfer key components to your new model architecture
                    with torch.no_grad():
                        # Map fc1 weights to actor first layer
                        self.actor_critic.actor[0].weight.copy_(legacy_model.fc1.weight)
                        self.actor_critic.actor[0].bias.copy_(legacy_model.fc1.bias)
                        
                        # Map fc2 weights to actor second layer 
                        self.actor_critic.actor[2].weight.copy_(legacy_model.fc2.weight)
                        self.actor_critic.actor[2].bias.copy_(legacy_model.fc2.bias)
                        
                        # Map fc3 weights to mean layer (output layer)
                        self.actor_critic.mean.weight.copy_(legacy_model.fc3.weight)
                        self.actor_critic.mean.bias.copy_(legacy_model.fc3.bias)
                        
                        # Initialize log_std with small values
                        self.actor_critic.log_std.weight.fill_(-5)
                        self.actor_critic.log_std.bias.fill_(-5)
                        
                    logger.info("Transferred weights from legacy model to new architecture")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error during legacy model loading: {e}")
                    
                    # If we can extract dimensions from the checkpoint directly
                    if isinstance(checkpoint, dict) and 'fc1.weight' in checkpoint:
                        try:
                            # Get dimensions from the weights in the checkpoint
                            fc1_weight = checkpoint['fc1.weight']
                            state_dim = fc1_weight.shape[1]  # Input dimension
                            
                            if 'fc3.weight' in checkpoint:
                                fc3_weight = checkpoint['fc3.weight']
                                action_dim = fc3_weight.shape[0]  # Output dimension
                            else:
                                # Default to 1 if we can't determine
                                action_dim = 1
                                
                            logger.info(f"Extracted dimensions from checkpoint: state_dim={state_dim}, action_dim={action_dim}")
                            
                            # Create a new model instance with the legacy architecture
                            self.actor_critic = LegacyModel(state_dim, action_dim).to(device)
                            
                            # Load the saved weights directly
                            self.actor_critic.load_state_dict(checkpoint)
                            logger.info("Successfully loaded by rebuilding model architecture")
                            
                            # Flag that we're using a legacy model so other components can adapt
                            self.using_legacy_model = True
                            
                            return True
                        except Exception as nested_e:
                            logger.error(f"Error rebuilding model architecture: {nested_e}")
                    else:
                        logger.error("Could not determine model dimensions from checkpoint")
                    
                    # Initialize with default values if all else fails
                    logger.info("Initializing with default values")
                    return False
            
            # If it's not a legacy format, try the normal loading process
            if "actor_critic_state_dict" in checkpoint:
                # If they're saved together (original expected format)
                self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
                logger.info("Loaded combined actor-critic model")
            elif "model_state_dict" in checkpoint:
                # Another common convention
                self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded model from 'model_state_dict'")
            else:
                # As a last resort, try loading the entire checkpoint as the state dict
                logger.warning("No recognized state dict keys found. Attempting direct load.")
                # Print keys to debug
                logger.info(f"Available keys in checkpoint: {list(checkpoint.keys())}")
                
                if isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
                    try:
                        self.actor_critic.load_state_dict(checkpoint)
                        logger.info("Successfully loaded using direct approach")
                    except Exception as e:
                        logger.error(f"Direct loading failed: {e}")
                        logger.info("Initializing with default values")
                        return False
                else:
                    logger.error("Checkpoint is not a valid state dict")
                    logger.info("Initializing with default values")
                    return False
            
            # Load other components with error handling
            try:
                if "target_value_state_dict" in checkpoint:
                    self.target_value.load_state_dict(checkpoint["target_value_state_dict"])
            except (KeyError, RuntimeError) as e:
                logger.warning(f"Could not load target_value: {e}")
                
            try:
                if "actor_optimizer_state_dict" in checkpoint:
                    self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
                if "critic_optimizer_state_dict" in checkpoint:
                    self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
                if "value_optimizer_state_dict" in checkpoint:
                    self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
            except (KeyError, RuntimeError) as e:
                logger.warning(f"Could not load optimizer state: {e}")
                
            if "alpha" in checkpoint:
                self.alpha = checkpoint["alpha"]
                
            if self.automatic_entropy_tuning and "log_alpha" in checkpoint:
                self.log_alpha = checkpoint["log_alpha"]
                if "alpha_optimizer_state_dict" in checkpoint:
                    try:
                        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
                    except RuntimeError as e:
                        logger.warning(f"Could not load alpha optimizer: {e}")
                        
            if "training_info" in checkpoint:
                self.training_info = checkpoint["training_info"]
                
            logger.info("Model loaded successfully with available components")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Initializing with default values")
            return False


class ExitOptimizationModel:
    """
    Advanced reinforcement learning model for optimizing trade exits.
    Implements Soft Actor-Critic algorithm for robust learning.
    """

    def __init__(self, model_path: Optional[str] = None, use_sac: bool = True):
        self.feature_names = [
            "profit_pct",
            "time_in_trade",
            "position_size",
            "price_to_entry",
            "price_to_high",
            "price_to_low",
            "rsi_14",
            "bb_position",
            "macd_histogram",
            "volatility_5d",
            "atr",
            "volume_ratio_5",
            "market_trend",
            "sector_trend",
            "sharpe_ratio",
            "max_drawdown",
        ]
        self.state_dim = len(self.feature_names)
        self.action_dim = 1
        self.use_sac = use_sac
        if self.use_sac:
            self.agent = SACAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=256,
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                lr=3e-4,
                batch_size=256,
                automatic_entropy_tuning=True,
            )
        else:
            self.actor = None
            self.critic = None
        self.version = "1.0.0-SAC" if self.use_sac else "0.9.0-PPO"
        self.metadata = {
            "features": self.feature_names,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "algorithm": "SAC" if self.use_sac else "PPO",
            "version": self.version,
            "last_updated": datetime.now().isoformat(),
        }
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "avg_reward": 0.0,
            "best_reward": 0.0,
            "avg_sharpe": 0.0,
            "win_rate": 0.0,
        }
        self.performance_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_profit_loss": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
        }
        if model_path:
            if not os.path.isabs(model_path):
                model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
            if os.path.exists(model_path):
                self.load_model(model_path)
                logger.info(f"Loaded exit optimization model from {model_path}")
        else:
            logger.warning("No model file provided, using untrained model")

    def save_model(self, model_path: str):
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            if self.use_sac:
                self.agent.save(model_path)
                metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(
                        {
                            "metadata": self.metadata,
                            "training_stats": self.training_stats,
                            "performance_metrics": self.performance_metrics,
                        },
                        f,
                        indent=2,
                    )
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, model_path: str):
        try:
            self.agent.load(model_path)
            self.use_sac = True
            metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", self.metadata)
                    self.training_stats = data.get("training_stats", self.training_stats)
                    self.performance_metrics = data.get("performance_metrics", self.performance_metrics)
            logger.info(f"Loaded SAC model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def train_sac(
        self,
        training_data: List[Dict],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        updates_per_step: int = 1,
        start_steps: int = 1000,
        eval_interval: int = 5000,
    ) -> Dict[str, List[float]]:
        epochs = epochs or settings.model.epochs
        batch_size = batch_size or settings.model.batch_size
        lr = lr or settings.model.learning_rate
        self.agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=lr,
            batch_size=batch_size,
            automatic_entropy_tuning=True,
        )
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        
        # Collect all states from training data to fit the scaler
        all_states = []
        for episode in training_data:
            all_states.extend(episode.get("states", []))
            
        if all_states:
            self.scaler.fit(np.array(all_states))
            logger.info(f"Fitted StandardScaler on {len(all_states)} states")
        else:
            logger.warning("No states found in training data to fit scaler")
            self.scaler = None # Ensure scaler is None if no data
            
        metrics = {
            "epoch_rewards": [],
            "epoch_losses": [],
            "epoch_value_losses": [],
            "epoch_critic_losses": [],
            "epoch_actor_losses": [],
            "epoch_alpha_losses": [],
            "epoch_entropies": [],
            "sharpe_ratios": [],
            "max_drawdowns": [],
            "win_rates": [],
        }
        total_steps = 0
        logger.info(f"Starting SAC training for {epochs} epochs")
        for epoch in range(epochs):
            epoch_reward = 0
            epoch_steps = 0
            epoch_profits = []
            random.shuffle(training_data)
            for episode_idx, episode in enumerate(training_data):
                states = episode.get("states", [])
                if not states:
                    continue
                episode_reward = 0
                position_size = 1.0
                entry_price = episode.get("entry_price", 100.0)
                current_price = entry_price
                highest_price = entry_price
                profit_pct = 0.0
                episode_done = False
                state = states[0].copy() if states else np.zeros(self.state_dim)
                for step_idx in range(len(states) - 1):
                    state = states[step_idx].copy()
                    state[0] = profit_pct
                    state[2] = position_size
                    if total_steps < start_steps:
                        action = np.random.uniform(0, 1, 1)[0]
                    else:
                        action = self.agent.select_action(state)
                    exit_size = float(action[0]) if isinstance(action, np.ndarray) else float(action)
                    exit_size = max(0.0, min(1.0, exit_size))
                    actual_exit = min(position_size, exit_size)
                    new_position_size = position_size - actual_exit
                    next_state = states[step_idx + 1].copy()
                    current_price = entry_price * (1 + next_state[3])
                    highest_price = max(highest_price, current_price)
                    reward = self._calculate_risk_adjusted_reward(
                        exit_size=actual_exit,
                        position_size=position_size,
                        new_position_size=new_position_size,
                        current_price=current_price,
                        entry_price=entry_price,
                        highest_price=highest_price,
                        state=state,
                        next_state=next_state,
                    )
                    profit_pct = (current_price / entry_price - 1) * 100
                    done = new_position_size <= 0 or step_idx == len(states) - 2
                    if done:
                        episode_done = True
                    next_state[0] = profit_pct
                    next_state[2] = new_position_size
                    self.agent.replay_buffer.push(state, np.array([exit_size]), reward, next_state, done)
                    # Scale states before pushing to replay buffer
                    scaled_state = self.scaler.transform(state.reshape(1, -1)).flatten() if self.scaler else state
                    scaled_next_state = self.scaler.transform(next_state.reshape(1, -1)).flatten() if self.scaler else next_state
                    
                    self.agent.replay_buffer.push(scaled_state, np.array([exit_size]), reward, scaled_next_state, done)
                    
                    if len(self.agent.replay_buffer) > batch_size:
                        for _ in range(updates_per_step):
                            # Pass scaler to update method if needed (though update works on already scaled data)
                            value_loss, critic_loss, actor_loss, alpha_loss = self.agent.update(batch_size)
                            
                    state = next_state # Use original next_state for the loop logic
                    episode_reward += reward
                    epoch_reward += reward
                    epoch_steps += 1
                    total_steps += 1
                    position_size = new_position_size
                    if actual_exit > 0:
                        exit_profit = (current_price / entry_price - 1) * actual_exit
                        epoch_profits.append(exit_profit)
                    if episode_done:
                        break
                if total_steps % eval_interval == 0:
                    eval_reward = self._evaluate_model(5)
                    logger.info(f"Step {total_steps}, Eval reward: {eval_reward:.4f}")
            avg_reward = epoch_reward / max(1, epoch_steps)
            sharpe_ratio = 0.0
            win_rate = 0.0
            max_drawdown = 0.0
            if epoch_profits:
                mean_profit = np.mean(epoch_profits)
                std_profit = np.std(epoch_profits) if len(epoch_profits) > 1 else 1.0
                sharpe_ratio = mean_profit / max(std_profit, 1e-6)
                win_rate = sum(1 for p in epoch_profits if p > 0) / len(epoch_profits)
                cumulative_returns = np.cumsum(epoch_profits)
                max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            metrics["epoch_rewards"].append(avg_reward)
            metrics["epoch_losses"].append(
                np.mean(self.agent.training_info["actor_losses"][-epoch_steps:] if epoch_steps > 0 else [0])
            )
            metrics["epoch_value_losses"].append(
                np.mean(self.agent.training_info["value_losses"][-epoch_steps:] if epoch_steps > 0 else [0])
            )
            metrics["epoch_critic_losses"].append(
                np.mean(self.agent.training_info["critic_losses"][-epoch_steps:] if epoch_steps > 0 else [0])
            )
            metrics["epoch_actor_losses"].append(
                np.mean(self.agent.training_info["actor_losses"][-epoch_steps:] if epoch_steps > 0 else [0])
            )
            metrics["epoch_alpha_losses"].append(
                np.mean(self.agent.training_info["alpha_losses"][-epoch_steps:] if epoch_steps > 0 else [0])
            )
            metrics["epoch_entropies"].append(
                np.mean(self.agent.training_info["entropies"][-epoch_steps:] if epoch_steps > 0 else [0])
            )
            metrics["sharpe_ratios"].append(sharpe_ratio)
            metrics["max_drawdowns"].append(max_drawdown)
            metrics["win_rates"].append(win_rate)
            logger.info(
                f"Epoch {epoch+1}/{epochs} - Reward: {avg_reward:.4f}, "
                f"Sharpe: {sharpe_ratio:.4f}, Win Rate: {win_rate:.2f}, Alpha: {self.agent.alpha:.4f}"
            )
        self.training_stats["episodes"] = len(training_data)
        self.training_stats["total_steps"] = total_steps
        self.training_stats["avg_reward"] = np.mean(metrics["epoch_rewards"])
        self.training_stats["best_reward"] = max(metrics["epoch_rewards"])
        self.training_stats["avg_sharpe"] = np.mean(metrics["sharpe_ratios"])
        self.training_stats["win_rate"] = np.mean(metrics["win_rates"])
        self.metadata["last_updated"] = datetime.now().isoformat()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(settings.models_dir, f"exit_optimization_sac_{timestamp}.pt")
        self.save_model(model_save_path)
        self.save_model(settings.model.exit_model_path)
        return metrics

    def _evaluate_model(self, num_episodes: int = 5) -> float:
        test_episodes = []
        for _ in range(num_episodes):
            steps = 20
            prices = [100.0]
            for _ in range(steps):
                price_change = np.random.uniform(-0.01, 0.01)
                prices.append(prices[-1] * (1 + price_change))
            states = []
            entry_price = prices[0]
            for i in range(steps):
                current_price = prices[i]
                price_to_entry = current_price / entry_price - 1
                profit_pct = price_to_entry * 100
                state = np.zeros(self.state_dim)
                state[0] = profit_pct
                state[1] = i / steps
                state[2] = 1.0
                state[3] = price_to_entry
                states.append(state)
            test_episodes.append({"states": states, "entry_price": entry_price, "prices": prices})
        total_reward = 0.0
        for episode in test_episodes:
            states = episode["states"]
            entry_price = episode["entry_price"]
            prices = episode["prices"]
            episode_reward = 0.0
            position_size = 1.0
            highest_price = entry_price
            for i in range(len(states) - 1):
                state = states[i].copy()
                state[2] = position_size
                
                # Select action using the agent, passing the scaler
                action = self.agent.select_action(state, evaluate=True, scaler=self.scaler)
                
                if isinstance(action, int):
                    # Discrete action mapping
                    exit_size = [0.0, 0.25, 0.33, 0.5, 1.0][action]
                else:
                    # Continuous action value (should be between 0 and 1 after tanh and scaling)
                    exit_size = float(action)
                    # Ensure exit_size is within valid range [0, 1]
                    exit_size = max(0.0, min(1.0, exit_size))
                actual_exit = min(position_size, exit_size)
                new_position_size = position_size - actual_exit
                current_price = prices[i]
                next_price = prices[i + 1]
                highest_price = max(highest_price, current_price)
                if actual_exit > 0:
                    exit_profit = (current_price / entry_price - 1) * actual_exit * 100
                    reward = exit_profit
                else:
                    price_change = (next_price - current_price) / current_price * 100
                    reward = price_change * position_size * 0.1
                episode_reward += reward
                position_size = new_position_size
                if position_size <= 0:
                    break
            total_reward += episode_reward
        return total_reward / num_episodes

    def _calculate_risk_adjusted_reward(
        self,
        exit_size: float,
        position_size: float,
        new_position_size: float,
        current_price: float,
        entry_price: float,
        highest_price: float,
        state: np.ndarray,
        next_state: np.ndarray,
    ) -> float:
        profit_pct = (current_price / entry_price - 1) * 100
        drawdown = (highest_price - current_price) / highest_price * 100
        rsi = state[6]
        bb_position = state[7]
        volatility = state[9]
        time_factor = state[1]
        if exit_size > 0:
            actual_exit = min(position_size, exit_size)
            exit_profit = profit_pct * actual_exit
            if profit_pct > 0:
                if rsi > 0.7:
                    exit_reward = exit_profit * 1.2
                elif drawdown > 2.0 and profit_pct > 2.0:
                    exit_reward = exit_profit * 1.1
                else:
                    exit_reward = exit_profit
            else:
                if rsi < 0.3:
                    exit_reward = exit_profit * 0.8
                else:
                    exit_reward = exit_profit
            risk_reduction_reward = 0.1 * actual_exit
            reward = exit_reward + risk_reduction_reward
        else:
            price_change = next_state[3] - state[3]
            price_change_pct = price_change * 100
            holding_reward = price_change_pct * position_size
            holding_reward *= (1.0 - 0.5 * time_factor)
            risk_penalty = volatility * position_size * 0.2
            reward = holding_reward - risk_penalty
        if hasattr(self, "_episode_returns"):
            self._episode_returns.append(reward)
            if len(self._episode_returns) > 1:
                mean_return = np.mean(self._episode_returns)
                std_return = np.std(self._episode_returns)
                sharpe = mean_return / max(std_return, 1e-6)
                sharpe_factor = min(max(sharpe, 0.5), 1.5)
                reward *= sharpe_factor
        else:
            self._episode_returns = [reward]
        return reward

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        cumulative_returns = np.array(cumulative_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (running_max - cumulative_returns) / np.maximum(running_max, 1e-10)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        return max_drawdown

    def _extract_features(self, ohlcv_data: pd.DataFrame, position_data: Dict) -> np.ndarray:
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in ohlcv_data.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return np.zeros(len(self.feature_names))
        if len(ohlcv_data) < 20:
            logger.error(f"Not enough data: {len(ohlcv_data)} < 20")
            return np.zeros(len(self.feature_names))
            
        # Try to use GPU acceleration if available
        if RAPIDS_AVAILABLE:
            try:
                # Convert to cuDF DataFrame for GPU acceleration
                gpu_df = cudf.DataFrame.from_pandas(ohlcv_data.copy())
                
                # Get basic values
                current_price = float(gpu_df["close"].iloc[-1])
                entry_price = position_data.get("entry_price", current_price)
                entry_time = position_data.get("entry_time", ohlcv_data.index[0])
                position_size = position_data.get("position_size", 1.0)
                
                # Handle entry_time
                if isinstance(entry_time, str):
                    try:
                        entry_time = pd.Timestamp(entry_time)
                    except:
                        entry_time = ohlcv_data.index[0]
                
                # Calculate time in trade (using pandas for timestamp operations)
                time_in_trade = (ohlcv_data.index[-1] - entry_time).total_seconds() / 86400
                time_in_trade = min(time_in_trade, 1.0)
                
                # Calculate profit percentage
                profit_pct = (current_price / entry_price - 1) * 100
                
                # Get high and low since entry (using pandas for loc operations)
                if entry_time in ohlcv_data.index:
                    high_since_entry = float(ohlcv_data["high"].loc[entry_time:].max())
                    low_since_entry = float(ohlcv_data["low"].loc[entry_time:].min())
                else:
                    high_since_entry = float(gpu_df["high"].max())
                    low_since_entry = float(gpu_df["low"].min())
                
                # Calculate price ratios
                price_to_entry = current_price / entry_price - 1
                price_to_high = current_price / high_since_entry - 1 if high_since_entry > 0 else 0
                price_to_low = current_price / low_since_entry - 1 if low_since_entry > 0 else 0
                
                # Calculate RSI on GPU
                delta = gpu_df["close"].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                rsi_14 = 100 - (100 / (1 + rs))
                
                # Calculate Bollinger Bands on GPU
                bb_middle = gpu_df["close"].rolling(20).mean()
                bb_std = gpu_df["close"].rolling(20).std()
                bb_upper = bb_middle + 2 * bb_std
                bb_lower = bb_middle - 2 * bb_std
                bb_position = (current_price - float(bb_lower.iloc[-1])) / (float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1]) + 1e-8)
                
                # Calculate MACD on GPU
                ema_12 = gpu_df["close"].ewm(span=12, adjust=False).mean()
                ema_26 = gpu_df["close"].ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                macd_histogram = macd - macd_signal
                
                # Calculate volatility on GPU
                gpu_df["returns"] = gpu_df["close"].pct_change()
                volatility_5d = float(gpu_df["returns"].rolling(5).std().iloc[-1]) * 100
                
                # Calculate ATR on GPU
                tr1 = gpu_df["high"] - gpu_df["low"]
                tr2 = (gpu_df["high"] - gpu_df["close"].shift(1)).abs()
                tr3 = (gpu_df["low"] - gpu_df["close"].shift(1)).abs()
                
                # Need to convert to pandas for max operation across columns
                tr_df = pd.DataFrame({
                    'tr1': tr1.to_pandas(),
                    'tr2': tr2.to_pandas(),
                    'tr3': tr3.to_pandas()
                })
                tr = tr_df.max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])
                normalized_atr = atr / current_price * 100
                
                # Calculate volume metrics on GPU
                volume_sma_5 = gpu_df["volume"].rolling(5).mean()
                volume_ratio_5 = float(gpu_df["volume"].iloc[-1]) / float(volume_sma_5.iloc[-1]) if float(volume_sma_5.iloc[-1]) else 1.0
                
                # Get market and sector trends
                market_trend = 0.0
                sector_trend = 0.0
                if "spy" in position_data or "market_trend" in position_data:
                    market_trend = position_data.get("market_trend", position_data.get("spy", 0.0))
                else:
                    if len(gpu_df) >= 50:
                        market_trend = (float(gpu_df["close"].iloc[-1]) / float(gpu_df["close"].iloc[-50]) - 1) * 100 / 10
                        market_trend = max(min(market_trend, 1.0), -1.0)
                if "sector_trend" in position_data:
                    sector_trend = position_data.get("sector_trend", 0.0)
                
                # Calculate Sharpe ratio (convert to pandas for dropna)
                returns = gpu_df["returns"].to_pandas().iloc[-20:].dropna()
                sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0.0
                
                # Calculate drawdown (using cupy for GPU acceleration)
                if entry_time in ohlcv_data.index:
                    close_prices = cp.array(ohlcv_data["close"].loc[entry_time:].values)
                else:
                    close_prices = cp.array(gpu_df["close"].values)
                    
                peak = cp.maximum.accumulate(close_prices)
                drawdown = (peak - close_prices) / peak
                max_drawdown = float(cp.max(drawdown)) if len(drawdown) > 0 else 0.0
                
                # Clip values
                profit_pct = cp.clip(profit_pct, -20, 20)
                rsi_14_value = cp.clip(float(rsi_14.iloc[-1]) / 100, 0, 1)
                bb_position = cp.clip(bb_position, 0, 1)
                macd_hist = cp.clip(float(macd_histogram.iloc[-1]), -1, 1)
                volume_ratio_5 = cp.clip(volume_ratio_5, 0, 5)
                
                # Create feature array
                features = cp.array([
                    profit_pct / 20,
                    time_in_trade,
                    position_size,
                    cp.clip(price_to_entry, -1, 1),
                    cp.clip(price_to_high, -1, 0),
                    cp.clip(price_to_low, 0, 1),
                    rsi_14_value,
                    bb_position,
                    macd_hist,
                    cp.clip(volatility_5d / 5, 0, 1),
                    cp.clip(normalized_atr / 5, 0, 1),
                    volume_ratio_5 / 5,
                    cp.clip(market_trend, -1, 1),
                    cp.clip(sector_trend, -1, 1),
                    cp.clip(sharpe_ratio, -1, 1),
                    cp.clip(max_drawdown, 0, 1),
                ], dtype=cp.float32)
                
                # Check for and handle NaN or infinite values
                if cp.any(cp.isnan(features)) or cp.any(cp.isinf(features)):
                    logger.warning(f"Detected NaN or infinite values in features")
                    # Replace NaN with 0 and infinite values with a large number (or clip)
                    features = cp.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
                    # Re-clip features after handling NaN/inf
                    features[0] = cp.clip(features[0], -20/20, 20/20)  # profit_pct / 20
                    features[1] = cp.clip(features[1], 0, 1)  # time_in_trade
                    features[3] = cp.clip(features[3], -1, 1)  # price_to_entry
                    features[4] = cp.clip(features[4], -1, 0)  # price_to_high
                    features[5] = cp.clip(features[5], 0, 1)  # price_to_low
                    features[6] = cp.clip(features[6], 0, 1)  # rsi_14
                    features[7] = cp.clip(features[7], 0, 1)  # bb_position
                    features[8] = cp.clip(features[8], -1, 1)  # macd_hist
                    features[9] = cp.clip(features[9], 0, 1)  # volatility_5d / 5
                    features[10] = cp.clip(features[10], 0, 1)  # normalized_atr / 5
                    features[11] = cp.clip(features[11], 0, 5/5)  # volume_ratio_5 / 5
                    features[12] = cp.clip(features[12], -1, 1)  # market_trend
                    features[13] = cp.clip(features[13], -1, 1)  # sector_trend
                    features[14] = cp.clip(features[14], -1, 1)  # sharpe_ratio
                    features[15] = cp.clip(features[15], 0, 1)  # max_drawdown
                
                # Convert back to numpy array
                logger.debug("Used GPU acceleration for feature extraction")
                return cp.asnumpy(features)
                
            except Exception as e:
                logger.warning(f"GPU acceleration failed, falling back to pandas: {e}")
                # Fall back to pandas implementation
        
        # Pandas implementation (CPU)
        df = ohlcv_data.copy()
        current_price = df["close"].iloc[-1]
        entry_price = position_data.get("entry_price", current_price)
        entry_time = position_data.get("entry_time", df.index[0])
        position_size = position_data.get("position_size", 1.0)
        if isinstance(entry_time, str):
            try:
                entry_time = pd.Timestamp(entry_time)
            except:
                entry_time = df.index[0]
        time_in_trade = (df.index[-1] - entry_time).total_seconds() / 86400
        time_in_trade = min(time_in_trade, 1.0)
        profit_pct = (current_price / entry_price - 1) * 100
        high_since_entry = df["high"].loc[entry_time:].max() if entry_time in df.index else df["high"].max()
        low_since_entry = df["low"].loc[entry_time:].min() if entry_time in df.index else df["low"].min()
        price_to_entry = current_price / entry_price - 1
        price_to_high = current_price / high_since_entry - 1 if high_since_entry > 0 else 0
        price_to_low = current_price / low_since_entry - 1 if low_since_entry > 0 else 0
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))
        bb_middle = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-8)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal
        df["returns"] = df["close"].pct_change()
        volatility_5d = df["returns"].rolling(5).std().iloc[-1] * 100
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        normalized_atr = atr / current_price * 100
        volume_sma_5 = df["volume"].rolling(5).mean()
        volume_ratio_5 = df["volume"].iloc[-1] / volume_sma_5.iloc[-1] if volume_sma_5.iloc[-1] else 1.0
        market_trend = 0.0
        sector_trend = 0.0
        if "spy" in position_data or "market_trend" in position_data:
            market_trend = position_data.get("market_trend", position_data.get("spy", 0.0))
        else:
            if len(df) >= 50:
                market_trend = (df["close"].iloc[-1] / df["close"].iloc[-50] - 1) * 100 / 10
                market_trend = max(min(market_trend, 1.0), -1.0)
        if "sector_trend" in position_data:
            sector_trend = position_data.get("sector_trend", 0.0)
        returns = df["returns"].iloc[-20:].dropna()
        sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0.0
        close_prices = df["close"].loc[entry_time:].values if entry_time in df.index else df["close"].values
        peak = np.maximum.accumulate(close_prices)
        drawdown = (peak - close_prices) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        profit_pct = np.clip(profit_pct, -20, 20)
        rsi_14 = np.clip(rsi_14.iloc[-1] / 100, 0, 1)
        bb_position = np.clip(bb_position, 0, 1)
        macd_hist = np.clip(macd_histogram.iloc[-1], -1, 1)
        volume_ratio_5 = np.clip(volume_ratio_5, 0, 5)
        features = [
            profit_pct / 20,
            time_in_trade,
            position_size,
            np.clip(price_to_entry, -1, 1),
            np.clip(price_to_high, -1, 0),
            np.clip(price_to_low, 0, 1),
            rsi_14,
            bb_position,
            macd_hist,
            np.clip(volatility_5d / 5, 0, 1),
            np.clip(normalized_atr / 5, 0, 1),
            volume_ratio_5 / 5,
            np.clip(market_trend, -1, 1),
            np.clip(sector_trend, -1, 1),
            np.clip(sharpe_ratio, -1, 1),
            np.clip(max_drawdown, 0, 1),
        ]
        # Check for and handle NaN or infinite values
        features = np.array(features, dtype=np.float32)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning(f"Detected NaN or infinite values in features: {features}")
            # Replace NaN with 0 and infinite values with a large number (or clip)
            features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            # Re-clip features after handling NaN/inf if necessary, based on original clipping ranges
            features[0] = np.clip(features[0], -20/20, 20/20) # profit_pct / 20
            features[1] = np.clip(features[1], 0, 1) # time_in_trade
            features[3] = np.clip(features[3], -1, 1) # price_to_entry
            features[4] = np.clip(features[4], -1, 0) # price_to_high
            features[5] = np.clip(features[5], 0, 1) # price_to_low
            features[6] = np.clip(features[6], 0, 1) # rsi_14
            features[7] = np.clip(features[7], 0, 1) # bb_position
            features[8] = np.clip(features[8], -1, 1) # macd_hist
            features[9] = np.clip(features[9], 0, 1) # volatility_5d / 5
            features[10] = np.clip(features[10], 0, 1) # normalized_atr / 5
            features[11] = np.clip(features[11], 0, 5/5) # volume_ratio_5 / 5
            features[12] = np.clip(features[12], -1, 1) # market_trend
            features[13] = np.clip(features[13], -1, 1) # sector_trend
            features[14] = np.clip(features[14], -1, 1) # sharpe_ratio
            features[15] = np.clip(features[15], 0, 1) # max_drawdown
            logger.warning(f"Handled features: {features}")
        return features

    def predict_exit_action(self, ohlcv_data: pd.DataFrame, position_data: Dict) -> Dict[str, Union[str, float]]:
        features = self._extract_features(ohlcv_data, position_data)
        features_tensor = torch.FloatTensor(features).to(device)
        if self.use_sac:
            with torch.no_grad():
                state = features_tensor.unsqueeze(0)
                action, _ = self.agent.actor_critic.get_action(state, evaluation=True)
                action_value = action.cpu().numpy()[0, 0]
                exit_size = float(action_value)
                exit_size = max(0.0, min(1.0, exit_size))
                if exit_size < 0.1:
                    action_idx = 0
                    action_name = "hold"
                elif exit_size < 0.3:
                    action_idx = 1
                    action_name = "exit_quarter"
                elif exit_size < 0.45:
                    action_idx = 2
                    action_name = "exit_third"
                elif exit_size < 0.75:
                    action_idx = 3
                    action_name = "exit_half"
                else:
                    action_idx = 4
                    action_name = "exit_full"
                result = {
                    "action": action_name,
                    "action_idx": action_idx,
                    "exit_size": exit_size,
                    "confidence": 0.8,
                    "probabilities": {
                        "hold": 1.0 - min(exit_size * 10, 1.0) if exit_size < 0.1 else 0.0,
                        "exit_quarter": 1.0 - min(abs(exit_size - 0.2) * 5, 1.0) if 0.1 <= exit_size < 0.3 else 0.0,
                        "exit_third": 1.0 - min(abs(exit_size - 0.37) * 5, 1.0) if 0.3 <= exit_size < 0.45 else 0.0,
                        "exit_half": 1.0 - min(abs(exit_size - 0.6) * 5, 1.0) if 0.45 <= exit_size < 0.75 else 0.0,
                        "exit_full": 1.0 - min((1.0 - exit_size) * 4, 1.0) if exit_size >= 0.75 else 0.0,
                    },
                }
        else:
            logger.error("PPO model not supported in this version")
            result = {
                "action": "hold",
                "action_idx": 0,
                "exit_size": 0.0,
                "confidence": 0.0,
                "probabilities": {action: 0.0 for action in ACTIONS.values()},
            }
        return result

    def evaluate_exit_conditions(
        self,
        ohlcv_data: pd.DataFrame,
        position_data: Dict,
        confidence_threshold: Optional[float] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        confidence_threshold = confidence_threshold or settings.model.confidence_threshold
        if market_data:
            position_data = self._add_market_context(position_data, market_data)
        prediction = self.predict_exit_action(ohlcv_data, position_data)
        action = prediction["action"]
        exit_size = prediction["exit_size"]
        confidence = prediction["confidence"]
        if action == "hold" or confidence < confidence_threshold:
            recommendation = {"exit": False, "size": 0.0, "reason": "hold_position", "confidence": confidence}
        elif action == "exit_quarter":
            recommendation = {"exit": True, "size": 0.25, "reason": "quarter_exit", "confidence": confidence}
        elif action == "exit_third":
            recommendation = {"exit": True, "size": 0.33, "reason": "third_exit", "confidence": confidence}
        elif action == "exit_half":
            recommendation = {"exit": True, "size": 0.5, "reason": "half_exit", "confidence": confidence}
        elif action == "exit_full":
            recommendation = {"exit": True, "size": 1.0, "reason": "full_exit", "confidence": confidence}
        else:
            recommendation = {"exit": False, "size": 0.0, "reason": "unknown_action", "confidence": confidence}
        recommendation["prediction"] = prediction
        if market_data:
            recommendation["market_context"] = self._get_market_summary(market_data)
        manual_checks = self._check_manual_exit_conditions(ohlcv_data, position_data)
        logger.info(f"Recommendation before update: {recommendation}")
        recommendation.update(manual_checks)
        logger.info(f"Recommendation after update: {recommendation}")
        logger.info(f"Recommendation before update: {recommendation}")
        recommendation.update(manual_checks)
        logger.info(f"Recommendation after update: {recommendation}")
        logger.info(f"Updated recommendation with manual checks: {recommendation}")
        if any(manual_checks.values()):
            exit_reasons = []
            if manual_checks.get("stop_loss_triggered", False):
                exit_reasons.append("stop_loss")
            if manual_checks.get("take_profit_triggered", False):
                exit_reasons.append("take_profit")
            if manual_checks.get("trailing_stop_triggered", False):
                exit_reasons.append("trailing_stop")
            if manual_checks.get("time_stop_triggered", False):
                exit_reasons.append("time_stop")
            if exit_reasons:
                recommendation["exit"] = True
                recommendation["size"] = 1.0
                recommendation["reason"] = "_".join(exit_reasons)
                recommendation["confidence"] = 1.0
        risk_metrics = self._calculate_risk_metrics(ohlcv_data, position_data)
        recommendation["risk_metrics"] = risk_metrics
        if risk_metrics["max_drawdown"] > 0.1 and risk_metrics["drawdown_ratio"] > 2.0:
            if recommendation["exit"]:
                recommendation["size"] = min(1.0, recommendation["size"] * 1.5)
                recommendation["reason"] += "_high_drawdown"
            else:
                recommendation["exit"] = True
                recommendation["size"] = 0.33
                recommendation["reason"] = "drawdown_protection"
                recommendation["confidence"] = 0.7
        return recommendation

    def _add_market_context(self, position_data: Dict, market_data: Dict[str, pd.DataFrame]) -> Dict:
        updated_data = position_data.copy()
        if "SPY" in market_data:
            spy_data = market_data["SPY"]
            if len(spy_data) > 20:
                spy_1d_return = spy_data["close"].pct_change(1).iloc[-1]
                spy_5d_return = spy_data["close"].pct_change(5).iloc[-1]
                updated_data["market_trend"] = spy_5d_return
                updated_data["market_trend_1d"] = spy_1d_return
        if "sector_etf" in position_data:
            sector_etf = position_data["sector_etf"]
            if sector_etf in market_data:
                sector_data = market_data[sector_etf]
                if len(sector_data) > 20:
                    sector_5d_return = sector_data["close"].pct_change(5).iloc[-1]
                    updated_data["sector_trend"] = sector_5d_return
        if "VIX" in market_data:
            vix_data = market_data["VIX"]
            if len(vix_data) > 1:
                updated_data["market_volatility"] = vix_data["close"].iloc[-1] / 100.0
                updated_data["market_volatility_change"] = vix_data["close"].pct_change(1).iloc[-1]
        return updated_data

    def _get_market_summary(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        summary = {}
        if "SPY" in market_data:
            spy_data = market_data["SPY"]
            if len(spy_data) > 20:
                spy_1d_return = spy_data["close"].pct_change(1).iloc[-1]
                spy_5d_return = spy_data["close"].pct_change(5).iloc[-1]
                spy_20d_return = spy_data["close"].pct_change(20).iloc[-1]
                spy_20d_ma = spy_data["close"].rolling(20).mean().iloc[-1]
                spy_price_to_ma = spy_data["close"].iloc[-1] / spy_20d_ma - 1
                summary["market"] = {
                    "trend_1d": spy_1d_return,
                    "trend_5d": spy_5d_return,
                    "trend_20d": spy_20d_return,
                    "price_to_ma": spy_price_to_ma,
                }
        if "VIX" in market_data:
            vix_data = market_data["VIX"]
            if len(vix_data) > 1:
                summary["volatility"] = {
                    "vix": vix_data["close"].iloc[-1],
                    "vix_change": vix_data["close"].pct_change(1).iloc[-1],
                }
        return summary

    def _check_manual_exit_conditions(self, ohlcv_data: pd.DataFrame, position_data: Dict) -> Dict[str, bool]:
        results = {
            "stop_loss_triggered": False,
            "take_profit_triggered": False,
            "trailing_stop_triggered": False,
            "time_stop_triggered": False,
        }
        current_price = ohlcv_data["close"].iloc[-1]
        entry_price = position_data.get("entry_price", current_price)
        entry_time = position_data.get("entry_time", ohlcv_data.index[0])
        stop_loss = position_data.get("stop_loss", 0.0)
        take_profit = position_data.get("take_profit", 0.0)
        trailing_stop = position_data.get("trailing_stop", 0.0)
        max_time = position_data.get("max_time", 0.0)
        if stop_loss > 0 and current_price <= stop_loss:
            results["stop_loss_triggered"] = True
        if take_profit > 0 and current_price >= take_profit:
            results["take_profit_triggered"] = True
        if trailing_stop > 0:
            high_since_entry = (
                ohlcv_data["high"].loc[entry_time:].max() if entry_time in ohlcv_data.index else ohlcv_data["high"].max()
            )
            trail_price = high_since_entry * (1 - trailing_stop / 100)
            if current_price <= trail_price:
                results["trailing_stop_triggered"] = True
        if max_time > 0:
            if isinstance(entry_time, str):
                try:
                    entry_time = pd.Timestamp(entry_time)
                except:
                    entry_time = ohlcv_data.index[0]
            time_diff = (ohlcv_data.index[-1] - entry_time).total_seconds() / 3600
            if time_diff >= max_time:
                results["time_stop_triggered"] = True
        logger.info(f"Manual exit checks: {results}")
        return results

    def _calculate_risk_metrics(self, ohlcv_data: pd.DataFrame, position_data: Dict) -> Dict:
        current_price = ohlcv_data["close"].iloc[-1]
        entry_price = position_data.get("entry_price", current_price)
        entry_time = position_data.get("entry_time", ohlcv_data.index[0])
        profit_pct = (current_price / entry_price - 1) * 100
        high_since_entry = (
            ohlcv_data["high"].loc[entry_time:].max() if entry_time in ohlcv_data.index else ohlcv_data["high"].max()
        )
        max_profit_pct = (high_since_entry / entry_price - 1) * 100
        drawdown = (high_since_entry - current_price) / high_since_entry * 100
        drawdown_ratio = drawdown / max(profit_pct, 0.01) if profit_pct > 0 else 0
        returns = ohlcv_data["close"].pct_change().loc[entry_time:] if entry_time in ohlcv_data.index else ohlcv_data["close"].pct_change()
        volatility = returns.std() * 100
        avg_return = returns.mean()
        sharpe_ratio = avg_return / returns.std() if returns.std() > 0 else 0
        stop_loss = position_data.get("stop_loss", entry_price * 0.95)
        initial_risk = (entry_price - stop_loss) / entry_price * 100
        r_multiple = profit_pct / initial_risk if initial_risk > 0 else 0
        return {
            "profit_pct": profit_pct,
            "max_profit_pct": max_profit_pct,
            "drawdown": drawdown,
            "drawdown_ratio": drawdown_ratio,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "r_multiple": r_multiple,
            "max_drawdown": drawdown / 100,
        }

    def backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        exit_strategy: str = "model",
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.2,
    ) -> Dict:
        results = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "trades": [],
        }
        capital = initial_capital
        equity_curve = [capital]
        trades = []
        for symbol, df in historical_data.items():
            if len(df) < 50:
                continue
            entry_threshold = 0.01
            stop_loss_pct = 0.05
            take_profit_pct = 0.1
            for i in range(20, len(df) - 20):
                entry_return = df["close"].iloc[i] / df["close"].iloc[i - 1] - 1
                if entry_return > entry_threshold:
                    entry_idx = i
                    entry_price = df["close"].iloc[entry_idx]
                    entry_time = df.index[entry_idx]
                    position_size = capital * position_size_pct / entry_price
                    stop_loss = entry_price * (1 - stop_loss_pct)
                    take_profit = entry_price * (1 + take_profit_pct)
                    position_data = {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "entry_time": entry_time,
                        "position_size": 1.0,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "trailing_stop": 2.0,
                    }
                    trade = {
                        "symbol": symbol,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "entry_idx": entry_idx,
                        "position_size": position_size,
                        "exit_time": None,
                        "exit_price": None,
                        "exit_idx": None,
                        "profit_loss": 0.0,
                        "profit_loss_pct": 0.0,
                        "duration": 0,
                        "exit_reason": "",
                    }
                    current_position = 1.0
                    exit_reasons = []
                    for j in range(entry_idx + 1, min(entry_idx + 20, len(df))):
                        if current_position <= 0:
                            break
                        current_idx = j
                        current_price = df["close"].iloc[current_idx]
                        current_time = df.index[current_idx]
                        current_data = df.iloc[: current_idx + 1].copy()
                        position_data["position_size"] = current_position
                        if exit_strategy == "model":
                            recommendation = self.evaluate_exit_conditions(current_data, position_data)
                            if recommendation["exit"]:
                                exit_size = recommendation["size"] * current_position
                                exit_reason = recommendation["reason"]
                                if exit_size > 0:
                                    exit_value = exit_size * position_size * current_price
                                    profit_loss = exit_value - (exit_size * position_size * entry_price)
                                    current_position -= exit_size
                                    if exit_reason not in exit_reasons:
                                        exit_reasons.append(exit_reason)
                                    if current_position <= 0:
                                        trade["exit_time"] = current_time
                                        trade["exit_price"] = current_price
                                        trade["exit_idx"] = current_idx
                                        trade["profit_loss"] = profit_loss
                                        trade["profit_loss_pct"] = profit_loss / (position_size * entry_price) * 100
                                        trade["duration"] = (current_time - entry_time).total_seconds() / 86400
                                        trade["exit_reason"] = "+".join(exit_reasons)
                                        capital += profit_loss
                                        equity_curve.append(capital)
                                        trades.append(trade)
                                        break
                        elif exit_strategy == "simple":
                            exit_size = 0.0
                            exit_reason = ""
                            if current_price <= stop_loss:
                                exit_size = 1.0
                                exit_reason = "stop_loss"
                            elif current_price >= take_profit:
                                exit_size = 1.0
                                exit_reason = "take_profit"
                            elif j - entry_idx >= 10:
                                exit_size = 1.0
                                exit_reason = "time_stop"
                            if exit_size > 0:
                                exit_value = position_size * current_price
                                profit_loss = exit_value - (position_size * entry_price)
                                current_position = 0.0
                                trade["exit_time"] = current_time
                                trade["exit_price"] = current_price
                                trade["exit_idx"] = current_idx
                                trade["profit_loss"] = profit_loss
                                trade["profit_loss_pct"] = profit_loss / (position_size * entry_price) * 100
                                trade["duration"] = (current_time - entry_time).total_seconds() / 86400
                                trade["exit_reason"] = exit_reason
                                capital += profit_loss
                                equity_curve.append(capital)
                                trades.append(trade)
                                break
                        elif exit_strategy == "hold":
                            if j == min(entry_idx + 19, len(df) - 1):
                                exit_value = position_size * current_price
                                profit_loss = exit_value - (position_size * entry_price)
                                trade["exit_time"] = current_time
                                trade["exit_price"] = current_price
                                trade["exit_idx"] = current_idx
                                trade["profit_loss"] = profit_loss
                                trade["profit_loss_pct"] = profit_loss / (position_size * entry_price) * 100
                                trade["duration"] = (current_time - entry_time).total_seconds() / 86400
                                trade["exit_reason"] = "hold_period_end"
                                capital += profit_loss
                                equity_curve.append(capital)
                                trades.append(trade)
                                break
                    i = max(i, trade.get("exit_idx", i) + 5)
        if trades:
            results["total_trades"] = len(trades)
            winning_trades = [t for t in trades if t["profit_loss"] > 0]
            losing_trades = [t for t in trades if t["profit_loss"] <= 0]
            results["winning_trades"] = len(winning_trades)
            results["losing_trades"] = len(losing_trades)
            results["win_rate"] = len(winning_trades)
            results["win_rate"] = len(winning_trades) / len(trades) if trades else 0.0
            results["avg_profit"] = sum(t["profit_loss"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
            results["avg_loss"] = sum(t["profit_loss"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
            total_profit = sum(t["profit_loss"] for t in winning_trades)
            total_loss = abs(sum(t["profit_loss"] for t in losing_trades))
            results["profit_factor"] = total_profit / total_loss if total_loss > 0 else float("inf")
            peak = equity_curve[0]
            max_drawdown = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            results["max_drawdown"] = max_drawdown
            returns = np.diff(equity_curve) / equity_curve[:-1]
            results["sharpe_ratio"] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            results["total_return"] = (capital - initial_capital) / initial_capital * 100
            if trades:
                start_date = min(t["entry_time"] for t in trades)
                end_date = max(t["exit_time"] for t in trades if t["exit_time"] is not None)
                if start_date and end_date:
                    years = (end_date - start_date).total_seconds() / (86400 * 365)
                    if years > 0:
                        results["annualized_return"] = ((1 + results["total_return"] / 100) ** (1 / years) - 1) * 100
            results["trades"] = trades
            results["equity_curve"] = equity_curve
        return results

    def export_model(self, export_dir: Optional[str] = None) -> str:
        if export_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(settings.models_dir, "exports", f"exit_optimization_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)
        model_path = os.path.join(export_dir, "model.pt")
        self.save_model(model_path)
        metadata_path = os.path.join(export_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "metadata": self.metadata,
                    "training_stats": self.training_stats,
                    "performance_metrics": self.performance_metrics,
                    "feature_names": self.feature_names,
                    "export_date": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
        logger.info(f"Model exported to {export_dir}")
        return export_dir


# Initialize the global exit optimization model instance without loading a model by default.
# The model will be loaded explicitly when needed (e.g., when the trading system starts).
exit_optimization_model = ExitOptimizationModel(model_path=None, use_sac=True)


def evaluate_exit_strategy(ohlcv_data: pd.DataFrame, position_data: Dict, confidence_threshold: Optional[float] = None) -> Dict:
    return exit_optimization_model.evaluate_exit_conditions(ohlcv_data, position_data, confidence_threshold)


def backtest_exit_strategies(
    historical_data: Dict[str, pd.DataFrame],
    strategies: List[str] = ["model", "simple", "hold"],
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.2,
) -> Dict[str, Dict]:
    results = {}
    for strategy in strategies:
        logger.info(f"Backtesting {strategy} strategy...")
        strategy_results = exit_optimization_model.backtest(
            historical_data, exit_strategy=strategy, initial_capital=initial_capital, position_size_pct=position_size_pct
        )
        results[strategy] = strategy_results
        logger.info(
            f"{strategy} strategy: {strategy_results['total_return']:.2f}% return, "
            f"{strategy_results['win_rate']*100:.2f}% win rate, "
            f"{strategy_results['sharpe_ratio']:.2f} Sharpe ratio"
        )
    return results


def train_exit_model(training_data: List[Dict], use_sac: bool = True, optimize_hyperparams: bool = False) -> Tuple[Dict, ExitOptimizationModel]:
    model = ExitOptimizationModel(use_sac=use_sac)
    if use_sac:
        metrics = model.train_sac(
            training_data=training_data,
            epochs=settings.model.epochs,
            batch_size=settings.model.batch_size,
            lr=settings.model.learning_rate,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            updates_per_step=1,
            start_steps=1000,
            eval_interval=5000,
        )
    else:
        logger.error("PPO training is not supported in this version")
        metrics = {}
    model.save_model(settings.model.exit_model_path)
    return metrics, model


async def schedule_exit_model_training(days: int = 5) -> bool:
    try:
        current_hour = datetime.now().hour
        is_off_hours = current_hour < 9 or current_hour > 16
        if is_off_hours:
            logger.info("Scheduling exit model training during off-hours")
            training_data = await generate_training_data(days)
            if not training_data:
                logger.warning("No training data available")
                return False
            optimize_hyperparams = datetime.now().weekday() == 5
            _, model = train_exit_model(training_data=training_data, use_sac=True, optimize_hyperparams=optimize_hyperparams)
            validation_data = await generate_validation_data(1)
            if validation_data:
                backtest_results = model.backtest(historical_data=validation_data, exit_strategy="model")
                logger.info(
                    f"Model validation: {backtest_results['total_return']:.2f}% return, "
                    f"{backtest_results['win_rate']*100:.2f}% win rate, "
                    f"{backtest_results['sharpe_ratio']:.2f} Sharpe ratio"
                )
            return True
        else:
            logger.info("Not in off-hours, skipping scheduled training")
            return False
    except Exception as e:
        logger.error(f"Error in schedule_exit_model_training: {e}")
        return False


async def generate_training_data(days: int = 5) -> List[Dict]:
    """
    Generate synthetic training data for exit optimization model.

    Args:
        days: Number of days of historical data to simulate

    Returns:
        List of training episodes
    """
    logger.info(f"Generating training data for {days} days")
    training_data = []
    num_episodes = 100  # Number of episodes to generate
    steps_per_episode = 20  # Number of steps per episode

    for _ in range(num_episodes):
        # Generate a random price path
        entry_price = np.random.uniform(50, 200)  # Random entry price between $50 and $200
        prices = [entry_price]
        for _ in range(steps_per_episode):
            price_change = np.random.normal(0, 0.01)  # Random walk with small volatility
            prices.append(prices[-1] * (1 + price_change))

        # Generate states
        states = []
        for i in range(steps_per_episode):
            current_price = prices[i]
            price_to_entry = current_price / entry_price - 1
            profit_pct = price_to_entry * 100
            # Simulate technical indicators
            rsi = np.random.uniform(0, 1)  # Simplified RSI
            bb_position = np.random.uniform(0, 1)  # Simplified Bollinger Band position
            macd_histogram = np.random.uniform(-1, 1)  # Simplified MACD
            volatility = np.random.uniform(0, 1)  # Simplified volatility
            atr = np.random.uniform(0, 1)  # Simplified ATR
            volume_ratio = np.random.uniform(0, 5)  # Simplified volume ratio
            market_trend = np.random.uniform(-1, 1)  # Simplified market trend
            sector_trend = np.random.uniform(-1, 1)  # Simplified sector trend
            sharpe_ratio = np.random.uniform(-1, 1)  # Simplified Sharpe ratio
            max_drawdown = np.random.uniform(0, 1)  # Simplified max drawdown

            state = np.array(
                [
                    profit_pct / 20,
                    i / steps_per_episode,
                    1.0,
                    price_to_entry,
                    np.clip(price_to_entry - 0.1, -1, 0),  # Simulated high
                    np.clip(price_to_entry + 0.1, 0, 1),  # Simulated low
                    rsi,
                    bb_position,
                    macd_histogram,
                    volatility,
                    atr,
                    volume_ratio / 5,
                    market_trend,
                    sector_trend,
                    sharpe_ratio,
                    max_drawdown,
                ],
                dtype=np.float32,
            )
            states.append(state)

        training_data.append({"states": states, "entry_price": entry_price, "prices": prices})

    logger.info(f"Generated {len(training_data)} training episodes")
    return training_data


async def generate_validation_data(days: int = 1) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic validation data for backtesting.

    Args:
        days: Number of days of historical data to simulate

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    logger.info(f"Generating validation data for {days} days")
    validation_data = {}
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]  # Example symbols
    bars_per_day = 390  # Approximate number of 1-minute bars in a trading day

    for symbol in symbols:
        # Generate synthetic OHLCV data
        dates = pd.date_range(
            end=datetime.now(), periods=days * bars_per_day, freq="1min"
        )
        prices = np.random.normal(100, 10, len(dates))  # Random walk starting at $100
        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(0, 2, len(dates)),
                "low": prices - np.random.uniform(0, 2, len(dates)),
                "close": prices + np.random.uniform(-1, 1, len(dates)),
                "volume": np.random.randint(1000, 10000, len(dates)),
            },
            index=dates,
        )
        validation_data[symbol] = df

    logger.info(f"Generated validation data for {len(validation_data)} symbols")
    return validation_data
