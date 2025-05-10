"""
Exit optimization model for determining optimal exit points.

This model uses advanced reinforcement learning (SAC) to optimize trade exit decisions
based on price action, indicators, position state, and market context.
"""
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import os
import pickle
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.optim import Adam
from collections import deque
import random

from src.config.settings import settings
from src.utils.logging import setup_logger, log_execution_time
from src.models.pattern_recognition import pattern_model  # Added import for pattern model

# Set up logger
logger = setup_logger("exit_optimization")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() and settings.advanced.use_gpu else "cpu")
logger.info(f"Using device: {device}")

# Define actions
ACTIONS = {
    0: "hold",        # No action, continue holding position
    1: "exit_quarter", # Exit 1/4 of the position
    2: "exit_third",  # Exit 1/3 of the position
    3: "exit_half",   # Exit 1/2 of the position
    4: "exit_full"    # Exit the entire position
}

# Continuous action space scaling
ACTION_SCALE = torch.tensor(1.0).to(device)  # Scale actions to [0, 1]
ACTION_BIAS = torch.tensor(0.5).to(device)   # Center around 0.5

class ActorCritic(nn.Module):
    """
    Actor-Critic network for SAC algorithm.
    Uses both a stochastic policy (the actor) and a value function (the critic).
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        """
        Initialize the Actor-Critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Hidden layer dimension
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log standard deviation for the Gaussian policy
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic networks (twin critics for SAC)
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Value network (for SAC)
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, state):
        """
        Forward pass through the actor-critic network.
        
        Args:
            state: State tensor
            
        Returns:
            mean, log_std: Parameters of the Gaussian policy
        """
        x = self.actor(state)
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def get_action(self, state, evaluation=False):
        """
        Sample an action from the policy.
        
        Args:
            state: State tensor
            evaluation: Whether to use deterministic policy (for evaluation)
            
        Returns:
            action, log_prob: Sampled action and its log probability
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if evaluation:
            action = torch.tanh(mean) * ACTION_SCALE + ACTION_BIAS
            return action, None
        
        # Create normal distribution
        normal = Normal(mean, std)
        
        # Sample an action
        x = normal.rsample()
        
        # Compute log probability, using the Tanh squashing correction
        log_prob = normal.log_prob(x)
        
        # Apply tanh squashing
        action = torch.tanh(x) * ACTION_SCALE + ACTION_BIAS
        
        # Correct log probability for the squashing transformation
        log_prob -= torch.log(ACTION_SCALE * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob
    
    def get_critic_value(self, state, action):
        """
        Get the Q-value from the critic networks.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            q1, q2: Q-values from both critics
        """
        state_action = torch.cat([state, action], 1)
        
        q1 = self.critic1(state_action)
        q2 = self.critic2(state_action)
        
        return q1, q2
    
    def get_value(self, state):
        """
        Get the value estimate.
        
        Args:
            state: State tensor
            
        Returns:
            value: Value estimate
        """
        return self.value(state)

class ReplayBuffer:
    """
    Experience replay buffer for reinforcement learning.
    """
    
    def __init__(self, capacity=100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        Store a transition in the buffer.
        
        Args:
            state: State tensor
            action: Action tensor
            reward: Reward value
            next_state: Next state tensor
            done: Whether the episode is done
        """
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            state, action, reward, next_state, done: Batches of transitions
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        """
        Get the current size of the buffer.
        
        Returns:
            Length of the buffer
        """
        return len(self.buffer)

class SACAgent:
    """
    Soft Actor-Critic (SAC) agent for reinforcement learning.
    
    SAC is an off-policy actor-critic deep RL algorithm based on the maximum entropy
    reinforcement learning framework. It adds an entropy maximization term to the 
    standard RL objective to encourage exploration, and uses a stochastic policy.
    """
    
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim=256, 
        gamma=0.99, 
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        batch_size=256,
        automatic_entropy_tuning=True
    ):
        """
        Initialize the SAC agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Hidden layer dimension
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy coefficient
            lr: Learning rate
            batch_size: Batch size for training
            automatic_entropy_tuning: Whether to automatically tune entropy coefficient
        """
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Actor-critic network
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        
        # Target value network
        self.target_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Initialize target value network parameters
        for target_param, param in zip(self.target_value.parameters(), self.actor_critic.value.parameters()):
            target_param.data.copy_(param.data)
        
        # Setup optimizers
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(
            list(self.actor_critic.critic1.parameters()) + list(self.actor_critic.critic2.parameters()), 
            lr=lr
        )
        self.value_optimizer = Adam(self.actor_critic.value.parameters(), lr=lr)
        
        # If automatic entropy tuning is enabled, create log alpha and optimizer
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.tensor(action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training info
        self.training_info = {
            'actor_losses': [],
            'critic_losses': [],
            'value_losses': [],
            'alpha_losses': [],
            'entropies': [],
            'rewards': []
        }
    
    def select_action(self, state, evaluate=False):
        """
        Select an action from the policy.
        
        Args:
            state: State tensor
            evaluate: Whether to use deterministic policy (for evaluation)
            
        Returns:
            action: Selected action
        """
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.actor_critic.get_action(state, evaluation=evaluate)
        
        # Convert to discrete action if needed (e.g., for portfolio management)
        if evaluate:
            # For evaluation, we can directly map the continuous action to a discrete action
            # Scale from [0, 1] to [0, 4] and round to the nearest integer
            discrete_action = torch.round(action.item() * 4).int().item()
            discrete_action = min(max(discrete_action, 0), 4)  # Clamp to valid range
            return discrete_action
        else:
            # For training, return the continuous action
            return action.cpu().data.numpy().flatten()
    
    def update(self, batch_size=None):
        """
        Update the agent's networks.
        
        Args:
            batch_size: Batch size for training
            
        Returns:
            value_loss, critic_loss, actor_loss, alpha_loss: Training losses
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(np.vstack(action)).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)
        
        # Update value network
        with torch.no_grad():
            new_actions, log_probs = self.actor_critic.get_action(state)
            q1, q2 = self.actor_critic.get_critic_value(state, new_actions)
            target_q = torch.min(q1, q2) - self.alpha * log_probs
        
        value = self.actor_critic.get_value(state)
        value_loss = F.mse_loss(value, target_q)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Update critic networks
        with torch.no_grad():
            target_value = self.target_value(next_state)
            target_q = reward + (1 - done) * self.gamma * target_value
        
        current_q1, current_q2 = self.actor_critic.get_critic_value(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor network
        new_actions, log_probs = self.actor_critic.get_action(state)
        q1, q2 = self.actor_critic.get_critic_value(state, new_actions)
        min_q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha if automatic entropy tuning is enabled
        alpha_loss = torch.tensor(0.0)
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target value network
        for target_param, param in zip(self.target_value.parameters(), self.actor_critic.value.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
        
        # Update training info
        self.training_info['actor_losses'].append(actor_loss.item())
        self.training_info['critic_losses'].append(critic_loss.item())
        self.training_info['value_losses'].append(value_loss.item())
        self.training_info['alpha_losses'].append(alpha_loss.item() if self.automatic_entropy_tuning else 0.0)
        
        return value_loss.item(), critic_loss.item(), actor_loss.item(), alpha_loss.item() if self.automatic_entropy_tuning else 0.0
    
    def save(self, path):
        """
        Save the agent to a file.
        
        Args:
            path: Path to save the agent
        """
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'target_value_state_dict': self.target_value.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'alpha': self.alpha,
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'training_info': self.training_info
        }, path)
    
    def load(self, path):
        """
        Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        checkpoint = torch.load(path, map_location=device)
        
        try:
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        except KeyError as e:
            logger.error(f"Error loading actor-critic state dict: {e}")
            raise ValueError("The model state dict does not match the expected format.")
        self.target_value.load_state_dict(checkpoint['target_value_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        self.alpha = checkpoint['alpha']
        
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.training_info = checkpoint['training_info']

class ExitOptimizationModel:
    """
    Advanced reinforcement learning model for optimizing trade exits.
    Implements Soft Actor-Critic algorithm for robust and sample-efficient learning.
    """
    
    def __init__(self, model_path: Optional[str] = None, use_sac: bool = True):
        """
        Initialize the exit optimization model.
        
        Args:
            model_path: Path to the model file
            use_sac: Whether to use SAC algorithm (vs. PPO)
        """
        # Define state features
        self.feature_names = [
            # Position features
            'profit_pct',          # Current profit percentage
            'time_in_trade',       # Time in trade (normalized)
            'position_size',       # Current position size (normalized)
            
            # Price features
            'price_to_entry',      # Current price relative to entry
            'price_to_high',       # Current price relative to high since entry
            'price_to_low',        # Current price relative to low since entry
            
            # Technical indicators
            'rsi_14',              # RSI (14)
            'bb_position',         # Position within Bollinger Bands
            'macd_histogram',      # MACD histogram
            
            # Volatility
            'volatility_5d',       # 5-day volatility
            'atr',                 # Average True Range (normalized)
            
            # Volume
            'volume_ratio_5',      # Volume relative to 5-day average
            
            # Market context
            'market_trend',        # Overall market trend indicator
            'sector_trend',        # Sector trend indicator
            
            # Risk metrics
            'sharpe_ratio',        # Sharpe ratio (for risk-adjusted returns)
            'max_drawdown',        # Maximum drawdown since entry
        ]
        
        # State and action dimensions
        self.state_dim = len(self.feature_names)
        self.action_dim = 1  # Continuous action space (exit size)
        
        # Use SAC algorithm by default
        self.use_sac = use_sac
        
        # Initialize agent
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
                automatic_entropy_tuning=True
            )
        else:
            # PPO model (included for backward compatibility)
            self.actor = None
            self.critic = None
        
        # Model version and metadata
        self.version = "1.0.0-SAC" if self.use_sac else "0.9.0-PPO"
        self.metadata = {
            "features": self.feature_names,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "algorithm": "SAC" if self.use_sac else "PPO",
            "version": self.version,
            "last_updated": datetime.now().isoformat()
        }
        
        # Training statistics
        self.training_stats = {
            "episodes": 0,
            "total_steps": 0,
            "avg_reward": 0.0,
            "best_reward": 0.0,
            "avg_sharpe": 0.0,
            "win_rate": 0.0
        }
        
        # Performance metrics
        self.performance_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "avg_profit_loss": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0
        }
        
        # Load model if path is provided
        if model_path:
            # Convert to absolute path if it's a relative path
            if not os.path.isabs(model_path):
                model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
            
            if os.path.exists(model_path):
                self.load_model(model_path)
                logger.info(f"Loaded exit optimization model from {model_path}")
        else:
            logger.warning("No model file provided, using untrained model")
    
    def save_model(self, model_path: str):
        """
        Save model to file.
        
        Args:
            model_path: Path to save the model
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            if self.use_sac:
                # Save SAC agent
                self.agent.save(model_path)
                
                # Save metadata
                metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "metadata": self.metadata,
                        "training_stats": self.training_stats,
                        "performance_metrics": self.performance_metrics
                    }, f, indent=2)
            else:
                # Legacy PPO model saving
                if self.actor is not None and self.critic is not None:
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'metadata': self.metadata,
                        'training_stats': self.training_stats,
                        'performance_metrics': self.performance_metrics
                    }, model_path)
            
            logger.info(f"Saved model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, model_path: str):
        """
        Load model from file.
        
        Args:
            model_path: Path to the model file
        """
        try:
            # Try to load SAC model first
            try:
                # Check if it's an SAC model
                self.agent.load(model_path)
                self.use_sac = True
                
                # Load metadata if available
                metadata_path = os.path.splitext(model_path)[0] + '_metadata.json'
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        data = json.load(f)
                        if 'metadata' in data:
                            self.metadata = data['metadata']
                        if 'training_stats' in data:
                            self.training_stats = data['training_stats']
                        if 'performance_metrics' in data:
                            self.performance_metrics = data['performance_metrics']
                
                logger.info(f"Loaded SAC model from {model_path}")
                return
            except Exception as sac_error:
                logger.warning(f"Failed to load as SAC model: {sac_error}")
                # Continue to try loading as PPO model
            
            # Try to load legacy PPO model
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'actor_state_dict' in checkpoint and 'critic_state_dict' in checkpoint:
                # Legacy PPO model
                self.use_sac = False
                
                # Initialize legacy PPO networks
                # from src.models.exit_optimization_legacy import PolicyNetwork, ValueNetwork
                self.actor = PolicyNetwork(self.state_dim, len(ACTIONS)).to(device)
                self.critic = ValueNetwork(self.state_dim).to(device)
                
                # Load state dictionaries
                self.actor.load_state_dict(checkpoint['actor_state_dict'])
                self.critic.load_state_dict(checkpoint['critic_state_dict'])
                
                # Load metadata if available
                if 'metadata' in checkpoint:
                    self.metadata = checkpoint['metadata']
                if 'training_stats' in checkpoint:
                    self.training_stats = checkpoint['training_stats']
                if 'performance_metrics' in checkpoint:
                    self.performance_metrics = checkpoint['performance_metrics']
                
                logger.info(f"Loaded legacy PPO model from {model_path}")
            else:
                raise ValueError("Unknown model format")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def train_sac(
        self,
        training_data: List[Dict],
        epochs: int = None,
        batch_size: int = None,
        lr: float = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        updates_per_step: int = 1,
        start_steps: int = 1000,
        eval_interval: int = 5000
    ):
        """
        Train the model using Soft Actor-Critic.
        
        Args:
            training_data: List of training episodes
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy coefficient
            updates_per_step: Number of updates per step
            start_steps: Number of steps for random exploration
            eval_interval: Evaluation interval
            
        Returns:
            Training metrics
        """
        # Use values from config if not provided
        epochs = epochs or settings.model.epochs
        batch_size = batch_size or settings.model.batch_size
        lr = lr or settings.model.learning_rate
        
        # Initialize agent with provided parameters
        self.agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=256,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            lr=lr,
            batch_size=batch_size,
            automatic_entropy_tuning=True
        )
        
        # Initialize metrics
        metrics = {
            'epoch_rewards': [],
            'epoch_losses': [],
            'epoch_value_losses': [],
            'epoch_critic_losses': [],
            'epoch_actor_losses': [],
            'epoch_alpha_losses': [],
            'epoch_entropies': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'win_rates': []
        }
        
        # Total steps counter
        total_steps = 0
        
        logger.info(f"Starting SAC training for {epochs} epochs")
        
        # Training loop
        for epoch in range(epochs):
            epoch_reward = 0
            epoch_steps = 0
            epoch_profits = []
            
            # Shuffle training episodes
            random.shuffle(training_data)
            
            # Process each episode
            for episode_idx, episode in enumerate(training_data):
                # Extract episode data
                states = episode.get('states', [])
                
                if not states:
                    continue
                
                # Reset episode variables
                episode_reward = 0
                position_size = 1.0
                entry_price = episode.get('entry_price', 100.0)
                current_price = entry_price
                highest_price = entry_price
                profit_pct = 0.0
                episode_done = False
                
                # Initial state
                state = states[0].copy() if states else np.zeros(self.state_dim)
                
                # Simulate the episode
                for step_idx in range(len(states) - 1):
                    # Get current state
                    state = states[step_idx].copy()
                    
                    # Update position features in state
                    state[0] = profit_pct  # profit_pct
                    state[2] = position_size  # position_size
                    
                    # Select action (randomly at first, then from policy)
                    if total_steps < start_steps:
                        action = np.random.uniform(0, 1, 1)[0]  # Random action in [0, 1]
                    else:
                        action = self.agent.select_action(state)
                    
                    # Convert continuous action to exit size
                    if isinstance(action, np.ndarray):
                        exit_size = float(action[0])  # First element of continuous action
                    else:
                        exit_size = float(action)
                    
                    # Ensure action is in valid range
                    exit_size = max(0.0, min(1.0, exit_size))
                    
                    # Calculate actual exit size based on current position
                    actual_exit = min(position_size, exit_size)
                    
                    # Update position size
                    new_position_size = position_size - actual_exit
                    
                    # Get next state
                    next_state = states[step_idx + 1].copy()
                    
                    # Update price information
                    current_price = entry_price * (1 + next_state[3])  # price_to_entry feature
                    highest_price = max(highest_price, current_price)
                    
                    # Calculate reward (risk-adjusted)
                    reward = self._calculate_risk_adjusted_reward(
                        exit_size=actual_exit,
                        position_size=position_size,
                        new_position_size=new_position_size,
                        current_price=current_price,
                        entry_price=entry_price,
                        highest_price=highest_price,
                        state=state,
                        next_state=next_state
                    )
                    
                    # Update profit percentage
                    profit_pct = (current_price / entry_price - 1) * 100
                    
                    # Check if episode is done
                    done = new_position_size <= 0 or step_idx == len(states) - 2
                    
                    if done:
                        episode_done = True
                    
                    # Update next state's position features
                    next_state[0] = profit_pct  # profit_pct
                    next_state[2] = new_position_size  # position_size
                    
                    # Store transition in replay buffer
                    self.agent.replay_buffer.push(state, np.array([exit_size]), reward, next_state, done)
                    
                    # Update agent if enough samples are available
                    if len(self.agent.replay_buffer) > batch_size:
                        for _ in range(updates_per_step):
                            value_loss, critic_loss, actor_loss, alpha_loss = self.agent.update(batch_size)
                    
                    # Update metrics
                    state = next_state
                    episode_reward += reward
                    epoch_reward += reward
                    epoch_steps += 1
                    total_steps += 1
                    position_size = new_position_size
                    
                    # If position is closed, record the profit
                    if actual_exit > 0:
                        exit_profit = (current_price / entry_price - 1) * actual_exit
                        epoch_profits.append(exit_profit)
                    
                    # Break if episode is done
                    if episode_done:
                        break
                
                # Evaluate periodically
                if total_steps % eval_interval == 0:
                    eval_reward = self._evaluate_model(5)
                    logger.info(f"Step {total_steps}, Eval reward: {eval_reward:.4f}")
            
            # Calculate epoch metrics
            avg_reward = epoch_reward / max(1, epoch_steps)
            
            # Calculate Sharpe ratio and win rate if profits available
            sharpe_ratio = 0.0
            win_rate = 0.0
            max_drawdown = 0.0
            
            if epoch_profits:
                # Sharpe ratio: mean / std (or 1.0 to avoid division by zero)
                mean_profit = np.mean(epoch_profits)
                std_profit = np.std(epoch_profits) if len(epoch_profits) > 1 else 1.0
                sharpe_ratio = mean_profit / max(std_profit, 1e-6)
                
                # Win rate: percentage of profitable exits
                win_rate = sum(1 for p in epoch_profits if p > 0) / len(epoch_profits)
                
                # Max drawdown simulation
                cumulative_returns = np.cumsum(epoch_profits)
                max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            
            # Update metrics
            metrics['epoch_rewards'].append(avg_reward)
            metrics['epoch_losses'].append(np.mean(self.agent.training_info['actor_losses'][-epoch_steps:] if epoch_steps > 0 else [0]))
            metrics['epoch_value_losses'].append(np.mean(self.agent.training_info['value_losses'][-epoch_steps:] if epoch_steps > 0 else [0]))
            metrics['epoch_critic_losses'].append(np.mean(self.agent.training_info['critic_losses'][-epoch_steps:] if epoch_steps > 0 else [0]))
            metrics['epoch_actor_losses'].append(np.mean(self.agent.training_info['actor_losses'][-epoch_steps:] if epoch_steps > 0 else [0]))
            metrics['epoch_alpha_losses'].append(np.mean(self.agent.training_info['alpha_losses'][-epoch_steps:] if epoch_steps > 0 else [0]))
            metrics['epoch_entropies'].append(np.mean(self.agent.training_info['entropies'][-epoch_steps:] if epoch_steps > 0 else [0]))
            metrics['sharpe_ratios'].append(sharpe_ratio)
            metrics['max_drawdowns'].append(max_drawdown)
            metrics['win_rates'].append(win_rate)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                        f"Reward: {avg_reward:.4f}, "
                        f"Sharpe: {sharpe_ratio:.4f}, "
                        f"Win Rate: {win_rate:.2f}, "
                        f"Alpha: {self.agent.alpha:.4f}")
        
        # Update training stats
        self.training_stats["episodes"] = len(training_data)
        self.training_stats["total_steps"] = total_steps
        self.training_stats["avg_reward"] = np.mean(metrics['epoch_rewards'])
        self.training_stats["best_reward"] = max(metrics['epoch_rewards'])
        self.training_stats["avg_sharpe"] = np.mean(metrics['sharpe_ratios'])
        self.training_stats["win_rate"] = np.mean(metrics['win_rates'])
        
        # Update metadata
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Save model after training
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(
            settings.models_dir,
            f"exit_optimization_sac_{timestamp}.pt"
        )
        self.save_model(model_save_path)
        
        # Also save as default model
        self.save_model(settings.model.exit_model_path)
        
        return metrics
    
    def _evaluate_model(self, num_episodes=5):
        """
        Evaluate the model performance.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Average reward
        """
        # Create simple test episodes
        test_episodes = []
        
        for _ in range(num_episodes):
            # Generate a random price path
            steps = 20
            prices = [100.0]
            for i in range(steps):
                # Random price change (-1% to +1%)
                price_change = np.random.uniform(-0.01, 0.01)
                prices.append(prices[-1] * (1 + price_change))
            
            # Create states
            states = []
            entry_price = prices[0]
            
            for i in range(steps):
                current_price = prices[i]
                price_to_entry = current_price / entry_price - 1
                profit_pct = price_to_entry * 100
                
                # Create a simple state
                state = np.zeros(self.state_dim)
                state[0] = profit_pct  # profit_pct
                state[1] = i / steps  # time_in_trade
                state[2] = 1.0  # position_size
                state[3] = price_to_entry  # price_to_entry
                
                states.append(state)
            
            test_episodes.append({
                'states': states,
                'entry_price': entry_price,
                'prices': prices
            })
        
        # Evaluate on test episodes
        total_reward = 0.0
        
        for episode in test_episodes:
            states = episode['states']
            entry_price = episode['entry_price']
            prices = episode['prices']
            
            episode_reward = 0.0
            position_size = 1.0
            highest_price = entry_price
            
            for i in range(len(states) - 1):
                state = states[i].copy()
                state[2] = position_size  # Update position size
                
                # Get action from policy
                action = self.agent.select_action(state, evaluate=True)
                
                # Convert to exit size
                if isinstance(action, int):
                    # Discrete action
                    if action == 0:  # hold
                        exit_size = 0.0
                    elif action == 1:  # exit_quarter
                        exit_size = 0.25
                    elif action == 2:  # exit_third
                        exit_size = 0.33
                    elif action == 3:  # exit_half
                        exit_size = 0.5
                    else:  # exit_full
                        exit_size = 1.0
                else:
                    # Continuous action
                    exit_size = float(action)
                    exit_size = max(0.0, min(1.0, exit_size))
                
                # Calculate actual exit
                actual_exit = min(position_size, exit_size)
                new_position_size = position_size - actual_exit
                
                # Update price info
                current_price = prices[i]
                next_price = prices[i + 1]
                highest_price = max(highest_price, current_price)
                
                # Calculate reward
                if actual_exit > 0:
                    # Reward for exiting
                    exit_profit = (current_price / entry_price - 1) * actual_exit * 100
                    reward = exit_profit
                else:
                    # Small reward for holding
                    price_change = (next_price - current_price) / current_price * 100
                    reward = price_change * position_size * 0.1
                
                # Update metrics
                episode_reward += reward
                position_size = new_position_size
                
                # Break if position is closed
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
        next_state: np.ndarray
    ) -> float:
        """
        Calculate risk-adjusted reward for reinforcement learning.
        
        Args:
            exit_size: Size of the exit
            position_size: Current position size
            new_position_size: New position size after exit
            current_price: Current price
            entry_price: Entry price
            highest_price: Highest price since entry
            state: Current state
            next_state: Next state
            
        Returns:
            Risk-adjusted reward
        """
        # Base profit/loss calculation
        profit_pct = (current_price / entry_price - 1) * 100
        
        # Calculate drawdown
        drawdown = (highest_price - current_price) / highest_price * 100
        
        # Extract relevant indicators from state
        rsi = state[6]  # RSI feature
        bb_position = state[7]  # Bollinger Band position
        volatility = state[9]  # Volatility feature
        
        # Time factor (penalize holding too long)
        time_factor = state[1]  # time_in_trade feature
        
        # Base reward components
        if exit_size > 0:
            # Reward for exiting (based on actual exit size and profit)
            actual_exit = min(position_size, exit_size)
            exit_profit = profit_pct * actual_exit
            
            # Adjust reward based on market conditions
            if profit_pct > 0:
                # For profitable exits
                if rsi > 0.7:  # Overbought
                    exit_reward = exit_profit * 1.2  # Bonus for exiting overbought
                elif drawdown > 2.0 and profit_pct > 2.0:
                    exit_reward = exit_profit * 1.1  # Bonus for exiting after drawdown from high
                else:
                    exit_reward = exit_profit
            else:
                # For losing exits
                if rsi < 0.3:  # Oversold
                    exit_reward = exit_profit * 0.8  # Reduced penalty for exiting oversold
                else:
                    exit_reward = exit_profit
            
            # Add a small reward for reducing risk
            risk_reduction_reward = 0.1 * actual_exit
            
            # Combine rewards
            reward = exit_reward + risk_reduction_reward
        else:
            # Reward for holding
            # Use next state's price change as indicator
            price_change = next_state[3] - state[3]  # Change in price_to_entry
            price_change_pct = price_change * 100
            
            # Base holding reward is proportional to price change and position size
            holding_reward = price_change_pct * position_size
            
            # Apply time decay factor to penalize holding too long
            holding_reward *= (1.0 - 0.5 * time_factor)
            
            # Add risk penalty for volatile conditions
            risk_penalty = volatility * position_size * 0.2
            
            # Combine rewards
            reward = holding_reward - risk_penalty
        
        # Add Sharpe ratio component for risk adjustment
        if hasattr(self, '_episode_returns'):
            self._episode_returns.append(reward)
            if len(self._episode_returns) > 1:
                mean_return = np.mean(self._episode_returns)
                std_return = np.std(self._episode_returns)
                sharpe = mean_return / max(std_return, 1e-6)
                
                # Scale the reward by the Sharpe ratio trend
                sharpe_factor = min(max(sharpe, 0.5), 1.5)  # Limit effect
                reward *= sharpe_factor
        else:
            self._episode_returns = [reward]
        
        return reward
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """
        Calculate maximum drawdown from a sequence of cumulative returns.
        
        Args:
            cumulative_returns: Array of cumulative returns
            
        Returns:
            Maximum drawdown as a percentage
        """
        # Convert to numpy array if not already
        cumulative_returns = np.array(cumulative_returns)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdown
        drawdown = (running_max - cumulative_returns) / np.maximum(running_max, 1e-10)
        
        # Get maximum drawdown
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        return max_drawdown
    
    def train_ppo(
        self,
        training_data: List[Dict],
        epochs: int = None,
        batch_size: int = None,
        lr: float = None,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4
    ):
        """
        Train the model using Proximal Policy Optimization (legacy).
        
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
        logger.warning("PPO training is included for backward compatibility. Consider using SAC instead.")
        
        # Import legacy PPO implementation
        from src.models.exit_optimization_legacy import train_ppo_legacy
        
        # Call legacy training function
        metrics = train_ppo_legacy(
            self, training_data, epochs, batch_size, lr, gamma, eps_clip, k_epochs
        )
        
        # Update metadata
        self.metadata["algorithm"] = "PPO"
        self.metadata["last_updated"] = datetime.now().isoformat()
        
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
        high_since_entry = df['high'].loc[entry_time:].max() if entry_time in df.index else df['high'].max()
        low_since_entry = df['low'].loc[entry_time:].min() if entry_time in df.index else df['low'].min()
        
        price_to_entry = current_price / entry_price - 1
        price_to_high = current_price / high_since_entry - 1 if high_since_entry > 0 else 0
        price_to_low = current_price / low_since_entry - 1 if low_since_entry > 0 else 0
        
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
        
        # ATR (Average True Range)
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        normalized_atr = atr / current_price * 100  # Convert to percentage
        
        # Volume
        volume_sma_5 = df['volume'].rolling(5).mean()
        volume_ratio_5 = df['volume'].iloc[-1] / volume_sma_5.iloc[-1] if volume_sma_5.iloc[-1] else 1.0
        
        # Market and sector context (placeholders - in real implementation, use market data)
        # Positive for uptrend, negative for downtrend, zero for sideways
        market_trend = 0.0
        sector_trend = 0.0
        
        # Try to get market trend from data if available
        if 'spy' in position_data or 'market_trend' in position_data:
            market_trend = position_data.get('market_trend', position_data.get('spy', 0.0))
        else:
            # Simple approximation using the 50-day trend
            if len(df) >= 50:
                market_trend = (df['close'].iloc[-1] / df['close'].iloc[-50] - 1) * 100 / 10  # Scale to [-1, 1]
                market_trend = max(min(market_trend, 1.0), -1.0)  # Clamp
        
        # Try to get sector trend from data if available
        if 'sector_trend' in position_data:
            sector_trend = position_data.get('sector_trend', 0.0)
        
        # Risk metrics
        # Sharpe ratio (approximation using recent returns)
        returns = df['returns'].iloc[-20:].dropna()
        sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0.0
        
        # Maximum drawdown
        close_prices = df['close'].loc[entry_time:].values if entry_time in df.index else df['close'].values
        peak = np.maximum.accumulate(close_prices)
        drawdown = (peak - close_prices) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Collect features
        features = [
            profit_pct,
            time_in_trade,
            position_size,
            price_to_entry,
            price_to_high,
            price_to_low,
            rsi_14.iloc[-1],
            bb_position,
            macd_histogram.iloc[-1],
            volatility_5d,
            normalized_atr,
            volume_ratio_5,
            market_trend,
            sector_trend,
            sharpe_ratio,
            max_drawdown
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
            position_size,    # Already [0, 1]
            np.clip(price_to_entry, -1, 1),
            np.clip(price_to_high, -1, 0),
            np.clip(price_to_low, 0, 1),
            rsi_14,           # Already [0, 1]
            bb_position,      # Already [0, 1]
            macd_hist,        # Already [-1, 1]
            np.clip(volatility_5d / 5, 0, 1),  # Scale to [0, 1]
            np.clip(normalized_atr / 5, 0, 1), # Scale to [0, 1]
            volume_ratio_5 / 5, # Scale to [0, 1]
            np.clip(market_trend, -1, 1),  # Already [-1, 1]
            np.clip(sector_trend, -1, 1),  # Already [-1, 1]
            np.clip(sharpe_ratio, -1, 1),  # Scale to [-1, 1]
            np.clip(max_drawdown, 0, 1)    # Already [0, 1]
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
        features_tensor = torch.FloatTensor(features).to(device)
        
        if self.use_sac:
            # Use SAC model
            with torch.no_grad():
                # Reshape for batch dimension
                state = features_tensor.unsqueeze(0)
                
                # Get action from policy
                action, _ = self.agent.actor_critic.get_action(state, evaluation=True)
                
                # Convert to numpy
                action_value = action.cpu().numpy()[0, 0]
                
                # Scale action to [0, 1]
                exit_size = float(action_value)
                exit_size = max(0.0, min(1.0, exit_size))
                
                # Map to discrete action (for interpretation)
                if exit_size < 0.1:
                    action_idx = 0  # hold
                    action_name = "hold"
                elif exit_size < 0.3:
                    action_idx = 1  # exit_quarter
                    action_name = "exit_quarter"
                elif exit_size < 0.45:
                    action_idx = 2  # exit_third
                    action_name = "exit_third"
                elif exit_size < 0.75:
                    action_idx = 3  # exit_half
                    action_name = "exit_half"
                else:
                    action_idx = 4  # exit_full
                    action_name = "exit_full"
                
                # Create result
                result = {
                    'action': action_name,
                    'action_idx': action_idx,
                    'exit_size': exit_size,
                    'confidence': 0.8,  # SAC doesn't directly provide confidence
                    'probabilities': {
                        # Approximate probabilities based on distance from thresholds
                        "hold": 1.0 - min(exit_size * 10, 1.0) if exit_size < 0.1 else 0.0,
                        "exit_quarter": 1.0 - min(abs(exit_size - 0.2) * 5, 1.0) if 0.1 <= exit_size < 0.3 else 0.0,
                        "exit_third": 1.0 - min(abs(exit_size - 0.37) * 5, 1.0) if 0.3 <= exit_size < 0.45 else 0.0,
                        "exit_half": 1.0 - min(abs(exit_size - 0.6) * 5, 1.0) if 0.45 <= exit_size < 0.75 else 0.0,
                        "exit_full": 1.0 - min((1.0 - exit_size) * 4, 1.0) if exit_size >= 0.75 else 0.0
                    }
                }
        else:
            # Use legacy PPO model
            if self.actor is None:
                logger.error("No actor model loaded")
                return {
                    'action': "hold",
                    'confidence': 0.0,
                    'probabilities': {action: 0.0 for action in ACTIONS.values()}
                }
            
            # Forward pass through actor network
            with torch.no_grad():
                action_probs = self.actor(features_tensor)
                dist = Categorical(action_probs)
                
                # Get action with highest probability
                action_idx = torch.argmax(action_probs).item()
                action_name = ACTIONS[action_idx]
                
                # Convert probabilities to dictionary
                probs_dict = {
                    ACTIONS[i]: float(prob) for i, prob in enumerate(action_probs.cpu().numpy())
                }
                
                # Calculate exit size based on action
                if action_idx == 0:  # hold
                    exit_size = 0.0
                elif action_idx == 1:  # exit_quarter
                    exit_size = 0.25
                elif action_idx == 2:  # exit_third
                    exit_size = 0.33
                elif action_idx == 3:  # exit_half
                    exit_size = 0.5
                else:  # exit_full
                    exit_size = 1.0
                
                # Create result
                result = {
                    'action': action_name,
                    'action_idx': action_idx,
                    'exit_size': exit_size,
                    'confidence': float(action_probs[action_idx].cpu().numpy()),
                    'probabilities': probs_dict
                }
        
        return result
    
    def evaluate_exit_conditions(
        self, 
        ohlcv_data: pd.DataFrame, 
        position_data: Dict,
        confidence_threshold: float = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate exit conditions for a position.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            position_data: Dictionary with position information
            confidence_threshold: Minimum confidence for taking action
            market_data: Optional dictionary with market data
            
        Returns:
            Dictionary with exit recommendation
        """
        # Use confidence threshold from config if not provided
        if confidence_threshold is None:
            confidence_threshold = settings.model.confidence_threshold
        
        # Enhance position data with market context if available
        if market_data:
            position_data = self._add_market_context(position_data, market_data)
        
        # Get model prediction
        prediction = self.predict_exit_action(ohlcv_data, position_data)
        # Get exit recommendation
        action = prediction['action']
        exit_size = prediction['exit_size']
        confidence = prediction['confidence']
        
        # Determine recommendation
        if action == "hold" or confidence < confidence_threshold:
            recommendation = {
                "exit": False,
                "size": 0.0,
                "reason": "hold_position",
                "confidence": confidence
            }
        elif action == "exit_quarter":
            recommendation = {
                "exit": True,
                "size": 0.25,  # Exit 1/4 of position
                "reason": "quarter_exit",
                "confidence": confidence
            }
        elif action == "exit_third":
            recommendation = {
                "exit": True,
                "size": 0.33,  # Exit 1/3 of position
                "reason": "third_exit",
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
        
        # Add market context if available
        if market_data:
            recommendation["market_context"] = self._get_market_summary(market_data)
        
        # Check manual exit conditions
        manual_checks = self._check_manual_exit_conditions(ohlcv_data, position_data)
        recommendation.update(manual_checks)
        
        # Modify recommendation based on manual checks
        if any(manual_checks.values()):
            # If any manual exit condition is triggered, recommend exit
            exit_reasons = []
            if manual_checks.get("stop_loss_triggered", False):
                exit_reasons.append("stop_loss")
            if manual_checks.get("take_profit_triggered", False):
                exit_reasons.append("take_profit")
            if manual_checks.get("trailing_stop_triggered", False):
                exit_reasons.append("trailing_stop")
            if manual_checks.get("time_stop_triggered", False):
                exit_reasons.append("time_stop")
            
            # If any manual exit is triggered, override the model's recommendation
            if exit_reasons:
                recommendation["exit"] = True
                recommendation["size"] = 1.0  # Full exit for manual conditions
                recommendation["reason"] = "_".join(exit_reasons)
                recommendation["confidence"] = 1.0  # High confidence for manual exits
        
        # Add risk metrics
        risk_metrics = self._calculate_risk_metrics(ohlcv_data, position_data)
        recommendation["risk_metrics"] = risk_metrics
        
        # Adjust recommendation based on risk metrics
        if risk_metrics["max_drawdown"] > 0.1 and risk_metrics["drawdown_ratio"] > 2.0:
            # Increase exit size if we're experiencing significant drawdown
            if recommendation["exit"]:
                recommendation["size"] = min(1.0, recommendation["size"] * 1.5)
                recommendation["reason"] += "_high_drawdown"
            else:
                # Consider exit due to drawdown, even if model suggests hold
                recommendation["exit"] = True
                recommendation["size"] = 0.33
                recommendation["reason"] = "drawdown_protection"
                recommendation["confidence"] = 0.7
        
        return recommendation
    
    def _add_market_context(self, position_data: Dict, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Add market context to position data.
        
        Args:
            position_data: Position data
            market_data: Market data dictionary
            
        Returns:
            Updated position data
        """
        # Create a copy to avoid modifying the original
        updated_data = position_data.copy()
        
        # Add major index trend
        if "SPY" in market_data:
            spy_data = market_data["SPY"]
            if len(spy_data) > 20:
                # Calculate 1-day and 5-day returns
                spy_1d_return = spy_data["close"].pct_change(1).iloc[-1]
                spy_5d_return = spy_data["close"].pct_change(5).iloc[-1]
                
                # Add to position data
                updated_data["market_trend"] = spy_5d_return
                updated_data["market_trend_1d"] = spy_1d_return
        
        # Add sector trend if available
        if "sector_etf" in position_data:
            sector_etf = position_data["sector_etf"]
            if sector_etf in market_data:
                sector_data = market_data[sector_etf]
                if len(sector_data) > 20:
                    # Calculate 5-day return
                    sector_5d_return = sector_data["close"].pct_change(5).iloc[-1]
                    
                    # Add to position data
                    updated_data["sector_trend"] = sector_5d_return
        
        # Add market volatility
        if "VIX" in market_data:
            vix_data = market_data["VIX"]
            if len(vix_data) > 1:
                updated_data["market_volatility"] = vix_data["close"].iloc[-1] / 100.0
                updated_data["market_volatility_change"] = vix_data["close"].pct_change(1).iloc[-1]
        
        return updated_data
    
    def _get_market_summary(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Get a summary of market conditions.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Market summary
        """
        summary = {}
        
        # Add market trend
        if "SPY" in market_data:
            spy_data = market_data["SPY"]
            if len(spy_data) > 20:
                # Calculate returns
                spy_1d_return = spy_data["close"].pct_change(1).iloc[-1]
                spy_5d_return = spy_data["close"].pct_change(5).iloc[-1]
                spy_20d_return = spy_data["close"].pct_change(20).iloc[-1]
                
                # Calculate 20-day moving average
                spy_20d_ma = spy_data["close"].rolling(20).mean().iloc[-1]
                spy_price_to_ma = spy_data["close"].iloc[-1] / spy_20d_ma - 1
                
                # Add to summary
                summary["market"] = {
                    "trend_1d": spy_1d_return,
                    "trend_5d": spy_5d_return,
                    "trend_20d": spy_20d_return,
                    "price_to_ma": spy_price_to_ma
                }
        
        # Add market volatility
        if "VIX" in market_data:
            vix_data = market_data["VIX"]
            if len(vix_data) > 1:
                summary["volatility"] = {
                    "vix": vix_data["close"].iloc[-1],
                    "vix_change": vix_data["close"].pct_change(1).iloc[-1]
                }
        
        return summary
    
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
            high_since_entry = ohlcv_data['high'].loc[entry_time:].max() if entry_time in ohlcv_data.index else ohlcv_data['high'].max()
            
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
    
    def _calculate_risk_metrics(self, ohlcv_data: pd.DataFrame, position_data: Dict) -> Dict:
        """
        Calculate risk metrics for a position.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            position_data: Dictionary with position information
            
        Returns:
            Dictionary with risk metrics
        """
        # Get current price
        current_price = ohlcv_data['close'].iloc[-1]
        
        # Get position details
        entry_price = position_data.get('entry_price', current_price)
        entry_time = position_data.get('entry_time', ohlcv_data.index[0])
        
        # Calculate profit percentage
        profit_pct = (current_price / entry_price - 1) * 100
        
        # Calculate highest price since entry
        high_since_entry = ohlcv_data['high'].loc[entry_time:].max() if entry_time in ohlcv_data.index else ohlcv_data['high'].max()
        
        # Calculate maximum profit percentage reached
        max_profit_pct = (high_since_entry / entry_price - 1) * 100
        
        # Calculate drawdown from peak
        drawdown = (high_since_entry - current_price) / high_since_entry * 100
        
        # Calculate drawdown ratio (drawdown / profit)
        drawdown_ratio = drawdown / max(profit_pct, 0.01) if profit_pct > 0 else 0
        
        # Calculate volatility
        returns = ohlcv_data['close'].pct_change().loc[entry_time:] if entry_time in ohlcv_data.index else ohlcv_data['close'].pct_change()
        volatility = returns.std() * 100  # Convert to percentage
        
        # Calculate Sharpe ratio (simplified)
        avg_return = returns.mean()
        sharpe_ratio = avg_return / returns.std() if returns.std() > 0 else 0
        
        # Calculate R multiple (profit / stop loss distance)
        stop_loss = position_data.get('stop_loss', entry_price * 0.95)  # Default to 5% stop loss
        initial_risk = (entry_price - stop_loss) / entry_price * 100  # As percentage
        r_multiple = profit_pct / initial_risk if initial_risk > 0 else 0
        
        return {
            "profit_pct": profit_pct,
            "max_profit_pct": max_profit_pct,
            "drawdown": drawdown,
            "drawdown_ratio": drawdown_ratio,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "r_multiple": r_multiple,
            "max_drawdown": drawdown / 100
        }
    
    def backtest(
        self, 
        historical_data: Dict[str, pd.DataFrame],
        exit_strategy: str = "model",
        initial_capital: float = 10000.0,
        position_size_pct: float = 0.2
    ) -> Dict:
        """
        Backtest the exit optimization model.
        
        Args:
            historical_data: Dictionary mapping symbols to DataFrames
            exit_strategy: Exit strategy to use ('model', 'hold', 'simple')
            initial_capital: Initial capital
            position_size_pct: Position size as percentage of capital
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize backtest results
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
            "trades": []
        }
        
        # Initialize portfolio
        capital = initial_capital
        equity_curve = [capital]
        trades = []
        
        # Process each symbol
        for symbol, df in historical_data.items():
            # Skip if not enough data
            if len(df) < 50:
                continue
            
            # Use a moving window to simulate entries and exits
            entry_threshold = 0.01  # 1% price increase for entry (simple strategy)
            stop_loss_pct = 0.05  # 5% stop loss
            take_profit_pct = 0.1  # 10% take profit
            
            # Loop through data with sliding window
            for i in range(20, len(df) - 20):
                # Check for entry signal
                entry_return = df['close'].iloc[i] / df['close'].iloc[i-1] - 1
                
                # Enter position if signal triggered and not already in a position
                if entry_return > entry_threshold:
                    # Entry point
                    entry_idx = i
                    entry_price = df['close'].iloc[entry_idx]
                    entry_time = df.index[entry_idx]
                    
                    # Calculate position size
                    position_size = capital * position_size_pct / entry_price
                    
                    # Set stop loss and take profit
                    stop_loss = entry_price * (1 - stop_loss_pct)
                    take_profit = entry_price * (1 + take_profit_pct)
                    
                    # Initialize position data
                    position_data = {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "entry_time": entry_time,
                        "position_size": 1.0,  # Full position
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "trailing_stop": 2.0  # 2% trailing stop
                    }
                    
                    # Initialize trade data
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
                        "exit_reason": ""
                    }
                    
                    # Simulate holding the position
                    current_position = 1.0
                    exit_reasons = []
                    
                    for j in range(entry_idx + 1, min(entry_idx + 20, len(df))):
                        # Current price and time
                        current_idx = j
                        current_price = df['close'].iloc[current_idx]
                        current_time = df.index[current_idx]
                        
                        # Skip if position fully closed
                        if current_position <= 0:
                            break
                        
                        # Current slice of data
                        current_data = df.iloc[:current_idx+1].copy()
                        
                        # Update position data
                        position_data["position_size"] = current_position
                        
                        # Evaluate exit conditions
                        if exit_strategy == "model":
                            # Use model to decide exit
                            recommendation = self.evaluate_exit_conditions(
                                current_data,
                                position_data
                            )
                            
                            # Check for exit signal
                            if recommendation["exit"]:
                                exit_size = recommendation["size"] * current_position
                                exit_reason = recommendation["reason"]
                                
                                # Execute exit
                                if exit_size > 0:
                                    # Partial exit
                                    exit_value = exit_size * position_size * current_price
                                    profit_loss = exit_value - (exit_size * position_size * entry_price)
                                    
                                    # Update position
                                    current_position -= exit_size
                                    
                                    # Record exit
                                    if exit_reason not in exit_reasons:
                                        exit_reasons.append(exit_reason)
                                    
                                    # If position fully closed, record trade
                                    if current_position <= 0:
                                        trade["exit_time"] = current_time
                                        trade["exit_price"] = current_price
                                        trade["exit_idx"] = current_idx
                                        trade["profit_loss"] = profit_loss
                                        trade["profit_loss_pct"] = profit_loss / (position_size * entry_price) * 100
                                        trade["duration"] = (current_time - entry_time).total_seconds() / 86400  # In days
                                        trade["exit_reason"] = "+".join(exit_reasons)
                                        
                                        # Update capital
                                        capital += profit_loss
                                        equity_curve.append(capital)
                                        
                                        # Add trade to list
                                        trades.append(trade)
                                        break
                        
                        elif exit_strategy == "simple":
                            # Simple exit strategy based on stop loss and take profit
                            exit_size = 0.0
                            exit_reason = ""
                            
                            # Check stop loss
                            if current_price <= stop_loss:
                                exit_size = 1.0
                                exit_reason = "stop_loss"
                            
                            # Check take profit
                            elif current_price >= take_profit:
                                exit_size = 1.0
                                exit_reason = "take_profit"
                            
                            # Check if max holding period reached
                            elif j - entry_idx >= 10:
                                exit_size = 1.0
                                exit_reason = "time_stop"
                            
                            # Execute exit if triggered
                            if exit_size > 0:
                                # Calculate profit/loss
                                exit_value = position_size * current_price
                                profit_loss = exit_value - (position_size * entry_price)
                                
                                # Update position
                                current_position = 0.0
                                
                                # Record trade
                                trade["exit_time"] = current_time
                                trade["exit_price"] = current_price
                                trade["exit_idx"] = current_idx
                                trade["profit_loss"] = profit_loss
                                trade["profit_loss_pct"] = profit_loss / (position_size * entry_price) * 100
                                trade["duration"] = (current_time - entry_time).total_seconds() / 86400  # In days
                                trade["exit_reason"] = exit_reason
                                
                                # Update capital
                                capital += profit_loss
                                equity_curve.append(capital)
                                
                                # Add trade to list
                                trades.append(trade)
                                break
                        
                        elif exit_strategy == "hold":
                            # Hold until end of simulation period
                            if j == min(entry_idx + 19, len(df) - 1):
                                # Calculate profit/loss
                                exit_value = position_size * current_price
                                profit_loss = exit_value - (position_size * entry_price)
                                
                                # Record trade
                                trade["exit_time"] = current_time
                                trade["exit_price"] = current_price
                                trade["exit_idx"] = current_idx
                                trade["profit_loss"] = profit_loss
                                trade["profit_loss_pct"] = profit_loss / (position_size * entry_price) * 100
                                trade["duration"] = (current_time - entry_time).total_seconds() / 86400  # In days
                                trade["exit_reason"] = "hold_period_end"
                                
                                # Update capital
                                capital += profit_loss
                                equity_curve.append(capital)
                                
                                # Add trade to list
                                trades.append(trade)
                                break
                    
                    # Skip the next 5 bars after a trade
                    i = max(i, trade.get("exit_idx", i) + 5)
        
        # Calculate performance metrics
        if trades:
            # Total trades
            results["total_trades"] = len(trades)
            
            # Winning and losing trades
            winning_trades = [t for t in trades if t["profit_loss"] > 0]
            losing_trades = [t for t in trades if t["profit_loss"] <= 0]
            
            results["winning_trades"] = len(winning_trades)
            results["losing_trades"] = len(losing_trades)
            
            # Win rate
            results["win_rate"] = len(winning_trades) / len(trades) if trades else 0.0
            
            # Average profit and loss
            results["avg_profit"] = sum(t["profit_loss"] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
            results["avg_loss"] = sum(t["profit_loss"] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
            
            # Profit factor
            total_profit = sum(t["profit_loss"] for t in winning_trades)
            total_loss = abs(sum(t["profit_loss"] for t in losing_trades))
            results["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate max drawdown
            peak = equity_curve[0]
            max_drawdown = 0
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            results["max_drawdown"] = max_drawdown
            
            # Calculate returns
            results["total_return"] = (capital - initial_capital) / initial_capital * 100
            
            # Calculate Sharpe ratio
            returns = np.diff(equity_curve) / equity_curve[:-1]
            results["sharpe_ratio"] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            
            # Calculate annualized return (assuming 252 trading days per year)
            if trades:
                start_date = min(t["entry_time"] for t in trades)
                end_date = max(t["exit_time"] for t in trades if t["exit_time"] is not None)
                
                if start_date and end_date:
                    years = (end_date - start_date).total_seconds() / (86400 * 365)
                    if years > 0:
                        results["annualized_return"] = ((1 + results["total_return"] / 100) ** (1 / years) - 1) * 100
            
            # Add trades to results
            results["trades"] = trades
            
            # Add equity curve
            results["equity_curve"] = equity_curve
        
        return results
    
    def export_model(self, export_dir: str = None):
        """
        Export the model for deployment.
        
        Args:
            export_dir: Directory to export the model to (defaults to a timestamped directory)
        """
        if export_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(settings.models_dir, "exports", f"exit_optimization_{timestamp}")
        
        # Create export directory
        os.makedirs(export_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(export_dir, "model.pt")
        self.save_model(model_path)
        
        # Save metadata
        metadata_path = os.path.join(export_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "metadata": self.metadata,
                "training_stats": self.training_stats,
                "performance_metrics": self.performance_metrics,
                "feature_names": self.feature_names,
                "export_date": datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f"Model exported to {export_dir}")
        
        return export_dir

# Create a global instance of the model
exit_optimization_model = ExitOptimizationModel(
    model_path=settings.model.exit_model_path,
    use_sac=True
)

# Utility functions
def evaluate_exit_strategy(ohlcv_data, position_data, confidence_threshold=None):
    """
    Evaluate exit strategy for a position.
    
    Args:
        ohlcv_data: DataFrame with OHLCV data
        position_data: Dictionary with position information
        confidence_threshold: Minimum confidence for taking action
    
    Returns:
        Dictionary with exit recommendation
    """
    return exit_optimization_model.evaluate_exit_conditions(
        ohlcv_data, position_data, confidence_threshold
    )

def backtest_exit_strategies(
    historical_data,
    strategies=["model", "simple", "hold"],
    initial_capital=10000.0,
    position_size_pct=0.2
):
    """
    Backtest multiple exit strategies and compare them.
    
    Args:
        historical_data: Dictionary mapping symbols to DataFrames
        strategies: List of strategies to test
        initial_capital: Initial capital
        position_size_pct: Position size as percentage of capital
    
    Returns:
        Dictionary with backtest results for each strategy
    """
    results = {}
    
    # Run backtest for each strategy
    for strategy in strategies:
        logger.info(f"Backtesting {strategy} strategy...")
        
        strategy_results = exit_optimization_model.backtest(
            historical_data,
            exit_strategy=strategy,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct
        )
        
        results[strategy] = strategy_results
        
        logger.info(f"{strategy} strategy: {strategy_results['total_return']:.2f}% return, "
                   f"{strategy_results['win_rate']*100:.2f}% win rate, "
                   f"{strategy_results['sharpe_ratio']:.2f} Sharpe ratio")
    
    return results

def train_exit_model(training_data, use_sac=True, optimize_hyperparams=False):
    """
    Train the exit optimization model.
    
    Args:
        training_data: List of training episodes
        use_sac: Whether to use SAC algorithm (vs. PPO)
        optimize_hyperparams: Whether to optimize hyperparameters
    
    Returns:
        Training metrics
    """
    model = ExitOptimizationModel(use_sac=use_sac)
    
    if use_sac:
        # Use SAC algorithm
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
            eval_interval=5000
        )
    else:
        # Use legacy PPO algorithm
        metrics = model.train_ppo(
            training_data=training_data,
            epochs=settings.model.epochs,
            batch_size=settings.model.batch_size,
            lr=settings.model.learning_rate,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4
        )
    
    # Save model as default
    model.save_model(settings.model.exit_model_path)
    
    return metrics, model

# Utility functions
def evaluate_exit_strategy(ohlcv_data, position_data, confidence_threshold=None):
    """
    Evaluate exit strategy for a position.
    
    Args:
        ohlcv_data: DataFrame with OHLCV data
        position_data: Dictionary with position information
        confidence_threshold: Minimum confidence for taking action
    
    Returns:
        Dictionary with exit recommendation
    """
    return exit_optimization_model.evaluate_exit_conditions(
        ohlcv_data, position_data, confidence_threshold
    )

def backtest_exit_strategies(
    historical_data,
    strategies=["model", "simple", "hold"],
    initial_capital=10000.0,
    position_size_pct=0.2
):
    """
    Backtest multiple exit strategies and compare them.
    
    Args:
        historical_data: Dictionary mapping symbols to DataFrames
        strategies: List of strategies to test
        initial_capital: Initial capital
        position_size_pct: Position size as percentage of capital
    
    Returns:
        Dictionary with backtest results for each strategy
    """
    results = {}
    
    # Run backtest for each strategy
    for strategy in strategies:
        logger.info(f"Backtesting {strategy} strategy...")
        
        strategy_results = exit_optimization_model.backtest(
            historical_data,
            exit_strategy=strategy,
            initial_capital=initial_capital,
            position_size_pct=position_size_pct
        )
        
        results[strategy] = strategy_results
        
        logger.info(f"{strategy} strategy: {strategy_results['total_return']:.2f}% return, "
                   f"{strategy_results['win_rate']*100:.2f}% win rate, "
                   f"{strategy_results['sharpe_ratio']:.2f} Sharpe ratio")
    
    return results

def train_exit_model(training_data, use_sac=True, optimize_hyperparams=False):
    """
    Train the exit optimization model.
    
    Args:
        training_data: List of training episodes
        use_sac: Whether to use SAC algorithm (vs. PPO)
        optimize_hyperparams: Whether to optimize hyperparameters
    
    Returns:
        Training metrics
    """
    model = ExitOptimizationModel(use_sac=use_sac)
    
    if use_sac:
        # Use SAC algorithm
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
            eval_interval=5000
        )
    else:
        # Use legacy PPO algorithm
        metrics = model.train_ppo(
            training_data=training_data,
            epochs=settings.model.epochs,
            batch_size=settings.model.batch_size,
            lr=settings.model.learning_rate,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4
        )
    
    # Save model as default
    model.save_model(settings.model.exit_model_path)
    
    return metrics, model

# Schedule daily training during off-hours
async def schedule_exit_model_training(days=5):
    """
    Schedule exit model training to run during off-hours.
    
    Args:
        days: Number of days of historical data to use for training
    """
    try:
        # Get current hour
        current_hour = datetime.now().hour
        
        # Define off-hours (typically outside of market hours)
        # US market hours are 9:30 AM - 4:00 PM Eastern Time
        is_off_hours = current_hour < 9 or current_hour > 16
        
        if is_off_hours:
            logger.info("Scheduling exit model training during off-hours")
            
            # Generate training data from historical trades
            training_data = await generate_training_data(days)
            
            if not training_data:
                logger.warning("No training data available")
                return False
            
            # Train model with SAC algorithm
            # Train with hyperparameter optimization every Saturday
            optimize_hyperparams = datetime.now().weekday() == 5  # Saturday
            
            _, model = train_exit_model(
                training_data=training_data,
                use_sac=True,
                optimize_hyperparams=optimize_hyperparams
            )
            
            # Validate model on recent data
            validation_data = await generate_validation_data(1)  # 1 day of validation data
            
            if validation_data:
                backtest_results = model.backtest(
                    historical_data=validation_data,
                    exit_strategy="model"
                )
                
                logger.info(f"Model validation: {backtest_results['total_return']:.2f}% return, "
                           f"{backtest_results['win_rate']*100:.2f}% win rate, "
                           f"{backtest_results['sharpe_ratio']:.2f} Sharpe ratio")
            
            return True
        else:
            logger.info("Not in off-hours, skipping scheduled training")
            return False
    except Exception as e:
        logger.error(f"Error in schedule_exit_model_training: {e}")
        return False

async def generate_training_data(days=5) -> List[Dict]:
    """
    Generate training data from historical trades.
    
    Args:
        days: Number of days of historical data to use
    
    Returns:
        List of training episodes
    """
    # In a real implementation, this would fetch data from database or API
    # Here we'll return a placeholder
    logger.info(f"Generating training data from {days} days of historical trades")
    
    # Placeholder for training data
    training_data = []
    
    # In a real implementation, you would:
    # 1. Fetch historical trades from database
    # 2. Fetch OHLCV data for those trades
    # 3. Convert to the expected format
    
    return training_data

async def generate_validation_data(days=1) -> Dict[str, pd.DataFrame]:
    """
    Generate validation data for backtesting.
    
    Args:
        days: Number of days of historical data to use
    
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    # In a real implementation, this would fetch data from database or API
    # Here we'll return a placeholder
    logger.info(f"Generating validation data from {days} days of historical data")
    
    # Placeholder for validation data
    validation_data = {}
    
    # In a real implementation, you would:
    # 1. Fetch recent OHLCV data for relevant symbols
    # 2. Convert to the expected format
    
    return validation_data
