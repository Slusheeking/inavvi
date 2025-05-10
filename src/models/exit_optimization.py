"""
Exit optimization model for determining optimal exit points.

This model uses advanced reinforcement learning (SAC) to optimize trade exit decisions
based on price action, indicators, position state, and market context.
"""
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.distributions import Normal
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler

from src.config.settings import settings
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("exit_optimization")

# Device configuration - using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Discrete action mapping
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
        lr: float = 3e-4,
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
        if scaler:
            state = scaler.transform(state.reshape(1, -1)).flatten()

        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor_critic.get_action(state, evaluation=evaluate)
        if evaluate:
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
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            state_dim = self.actor_critic.critic1[0].in_features - self.actor_critic.mean.out_features
            action_dim = self.actor_critic.mean.out_features
            
            checkpoint = {
                'model_version': 'v2',
                'architecture': 'ActorCritic',
                'state_dim': state_dim,
                'action_dim': action_dim,
                'hidden_dim': 256,
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
            
            checkpoint = {k: v for k, v in checkpoint.items() if v is not None}
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                logger.error(f"Model file not found at {path}")
                return False
            
            checkpoint = torch.load(path, map_location=device)
            
            if "actor_critic_state_dict" in checkpoint:
                self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
                logger.info("Loaded combined actor-critic model")
            else:
                logger.warning(f"No recognized state dict keys found in {path}")
                return False
            
            try:
                if "target_value_state_dict" in checkpoint:
                    self.target_value.load_state_dict(checkpoint["target_value_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load target_value: {e}")
                
            try:
                if "actor_optimizer_state_dict" in checkpoint:
                    self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
                if "critic_optimizer_state_dict" in checkpoint:
                    self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
                if "value_optimizer_state_dict" in checkpoint:
                    self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not load optimizer state: {e}")
                
            if "alpha" in checkpoint:
                self.alpha = checkpoint["alpha"]
                
            if self.automatic_entropy_tuning and "log_alpha" in checkpoint:
                self.log_alpha = checkpoint["log_alpha"]
                if "alpha_optimizer_state_dict" in checkpoint:
                    try:
                        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
                    except Exception as e:
                        logger.warning(f"Could not load alpha optimizer: {e}")
                        
            if "training_info" in checkpoint:
                self.training_info = checkpoint["training_info"]
                
            logger.info("Model loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
            
        self.version = "1.0.0-SAC"
        self.metadata = {
            "features": self.feature_names,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "algorithm": "SAC",
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
        self.scaler = None
        
        if model_path:
            if not os.path.isabs(model_path):
                model_path = os.path.join(settings.models_dir, os.path.basename(model_path))
            if os.path.exists(model_path):
                self.load_model(model_path)
                logger.info(f"Loaded exit optimization model from {model_path}")
        else:
            logger.warning("No model file provided, using untrained model")

    def save_model(self, model_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.agent.save(model_path)
            metadata_path = os.path.join(os.path.dirname(model_path), f"{Path(model_path).stem}_metadata.json")
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
            if self.scaler:
                import joblib
                scaler_path = os.path.join(os.path.dirname(model_path), f"{Path(model_path).stem}_scaler.joblib")
                joblib.dump(self.scaler, scaler_path)
                
            logger.info(f"Saved model to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        try:
            success = self.agent.load(model_path)
            
            metadata_path = os.path.join(os.path.dirname(model_path), f"{Path(model_path).stem}_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                    self.metadata = data.get("metadata", self.metadata)
                    self.training_stats = data.get("training_stats", self.training_stats)
                    self.performance_metrics = data.get("performance_metrics", self.performance_metrics)
            
            # Load scaler if it exists
            import joblib
            scaler_path = os.path.join(os.path.dirname(model_path), f"{Path(model_path).stem}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
                
            return success
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

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
            self.scaler = None
            
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
                    
                    # Scale states before agent selection
                    scaled_state = self.scaler.transform(state.reshape(1, -1)).flatten() if self.scaler else state
                    
                    if total_steps < start_steps:
                        action = np.random.uniform(0, 1, 1)[0]
                    else:
                        action = self.agent.select_action(scaled_state)
                        
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
                    
                    # Scale states before pushing to replay buffer
                    scaled_next_state = self.scaler.transform(next_state.reshape(1, -1)).flatten() if self.scaler else next_state
                    
                    self.agent.replay_buffer.push(scaled_state, np.array([exit_size]), reward, scaled_next_state, done)
                    
                    if len(self.agent.replay_buffer) > batch_size:
                        for _ in range(updates_per_step):
                            self.agent.update(batch_size)
                            
                    state = next_state
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
                
                action = self.agent.select_action(state, evaluate=True, scaler=self.scaler)
                
                if isinstance(action, int):
                    exit_size = [0.0, 0.25, 0.33, 0.5, 1.0][action]
                else:
                    exit_size = float(action)
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
            
        df = ohlcv_data.copy()
        current_price = df["close"].iloc[-1]
        entry_price = position_data.get("entry_price", current_price)
        entry_time = position_data.get("entry_time", df.index[0])
        position_size = position_data.get("position_size", 1.0)
        if isinstance(entry_time, str):
            try:
                entry_time = pd.Timestamp(entry_time)
            except Exception:
                entry_time = df.index[0]
                
        time_in_trade = (df.index[-1] - entry_time).total_seconds() / 86400
        time_in_trade = min(time_in_trade, 1.0)
        profit_pct = (current_price / entry_price - 1) * 100
        high_since_entry = df["high"].loc[entry_time:].max() if entry_time in df.index else df["high"].max()
        low_since_entry = df["low"].loc[entry_time:].min() if entry_time in df.index else df["low"].min()
        price_to_entry = current_price / entry_price - 1
        price_to_high = current_price / high_since_entry - 1 if high_since_entry > 0 else 0
        price_to_low = current_price / low_since_entry - 1 if low_since_entry > 0 else 0
        
        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        bb_middle = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-8)
        
        # Calculate MACD
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        # Calculate volatility
        df["returns"] = df["close"].pct_change()
        volatility_5d = df["returns"].rolling(5).std().iloc[-1] * 100
        
        # Calculate ATR
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift(1))
        tr3 = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        normalized_atr = atr / current_price * 100
        
        # Calculate volume metrics
        volume_sma_5 = df["volume"].rolling(5).mean()
        volume_ratio_5 = df["volume"].iloc[-1] / volume_sma_5.iloc[-1] if volume_sma_5.iloc[-1] else 1.0
        
        # Get market and sector trends
        market_trend = 0.0
        sector_trend = 0.0
        if "market_trend" in position_data:
            market_trend = position_data.get("market_trend", 0.0)
        else:
            if len(df) >= 50:
                market_trend = (df["close"].iloc[-1] / df["close"].iloc[-50] - 1) * 100 / 10
                market_trend = max(min(market_trend, 1.0), -1.0)
                
        if "sector_trend" in position_data:
            sector_trend = position_data.get("sector_trend", 0.0)
            
        # Calculate Sharpe ratio and max drawdown
        returns = df["returns"].iloc[-20:].dropna()
        sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0.0
        close_prices = df["close"].loc[entry_time:].values if entry_time in df.index else df["close"].values
        peak = np.maximum.accumulate(close_prices)
        drawdown = (peak - close_prices) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Normalize features
        profit_pct = np.clip(profit_pct, -20, 20)
        rsi_14 = np.clip(rsi_14.iloc[-1] / 100, 0, 1)
        bb_position = np.clip(bb_position, 0, 1)
        macd_hist = np.clip(macd_histogram.iloc[-1], -1, 1)
        volume_ratio_5 = np.clip(volume_ratio_5, 0, 5)
        
        # Create feature vector
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
        
        # Handle NaN values
        features = np.array(features, dtype=np.float32)
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Detected NaN or infinite values in features")
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return features

    def predict_exit_action(self, ohlcv_data: pd.DataFrame, position_data: Dict) -> Dict[str, Union[str, float]]:
        features = self._extract_features(ohlcv_data, position_data)
        
        # Apply scaling if available
        if self.scaler:
            scaled_features = self.scaler.transform(features.reshape(1, -1)).flatten()
        else:
            scaled_features = features
            
        features_tensor = torch.FloatTensor(scaled_features).to(device)
        
        with torch.no_grad():
            state = features_tensor.unsqueeze(0)
            action, _ = self.agent.actor_critic.get_action(state, evaluation=True)
            action_value = action.cpu().numpy()[0, 0]
            exit_size = float(action_value)
            exit_size = max(0.0, min(1.0, exit_size))
            
            # Map continuous action to discrete action
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
                
            # Calculate confidence scores
            probabilities = {
                "hold": 1.0 - min(exit_size * 10, 1.0) if exit_size < 0.1 else 0.0,
                "exit_quarter": 1.0 - min(abs(exit_size - 0.2) * 5, 1.0) if 0.1 <= exit_size < 0.3 else 0.0,
                "exit_third": 1.0 - min(abs(exit_size - 0.37) * 5, 1.0) if 0.3 <= exit_size < 0.45 else 0.0,
                "exit_half": 1.0 - min(abs(exit_size - 0.6) * 5, 1.0) if 0.45 <= exit_size < 0.75 else 0.0,
                "exit_full": 1.0 - min((1.0 - exit_size) * 4, 1.0) if exit_size >= 0.75 else 0.0,
            }
            
            result = {
                "action": action_name,
                "action_idx": action_idx,
                "exit_size": exit_size,
                "confidence": 0.8,
                "probabilities": probabilities,
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
        # exit_size is available from prediction but not used in this function
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
        recommendation.update(manual_checks)
        
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
        # entry_price is retrieved but not used in this function
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
                except Exception:
                    entry_time = ohlcv_data.index[0]
            time_diff = (ohlcv_data.index[-1] - entry_time).total_seconds() / 3600
            if time_diff >= max_time:
                results["time_stop_triggered"] = True
                
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


# Create global instance
exit_optimization_model = ExitOptimizationModel(
    model_path=getattr(settings.model, "exit_model_path", None),
    use_sac=True
)


def evaluate_exit_strategy(ohlcv_data: pd.DataFrame, position_data: Dict, confidence_threshold: Optional[float] = None) -> Dict:
    """Evaluate exit strategy for a given position."""
    return exit_optimization_model.evaluate_exit_conditions(ohlcv_data, position_data, confidence_threshold)


def backtest_exit_strategies(
    historical_data: Dict[str, pd.DataFrame],
    strategies: List[str] = ["model", "simple", "hold"],
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.2,
) -> Dict[str, Dict]:
    """Backtest different exit strategies."""
    results = {}
    
    for strategy in strategies:
        logger.info(f"Backtesting {strategy} strategy...")
        # Create a simplified backtest here
        total_return = 0.0
        win_rate = 0.0
        max_drawdown = 0.0
        sharpe_ratio = 0.0
        trades = []
        
        # Simple model-free backtesting implementation
        for symbol, ohlcv_data in historical_data.items():
            # Implement a basic backtesting algorithm for each strategy
            if strategy == "model":
                # Use the model for exit decisions
                pass
            elif strategy == "simple":
                # Use simple rules (stop loss, take profit)
                pass
            elif strategy == "hold":
                # Hold until the end of the period
                pass
        
        # Store strategy results        
        results[strategy] = {
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "trades": trades
        }
        
        logger.info(
            f"{strategy} strategy: {total_return:.2f}% return, "
            f"{win_rate*100:.2f}% win rate, "
            f"{sharpe_ratio:.2f} Sharpe ratio"
        )
        
    return results


def train_exit_model(training_data: List[Dict], epochs: int = 10, batch_size: int = 256) -> Tuple[Dict, ExitOptimizationModel]:
    """Train the exit optimization model with the given data."""
    model = ExitOptimizationModel(use_sac=True)
    metrics = model.train_sac(
        training_data=training_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        updates_per_step=1,
        start_steps=1000,
        eval_interval=5000,
    )
    model.save_model(settings.model.exit_model_path)
    return metrics, model