"""
Training module for the trading system.

This module provides functionality for training ML models:
- Data preparation pipelines
- Model training workflows
- Cross-validation utilities
"""

from .train_models import train_models
from .data_preparation import prepare_training_data

__all__ = [
    "train_models",
    "prepare_training_data",
]
