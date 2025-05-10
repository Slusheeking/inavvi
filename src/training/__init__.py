"""
Training module for the trading system.

This module provides functionality for training ML models:
- Data preparation pipelines
- Model training workflows
- Cross-validation utilities
"""

from .data_preparation import prepare_training_data
from .train_models import train_models

__all__ = [
    "train_models",
    "prepare_training_data",
]
