"""
Training module for the trading system.

This module provides functionality for training ML models:
- Data preparation pipelines
- Model training workflows
- Cross-validation utilities
"""

from .train_models import train_all_models

__all__ = [
    "train_all_models",

]
