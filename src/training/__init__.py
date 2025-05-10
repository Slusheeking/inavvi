"""
Training module for the trading system.

This module provides functionality for training ML models:
- Data preparation pipelines
- Model training workflows
- Cross-validation utilities
- Hyperparameter optimization
- Scheduled off-hours training
- Model evaluation and metrics tracking
"""

from .train_models import (
    ModelTrainer,
    parse_args,
    main,
    run_scheduled_training,
    schedule_training
)

# Define public API
__all__ = [
    # Main classes
    "ModelTrainer",
    
    # Main functions
    "main",
    "parse_args",
    "run_scheduled_training",
    "schedule_training",
]