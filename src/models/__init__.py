"""
Models module for the trading system.

This module contains machine learning and quantitative models:
- Ranking models for stock selection
- Sentiment analysis models
- Pattern recognition models
- Exit optimization models
"""

import importlib
import functools

# Lazy loading mechanism for models to improve import performance
@functools.lru_cache(maxsize=None)
def get_model(name):
    """
    Lazily import and return a model class.
    This improves startup performance by only loading modules when needed.
    
    Args:
        name (str): Name of the model to load
        
    Returns:
        class: The requested model class
    """
    if name == "RankingModel":
        from .ranking_model import RankingModel
        return RankingModel
    elif name == "MultiFactorRankingModel":
        from .multi_factor_ranking import MultiFactorRankingModel
        return MultiFactorRankingModel
    elif name == "SentimentModel":
        from .sentiment import SentimentModel
        return SentimentModel
    elif name == "PatternRecognitionModel":
        from .pattern_recognition import PatternRecognitionModel
        return PatternRecognitionModel
    elif name == "ExitOptimizationModel":
        from .exit_optimization import ExitOptimizationModel
        return ExitOptimizationModel
    else:
        raise ValueError(f"Unknown model: {name}")

# Define proxy classes that load the actual implementation on demand
class RankingModel:
    def __new__(cls, *args, **kwargs):
        return get_model("RankingModel")(*args, **kwargs)

class SentimentModel:
    def __new__(cls, *args, **kwargs):
        return get_model("SentimentModel")(*args, **kwargs)

class PatternRecognitionModel:
    def __new__(cls, *args, **kwargs):
        return get_model("PatternRecognitionModel")(*args, **kwargs)

class ExitOptimizationModel:
    def __new__(cls, *args, **kwargs):
        return get_model("ExitOptimizationModel")(*args, **kwargs)

# Import key functions
from .multi_factor_ranking import rank_opportunities, get_model_weights

__all__ = [
    # Model classes
    "RankingModel",
    "MultiFactorRankingModel",
    "SentimentModel",
    "PatternRecognitionModel",
    "ExitOptimizationModel",
    
    # Factory function
    "get_model",
    
    # Utility functions
    "rank_opportunities",
    "get_model_weights",
]
