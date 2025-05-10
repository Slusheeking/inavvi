"""
Models module for the trading system.
This module contains machine learning and quantitative models:
- Ranking models for stock selection
- Sentiment analysis models
- Pattern recognition models
- Exit optimization models
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.sentiment import (
    FinancialSentimentModel,
    analyze_sentiment,
    analyze_news_batch,
    generate_sentiment_report,
    get_entity_sentiment_network,
    sentiment_model,
)
from src.models.pattern_recognition import PatternRecognitionModel, pattern_recognition_model
from src.models.exit_optimization import (
    ExitOptimizationModel,
    evaluate_exit_strategy,
    backtest_exit_strategies,
    train_exit_model,
    exit_optimization_model,
)

__all__ = [
    # Model classes
    "FinancialSentimentModel",
    "PatternRecognitionModel",
    "ExitOptimizationModel",
    # Utility functions
    "analyze_sentiment",
    "analyze_news_batch",
    "generate_sentiment_report",
    "get_entity_sentiment_network",
    "evaluate_exit_strategy",
    "backtest_exit_strategies",
    "train_exit_model",
    # Global instances
    "sentiment_model",
    "pattern_recognition_model",
    "exit_optimization_model",
]