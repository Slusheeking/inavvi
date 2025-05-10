"""
LLM (Large Language Model) integration module.

This module provides functionality for working with large language models:
- Prompt templates and generation
- Response parsing and extraction
- LLM provider routing
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.llm.parsing import (
    extract_json_from_text,
    extract_market_analysis,
    extract_sentiment_score,
    extract_structured_data,
    extract_trading_signals,
    parse_exit_decision,
    parse_llm_response,
    parse_market_analysis,
    parse_trade_decision,
)
from src.llm.prompts import PromptTemplates
from src.llm.router import LLMRouter, get_llm_router, llm_router, openrouter_client


# Create module-level convenience functions that delegate to PromptTemplates class methods
def get_prompt_template(task_type: str) -> str:
    """
    Get the appropriate prompt template for a specific task.

    Args:
        task_type: The type of task (e.g., "market_analysis", "pattern_recognition")

    Returns:
        The corresponding prompt template string
    """
    return PromptTemplates.get_prompt_template(task_type)


def render_prompt(template: str, data: dict) -> str:
    """
    Format a prompt template with provided data.

    Args:
        template: The prompt template string
        data: A dictionary of values to insert into the template

    Returns:
        The formatted prompt string
    """
    return PromptTemplates.render_prompt(template, data)


# Create module-level constants that reference class attributes
MARKET_ANALYSIS_PROMPT = PromptTemplates.MARKET_ANALYSIS_PROMPT
SENTIMENT_ANALYSIS_PROMPT = PromptTemplates.SENTIMENT_ANALYSIS_PROMPT
PATTERN_RECOGNITION_PROMPT = PromptTemplates.PATTERN_RECOGNITION_PROMPT
TRADING_DECISION_PROMPT = PromptTemplates.TRADING_DECISION_PROMPT

__all__ = [
    # Prompts
    "PromptTemplates",
    "get_prompt_template",
    "render_prompt",
    "MARKET_ANALYSIS_PROMPT",
    "SENTIMENT_ANALYSIS_PROMPT",
    "PATTERN_RECOGNITION_PROMPT",
    "TRADING_DECISION_PROMPT",
    # Parsing
    "parse_trade_decision",
    "parse_exit_decision",
    "parse_market_analysis",
    "extract_json_from_text",
    "parse_llm_response",
    "extract_market_analysis",
    "extract_sentiment_score",
    "extract_trading_signals",
    "extract_structured_data",
    # Router
    "openrouter_client",
    "get_llm_router",
    "LLMRouter",
    "llm_router",
]
