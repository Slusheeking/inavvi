"""
LLM (Large Language Model) integration module.

This module provides functionality for working with large language models:
- Prompt templates and generation
- Response parsing and extraction
- LLM provider routing
"""

from .prompts import PromptTemplates
from .parsing import (
    parse_trade_decision,
    parse_exit_decision,
    parse_market_analysis,
    extract_json_from_text,
    parse_llm_response,
    extract_market_analysis,
    extract_sentiment_score,
    extract_trading_signals,
    extract_structured_data
)
from .router import (
    openrouter_client,
    get_llm_router,
    LLMRouter,
    llm_router
)

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
    "llm_router"
]
