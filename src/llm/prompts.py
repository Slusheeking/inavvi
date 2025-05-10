"""
Prompt templates for LLM integration.
"""

import os
import sys
from typing import Any, Dict, List

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


class PromptTemplates:
    """
    Collection of prompt templates for LLM interactions.
    """

    # Specific prompt templates
    MARKET_ANALYSIS_PROMPT = """
    Analyze the provided market data and identify the current market regime.
    Consider key indicators such as volatility (VIX), market breadth, sector rotations,
    volume patterns, and major index performance. Evaluate how these factors may impact
    trading opportunities.
    
    Focus on:
    1. Market structure (bullish, bearish, sideways)
    2. Volatility regime (high, normal, low)
    3. Liquidity conditions and volume patterns
    4. Relative sector strength
    5. Key support/resistance levels
    
    Provide your analysis in the following JSON format:
    {
        "market_regime": "bullish/bearish/range_bound/volatile/uncertain",
        "volatility_state": "high/normal/low",
        "key_levels": {
            "support": [level1, level2],
            "resistance": [level1, level2]
        },
        "strongest_sectors": ["sector1", "sector2"],
        "weakest_sectors": ["sector1", "sector2"],
        "recommendation": "brief trading approach based on current conditions",
        "risk_level": "high/medium/low"
    }
    """

    PATTERN_RECOGNITION_PROMPT = """
    Analyze the provided price action data and identify any significant technical patterns.
    Focus on classical chart patterns, candlestick formations, and technical indicator setups
    that may signal potential trading opportunities.
    
    Evaluate the following:
    1. Chart patterns (e.g., head and shoulders, triangles, flags)
    2. Candlestick patterns (e.g., doji, engulfing, morning star)
    3. Support/resistance breaches
    4. Indicator divergences (e.g., price/RSI divergence)
    5. Volume confirmation signals
    
    Provide your analysis in the following JSON format:
    {
        "detected_patterns": [
            {
                "name": "pattern name",
                "type": "reversal/continuation",
                "confidence": float (0.0-1.0),
                "price_target": float or null,
                "timeframe": "intraday/daily/weekly"
            }
        ],
        "key_levels": {
            "support": [level1, level2],
            "resistance": [level1, level2]
        },
        "indicator_signals": [
            {
                "indicator": "indicator name",
                "signal": "bullish/bearish/neutral",
                "strength": float (0.0-1.0)
            }
        ],
        "trading_implication": "brief description of pattern significance"
    }
    """

    SENTIMENT_ANALYSIS_PROMPT = """
    Analyze the provided news articles, social media data, and market sentiment indicators
    to determine the current sentiment around the specified asset or market segment.
    
    Consider the following sentiment sources:
    1. Recent news headlines and their tone
    2. Social media sentiment (Twitter, Reddit, StockTwits)
    3. Analyst ratings and price target changes
    4. Options market sentiment (put/call ratio)
    5. Institutional flows and positioning
    
    Provide your analysis in the following JSON format:
    {
        "overall_sentiment": "bullish/bearish/neutral/mixed",
        "sentiment_score": float (-1.0 to 1.0, where -1 is extremely bearish, +1 is extremely bullish),
        "news_sentiment": {
            "score": float (-1.0 to 1.0),
            "key_themes": ["theme1", "theme2"]
        },
        "social_sentiment": {
            "score": float (-1.0 to 1.0),
            "volume": "high/medium/low",
            "key_topics": ["topic1", "topic2"]
        },
        "institutional_sentiment": "bullish/bearish/neutral",
        "sentiment_change": "improving/deteriorating/stable",
        "contrarian_indicators": ["indicator1", "indicator2"] or []
    }
    """

    TRADING_DECISION_PROMPT = """
    Based on the provided market analysis, pattern recognition, sentiment analysis, and portfolio state,
    make a trading decision for the specified asset. Consider risk parameters, current market conditions,
    and portfolio constraints.
    
    Evaluate the following:
    1. Signal strength and conviction
    2. Market regime compatibility
    3. Risk/reward ratio
    4. Position sizing based on volatility
    5. Entry timing and execution strategy
    6. Potential exit points and stop loss levels
    
    Provide your decision in the following JSON format:
    {
        "decision": "buy/sell/hold",
        "conviction": float (0.0-1.0),
        "position_size": float (0.0-1.0, as fraction of max allocation),
        "entry": {
            "price_target": float or "market",
            "valid_until": "time limit for entry",
            "execution_strategy": "market/limit/conditional",
        },
        "exit": {
            "stop_loss": float,
            "take_profit": float or null,
            "trailing_stop": float or null,
            "time_stop": "time limit for position" or null
        },
        "risk_per_trade": float (dollar amount or percentage),
        "key_reasons": ["reason1", "reason2", "reason3"],
        "key_risks": ["risk1", "risk2"]
    }
    """

    # System prompts

    SYSTEM_TRADE_DECISION = """You are an expert day trader assistant specializing in short-term trading.
    Your goal is to analyze the provided stock data, market context, and portfolio state to
    make a binary trading decision (trade or do not trade) along with a specified position size.
    Your analysis should be concise and focus on the most relevant factors.
    
    Provide your response in the following JSON format:
    {
        "decision": "trade" or "no_trade",
        "position_size": float (0.0 to 1.0, as a fraction of max position size),
        "confidence": float (0.0 to 1.0),
        "reasoning": "Brief explanation of your decision",
        "key_factors": ["factor1", "factor2", "factor3"]
    }
    
    The position_size should be 0 for no_trade decisions, and between 0.25 and 1.0 for trade decisions.
    """

    SYSTEM_EXIT_DECISION = """You are an expert day trader assistant specializing in exit decisions.
    Your goal is to analyze the provided position data, current stock data, market context,
    and exit signals to make an exit decision. Your analysis should be concise and focus on
    the most relevant factors.
    
    Provide your response in the following JSON format:
    {
        "decision": "exit" or "hold",
        "exit_size": float (0.0 to 1.0, as a fraction of current position),
        "confidence": float (0.0 to 1.0),
        "reasoning": "Brief explanation of your decision",
        "key_factors": ["factor1", "factor2", "factor3"]
    }
    
    The exit_size should be 0.0 for hold decisions, and between 0.25 and 1.0 for exit decisions.
    """

    SYSTEM_MARKET_ANALYSIS = """You are an expert market analyst specializing in identifying market regimes and conditions.
    Your goal is to analyze the provided market data to determine the current market regime and provide
    trading recommendations based on the identified regime. Your analysis should be concise and focus on
    the most relevant factors.
    
    Provide your response in the following JSON format:
    {
        "market_regime": "bullish", "bearish", "range_bound", "volatile", or "uncertain",
        "confidence": float (0.0 to 1.0),
        "reasoning": "Brief explanation of your analysis",
        "key_indicators": ["indicator1", "indicator2", "indicator3"],
        "trading_recommendation": "Brief trading recommendation based on the regime"
    }
    """

    # Function to format the stock data message
    @staticmethod
    def format_stock_data(stock_data: Dict[str, Any]) -> str:
        """
        Format stock data for inclusion in prompts.

        Args:
            stock_data: Dictionary containing stock data

        Returns:
            Formatted string
        """
        return f"""
        # Stock Data
        Symbol: {stock_data.get("symbol", "Unknown")}
        Current Price: ${stock_data.get("price", {}).get("last", 0):.2f}
        Daily Change: {stock_data.get("price", {}).get("change_pct", 0):.2f}%
        Volume: {stock_data.get("price", {}).get("volume", 0):,}
        
        # Technical Analysis
        RSI(14): {stock_data.get("indicators", {}).get("rsi_14", 0):.2f}
        MACD Histogram: {stock_data.get("indicators", {}).get("macd_histogram", 0):.4f}
        Bollinger Band Position: {stock_data.get("indicators", {}).get("bb_position", 0):.2f}
        
        # Pattern Recognition
        Detected Pattern: {stock_data.get("pattern", {}).get("name", "None")}
        Pattern Confidence: {stock_data.get("pattern", {}).get("confidence", 0):.2f}
        
        # News Sentiment
        Overall Sentiment: {stock_data.get("sentiment", {}).get("overall_score", 0):.2f}
        Recent News Count: {len(stock_data.get("news", []))}
        """

    # Function to format the market context message
    @staticmethod
    def format_market_context(market_context: Dict[str, Any]) -> str:
        """
        Format market context for inclusion in prompts.

        Args:
            market_context: Dictionary containing market context

        Returns:
            Formatted string
        """
        return f"""
        # Market Context
        Market State: {market_context.get("state", "Unknown")}
        Sector Performance: {market_context.get("sector_performance", 0):.2f}%
        VIX: {market_context.get("vix", 0):.2f}
        Market Breadth: {market_context.get("breadth", 0):.2f}
        Time Until Close: {market_context.get("time_until_close", "Unknown")} hours
        """

    # Function to format the portfolio state message
    @staticmethod
    def format_portfolio_state(portfolio_state: Dict[str, Any]) -> str:
        """
        Format portfolio state for inclusion in prompts.

        Args:
            portfolio_state: Dictionary containing portfolio state

        Returns:
            Formatted string
        """
        return f"""
        # Portfolio State
        Current Positions: {portfolio_state.get("position_count", 0)} / {portfolio_state.get("max_positions", 3)}
        Available Capital: ${portfolio_state.get("available_capital", 0):.2f}
        Daily P&L: ${portfolio_state.get("daily_pnl", 0):.2f} ({portfolio_state.get("daily_pnl_pct", 0):.2f}%)
        Daily Risk Remaining: ${portfolio_state.get("risk_remaining", 0):.2f}
        """

    # Function to format the position data message
    @staticmethod
    def format_position_data(position_data: Dict[str, Any]) -> str:
        """
        Format position data for inclusion in prompts.

        Args:
            position_data: Dictionary containing position data

        Returns:
            Formatted string
        """
        return f"""
        # Position Data
        Symbol: {position_data.get("symbol", "Unknown")}
        Entry Price: ${position_data.get("entry_price", 0):.2f}
        Current Price: ${position_data.get("current_price", 0):.2f}
        Quantity: {position_data.get("quantity", 0)}
        Unrealized P&L: ${position_data.get("unrealized_pnl", 0):.2f} ({position_data.get("unrealized_pnl_pct", 0):.2f}%)
        Time in Trade: {position_data.get("time_in_trade", 0):.2f} hours
        """

    # Function to format the exit signals message
    @staticmethod
    def format_exit_signals(exit_signals: Dict[str, Any]) -> str:
        """
        Format exit signals for inclusion in prompts.

        Args:
            exit_signals: Dictionary containing exit signals

        Returns:
            Formatted string
        """
        return f"""
        # Exit Signals
        ML Model Recommendation: {exit_signals.get("recommendation", {}).get("reason", "hold_position")}
        ML Confidence: {exit_signals.get("recommendation", {}).get("confidence", 0):.2f}
        Stop Loss Triggered: {exit_signals.get("stop_loss_triggered", False)}
        Take Profit Triggered: {exit_signals.get("take_profit_triggered", False)}
        Trailing Stop Triggered: {exit_signals.get("trailing_stop_triggered", False)}
        Time Stop Triggered: {exit_signals.get("time_stop_triggered", False)}
        """

    # Function to create a complete trade decision prompt
    @staticmethod
    def create_trade_decision_prompt(
        stock_data: Dict[str, Any], market_context: Dict[str, Any], portfolio_state: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Create a complete prompt for trade decisions.

        Args:
            stock_data: Dictionary containing stock data
            market_context: Dictionary containing market context
            portfolio_state: Dictionary containing portfolio state

        Returns:
            List of message dictionaries for LLM
        """
        # Create system message
        system_message = {"role": "system", "content": PromptTemplates.SYSTEM_TRADE_DECISION}

        # Create user message
        user_content = (
            PromptTemplates.format_stock_data(stock_data)
            + PromptTemplates.format_market_context(market_context)
            + PromptTemplates.format_portfolio_state(portfolio_state)
        )

        user_message = {"role": "user", "content": user_content}

        return [system_message, user_message]

    # Function to create a complete exit decision prompt
    @staticmethod
    def create_exit_decision_prompt(
        position_data: Dict[str, Any],
        current_data: Dict[str, Any],
        market_context: Dict[str, Any],
        exit_signals: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Create a complete prompt for exit decisions.

        Args:
            position_data: Dictionary containing position data
            current_data: Dictionary containing current stock data
            market_context: Dictionary containing market context
            exit_signals: Dictionary containing exit signals

        Returns:
            List of message dictionaries for LLM
        """
        # Create system message
        system_message = {"role": "system", "content": PromptTemplates.SYSTEM_EXIT_DECISION}

        # Create user message
        user_content = (
            PromptTemplates.format_position_data(position_data)
            + PromptTemplates.format_stock_data(current_data)
            + PromptTemplates.format_market_context(market_context)
            + PromptTemplates.format_exit_signals(exit_signals)
        )

        user_message = {"role": "user", "content": user_content}

        return [system_message, user_message]

    # Function to create a complete market analysis prompt
    @staticmethod
    def create_market_analysis_prompt(
        market_data: Dict[str, Any], market_context: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Create a complete prompt for market analysis.

        Args:
            market_data: Dictionary containing market data
            market_context: Dictionary containing market context

        Returns:
            List of message dictionaries for LLM
        """
        # Create system message
        system_message = {"role": "system", "content": PromptTemplates.SYSTEM_MARKET_ANALYSIS}

        # Create user message
        user_content = PromptTemplates.format_stock_data(
            market_data
        ) + PromptTemplates.format_market_context(market_context)

        user_message = {"role": "user", "content": user_content}

        return [system_message, user_message]

    @staticmethod
    def get_prompt_template(task_type: str) -> str:
        """
        Get the appropriate prompt template for a specific task.

        Args:
            task_type: The type of task (e.g., "market_analysis", "pattern_recognition")

        Returns:
            The corresponding prompt template string

        Raises:
            ValueError: If the task type is not recognized
        """
        task_type_mapping = {
            "market_analysis": PromptTemplates.MARKET_ANALYSIS_PROMPT,
            "pattern_recognition": PromptTemplates.PATTERN_RECOGNITION_PROMPT,
            "sentiment_analysis": PromptTemplates.SENTIMENT_ANALYSIS_PROMPT,
            "trading_decision": PromptTemplates.TRADING_DECISION_PROMPT,
        }

        if task_type not in task_type_mapping:
            raise ValueError(
                f"Unknown task type: {task_type}. Available types: {list(task_type_mapping.keys())}"
            )

        return task_type_mapping[task_type]

    @staticmethod
    def render_prompt(template: str, data: Dict[str, Any]) -> str:
        """
        Format a prompt template with provided data.

        Args:
            template: The prompt template string
            data: A dictionary of values to insert into the template

        Returns:
            The formatted prompt string

        Example:
            template = "Analyze {symbol} with price {price}"
            data = {"symbol": "AAPL", "price": 150.25}
            result = "Analyze AAPL with price 150.25"
        """
        # Create a simplified template with values directly from the data dictionary
        try:
            return template.format(**data)
        except KeyError as e:
            # Handle missing keys gracefully
            raise KeyError(f"Missing required data field: {e}")
        except Exception as e:
            # Handle other formatting errors
            raise ValueError(f"Error formatting prompt template: {e}")

# If this file is run directly, run a simple test
if __name__ == "__main__":
    print("LLM Prompts module initialized successfully.")
    print("This module provides prompt templates for LLM interactions.")
    print("\nTo use this module, import it in your code:")
    print("from src.llm.prompts import PromptTemplates")
    print("\nExample usage:")
    print("template = PromptTemplates.get_prompt_template('market_analysis')")
    print("formatted_prompt = PromptTemplates.render_prompt(template, data)")
