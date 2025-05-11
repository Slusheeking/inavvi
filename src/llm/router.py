"""
LLM Router for integrating multiple LLM providers.

This module provides classes and functions for routing LLM requests to different
providers and models based on the task type and other criteria.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.metrics.ml_metrics import get_collector, MetricsTimer

# Set up logger
logger = setup_logger("llm_router")


class OpenRouterClient:
    """
    Client for OpenRouter API using OpenAI SDK.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key or settings.api.openrouter_api_key
        
        # Create OpenAI client with OpenRouter base URL
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )

        # Default model
        self.default_model = settings.llm.model

        logger.info(f"OpenRouter client initialized with default model: {self.default_model}")

    async def close(self):
        """Close the client session."""
        # The AsyncOpenAI client handles session cleanup automatically
        logger.info("OpenRouter session closed")

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Get chat completion from OpenRouter API using OpenAI SDK.

        Args:
            messages: List of message dictionaries
            model: Model to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Returns:
            API response
        """
        # Use default values if not provided
        model = model or self.default_model
        temperature = temperature or settings.llm.temperature
        max_tokens = max_tokens or settings.llm.max_tokens

        try:
            # Use metrics timer to measure request latency
            with MetricsTimer("llm_router", "api_request"):
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    extra_headers={
                        "HTTP-Referer": "https://trading-system.com",  # Replace with your domain
                        "X-Title": settings.app_name,
                    }
                )
            
            # Convert the response to a dictionary for compatibility with existing code
            if stream:
                return response  # Return the streaming response object
            else:
                # Convert Pydantic model to dict
                response_dict = {
                    "id": response.id,
                    "object": response.object,
                    "created": response.created,
                    "model": response.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "message": {
                                "role": choice.message.role,
                                "content": choice.message.content
                            },
                            "finish_reason": choice.finish_reason
                        }
                        for choice in response.choices
                    ]
                }
                return response_dict
                
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            return {"error": str(e)}

    async def get_trade_decision(
        self,
        stock_data: Dict[str, Any],
        market_context: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get trade decision from LLM.

        Args:
            stock_data: Data for the stock
            market_context: Current market context
            portfolio_state: Current portfolio state

        Returns:
            Trade decision
        """
        # Prepare system message
        system_message = {
            "role": "system",
            "content": """You are an expert day trader assistant specializing in short-term trading.
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
            """,
        }

        # Prepare user message with stock data
        stock_message = f"""
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

        # Add market context
        market_message = f"""
        # Market Context
        Market State: {market_context.get("state", "Unknown")}
        Sector Performance: {market_context.get("sector_performance", 0):.2f}%
        VIX: {market_context.get("vix", 0):.2f}
        Market Breadth: {market_context.get("breadth", 0):.2f}
        Time Until Close: {market_context.get("time_until_close", "Unknown")} hours
        """

        # Add portfolio state
        portfolio_message = f"""
        # Portfolio State
        Current Positions: {portfolio_state.get("position_count", 0)} / {portfolio_state.get("max_positions", 3)}
        Available Capital: ${portfolio_state.get("available_capital", 0):.2f}
        Daily P&L: ${portfolio_state.get("daily_pnl", 0):.2f} ({portfolio_state.get("daily_pnl_pct", 0):.2f}%)
        Daily Risk Remaining: ${portfolio_state.get("risk_remaining", 0):.2f}
        """

        # Combine all messages
        user_message = {
            "role": "user",
            "content": stock_message + market_message + portfolio_message,
        }

        # Make request to OpenRouter
        messages = [system_message, user_message]
        
        try:
            # Get metrics collector
            metrics_collector = get_collector("llm_trade_decision")
            # Use metrics timer to measure the entire process
            with MetricsTimer("llm_trade_decision", "full_process"):
                # Use the OpenAI SDK to make the request
                response = await self.chat_completion(messages)
                
                # Parse response
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Extract JSON
                if "{" in content and "}" in content:
                    json_str = content[content.find("{") : content.rfind("}") + 1]
                    decision = json.loads(json_str)
                else:
                    # If no JSON found, create a basic response
                    decision = {
                        "decision": "no_trade",
                        "position_size": 0.0,
                        "confidence": 0.0,
                        "reasoning": "Failed to parse LLM response",
                        "key_factors": [],
                    }

                # Add raw response for debugging
                decision["raw_response"] = content
                
                # Record confidence metric
                metrics_collector.record_confidence(
                    confidence=decision.get("confidence", 0.0),
                    correct=True,  # We don't know if it's correct in real-time
                    prediction_type="trade_decision"
                )

            return decision
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            
            # Record error in metrics
            metrics_collector.record_error(
                error_type="LLMResponseParsingError",
                error_message=str(e),
                context={"model": self.default_model}
            )
            
            return {
                "decision": "no_trade",
                "position_size": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "key_factors": [],
                "raw_response": str(e),
            }

    async def get_exit_decision(
        self,
        position_data: Dict[str, Any],
        current_data: Dict[str, Any],
        market_context: Dict[str, Any],
        exit_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get exit decision from LLM.

        Args:
            position_data: Data for the current position
            current_data: Current stock data
            market_context: Current market context
            exit_signals: Exit signals from monitoring

        Returns:
            Exit decision
        """
        # Prepare system message
        system_message = {
            "role": "system",
            "content": """You are an expert day trader assistant specializing in exit decisions.
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
            """,
        }

        # Prepare user message with position data
        position_message = f"""
        # Position Data
        Symbol: {position_data.get("symbol", "Unknown")}
        Entry Price: ${position_data.get("entry_price", 0):.2f}
        Current Price: ${current_data.get("price", {}).get("last", 0):.2f}
        Quantity: {position_data.get("quantity", 0)}
        Unrealized P&L: ${position_data.get("unrealized_pnl", 0):.2f} ({position_data.get("unrealized_pnl_pct", 0):.2f}%)
        Time in Trade: {position_data.get("time_in_trade", 0):.2f} hours
        """

        # Add current data
        current_message = f"""
        # Current Data
        Price Change: {current_data.get("price", {}).get("change_pct", 0):.2f}%
        Volume: {current_data.get("price", {}).get("volume", 0):,}
        RSI(14): {current_data.get("indicators", {}).get("rsi_14", 0):.2f}
        MACD Histogram: {current_data.get("indicators", {}).get("macd_histogram", 0):.4f}
        Bollinger Band Position: {current_data.get("indicators", {}).get("bb_position", 0):.2f}
        """

        # Add market context
        market_message = f"""
        # Market Context
        Market State: {market_context.get("state", "Unknown")}
        Sector Performance: {market_context.get("sector_performance", 0):.2f}%
        VIX: {market_context.get("vix", 0):.2f}
        Time Until Close: {market_context.get("time_until_close", "Unknown")} hours
        """

        # Add exit signals
        exit_message = f"""
        # Exit Signals
        ML Model Recommendation: {exit_signals.get("recommendation", {}).get("reason", "hold_position")}
        ML Confidence: {exit_signals.get("recommendation", {}).get("confidence", 0):.2f}
        Stop Loss Triggered: {exit_signals.get("stop_loss_triggered", False)}
        Take Profit Triggered: {exit_signals.get("take_profit_triggered", False)}
        Trailing Stop Triggered: {exit_signals.get("trailing_stop_triggered", False)}
        Time Stop Triggered: {exit_signals.get("time_stop_triggered", False)}
        """

        # Combine all messages
        user_message = {
            "role": "user",
            "content": position_message + current_message + market_message + exit_message,
        }

        # Make request to OpenRouter
        messages = [system_message, user_message]
        
        # Get metrics collector
        metrics_collector = get_collector("llm_exit_decision")
        
        try:
            # Use metrics timer to measure the entire process
            with MetricsTimer("llm_exit_decision", "full_process"):
                # Use the OpenAI SDK to make the request
                response = await self.chat_completion(messages)
                
                # Parse response
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Extract JSON
                if "{" in content and "}" in content:
                    json_str = content[content.find("{") : content.rfind("}") + 1]
                    decision = json.loads(json_str)
                else:
                    # If no JSON found, create a basic response
                    decision = {
                        "decision": "hold",
                        "exit_size": 0.0,
                        "confidence": 0.0,
                        "reasoning": "Failed to parse LLM response",
                        "key_factors": [],
                    }

                # Add raw response for debugging
                decision["raw_response"] = content
                
                # Record confidence metric
                metrics_collector.record_confidence(
                    confidence=decision.get("confidence", 0.0),
                    correct=True,  # We don't know if it's correct in real-time
                    prediction_type="exit_decision"
                )

            return decision
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            
            # Record error in metrics
            metrics_collector.record_error(
                error_type="LLMResponseParsingError",
                error_message=str(e),
                context={"model": self.default_model}
            )
            
            return {
                "decision": "hold",
                "exit_size": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "key_factors": [],
                "raw_response": str(e),
            }

    async def get_market_analysis(
        self, market_data: Dict[str, Any], market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get market analysis from LLM.

        Args:
            market_data: Dictionary containing market data
            market_context: Dictionary containing market context

        Returns:
            Market analysis with regime identification and recommendations
        """
        from .parsing import parse_market_analysis
        from .prompts import PromptTemplates

        # Generate prompt using PromptTemplates
        messages = PromptTemplates.create_market_analysis_prompt(
            market_data=market_data, market_context=market_context
        )

        # Get metrics collector
        metrics_collector = get_collector("llm_market_analysis")
        
        try:
            # Use metrics timer to measure the entire process
            with MetricsTimer("llm_market_analysis", "full_process"):
                # Use the OpenAI SDK to make the request
                response = await self.chat_completion(messages)
                
                # Parse response
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                analysis = parse_market_analysis(content)
                
                # Record confidence metric
                metrics_collector.record_confidence(
                    confidence=analysis.get("confidence", 0.0),
                    correct=True,  # We don't know if it's correct in real-time
                    prediction_type="market_analysis"
                )
                
                return analysis
        except Exception as e:
            logger.error(f"Error processing market analysis response: {e}")
            
            # Record error in metrics
            metrics_collector.record_error(
                error_type="MarketAnalysisError",
                error_message=str(e),
                context={"model": self.default_model}
            )
            
            return {
                "market_regime": "uncertain",
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "key_indicators": [],
                "trading_recommendation": "Proceed with caution",
                "raw_response": str(e),
            }


class LLMRouter:
    """
    Router for LLM requests that distributes tasks between different LLM providers
    and models based on task type and other criteria.

    This class manages multiple LLM clients (like OpenRouterClient) and routes
    requests to appropriate models based on task type, optimizing for cost,
    performance, and specialized capabilities.
    """

    # Task type constants
    TASK_TRADE_DECISION = "trade_decision"
    TASK_EXIT_DECISION = "exit_decision"
    TASK_MARKET_ANALYSIS = "market_analysis"

    def __init__(self, main_llm_client=None, trade_llm_client=None):
        """
        Initialize the LLM router.

        Args:
            main_llm_client: Client for main LLM interactions (default OpenRouterClient)
            trade_llm_client: Client for trade-specific LLM interactions
        """
        # Initialize clients
        self.main_llm_client = main_llm_client or OpenRouterClient()
        self.trade_llm_client = trade_llm_client or self.main_llm_client

        # Task to LLM client mapping
        self.task_routing = {
            self.TASK_TRADE_DECISION: self.trade_llm_client,
            self.TASK_EXIT_DECISION: self.trade_llm_client,
            self.TASK_MARKET_ANALYSIS: self.main_llm_client,
        }

        logger.info("LLM Router initialized with:")
        logger.info(f"- Main LLM: {self.main_llm_client.default_model}")
        if self.trade_llm_client != self.main_llm_client:
            logger.info(f"- Trade LLM: {self.trade_llm_client.default_model}")
        else:
            logger.info("- Trade LLM: Using Main LLM")

    async def close(self):
        """Close all client connections."""
        await self.main_llm_client.close()
        if self.trade_llm_client != self.main_llm_client:
            await self.trade_llm_client.close()

    def get_client_for_task(self, task_type: str) -> OpenRouterClient:
        """
        Get the appropriate LLM client for a specific task.

        Args:
            task_type: Type of task (use class constants)

        Returns:
            OpenRouterClient: The appropriate client for the task
        """
        return self.task_routing.get(task_type, self.main_llm_client)

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        task_type: str = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Get chat completion from appropriate LLM based on task type.

        Args:
            messages: List of message dictionaries
            task_type: Type of task (determines which LLM to use)
            model: Model to use (will override task-based selection)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response

        Returns:
            API response
        """
        client = self.main_llm_client
        if task_type:
            client = self.get_client_for_task(task_type)
            logger.debug(f"Using {client.__class__.__name__} for {task_type} task")

        return await client.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

    async def get_trade_decision(
        self,
        stock_data: Dict[str, Any],
        market_context: Dict[str, Any],
        portfolio_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get trade decision from appropriate LLM.

        Args:
            stock_data: Data for the stock
            market_context: Current market context
            portfolio_state: Current portfolio state

        Returns:
            Trade decision
        """
        client = self.get_client_for_task(self.TASK_TRADE_DECISION)
        return await client.get_trade_decision(stock_data, market_context, portfolio_state)

    async def get_exit_decision(
        self,
        position_data: Dict[str, Any],
        current_data: Dict[str, Any],
        market_context: Dict[str, Any],
        exit_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get exit decision from LLM.

        Args:
            position_data: Data for the current position
            current_data: Current stock data
            market_context: Current market context
            exit_signals: Exit signals from monitoring

        Returns:
            Exit decision
        """
        client = self.get_client_for_task(self.TASK_EXIT_DECISION)
        return await client.get_exit_decision(
            position_data, current_data, market_context, exit_signals
        )

    async def get_market_analysis(
        self, market_data: Dict[str, Any], market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get market analysis from LLM.

        Args:
            market_data: Dictionary containing market data
            market_context: Dictionary containing market context

        Returns:
            Market analysis with regime identification and recommendations
        """
        client = self.get_client_for_task(self.TASK_MARKET_ANALYSIS)
        return await client.get_market_analysis(market_data, market_context)


def get_llm_router() -> LLMRouter:
    """
    Create and return an LLM router instance configured from settings.

    This factory function reads configuration and creates the appropriate
    router with specialized LLM clients if needed.

    Returns:
        LLMRouter: Configured router instance
    """
    # Create main LLM client
    main_llm_client = OpenRouterClient()

    # Create trade-specific LLM client if configured differently
    # In the future, may differentiate based on settings
    # For now, use the same client for both
    trade_llm_client = main_llm_client

    # Instantiate and return the router
    return LLMRouter(main_llm_client=main_llm_client, trade_llm_client=trade_llm_client)


# Create a global instance of the OpenRouterClient for compatibility
openrouter_client = OpenRouterClient()

# Create a global instance of the LLM router
llm_router = get_llm_router()

# If this file is run directly, run a simple test
if __name__ == "__main__":
    print("LLM Router module initialized successfully.")
    print("This module provides routing capabilities for LLM requests.")
    print("\nTo use this module, import it in your code:")
    print("from src.llm.router import llm_router, get_llm_router, OpenRouterClient")
    print("\nExample usage:")
    print("router = get_llm_router()")
    print("decision = await router.get_trade_decision(stock_data, market_context, portfolio_state)")
