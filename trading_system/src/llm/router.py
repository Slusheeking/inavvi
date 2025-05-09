"""
OpenRouter API client for LLM integration.

This module provides functions for interacting with the OpenRouter API
to get LLM responses for trade decisions.
"""
import json
import time
from typing import Any, Dict, List, Optional, Union

import aiohttp
import requests

from src.config.settings import settings
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("llm_router")

class OpenRouterClient:
    """
    Client for OpenRouter API.
    """
    
    # Base URL for OpenRouter API
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
        """
        self.api_key = api_key or settings.api.openrouter_api_key
        self.session = None
        
        # Default model
        self.default_model = settings.llm.model
        
        logger.info(f"OpenRouter client initialized with default model: {self.default_model}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create an aiohttp session.
        
        Returns:
            aiohttp.ClientSession: The session
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("OpenRouter session closed")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Get chat completion from OpenRouter API.
        
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
        
        # Prepare request data
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://trading-system.com",  # Replace with your domain
            "X-Title": settings.app_name
        }
        
        try:
            # Get session
            session = await self._get_session()
            
            # Make request
            url = f"{self.BASE_URL}/chat/completions"
            start_time = time.time()
            
            async with session.post(url, json=data, headers=headers) as response:
                elapsed_time = time.time() - start_time
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error {response.status} from OpenRouter: {error_text}")
                    return {"error": error_text}
                
                # Handle streaming responses
                if stream:
                    return response  # Return the response object for streaming
                
                # Parse response
                result = await response.json()
                
                logger.info(f"OpenRouter request completed in {elapsed_time:.2f}s")
                return result
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            return {"error": str(e)}
    
    async def get_trade_decision(
        self,
        stock_data: Dict[str, Any],
        market_context: Dict[str, Any],
        portfolio_state: Dict[str, Any]
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
            """
        }
        
        # Prepare user message with stock data
        stock_message = f"""
        # Stock Data
        Symbol: {stock_data.get('symbol', 'Unknown')}
        Current Price: ${stock_data.get('price', {}).get('last', 0):.2f}
        Daily Change: {stock_data.get('price', {}).get('change_pct', 0):.2f}%
        Volume: {stock_data.get('price', {}).get('volume', 0):,}
        
        # Technical Analysis
        RSI(14): {stock_data.get('indicators', {}).get('rsi_14', 0):.2f}
        MACD Histogram: {stock_data.get('indicators', {}).get('macd_histogram', 0):.4f}
        Bollinger Band Position: {stock_data.get('indicators', {}).get('bb_position', 0):.2f}
        
        # Pattern Recognition
        Detected Pattern: {stock_data.get('pattern', {}).get('name', 'None')}
        Pattern Confidence: {stock_data.get('pattern', {}).get('confidence', 0):.2f}
        
        # News Sentiment
        Overall Sentiment: {stock_data.get('sentiment', {}).get('overall_score', 0):.2f}
        Recent News Count: {len(stock_data.get('news', []))}
        """
        
        # Add market context
        market_message = f"""
        # Market Context
        Market State: {market_context.get('state', 'Unknown')}
        Sector Performance: {market_context.get('sector_performance', 0):.2f}%
        VIX: {market_context.get('vix', 0):.2f}
        Market Breadth: {market_context.get('breadth', 0):.2f}
        Time Until Close: {market_context.get('time_until_close', 'Unknown')} hours
        """
        
        # Add portfolio state
        portfolio_message = f"""
        # Portfolio State
        Current Positions: {portfolio_state.get('position_count', 0)} / {portfolio_state.get('max_positions', 3)}
        Available Capital: ${portfolio_state.get('available_capital', 0):.2f}
        Daily P&L: ${portfolio_state.get('daily_pnl', 0):.2f} ({portfolio_state.get('daily_pnl_pct', 0):.2f}%)
        Daily Risk Remaining: ${portfolio_state.get('risk_remaining', 0):.2f}
        """
        
        # Combine all messages
        user_message = {
            "role": "user",
            "content": stock_message + market_message + portfolio_message
        }
        
        # Make request to OpenRouter
        messages = [system_message, user_message]
        response = await self.chat_completion(messages)
        
        # Parse response
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Extract JSON
            if '{' in content and '}' in content:
                json_str = content[content.find('{'):content.rfind('}')+1]
                decision = json.loads(json_str)
            else:
                # If no JSON found, create a basic response
                decision = {
                    "decision": "no_trade",
                    "position_size": 0.0,
                    "confidence": 0.0,
                    "reasoning": "Failed to parse LLM response",
                    "key_factors": []
                }
            
            # Add raw response for debugging
            decision['raw_response'] = content
            
            return decision
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "decision": "no_trade",
                "position_size": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "key_factors": [],
                "raw_response": response
            }
    
    async def get_exit_decision(
        self,
        position_data: Dict[str, Any],
        current_data: Dict[str, Any],
        market_context: Dict[str, Any],
        exit_signals: Dict[str, Any]
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
            """
        }
        
        # Prepare user message with position data
        position_message = f"""
        # Position Data
        Symbol: {position_data.get('symbol', 'Unknown')}
        Entry Price: ${position_data.get('entry_price', 0):.2f}
        Current Price: ${current_data.get('price', {}).get('last', 0):.2f}
        Quantity: {position_data.get('quantity', 0)}
        Unrealized P&L: ${position_data.get('unrealized_pnl', 0):.2f} ({position_data.get('unrealized_pnl_pct', 0):.2f}%)
        Time in Trade: {position_data.get('time_in_trade', 0):.2f} hours
        """
        
        # Add current data
        current_message = f"""
        # Current Data
        Price Change: {current_data.get('price', {}).get('change_pct', 0):.2f}%
        Volume: {current_data.get('price', {}).get('volume', 0):,}
        RSI(14): {current_data.get('indicators', {}).get('rsi_14', 0):.2f}
        MACD Histogram: {current_data.get('indicators', {}).get('macd_histogram', 0):.4f}
        Bollinger Band Position: {current_data.get('indicators', {}).get('bb_position', 0):.2f}
        """
        
        # Add market context
        market_message = f"""
        # Market Context
        Market State: {market_context.get('state', 'Unknown')}
        Sector Performance: {market_context.get('sector_performance', 0):.2f}%
        VIX: {market_context.get('vix', 0):.2f}
        Time Until Close: {market_context.get('time_until_close', 'Unknown')} hours
        """
        
        # Add exit signals
        exit_message = f"""
        # Exit Signals
        ML Model Recommendation: {exit_signals.get('recommendation', {}).get('reason', 'hold_position')}
        ML Confidence: {exit_signals.get('recommendation', {}).get('confidence', 0):.2f}
        Stop Loss Triggered: {exit_signals.get('stop_loss_triggered', False)}
        Take Profit Triggered: {exit_signals.get('take_profit_triggered', False)}
        Trailing Stop Triggered: {exit_signals.get('trailing_stop_triggered', False)}
        Time Stop Triggered: {exit_signals.get('time_stop_triggered', False)}
        """
        
        # Combine all messages
        user_message = {
            "role": "user",
            "content": position_message + current_message + market_message + exit_message
        }
        
        # Make request to OpenRouter
        messages = [system_message, user_message]
        response = await self.chat_completion(messages)
        
        # Parse response
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Extract JSON
            if '{' in content and '}' in content:
                json_str = content[content.find('{'):content.rfind('}')+1]
                decision = json.loads(json_str)
            else:
                # If no JSON found, create a basic response
                decision = {
                    "decision": "hold",
                    "exit_size": 0.0,
                    "confidence": 0.0,
                    "reasoning": "Failed to parse LLM response",
                    "key_factors": []
                }
            
            # Add raw response for debugging
            decision['raw_response'] = content
            
            return decision
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "decision": "hold",
                "exit_size": 0.0,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "key_factors": [],
                "raw_response": response
            }

# Create a global instance of the client
openrouter_client = OpenRouterClient()