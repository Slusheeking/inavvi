"""
Prompt templates for LLM integration.
"""
from typing import Dict, List, Optional, Union, Any

class PromptTemplates:
    """
    Collection of prompt templates for LLM interactions.
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
        Market State: {market_context.get('state', 'Unknown')}
        Sector Performance: {market_context.get('sector_performance', 0):.2f}%
        VIX: {market_context.get('vix', 0):.2f}
        Market Breadth: {market_context.get('breadth', 0):.2f}
        Time Until Close: {market_context.get('time_until_close', 'Unknown')} hours
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
        Current Positions: {portfolio_state.get('position_count', 0)} / {portfolio_state.get('max_positions', 3)}
        Available Capital: ${portfolio_state.get('available_capital', 0):.2f}
        Daily P&L: ${portfolio_state.get('daily_pnl', 0):.2f} ({portfolio_state.get('daily_pnl_pct', 0):.2f}%)
        Daily Risk Remaining: ${portfolio_state.get('risk_remaining', 0):.2f}
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
        Symbol: {position_data.get('symbol', 'Unknown')}
        Entry Price: ${position_data.get('entry_price', 0):.2f}
        Current Price: ${position_data.get('current_price', 0):.2f}
        Quantity: {position_data.get('quantity', 0)}
        Unrealized P&L: ${position_data.get('unrealized_pnl', 0):.2f} ({position_data.get('unrealized_pnl_pct', 0):.2f}%)
        Time in Trade: {position_data.get('time_in_trade', 0):.2f} hours
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
        ML Model Recommendation: {exit_signals.get('recommendation', {}).get('reason', 'hold_position')}
        ML Confidence: {exit_signals.get('recommendation', {}).get('confidence', 0):.2f}
        Stop Loss Triggered: {exit_signals.get('stop_loss_triggered', False)}
        Take Profit Triggered: {exit_signals.get('take_profit_triggered', False)}
        Trailing Stop Triggered: {exit_signals.get('trailing_stop_triggered', False)}
        Time Stop Triggered: {exit_signals.get('time_stop_triggered', False)}
        """
    
    # Function to create a complete trade decision prompt
    @staticmethod
    def create_trade_decision_prompt(
        stock_data: Dict[str, Any],
        market_context: Dict[str, Any],
        portfolio_state: Dict[str, Any]
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
        system_message = {
            "role": "system",
            "content": PromptTemplates.SYSTEM_TRADE_DECISION
        }
        
        # Create user message
        user_content = (
            PromptTemplates.format_stock_data(stock_data) +
            PromptTemplates.format_market_context(market_context) +
            PromptTemplates.format_portfolio_state(portfolio_state)
        )
        
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        return [system_message, user_message]
    
    # Function to create a complete exit decision prompt
    @staticmethod
    def create_exit_decision_prompt(
        position_data: Dict[str, Any],
        current_data: Dict[str, Any],
        market_context: Dict[str, Any],
        exit_signals: Dict[str, Any]
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
        system_message = {
            "role": "system",
            "content": PromptTemplates.SYSTEM_EXIT_DECISION
        }
        
        # Create user message
        user_content = (
            PromptTemplates.format_position_data(position_data) +
            PromptTemplates.format_stock_data(current_data) +
            PromptTemplates.format_market_context(market_context) +
            PromptTemplates.format_exit_signals(exit_signals)
        )
        
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        return [system_message, user_message]