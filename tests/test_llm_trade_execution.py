"""
Tests for LLM trade execution with real Alpaca API integration.
"""

import asyncio
import unittest
import os

from unittest.mock import patch, AsyncMock

from src.config.settings import settings
from src.core.trade_execution import TradeExecutor
from src.llm.router import OpenRouterClient


class TestLLMTradeExecution(unittest.TestCase):
    """Test cases for LLM trade execution with real Alpaca API integration."""

    def setUp(self):
        """Set up test environment."""
        # Create test client with real API key from settings
        self.llm_client = OpenRouterClient(api_key=settings.api.openrouter_api_key)
        
        # Mock data for a trade that should pass (bullish conditions)
        # Using AMD as the symbol, assuming you don't already have a position in it
        self.pass_trade_data = {
            "symbol": "AMD",
            "price": {"last": 150.0, "change_pct": 1.5, "volume": 10000000},
            "indicators": {"rsi_14": 55.0, "macd_histogram": 0.5, "bb_position": 0.5},
            "pattern": {
                "name": "breakout",
                "confidence": 0.85,
                "pattern_idx": 15,
                "bullish": True,
                "reliability": 0.8,
                "probabilities": {"breakout": 0.85, "none": 0.05}
            },
            "sentiment": {
                "positive": 0.75,
                "neutral": 0.20,
                "negative": 0.05,
                "sentiment_score": 0.70
            },
            "news": [
                {"title": "AMD Announces New Product Line", "summary": "Positive reception expected"},
                {"title": "Tech Sector Rally Continues", "summary": "Market showing strength"}
            ],
        }
        
        # Mock data for a trade that should fail (overbought conditions)
        self.fail_trade_data = {
            "symbol": "MSFT",
            "price": {"last": 350.0, "change_pct": 2.5, "volume": 8000000},
            "indicators": {"rsi_14": 78.0, "macd_histogram": 0.2, "bb_position": 0.95},
            "pattern": {
                "name": "double_top",
                "confidence": 0.75,
                "pattern_idx": 1,
                "bullish": False,
                "reliability": 0.7,
                "probabilities": {"double_top": 0.75, "none": 0.15}
            },
            "sentiment": {
                "positive": 0.45,
                "neutral": 0.30,
                "negative": 0.25,
                "sentiment_score": 0.20
            },
            "news": [
                {"title": "Microsoft Stock Reaches All-Time High", "summary": "Analysts warn of potential pullback"},
                {"title": "Tech Sector Showing Signs of Overheating", "summary": "Valuations at concerning levels"}
            ],
        }
        
        # Mock market context
        self.market_context = {
            "state": "bullish",
            "sector_performance": 1.8,
            "vix": 15.0,
            "breadth": 0.65,
            "time_until_close": 3.5,
        }
        
        # Mock portfolio state
        self.portfolio_state = {
            "position_count": 2,
            "max_positions": 5,
            "available_capital": 25000.0,
            "daily_pnl": 500.0,
            "daily_pnl_pct": 2.0,
            "risk_remaining": 1000.0,
        }

    @patch("src.core.trade_execution.openrouter_client")
    def test_llm_trade_decisions_with_real_alpaca(self, mock_openrouter):
        """Test LLM trade decisions with real Alpaca API."""
        # Skip this test if we're in CI environment or if explicitly disabled
        if os.environ.get("SKIP_LIVE_TESTS", "").lower() == "true":
            self.skipTest("Skipping live API test")

        # Set up the mock LLM client
        mock_openrouter.chat_completion = AsyncMock()
        
        # Set up the pass response
        pass_response = {
            "choices": [
                {
                    "message": {
                        "content": """```json
{
    "decision": "trade",
    "position_size": 0.5,
    "confidence": 0.75,
    "reasoning": "AMD shows a breakout pattern in a bullish market. The RSI and MACD are neutral, but the breakout pattern and sector performance suggest upward momentum. Portfolio has room for another position.",
    "key_factors": ["breakout", "bullish market", "sector performance"]
}
```"""
                    }
                }
            ]
        }
        
        # Set up the fail response
        fail_response = {
            "choices": [
                {
                    "message": {
                        "content": """```json
{
    "decision": "no_trade",
    "position_size": 0.0,
    "confidence": 0.6,
    "reasoning": "A double top pattern has been detected, which is a bearish signal. Although the market is bullish, the pattern confidence is high enough to warrant caution. The neutral news sentiment and RSI around 50 do not provide additional support for a long position.",
    "key_factors": ["double_top", "market_state", "time_until_close"]
}
```"""
                    }
                }
            ]
        }
        
        # Run the async test
        async def run_test():
            # Set up the mock to return different responses for different inputs
            mock_openrouter.chat_completion.side_effect = [pass_response, fail_response]
            
            # Create a TradeExecutor with mocked dependencies but real Alpaca API
            with patch("src.core.data_pipeline.data_pipeline.get_stock_data", return_value=self.pass_trade_data), \
                 patch("src.core.data_pipeline.data_pipeline.get_market_context", return_value=self.market_context), \
                 patch("src.core.position_monitor.position_monitor.get_portfolio_state", return_value=self.portfolio_state):
                
                # Create the executor with real Alpaca API
                executor = TradeExecutor()
                executor.trading_mode = "paper"
                
                # Test the pass case
                pass_decision = await executor.make_trade_decision(self.pass_trade_data)
                print(f"Pass Decision: {pass_decision}")
                self.assertEqual(pass_decision["decision"], "trade")
                self.assertEqual(pass_decision["position_size"], 0.5)
                
                # Test trade execution with real Alpaca API
                pass_result = await executor.execute_trade(self.pass_trade_data["symbol"], pass_decision)
                print(f"Pass Result: {pass_result}")
                
                # Test the fail case
                fail_decision = await executor.make_trade_decision(self.fail_trade_data)
                print(f"Fail Decision: {fail_decision}")
                self.assertEqual(fail_decision["decision"], "no_trade")
                self.assertEqual(fail_decision["position_size"], 0.0)
                
                # Test that no trade is executed
                fail_result = await executor.execute_trade(self.fail_trade_data["symbol"], fail_decision)
                print(f"Fail Result: {fail_result}")
                
                return pass_result, fail_result
            
        # Run the async test
        loop = asyncio.get_event_loop()
        pass_result, fail_result = loop.run_until_complete(run_test())
        
        # Print results for debugging
        print(f"Pass Result: {pass_result}")
        print(f"Fail Result: {fail_result}")


if __name__ == "__main__":
    # Run async tests
    loop = asyncio.get_event_loop()
    unittest.main()
