"""
Tests for LLM integration.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from src.llm.parsing import extract_json_from_text, parse_exit_decision, parse_trade_decision
from src.llm.prompts import PromptTemplates
from src.llm.router import OpenRouterClient


class TestOpenRouterClient(unittest.TestCase):
    """Test cases for OpenRouterClient."""

    def setUp(self):
        """Set up test environment."""
        # Create test client with dummy API key
        self.client = OpenRouterClient(api_key="test_api_key")

        # Sample data for tests
        self.sample_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]

        self.sample_stock_data = {
            "symbol": "AAPL",
            "price": {"last": 150.0, "change_pct": 1.5, "volume": 10000000},
            "indicators": {"rsi_14": 65.0, "macd_histogram": 0.5, "bb_position": 0.7},
            "pattern": {"name": "breakout", "confidence": 0.8},
            "sentiment": {"overall_score": 0.6},
            "news": [{"title": "Sample news", "summary": "Sample summary"}],
        }

        self.sample_market_context = {
            "state": "bullish",
            "sector_performance": 1.2,
            "vix": 15.0,
            "breadth": 0.65,
            "time_until_close": 3,
        }

        self.sample_portfolio_state = {
            "position_count": 1,
            "max_positions": 3,
            "available_capital": 10000.0,
            "daily_pnl": 500.0,
            "daily_pnl_pct": 5.0,
            "risk_remaining": 1000.0,
        }

        self.sample_position_data = {
            "symbol": "AAPL",
            "entry_price": 145.0,
            "current_price": 150.0,
            "quantity": 10,
            "unrealized_pnl": 50.0,
            "unrealized_pnl_pct": 3.45,
            "time_in_trade": 2.5,
        }

        self.sample_exit_signals = {
            "recommendation": {"reason": "partial_exit", "confidence": 0.7},
            "stop_loss_triggered": False,
            "take_profit_triggered": False,
            "trailing_stop_triggered": False,
            "time_stop_triggered": False,
        }

    @patch("aiohttp.ClientSession.post")
    async def test_chat_completion(self, mock_post):
        """Test chat completion request."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "id": "test_id",
            "object": "chat.completion",
            "created": 1677858242,
            "model": "test-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
        mock_post.return_value.__aenter__.return_value = mock_response

        # Call chat completion
        response = await self.client.chat_completion(
            messages=self.sample_messages, model="test-model", temperature=0.7, max_tokens=1000
        )

        # Check that POST was called with correct arguments
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args

        # Check URL
        self.assertEqual(kwargs["url"], "https://openrouter.ai/api/v1/chat/completions")

        # Check data
        data = kwargs["json"]
        self.assertEqual(data["model"], "test-model")
        self.assertEqual(data["messages"], self.sample_messages)
        self.assertEqual(data["temperature"], 0.7)
        self.assertEqual(data["max_tokens"], 1000)

        # Check headers
        headers = kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test_api_key")

        # Check response
        self.assertEqual(
            response["choices"][0]["message"]["content"], "Hello! How can I help you today?"
        )

    @patch("src.llm.router.OpenRouterClient.chat_completion")
    async def test_get_trade_decision(self, mock_chat_completion):
        """Test trade decision method."""
        # Mock response
        mock_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """Based on the analysis, here is my trading decision:

```json
{
    "decision": "trade",
    "position_size": 0.5,
    "confidence": 0.85,
    "reasoning": "Strong breakout pattern with positive sentiment",
    "key_factors": ["breakout pattern", "positive sentiment", "bullish market"]
}
```

Let me know if you need any other information!"""
                    }
                }
            ]
        }

        # Call get_trade_decision
        decision = await self.client.get_trade_decision(
            self.sample_stock_data, self.sample_market_context, self.sample_portfolio_state
        )

        # Check that chat_completion was called
        mock_chat_completion.assert_called_once()

        # Check decision
        self.assertEqual(decision["decision"], "trade")
        self.assertEqual(decision["position_size"], 0.5)
        self.assertEqual(decision["confidence"], 0.85)
        self.assertEqual(decision["reasoning"], "Strong breakout pattern with positive sentiment")
        self.assertEqual(
            decision["key_factors"], ["breakout pattern", "positive sentiment", "bullish market"]
        )

    @patch("src.llm.router.OpenRouterClient.chat_completion")
    async def test_get_exit_decision(self, mock_chat_completion):
        """Test exit decision method."""
        # Mock response
        mock_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """Based on the analysis, here is my exit decision:

```json
{
    "decision": "exit",
    "exit_size": 0.5,
    "confidence": 0.75,
    "reasoning": "Taking partial profits with RSI approaching overbought",
    "key_factors": ["profit target reached", "overbought indicators", "time in trade"]
}
```

Let me know if you need any other information!"""
                    }
                }
            ]
        }

        # Call get_exit_decision
        decision = await self.client.get_exit_decision(
            self.sample_position_data,
            self.sample_stock_data,
            self.sample_market_context,
            self.sample_exit_signals,
        )

        # Check that chat_completion was called
        mock_chat_completion.assert_called_once()

        # Check decision
        self.assertEqual(decision["decision"], "exit")
        self.assertEqual(decision["exit_size"], 0.5)
        self.assertEqual(decision["confidence"], 0.75)
        self.assertEqual(
            decision["reasoning"], "Taking partial profits with RSI approaching overbought"
        )
        self.assertEqual(
            decision["key_factors"],
            ["profit target reached", "overbought indicators", "time in trade"],
        )

    @patch("src.llm.router.OpenRouterClient.chat_completion")
    async def test_get_market_analysis(self, mock_chat_completion):
        """Test market analysis method."""
        # Mock response
        mock_chat_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """Based on the analysis, here is my market analysis:

```json
{
    "market_regime": "bullish",
    "confidence": 0.8,
    "reasoning": "Strong uptrend with positive economic data",
    "key_indicators": ["rising prices", "positive news", "high volume"],
    "trading_recommendation": "Buy the dips"
}
```

Let me know if you need any other information!"""
                    }
                }
            ]
        }

        # Call get_market_analysis
        analysis = await self.client.get_market_analysis(
            self.sample_stock_data, self.sample_market_context
        )

        # Check that chat_completion was called
        mock_chat_completion.assert_called_once()

        # Check analysis
        self.assertEqual(analysis["market_regime"], "bullish")
        self.assertEqual(analysis["confidence"], 0.8)
        self.assertEqual(analysis["reasoning"], "Strong uptrend with positive economic data")
        self.assertEqual(
            analysis["key_indicators"], ["rising prices", "positive news", "high volume"]
        )
        self.assertEqual(analysis["trading_recommendation"], "Buy the dips")


class TestPromptTemplates(unittest.TestCase):
    """Test cases for PromptTemplates."""

    def setUp(self):
        """Set up test environment."""
        # Sample data for tests
        self.sample_stock_data = {
            "symbol": "AAPL",
            "price": {"last": 150.0, "change_pct": 1.5, "volume": 10000000},
            "indicators": {"rsi_14": 65.0, "macd_histogram": 0.5, "bb_position": 0.7},
            "pattern": {"name": "breakout", "confidence": 0.8},
            "sentiment": {"overall_score": 0.6},
            "news": [{"title": "Sample news", "summary": "Sample summary"}],
        }

        self.sample_market_context = {
            "state": "bullish",
            "sector_performance": 1.2,
            "vix": 15.0,
            "breadth": 0.65,
            "time_until_close": 3,
        }

        self.sample_portfolio_state = {
            "position_count": 1,
            "max_positions": 3,
            "available_capital": 10000.0,
            "daily_pnl": 500.0,
            "daily_pnl_pct": 5.0,
            "risk_remaining": 1000.0,
        }

        self.sample_position_data = {
            "symbol": "AAPL",
            "entry_price": 145.0,
            "current_price": 150.0,
            "quantity": 10,
            "unrealized_pnl": 50.0,
            "unrealized_pnl_pct": 3.45,
            "time_in_trade": 2.5,
        }

        self.sample_exit_signals = {
            "recommendation": {"reason": "partial_exit", "confidence": 0.7},
            "stop_loss_triggered": False,
            "take_profit_triggered": False,
            "trailing_stop_triggered": False,
            "time_stop_triggered": False,
        }

    def test_format_stock_data(self):
        """Test formatting stock data."""
        formatted = PromptTemplates.format_stock_data(self.sample_stock_data)

        # Check that it includes key information
        self.assertIn("AAPL", formatted)
        self.assertIn("$150.00", formatted)
        self.assertIn("1.50%", formatted)
        self.assertIn("10,000,000", formatted)
        self.assertIn("65.00", formatted)
        self.assertIn("breakout", formatted)
        self.assertIn("0.80", formatted)
        self.assertIn("0.60", formatted)

    def test_format_market_context(self):
        """Test formatting market context."""
        formatted = PromptTemplates.format_market_context(self.sample_market_context)

        # Check that it includes key information
        self.assertIn("bullish", formatted)
        self.assertIn("1.20%", formatted)
        self.assertIn("15.00", formatted)
        self.assertIn("0.65", formatted)
        self.assertIn("3", formatted)

    def test_format_portfolio_state(self):
        """Test formatting portfolio state."""
        formatted = PromptTemplates.format_portfolio_state(self.sample_portfolio_state)

        # Check that it includes key information
        self.assertIn("1 / 3", formatted)
        self.assertIn("$10000.00", formatted)
        self.assertIn("$500.00", formatted)
        self.assertIn("5.00%", formatted)
        self.assertIn("$1000.00", formatted)

    def test_create_trade_decision_prompt(self):
        """Test creating trade decision prompt."""
        messages = PromptTemplates.create_trade_decision_prompt(
            self.sample_stock_data, self.sample_market_context, self.sample_portfolio_state
        )

        # Check message structure
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

        # Check content
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]

        self.assertIn("expert day trader", system_content)
        self.assertIn("AAPL", user_content)
        self.assertIn("bullish", user_content)
        self.assertIn("$10000.00", user_content)


class TestParsing(unittest.TestCase):
    """Test cases for parsing functions."""

    def setUp(self):
        """Set up test environment."""
        # Sample responses for tests
        self.json_in_code_block = """Based on the analysis, here is my trading decision:

```json
{
    "decision": "trade",
    "position_size": 0.5,
    "confidence": 0.85,
    "reasoning": "Strong breakout pattern with positive sentiment",
    "key_factors": ["breakout pattern", "positive sentiment", "bullish market"]
}
```

Let me know if you need any other information!"""

        self.json_in_text = """Based on the analysis, here is my trading decision:

{
    "decision": "trade",
    "position_size": 0.5,
    "confidence": 0.85,
    "reasoning": "Strong breakout pattern with positive sentiment",
    "key_factors": ["breakout pattern", "positive sentiment", "bullish market"]
}

Let me know if you need any other information!"""

        self.malformed_json = """Based on the analysis, here is my trading decision:

{
    "decision": "trade",
    "position_size": 0.5,
    "confidence": 0.85,
    "reasoning": "Strong breakout pattern with positive sentiment",
    "key_factors": ["breakout pattern", "positive sentiment", "bullish market"
}

Let me know if you need any other information!"""

        self.no_json = """Based on the analysis, I recommend entering a trade with 50% position size.
The main factors are the strong breakout pattern, positive sentiment, and bullish market conditions."""

    def test_extract_json_from_text_code_block(self):
        """Test extracting JSON from code block."""
        result = extract_json_from_text(self.json_in_code_block)

        # Check result
        self.assertIsInstance(result, dict)
        self.assertEqual(result["decision"], "trade")
        self.assertEqual(result["position_size"], 0.5)
        self.assertEqual(result["confidence"], 0.85)

    def test_extract_json_from_text_in_text(self):
        """Test extracting JSON from text."""
        result = extract_json_from_text(self.json_in_text)

        # Check result
        self.assertIsInstance(result, dict)
        self.assertEqual(result["decision"], "trade")
        self.assertEqual(result["position_size"], 0.5)
        self.assertEqual(result["confidence"], 0.85)

    def test_extract_json_from_text_malformed(self):
        """Test extracting JSON from malformed JSON."""
        result = extract_json_from_text(self.malformed_json)

        # Should return None for malformed JSON
        self.assertIsNone(result)

    def test_extract_json_from_text_no_json(self):
        """Test extracting JSON from text with no JSON."""
        result = extract_json_from_text(self.no_json)

        # Should return None when no JSON is found
        self.assertIsNone(result)

    def test_parse_trade_decision_valid(self):
        """Test parsing valid trade decision."""
        decision = parse_trade_decision(self.json_in_code_block)

        # Check result
        self.assertEqual(decision["decision"], "trade")
        self.assertEqual(decision["position_size"], 0.5)
        self.assertEqual(decision["confidence"], 0.85)
        self.assertEqual(decision["reasoning"], "Strong breakout pattern with positive sentiment")
        self.assertEqual(len(decision["key_factors"]), 3)

    def test_parse_trade_decision_no_json(self):
        """Test parsing trade decision with no JSON."""
        decision = parse_trade_decision(self.no_json)

        # Should return default decision
        self.assertEqual(decision["decision"], "no_trade")
        self.assertEqual(decision["position_size"], 0.0)
        self.assertEqual(decision["reasoning"], "Failed to parse response")

    def test_parse_exit_decision_valid(self):
        """Test parsing valid exit decision."""
        exit_json = self.json_in_code_block.replace("trade", "exit").replace(
            "position_size", "exit_size"
        )
        decision = parse_exit_decision(exit_json)

        # Check result
        self.assertEqual(decision["decision"], "exit")
        self.assertEqual(decision["exit_size"], 0.5)
        self.assertEqual(decision["confidence"], 0.85)
        self.assertEqual(decision["reasoning"], "Strong breakout pattern with positive sentiment")
        self.assertEqual(len(decision["key_factors"]), 3)

    def test_parse_exit_decision_no_json(self):
        """Test parsing exit decision with no JSON."""
        decision = parse_exit_decision(self.no_json)

        # Should return default decision
        self.assertEqual(decision["decision"], "hold")
        self.assertEqual(decision["exit_size"], 0.0)
        self.assertEqual(decision["reasoning"], "Failed to parse response")


if __name__ == "__main__":
    # Run async tests
    loop = asyncio.get_event_loop()
    unittest.main()
