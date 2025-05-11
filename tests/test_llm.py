"""
Tests for LLM integration.
"""

import asyncio
import unittest
from unittest.mock import patch, MagicMock

from src.llm.parsing import extract_json_from_text, parse_exit_decision, parse_trade_decision
from src.llm.prompts import PromptTemplates
from src.llm.router import OpenRouterClient


class TestOpenRouterClient(unittest.TestCase):
    """Test cases for OpenRouterClient."""

    def setUp(self):
        """Set up test environment."""
        # Create test client with real API key from settings
        from src.config.settings import settings
        self.client = OpenRouterClient(api_key=settings.api.openrouter_api_key)

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

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    def test_chat_completion(self, mock_create):
        """Test chat completion request."""
        # Set up the expected response content
        expected_content = "Hello! How can I help you today?"
        mock_create.return_value = {"choices": [{"message": {"content": expected_content}}]}
        
        # Create a mock response object
        mock_response = MagicMock()
        mock_response.id = "test_id"
        mock_response.object = "chat.completion"
        mock_response.created = 1677858242
        mock_response.model = "test-model"
        
        # Create a mock choice
        mock_choice = MagicMock()
        mock_choice.index = 0
        mock_choice.finish_reason = "stop"
        
        # Create a mock message
        mock_message = MagicMock()
        mock_message.role = "assistant"
        mock_message.content = "Hello! How can I help you today?"
        mock_choice.message = mock_message
        
        # Set choices on the response
        mock_response.choices = [mock_choice]
        
        # Set up the mock to return the response object
        mock_create.return_value = mock_response
        
        # Run the async test
        async def run_test():
            # Call chat completion
            response = await self.client.chat_completion(
                messages=self.sample_messages, model="test-model", temperature=0.7, max_tokens=1000
            )

            # Check that create was called with correct arguments
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args

            # Check parameters
            self.assertEqual(kwargs["model"], "test-model")
            self.assertEqual(kwargs["messages"], self.sample_messages)
            self.assertEqual(kwargs["temperature"], 0.7)
            self.assertEqual(kwargs["max_tokens"], 1000)

            # Check extra headers
            self.assertIn("extra_headers", kwargs)
            self.assertIn("HTTP-Referer", kwargs["extra_headers"])
            self.assertIn("X-Title", kwargs["extra_headers"])

            # Check response
            self.assertEqual(
                response["choices"][0]["message"]["content"], "Hello! How can I help you today?"
            )
            
            return response
            
        # Run the async test
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_test())

    @patch("src.llm.router.OpenRouterClient.chat_completion")
    def test_get_trade_decision(self, mock_chat_completion):
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

        # Run the async test
        async def run_test():
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
            
            return decision
            
        # Run the async test
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_test())

    @patch("src.llm.router.OpenRouterClient.chat_completion")
    def test_get_exit_decision(self, mock_chat_completion):
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

        # Run the async test
        async def run_test():
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
            
            return decision
            
        # Run the async test
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_test())

    @patch("src.llm.router.OpenRouterClient.chat_completion")
    def test_get_market_analysis(self, mock_chat_completion):
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

        # Run the async test
        async def run_test():
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
            
            return analysis
            
        # Run the async test
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_test())


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


class TestLiveOpenRouterClient(unittest.TestCase):
    """Test cases for OpenRouterClient with live API calls."""

    def setUp(self):
        """Set up test environment."""
        # Create test client with real API key from settings
        from src.config.settings import settings
        self.client = OpenRouterClient(api_key=settings.api.openrouter_api_key)

        # Sample data for tests
        self.sample_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."},
        ]

    def test_live_chat_completion(self):
        """Test live chat completion request."""
        # Skip this test if we're in CI environment or if explicitly disabled
        import os
        if os.environ.get("SKIP_LIVE_TESTS", "").lower() == "true":
            self.skipTest("Skipping live API test")

        # Run the async test
        async def run_test():
            response = await self.client.chat_completion(
                messages=self.sample_messages,
                temperature=0.0,  # Use 0 temperature for deterministic results
                max_tokens=10,    # Limit tokens for faster response
            )
            
            # Check that we got a valid response
            self.assertIn("choices", response)
            self.assertGreater(len(response["choices"]), 0)
            
            # Check content (should be "4" or contain "4")
            content = response["choices"][0]["message"]["content"]
            self.assertIn("4", content)
            
            return response
            
        # Run the async test
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(run_test())
        
        # Print response for debugging
        print(f"Live API Response: {response}")


class TestSystemIntegration(unittest.TestCase):
    """Test cases for LLM integration with the entire system flow."""

    def setUp(self):
        """Set up test environment."""
        # Create test client with real API key from settings
        from src.config.settings import settings
        
        # Create Main LLM and Trade LLM clients
        self.main_llm = OpenRouterClient(api_key=settings.api.openrouter_api_key)
        self.trade_llm = OpenRouterClient(api_key=settings.api.openrouter_api_key)
        
        # Mock data for pattern recognition model
        self.mock_pattern_recognition = {
            "pattern": "cup_with_handle",
            "confidence": 0.87,
            "pattern_idx": 14,
            "bullish": True,
            "reliability": 0.85,
            "probabilities": {
                "none": 0.02,
                "double_top": 0.01,
                "double_bottom": 0.03,
                "head_shoulders": 0.01,
                "inv_head_shoulders": 0.04,
                "triangle_ascending": 0.05,
                "triangle_descending": 0.01,
                "triangle_symmetrical": 0.02,
                "flag_bullish": 0.06,
                "flag_bearish": 0.01,
                "wedge_rising": 0.03,
                "wedge_falling": 0.01,
                "channel_up": 0.07,
                "channel_down": 0.01,
                "cup_with_handle": 0.87,
                "breakout": 0.12
            }
        }
        
        # Mock data for sentiment analysis
        self.mock_sentiment = {
            "sentiment": {
                "positive": 0.75,
                "neutral": 0.20,
                "negative": 0.05,
                "sentiment_score": 0.70
            },
            "entities": [
                {
                    "text": "AAPL",
                    "type": "ORG",
                    "start": 10,
                    "end": 14,
                    "sentiment_score": 0.82,
                    "relevance_score": 1.5,
                    "context": ["Apple announced strong quarterly results"]
                }
            ]
        }
        
        # Mock data for exit optimization model
        self.mock_exit_optimization = {
            "action": "exit_half",
            "action_idx": 3,
            "exit_size": 0.65,
            "confidence": 0.78,
            "probabilities": {
                "hold": 0.0,
                "exit_quarter": 0.0,
                "exit_third": 0.0,
                "exit_half": 0.75,
                "exit_full": 0.25
            }
        }
        
        # Mock exit recommendation
        self.mock_exit_recommendation = {
            "exit": True,
            "size": 0.5,
            "reason": "half_exit",
            "confidence": 0.78,
            "prediction": self.mock_exit_optimization,
            "stop_loss_triggered": False,
            "take_profit_triggered": False,
            "trailing_stop_triggered": False,
            "time_stop_triggered": False,
            "risk_metrics": {
                "max_drawdown": 0.05,
                "drawdown_ratio": 1.2,
                "volatility": 0.015,
                "sharpe_ratio": 1.8
            }
        }
        
        # Mock stock data
        self.mock_stock_data = {
            "symbol": "AAPL",
            "price": {"last": 178.72, "change_pct": 2.35, "volume": 15782400},
            "indicators": {"rsi_14": 72.8, "macd_histogram": 0.85, "bb_position": 0.92},
            "pattern": self.mock_pattern_recognition,
            "sentiment": self.mock_sentiment["sentiment"],
            "news": [
                {"title": "Apple Announces New Product Line", "summary": "Positive reception expected"},
                {"title": "Tech Sector Rally Continues", "summary": "Market showing strength"}
            ],
        }
        
        # Mock market context
        self.mock_market_context = {
            "state": "bullish",
            "sector_performance": 2.8,
            "vix": 12.5,
            "breadth": 0.78,
            "time_until_close": 2.5,
        }
        
        # Mock portfolio state
        self.mock_portfolio_state = {
            "position_count": 2,
            "max_positions": 5,
            "available_capital": 25000.0,
            "daily_pnl": 850.0,
            "daily_pnl_pct": 3.4,
            "risk_remaining": 1500.0,
        }
        
        # Mock position data
        self.mock_position_data = {
            "symbol": "AAPL",
            "entry_price": 170.25,
            "current_price": 178.72,
            "quantity": 100,
            "unrealized_pnl": 847.0,
            "unrealized_pnl_pct": 4.97,
            "time_in_trade": 1.5,
        }

    def test_main_llm_trade_decision_with_mock_data(self):
        """Test Main LLM trade decision with mock data from all models."""
        # Skip this test if we're in CI environment or if explicitly disabled
        import os
        if os.environ.get("SKIP_LIVE_TESTS", "").lower() == "true":
            self.skipTest("Skipping live API test")

        # Run the async test
        async def run_test():
            # Prepare system message
            system_message = {
                "role": "system",
                "content": """You are an expert trading assistant analyzing market data.
                Based on the pattern recognition data, market context, and portfolio state,
                provide a trading recommendation in JSON format with the following structure:
                {
                    "decision": "trade" or "no_trade",
                    "position_size": float (0.0 to 1.0, as a fraction of max position size),
                    "confidence": float (0.0 to 1.0),
                    "reasoning": "Brief explanation of your decision",
                    "key_factors": ["factor1", "factor2", "factor3"]
                }
                """
            }
            
            # Format the data for the prompt
            from src.llm.prompts import PromptTemplates
            stock_data_str = PromptTemplates.format_stock_data(self.mock_stock_data)
            market_context_str = PromptTemplates.format_market_context(self.mock_market_context)
            portfolio_state_str = PromptTemplates.format_portfolio_state(self.mock_portfolio_state)
            
            # Combine all data
            user_message = {
                "role": "user",
                "content": stock_data_str + market_context_str + portfolio_state_str
            }
            
            # Measure latency
            import time
            start_time = time.time()
            
            # Make request to Main LLM
            response = await self.main_llm.chat_completion(
                messages=[system_message, user_message],
                temperature=0.2,  # Lower temperature for more deterministic results
                max_tokens=1000
            )
            
            # Calculate latency
            latency = time.time() - start_time
            print(f"Main LLM Response Latency: {latency:.2f} seconds")
            
            # Check that we got a valid response
            self.assertIn("choices", response)
            self.assertGreater(len(response["choices"]), 0)
            
            # Get content
            content = response["choices"][0]["message"]["content"]
            print(f"Main LLM Response Content: {content}")
            
            # Extract JSON from response
            from src.llm.parsing import extract_json_from_text
            json_data = extract_json_from_text(content)
            
            # Verify JSON was extracted
            self.assertIsNotNone(json_data, "Failed to extract JSON from LLM response")
            
            # Check for expected fields
            self.assertIn("decision", json_data)
            self.assertIn("position_size", json_data)
            self.assertIn("confidence", json_data)
            self.assertIn("reasoning", json_data)
            self.assertIn("key_factors", json_data)
            
            # Check confidence is a float between 0 and 1
            self.assertIsInstance(json_data["confidence"], (int, float))
            self.assertGreaterEqual(json_data["confidence"], 0.0)
            self.assertLessEqual(json_data["confidence"], 1.0)
            
            # Check key_factors is a list
            self.assertIsInstance(json_data["key_factors"], list)
            
            # Verify the decision is valid (either trade or no_trade)
            self.assertIn(json_data["decision"], ["trade", "no_trade"])
            
            # If it's a trade decision, position size should be > 0
            # If it's a no_trade decision, position size should be 0
            if json_data["decision"] == "trade":
                self.assertGreater(json_data["position_size"], 0.0)
            else:
                self.assertEqual(json_data["position_size"], 0.0)
            
            return response, latency
            
        # Run the async test
        loop = asyncio.get_event_loop()
        response, latency = loop.run_until_complete(run_test())
        
        # Print response for debugging
        print(f"Main LLM Response: {response}")
        print(f"Response Latency: {latency:.2f} seconds")

    def test_trade_llm_exit_decision_with_mock_data(self):
        """Test Trade LLM exit decision with mock data from all models."""
        # Skip this test if we're in CI environment or if explicitly disabled
        import os
        if os.environ.get("SKIP_LIVE_TESTS", "").lower() == "true":
            self.skipTest("Skipping live API test")

        # Run the async test
        async def run_test():
            # Prepare system message
            system_message = {
                "role": "system",
                "content": """You are an expert trading assistant specializing in exit decisions.
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
            
            # Format the data for the prompt
            from src.llm.prompts import PromptTemplates
            position_message = PromptTemplates.format_position_data(self.mock_position_data)
            stock_data_str = PromptTemplates.format_stock_data(self.mock_stock_data)
            market_context_str = PromptTemplates.format_market_context(self.mock_market_context)
            
            # Format exit signals
            exit_message = f"""
            # Exit Signals
            ML Model Recommendation: {self.mock_exit_recommendation["reason"]}
            ML Confidence: {self.mock_exit_recommendation["confidence"]:.2f}
            Stop Loss Triggered: {self.mock_exit_recommendation["stop_loss_triggered"]}
            Take Profit Triggered: {self.mock_exit_recommendation["take_profit_triggered"]}
            Trailing Stop Triggered: {self.mock_exit_recommendation["trailing_stop_triggered"]}
            Time Stop Triggered: {self.mock_exit_recommendation["time_stop_triggered"]}
            """
            
            # Combine all data
            user_message = {
                "role": "user",
                "content": position_message + stock_data_str + market_context_str + exit_message
            }
            
            # Measure latency
            import time
            start_time = time.time()
            
            # Make request to Trade LLM
            response = await self.trade_llm.chat_completion(
                messages=[system_message, user_message],
                temperature=0.2,  # Lower temperature for more deterministic results
                max_tokens=1000
            )
            
            # Calculate latency
            latency = time.time() - start_time
            print(f"Trade LLM Response Latency: {latency:.2f} seconds")
            
            # Check that we got a valid response
            self.assertIn("choices", response)
            self.assertGreater(len(response["choices"]), 0)
            
            # Get content
            content = response["choices"][0]["message"]["content"]
            print(f"Trade LLM Response Content: {content}")
            
            # Extract JSON from response
            from src.llm.parsing import extract_json_from_text
            json_data = extract_json_from_text(content)
            
            # Verify JSON was extracted
            self.assertIsNotNone(json_data, "Failed to extract JSON from LLM response")
            
            # Check for expected fields
            self.assertIn("decision", json_data)
            self.assertIn("exit_size", json_data)
            self.assertIn("confidence", json_data)
            self.assertIn("reasoning", json_data)
            self.assertIn("key_factors", json_data)
            
            # Check confidence is a float between 0 and 1
            self.assertIsInstance(json_data["confidence"], (int, float))
            self.assertGreaterEqual(json_data["confidence"], 0.0)
            self.assertLessEqual(json_data["confidence"], 1.0)
            
            # Check key_factors is a list
            self.assertIsInstance(json_data["key_factors"], list)
            
            # Since we provided exit signals, we expect an exit decision
            self.assertEqual(json_data["decision"], "exit")
            self.assertGreater(json_data["exit_size"], 0.0)
            
            return response, latency
            
        # Run the async test
        loop = asyncio.get_event_loop()
        response, latency = loop.run_until_complete(run_test())
        
        # Print response for debugging
        print(f"Trade LLM Response: {response}")
        print(f"Response Latency: {latency:.2f} seconds")


if __name__ == "__main__":
    # Run async tests
    loop = asyncio.get_event_loop()
    unittest.main()
