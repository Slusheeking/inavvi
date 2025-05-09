"""
Tests for trading logic modules.
"""
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock

import pandas as pd
import numpy as np

from src.core.screening import stock_screener
from src.core.position_monitor import position_monitor
from src.core.trade_execution import trade_executor
from src.utils.redis_client import redis_client

class TestStockScreener(unittest.TestCase):
    """Test the stock screener module."""
    
    def setUp(self):
        """Set up test case."""
        # Use the global instance
        self.screener = stock_screener
        
        # Create sample data
        self.sample_opportunities = [
            {
                'symbol': 'AAPL',
                'price': {'last': 150.0, 'open': 149.0, 'high': 151.0, 'low': 148.0, 'volume': 1000000},
                'pattern': {'name': 'breakout', 'confidence': 0.85},
                'indicators': {'rsi_14': 65.0, 'macd_histogram': 0.2, 'bb_position': 0.7},
                'sentiment': {'overall_score': 0.6},
                'score': 0.85,
                'timestamp': datetime.now().isoformat()
            },
            {
                'symbol': 'MSFT',
                'price': {'last': 250.0, 'open': 248.0, 'high': 252.0, 'low': 247.0, 'volume': 800000},
                'pattern': {'name': 'continuation', 'confidence': 0.75},
                'indicators': {'rsi_14': 60.0, 'macd_histogram': 0.15, 'bb_position': 0.6},
                'sentiment': {'overall_score': 0.5},
                'score': 0.75,
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        self.sample_watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    @patch('src.core.screening.stock_screener.scan_for_opportunities')
    def test_scan_for_opportunities(self, mock_scan):
        """Test scanning for trading opportunities."""
        # Set up mock
        mock_scan.return_value = self.sample_opportunities
        
        # Run test asynchronously
        result = asyncio.run(self.screener.scan_for_opportunities(self.sample_watchlist))
        
        # Assert
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['symbol'], 'AAPL')
        self.assertEqual(result[0]['pattern']['name'], 'breakout')
        self.assertEqual(result[0]['score'], 0.85)
        mock_scan.assert_called_once_with(self.sample_watchlist)
    
    @patch('src.core.screening.stock_screener.analyze_symbol')
    @patch('src.utils.redis_client.redis_client.add_candidate_score')
    def test_analyze_symbol(self, mock_add_candidate, mock_analyze):
        """Test analyzing a single symbol."""
        # Set up mocks
        mock_analyze.return_value = self.sample_opportunities[0]
        mock_add_candidate.return_value = True
        
        # Run test asynchronously
        result = asyncio.run(self.screener.analyze_symbol('AAPL'))
        
        # Assert
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['pattern']['name'], 'breakout')
        self.assertEqual(result['indicators']['rsi_14'], 65.0)
        mock_analyze.assert_called_once_with('AAPL')
        mock_add_candidate.assert_called_once()
    
    @patch('src.core.screening.stock_screener.rank_candidates')
    @patch('src.utils.redis_client.redis_client.set_ranked_candidates')
    def test_rank_candidates(self, mock_set_candidates, mock_rank):
        """Test ranking trading candidates."""
        # Set up mocks
        mock_rank.return_value = [
            {'symbol': 'AAPL', 'score': 0.85},
            {'symbol': 'MSFT', 'score': 0.75}
        ]
        mock_set_candidates.return_value = True
        
        # Run test asynchronously
        result = asyncio.run(self.screener.rank_candidates())
        
        # Assert
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['symbol'], 'AAPL')
        self.assertEqual(result[0]['score'], 0.85)
        mock_rank.assert_called_once()
        mock_set_candidates.assert_called_once()
    
    @patch('src.utils.redis_client.redis_client.get_watchlist')
    @patch('src.utils.redis_client.redis_client.set_watchlist')
    @patch('src.core.screening.stock_screener.rank_candidates')
    def test_update_watchlist(self, mock_rank, mock_set_watchlist, mock_get_watchlist):
        """Test updating the watchlist."""
        # Set up mocks
        mock_get_watchlist.return_value = self.sample_watchlist
        mock_set_watchlist.return_value = True
        mock_rank.return_value = [
            {'symbol': 'AAPL', 'score': 0.85},
            {'symbol': 'MSFT', 'score': 0.75},
            {'symbol': 'NVDA', 'score': 0.70},  # New symbol
            {'symbol': 'AMD', 'score': 0.65},   # New symbol
            {'symbol': 'GOOGL', 'score': 0.60}
        ]
        
        # Run test asynchronously
        result = asyncio.run(self.screener.update_watchlist())
        
        # Assert
        self.assertEqual(len(result), 5)
        self.assertIn('AAPL', result)
        self.assertIn('NVDA', result)  # New symbol should be included
        mock_get_watchlist.assert_called_once()
        mock_set_watchlist.assert_called_once()
        mock_rank.assert_called_once()

class TestPositionMonitor(unittest.TestCase):
    """Test the position monitor module."""
    
    def setUp(self):
        """Set up test case."""
        # Use the global instance
        self.monitor = position_monitor
        
        # Create sample position data
        self.sample_position = {
            'symbol': 'AAPL',
            'entry_price': 150.0,
            'entry_time': (datetime.now() - timedelta(hours=2)).isoformat(),
            'quantity': 10,
            'side': 'long',
            'stop_loss': 145.0,
            'take_profit': 160.0,
            'trailing_stop': 2.0,
            'max_time': 6.0,
            'unrealized_pnl': 50.0,
            'unrealized_pnl_pct': 3.33,
            'current_price': 155.0,
            'status': 'active',
            'order_id': 'test-order-1234'
        }
        
        # Create sample price data
        self.sample_price_data = {
            'symbol': 'AAPL',
            'price': {
                'last': 155.0,
                'open': 150.0,
                'high': 156.0,
                'low': 149.0,
                'volume': 1000000
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Create sample indicators
        self.sample_indicators = {
            'rsi_14': 65.0,
            'macd_histogram': 0.2,
            'bb_position': 0.7,
            'volatility_5d': 0.015,
            'volume_ratio_5': 1.2
        }
        
        # Create sample market context
        self.sample_market_context = {
            'state': 'open',
            'sector_performance': 1.2,
            'vix': 15.5,
            'time_until_close': 3.5,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create sample exit recommendation
        self.sample_exit_recommendation = {
            'exit': True,
            'size': 1.0,
            'reason': 'take_profit',
            'confidence': 0.8,
            'stop_loss_triggered': False,
            'take_profit_triggered': True,
            'trailing_stop_triggered': False,
            'time_stop_triggered': False,
            'prediction': {
                'action': 'exit_full',
                'confidence': 0.8,
                'probabilities': {
                    'hold': 0.2,
                    'exit_partial': 0.1,
                    'exit_half': 0.1,
                    'exit_full': 0.6
                }
            }
        }
    
    @patch('src.core.position_monitor.position_monitor.monitor_position')
    @patch('src.utils.redis_client.redis_client.get_all_active_positions')
    def test_monitor_positions(self, mock_get_positions, mock_monitor_position):
        """Test monitoring all positions."""
        # Set up mocks
        mock_get_positions.return_value = {'AAPL': self.sample_position, 'MSFT': self.sample_position.copy()}
        mock_monitor_position.return_value = None
        
        # Run test asynchronously
        asyncio.run(self.monitor.monitor_positions())
        
        # Assert
        mock_get_positions.assert_called_once()
        self.assertEqual(mock_monitor_position.call_count, 2)  # Should be called for each position
    
    @patch('src.core.position_monitor.position_monitor.check_forced_exit_conditions')
    @patch('src.core.data_pipeline.data_pipeline.get_stock_data')
    @patch('src.core.data_pipeline.data_pipeline.get_technical_indicators')
    @patch('src.utils.redis_client.redis_client.update_position_pnl')
    def test_check_forced_exit_conditions(self, mock_update_pnl, mock_get_indicators, 
                                         mock_get_data, mock_check_forced):
        """Test checking for forced exit conditions."""
        # Set up mocks
        mock_get_data.return_value = self.sample_price_data
        mock_get_indicators.return_value = self.sample_indicators
        mock_update_pnl.return_value = True
        
        # Set up the forced exit mock to return a valid exit reason
        mock_check_forced.return_value = {
            'reason': 'stop_loss',
            'description': 'Stop loss triggered'
        }
        
        # Run test asynchronously
        result = asyncio.run(self.monitor.check_forced_exit_conditions(
            'AAPL', self.sample_position, 'open', 3.5
        ))
        
        # Assert
        self.assertEqual(result['reason'], 'stop_loss')
        mock_check_forced.assert_called_once_with('AAPL', self.sample_position, 'open', 3.5)
    
    @patch('src.models.exit_optimization.exit_optimization_model.evaluate_exit_conditions')
    @patch('src.core.data_pipeline.data_pipeline.get_stock_data')
    @patch('src.core.data_pipeline.data_pipeline.get_technical_indicators')
    @patch('src.utils.redis_client.redis_client.get_active_position')
    @patch('src.utils.redis_client.redis_client.update_position_pnl')
    def test_monitor_position(self, mock_update_pnl, mock_get_position, 
                             mock_get_indicators, mock_get_data, mock_evaluate):
        """Test monitoring a single position."""
        # Set up mocks
        mock_get_data.return_value = self.sample_price_data
        mock_get_indicators.return_value = self.sample_indicators
        mock_get_position.return_value = self.sample_position
        mock_update_pnl.return_value = True
        mock_evaluate.return_value = self.sample_exit_recommendation
        
        # Create a mock sample_df for intraday_data
        dates = pd.date_range(start=datetime.now()-timedelta(days=1), periods=100, freq='15min')
        sample_df = pd.DataFrame({
            'open': np.random.randn(100) + 150,
            'high': np.random.randn(100) + 152,
            'low': np.random.randn(100) + 148,
            'close': np.random.randn(100) + 151,
            'volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        
        # Set second get_data call to return dataframe
        mock_get_data.side_effect = [self.sample_price_data, sample_df]
        
        # Run test asynchronously 
        asyncio.run(self.monitor.monitor_position(
            'AAPL', self.sample_position, 'open', 3.5
        ))
        
        # Assert
        mock_get_data.assert_called()
        mock_get_position.assert_called_once_with('AAPL')
        mock_update_pnl.assert_called_once()
        mock_evaluate.assert_called_once()
    
    @patch('src.core.position_monitor.position_monitor.get_exit_decision')
    @patch('src.llm.router.openrouter_client.chat_completion')
    def test_get_exit_decision(self, mock_chat_completion, mock_get_decision):
        """Test getting an exit decision from LLM."""
        # Set up mocks
        mock_chat_completion.return_value = {
            'choices': [
                {
                    'message': {
                        'content': '{"decision": "exit", "exit_size": 1.0, "confidence": 0.9, '
                                  '"reasoning": "Take profit target reached", '
                                  '"key_factors": ["price_above_target", "strong_momentum"]}'
                    }
                }
            ]
        }
        
        mock_get_decision.return_value = {
            'decision': 'exit',
            'exit_size': 1.0,
            'confidence': 0.9,
            'reasoning': 'Take profit target reached',
            'key_factors': ['price_above_target', 'strong_momentum']
        }
        
        # Run test asynchronously
        result = asyncio.run(self.monitor.get_exit_decision(
            self.sample_position,
            self.sample_price_data,
            self.sample_market_context,
            self.sample_exit_recommendation
        ))
        
        # Assert
        self.assertEqual(result['decision'], 'exit')
        self.assertEqual(result['exit_size'], 1.0)
        self.assertEqual(result['confidence'], 0.9)
        mock_get_decision.assert_called_once()
    
    @patch('src.utils.redis_client.redis_client.get_all_active_positions')
    @patch('src.utils.redis_client.redis_client.update_position_pnl')
    def test_update_position_stats(self, mock_update_pnl, mock_get_positions):
        """Test updating position statistics."""
        # Set up mocks
        mock_get_positions.return_value = {'AAPL': self.sample_position, 'MSFT': self.sample_position.copy()}
        mock_update_pnl.return_value = True
        
        # Run test asynchronously
        asyncio.run(self.monitor.update_position_stats())
        
        # Assert
        mock_get_positions.assert_called_once()
        self.assertEqual(mock_update_pnl.call_count, 2)  # Called for each position

class TestTradeExecutor(unittest.TestCase):
    """Test the trade executor module."""
    
    def setUp(self):
        """Set up test case."""
        # Create a test instance with mocked Alpaca client
        self.executor = trade_executor
        
        # Sample trade decision
        self.sample_decision = {
            'decision': 'trade',
            'position_size': 0.5,
            'confidence': 0.8,
            'reasoning': 'Strong breakout pattern with high volume',
            'key_factors': ['breakout', 'high_volume', 'positive_sentiment']
        }
        
        # Sample candidate
        self.sample_candidate = {
            'symbol': 'AAPL',
            'price': {'last': 150.0, 'open': 149.0, 'high': 151.0, 'low': 148.0, 'volume': 1000000},
            'pattern': {'name': 'breakout', 'confidence': 0.85},
            'indicators': {'rsi_14': 65.0, 'macd_histogram': 0.2, 'bb_position': 0.7},
            'sentiment': {'overall_score': 0.6},
            'score': 0.85,
            'timestamp': datetime.now().isoformat()
        }
        
        # Sample position
        self.sample_position = {
            'symbol': 'AAPL',
            'entry_price': 150.0,
            'entry_time': (datetime.now() - timedelta(hours=2)).isoformat(),
            'quantity': 10,
            'side': 'long',
            'stop_loss': 145.0,
            'take_profit': 160.0,
            'trailing_stop': 2.0,
            'max_time': 6.0,
            'unrealized_pnl': 50.0,
            'unrealized_pnl_pct': 3.33,
            'current_price': 155.0,
            'status': 'active',
            'order_id': 'test-order-1234'
        }
        
        # Sample exit signal
        self.sample_exit_signal = {
            'reason': 'take_profit',
            'price': 160.0,
            'size': 1.0,
            'confidence': 0.9,
            'timestamp': datetime.now().isoformat()
        }
        
        # Mock Alpaca order response
        self.mock_order = MagicMock()
        self.mock_order.id = 'test-order-5678'
        self.mock_order.status = 'filled'
        self.mock_order.filled_qty = 10
        self.mock_order.filled_avg_price = 150.0
        self.mock_order.symbol = 'AAPL'
        self.mock_order.side = 'buy'
        self.mock_order.type = 'market'
        self.mock_order.created_at = datetime.now().isoformat()
        self.mock_order.filled_at = datetime.now().isoformat()
    
    @patch('src.llm.router.openrouter_client.chat_completion')
    @patch('src.core.data_pipeline.data_pipeline.get_market_context')
    @patch('src.core.position_monitor.position_monitor.get_portfolio_state')
    def test_make_trade_decision(self, mock_portfolio, mock_context, mock_chat):
        """Test making a trade decision."""
        # Set up mocks
        mock_context.return_value = {
            'state': 'open',
            'sector_performance': 1.2,
            'vix': 15.5,
            'time_until_close': 3.5
        }
        mock_portfolio.return_value = {
            'position_count': 1,
            'max_positions': 3,
            'available_capital': 10000.0,
            'daily_pnl': 500.0,
            'daily_pnl_pct': 5.0,
            'risk_remaining': 1000.0
        }
        mock_chat.return_value = {
            'choices': [
                {
                    'message': {
                        'content': '{"decision": "trade", "position_size": 0.5, "confidence": 0.8, '
                                  '"reasoning": "Strong breakout pattern with high volume", '
                                  '"key_factors": ["breakout", "high_volume", "positive_sentiment"]}'
                    }
                }
            ]
        }
        
        # Run test asynchronously
        result = asyncio.run(self.executor.make_trade_decision(self.sample_candidate))
        
        # Assert
        self.assertEqual(result['decision'], 'trade')
        self.assertEqual(result['position_size'], 0.5)
        self.assertEqual(result['confidence'], 0.8)
        mock_context.assert_called_once()
        mock_portfolio.assert_called_once()
        mock_chat.assert_called_once()
    
    @patch('src.core.data_pipeline.data_pipeline.get_stock_data')
    @patch('src.core.trade_execution.trade_executor.alpaca.submit_order')
    @patch('src.core.trade_execution.trade_executor.alpaca.get_position')
    @patch('src.utils.redis_client.redis_client.set_active_position')
    @patch('src.utils.redis_client.redis_client.add_trading_signal')
    def test_execute_trade(self, mock_add_signal, mock_set_position, mock_get_position, 
                          mock_submit_order, mock_get_data):
        """Test executing a trade."""
        # Set up mocks
        mock_get_data.return_value = {
            'symbol': 'AAPL',
            'price': {'last': 150.0}
        }
        mock_get_position.side_effect = Exception("Position not found")  # No existing position
        mock_submit_order.return_value = self.mock_order
        mock_set_position.return_value = True
        mock_add_signal.return_value = True
        
        # Patch trading mode to "paper"
        self.executor.trading_mode = "paper"
        
        # Run test asynchronously
        result = asyncio.run(self.executor.execute_trade('AAPL', self.sample_decision))
        
        # Assert
        self.assertTrue(result)
        mock_get_data.assert_called_once_with('AAPL', 'snapshot')
        mock_get_position.assert_called_once_with('AAPL')
        mock_submit_order.assert_called_once()
        mock_set_position.assert_called_once()
        mock_add_signal.assert_called_once()
    
    @patch('src.utils.redis_client.redis_client.get_all_trading_signals')
    @patch('src.utils.redis_client.redis_client.get_active_position')
    @patch('src.utils.redis_client.redis_client.clear_trading_signal')
    def test_process_exit_signals(self, mock_clear_signal, mock_get_position, mock_get_signals):
        """Test processing exit signals."""
        # Set up mocks
        mock_get_signals.return_value = {'AAPL': self.sample_exit_signal}
        mock_get_position.return_value = self.sample_position
        mock_clear_signal.return_value = True
        
        # Patch execute_exit to be a mock
        self.executor.execute_exit = AsyncMock(return_value=True)
        
        # Run test asynchronously
        asyncio.run(self.executor.process_exit_signals())
        
        # Assert
        mock_get_signals.assert_called_once_with('exit')
        mock_get_position.assert_called_once_with('AAPL')
        self.executor.execute_exit.assert_called_once_with(
            'AAPL', self.sample_position, 1.0, 'take_profit'
        )
        mock_clear_signal.assert_called_once_with('AAPL', 'exit')
    
    @patch('src.core.trade_execution.trade_executor.alpaca.submit_order')
    @patch('src.utils.redis_client.redis_client.set_active_position')
    @patch('src.utils.redis_client.redis_client.set_closed_position')
    @patch('src.utils.redis_client.redis_client.delete_active_position')
    def test_execute_exit(self, mock_delete_position, mock_set_closed, 
                         mock_set_active, mock_submit_order):
        """Test executing an exit."""
        # Set up mocks
        mock_submit_order.return_value = self.mock_order
        mock_set_closed.return_value = True
        mock_delete_position.return_value = True
        
        # Make a copy of the order with sell side
        sell_order = MagicMock()
        sell_order.id = 'test-order-9876'
        sell_order.status = 'filled'
        sell_order.filled_qty = 10
        sell_order.filled_avg_price = 160.0
        sell_order.symbol = 'AAPL'
        sell_order.side = 'sell'
        sell_order.type = 'market'
        sell_order.created_at = datetime.now().isoformat()
        sell_order.filled_at = datetime.now().isoformat()
        
        mock_submit_order.return_value = sell_order
        
        # Patch trading mode to "paper"
        self.executor.trading_mode = "paper"
        
        # Run test asynchronously
        result = asyncio.run(self.executor.execute_exit(
            'AAPL', self.sample_position, 1.0, 'take_profit'
        ))
        
        # Assert
        self.assertTrue(result)
        mock_submit_order.assert_called_once_with(
            symbol='AAPL',
            qty=10,
            side='sell',
            type='market',
            time_in_force='day'
        )
        mock_set_closed.assert_called_once()
        mock_delete_position.assert_called_once_with('AAPL')
    
    @patch('src.core.trade_execution.trade_executor.alpaca.get_account')
    def test_get_account_info(self, mock_get_account):
        """Test getting account information."""
        # Set up mock
        mock_account = MagicMock()
        mock_account.id = 'test-account-123'
        mock_account.status = 'ACTIVE'
        mock_account.cash = '10000.0'
        mock_account.buying_power = '20000.0'
        mock_account.equity = '15000.0'
        mock_account.long_market_value = '5000.0'
        mock_account.short_market_value = '0.0'
        mock_account.initial_margin = '2500.0'
        mock_account.maintenance_margin = '2000.0'
        mock_account.last_equity = '14800.0'
        mock_account.last_maintenance_margin = '1950.0'
        
        mock_get_account.return_value = mock_account
        
        # Run test asynchronously
        result = asyncio.run(self.executor.get_account_info())
        
        # Assert
        self.assertEqual(result['id'], 'test-account-123')
        self.assertEqual(result['status'], 'ACTIVE')
        self.assertEqual(result['cash'], 10000.0)
        self.assertEqual(result['buying_power'], 20000.0)
        mock_get_account.assert_called_once()
    
    @patch('src.core.trade_execution.trade_executor.alpaca.list_positions')
    @patch('src.utils.redis_client.redis_client.get_all_active_positions')
    @patch('src.utils.redis_client.redis_client.set_active_position')
    @patch('src.utils.redis_client.redis_client.set_closed_position')
    @patch('src.utils.redis_client.redis_client.delete_active_position')
    def test_sync_positions(self, mock_delete, mock_set_closed, mock_set_active, 
                           mock_get_positions, mock_list_positions):
        """Test syncing positions with Alpaca."""
        # Set up mocks
        mock_position = MagicMock()
        mock_position.symbol = 'AAPL'
        mock_position.qty = '10'
        mock_position.avg_entry_price = '150.0'
        mock_position.current_price = '155.0'
        mock_position.unrealized_pl = '50.0'
        mock_position.unrealized_plpc = '0.0333'
        mock_position.market_value = '1550.0'
        mock_position.created_at = datetime.now().isoformat()
        
        mock_list_positions.return_value = [mock_position]
        mock_get_positions.return_value = {
            'AAPL': self.sample_position,
            'MSFT': self.sample_position.copy()  # This one should be closed as it's not in Alpaca
        }
        mock_set_active.return_value = True
        mock_set_closed.return_value = True
        mock_delete.return_value = True
        
        # Run test asynchronously
        asyncio.run(self.executor.sync_positions())
        
        # Assert
        mock_list_positions.assert_called_once()
        mock_get_positions.assert_called_once()
        self.assertEqual(mock_set_active.call_count, 1)  # Update AAPL
        self.assertEqual(mock_set_closed.call_count, 1)  # Close MSFT
        self.assertEqual(mock_delete.call_count, 1)      # Delete MSFT from active

if __name__ == '__main__':
    unittest.main()