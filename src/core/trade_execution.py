"""
Trade execution logic for the trading system.

Handles:
- Trade decisions and execution with Alpaca
- Position management
- Order processing
- Risk management
"""
import os
from datetime import datetime
from typing import Dict, Any
from enum import Enum, auto

import pandas as pd
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

from src.config.settings import settings
from src.data_sources.polygon import PolygonAPI
from src.core.data_pipeline import data_pipeline
from src.core.position_monitor import position_monitor
from src.llm.router import openrouter_client
from src.llm.prompts import PromptTemplates
from src.llm.parsing import parse_trade_decision
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Define enum classes needed for trading system
class OrderStatus(Enum):
    """Order status enum"""
    PENDING = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()
    PARTIAL = auto()
    
class OrderType(Enum):
    """Order type enum"""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()
    
class TradeResult(Enum):
    """Trade execution result enum"""
    SUCCESS = auto()
    FAILURE = auto()
    PARTIAL = auto()
    REJECTED = auto()
    
class TradeDirection(Enum):
    """Trade direction enum"""
    LONG = auto()
    SHORT = auto()
    EXIT_LONG = auto()
    EXIT_SHORT = auto()

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("trade_execution")

class TradeExecutor:
    """
    Trade execution module for the trading system.
    
    Responsibilities:
    - Make trade decisions
    - Execute trades via Alpaca
    - Manage risk
    - Process exit signals
    """
    
    def __init__(self):
        """Initialize the trade executor with Alpaca API connection."""
        # Connect to Polygon for market data
        self.polygon_client = PolygonAPI()
        
        # Initialize Alpaca API connection
        self.alpaca_api_key = os.getenv('ALPACA_API_KEY')
        self.alpaca_api_secret = os.getenv('ALPACA_API_SECRET')
        self.alpaca_base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        
        if not self.alpaca_api_key or not self.alpaca_api_secret:
            logger.error("Alpaca API credentials not found in environment")
            raise ValueError("Alpaca API credentials not found in environment")
            
        # Create Alpaca API client
        self.alpaca = tradeapi.REST(
            self.alpaca_api_key,
            self.alpaca_api_secret,
            self.alpaca_base_url,
            api_version='v2'
        )
        
        # Trading parameters
        self.max_positions = settings.trading.max_positions
        self.max_position_size = settings.trading.max_position_size
        self.max_daily_risk = settings.trading.max_daily_risk
        
        # Track trade processing state
        self.symbols_processing = set()
        
        # Initialize trading statistics
        self.trading_stats = {
            'trades_executed': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'win_rate': 0.0
        }
        
        logger.info(f"Trade executor initialized with Alpaca API ({self.alpaca_base_url})")
    
    async def initialize(self):
        """Initialize the trade executor."""
        logger.info("Initializing trade executor")
        
        # Check Alpaca API connection
        try:
            account = self.alpaca.get_account()
            logger.info(f"Connected to Alpaca account: {account.id}")
            logger.info(f"Account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):.2f}")
            logger.info(f"Cash: ${float(account.cash):.2f}")
            
            # Store account information in Redis
            redis_client.set("alpaca:account", {
                'id': account.id,
                'status': account.status,
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'equity': float(account.equity),
                'last_updated': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error connecting to Alpaca API: {e}")
            raise
        
        # Validate trading mode
        self.trading_mode = settings.trading_mode
        logger.info(f"Trading mode: {self.trading_mode}")
        
        # Check for any pending orders
        await self.check_pending_orders()
        
        # Sync Alpaca positions with our system
        await self.sync_positions()
        
        logger.info("Trade executor initialization complete")
    
    async def check_pending_orders(self):
        """Check for any pending orders and update their status."""
        logger.info("Checking pending orders")
        
        try:
            # Get all open orders from Alpaca
            orders = self.alpaca.list_orders(status='open')
            
            # Process each order
            for order in orders:
                logger.info(f"Found open order: {order.id} for {order.symbol} ({order.side} {order.qty} @ {order.type})")
                
                # Store order in Redis
                redis_client.set(f"alpaca:orders:{order.id}", {
                    'id': order.id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'qty': float(order.qty),
                    'type': order.type,
                    'status': order.status,
                    'created_at': order.created_at,
                    'updated_at': order.updated_at
                })
            
            # Clear any stale signals that don't correspond to open orders
            order_symbols = {order.symbol for order in orders}
            
            # Clear stale entry signals
            signals = redis_client.get_all_trading_signals('entry')
            for symbol, signal in signals.items():
                if symbol not in order_symbols:
                    # Check if signal is stale (older than 5 minutes)
                    try:
                        timestamp = pd.Timestamp(signal.get('timestamp', ''))
                        if (pd.Timestamp.now() - timestamp).total_seconds() > 300:
                            redis_client.clear_trading_signal(symbol, 'entry')
                            logger.info(f"Cleared stale entry signal for {symbol}")
                    except Exception as e:
                        logger.error(f"Error processing signal timestamp for {symbol}: {e}")
                        redis_client.clear_trading_signal(symbol, 'entry')
            
            # Clear stale exit signals
            signals = redis_client.get_all_trading_signals('exit')
            for symbol, signal in signals.items():
                if symbol not in order_symbols:
                    # Check if signal is stale (older than 5 minutes)
                    try:
                        timestamp = pd.Timestamp(signal.get('timestamp', ''))
                        if (pd.Timestamp.now() - timestamp).total_seconds() > 300:
                            redis_client.clear_trading_signal(symbol, 'exit')
                            logger.info(f"Cleared stale exit signal for {symbol}")
                    except Exception as e:
                        logger.error(f"Error processing signal timestamp for {symbol}: {e}")
                        redis_client.clear_trading_signal(symbol, 'exit')
                        
        except Exception as e:
            logger.error(f"Error checking pending orders: {e}")
            
    async def sync_positions(self):
        """Synchronize Alpaca positions with our system."""
        logger.info("Syncing positions with Alpaca")
        
        try:
            # Get all positions from Alpaca
            alpaca_positions = self.alpaca.list_positions()
            
            # Create a set of symbols from Alpaca positions
            alpaca_symbols = {position.symbol for position in alpaca_positions}
            
            # Get our tracked positions
            our_positions = redis_client.get_all_active_positions()
            our_symbols = set(our_positions.keys())
            
            # Find positions in Alpaca that we don't track
            missing_symbols = alpaca_symbols - our_symbols
            
            # Find positions we track that aren't in Alpaca
            extra_symbols = our_symbols - alpaca_symbols
            
            # Add missing positions to our system
            for symbol in missing_symbols:
                position = next((p for p in alpaca_positions if p.symbol == symbol), None)
                if position:
                    # Create position data
                    position_data = {
                        'symbol': position.symbol,
                        'entry_price': float(position.avg_entry_price),
                        'entry_time': pd.Timestamp(position.created_at).isoformat(),
                        'quantity': float(position.qty),
                        'side': 'long' if float(position.qty) > 0 else 'short',
                        'current_price': float(position.current_price),
                        'unrealized_pnl': float(position.unrealized_pl),
                        'unrealized_pnl_pct': float(position.unrealized_plpc) * 100,
                        'market_value': float(position.market_value),
                        'source': 'alpaca_sync',
                        'last_update': datetime.now().isoformat()
                    }
                    
                    # Save to Redis
                    redis_client.set_active_position(symbol, position_data)
                    logger.info(f"Added position from Alpaca: {symbol} ({position_data['quantity']} @ ${position_data['entry_price']:.2f})")
            
            # Remove extra positions from our system
            for symbol in extra_symbols:
                logger.warning(f"Found position in our system that doesn't exist in Alpaca: {symbol}")
                # We can either delete it or mark it as closed
                position = our_positions[symbol]
                position['status'] = 'closed'
                position['close_reason'] = 'not_found_in_alpaca'
                position['close_time'] = datetime.now().isoformat()
                
                # Move to closed positions
                redis_client.set_closed_position(symbol, position)
                
                # Remove from active positions
                redis_client.delete_active_position(symbol)
                logger.info(f"Moved {symbol} to closed positions")
                
            # Update existing positions with latest data from Alpaca
            for symbol in alpaca_symbols.intersection(our_symbols):
                position = next((p for p in alpaca_positions if p.symbol == symbol), None)
                our_position = our_positions[symbol]
                
                if position:
                    # Update current price and P&L
                    our_position['current_price'] = float(position.current_price)
                    our_position['unrealized_pnl'] = float(position.unrealized_pl)
                    our_position['unrealized_pnl_pct'] = float(position.unrealized_plpc) * 100
                    our_position['market_value'] = float(position.market_value)
                    our_position['last_update'] = datetime.now().isoformat()
                    
                    # Save to Redis
                    redis_client.set_active_position(symbol, our_position)
            
            logger.info(f"Position sync complete: {len(alpaca_symbols)} positions from Alpaca, {len(our_symbols)} in our system")
            
        except Exception as e:
            logger.error(f"Error syncing positions with Alpaca: {e}")
    
    async def make_trade_decision(self, candidate: dict):
        """
        Make a trade decision for a candidate using LLM.
        
        Args:
            candidate: Candidate data
            
        Returns:
            Trade decision
        """
        logger.info(f"Making trade decision for {candidate['symbol']}...")
        
        # Get market context
        market_context = await data_pipeline.get_market_context()
        
        # Get portfolio state
        portfolio_state = await position_monitor.get_portfolio_state()
        
        # Create input data for LLM
        stock_data = {
            'symbol': candidate['symbol'],
            'price': candidate.get('price', {}),
            'pattern': candidate.get('pattern', {}),
            'indicators': {
                'rsi_14': candidate.get('rsi_14', 50),
                'macd_histogram': candidate.get('macd_histogram', 0),
                'bb_position': candidate.get('bb_position', 0.5)
            },
            'sentiment': candidate.get('sentiment', {}),
            'news': candidate.get('sentiment', {}).get('news', [])
        }
        
        # Create messages for LLM
        messages = PromptTemplates.create_trade_decision_prompt(
            stock_data,
            market_context,
            portfolio_state
        )
        
        # Get decision from LLM
        response = await openrouter_client.chat_completion(messages)
        
        # Parse decision
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        decision = parse_trade_decision(content)
        
        # Log decision
        if decision['decision'] == 'trade':
            logger.info(f"Decision: TRADE {candidate['symbol']} with {decision['position_size']*100:.0f}% position size")
            logger.info(f"Reason: {decision['reasoning']}")
        else:
            logger.info(f"Decision: NO TRADE for {candidate['symbol']}")
            logger.info(f"Reason: {decision['reasoning']}")
        
        # Store decision in Redis
        redis_client.set(f"decisions:{candidate['symbol']}", {
            'symbol': candidate['symbol'],
            'decision': decision,
            'timestamp': datetime.now().isoformat()
        }, expiry=3600)  # Expire after 1 hour
        
        return decision
    
    async def execute_trade(self, symbol: str, decision: dict):
        """
        Execute a trade based on decision using Alpaca.
        
        Args:
            symbol: Stock symbol
            decision: Trade decision
            
        Returns:
            True if trade executed, False otherwise
        """
        logger.info(f"Executing trade for {symbol}...")
        
        # Skip if decision is not to trade
        if decision['decision'] != 'trade' or decision['position_size'] <= 0:
            logger.info(f"Decision is not to trade {symbol}, skipping execution")
            return False
        
        # Check if we're already processing this symbol
        if symbol in self.symbols_processing:
            logger.warning(f"Already processing a trade for {symbol}, skipping")
            return False
        
        # Mark symbol as being processed
        self.symbols_processing.add(symbol)
        
        try:
            # Get current price
            price_data = await data_pipeline.get_stock_data(symbol, 'snapshot')
            current_price = price_data.get('price', {}).get('last', 0) if price_data else 0
            
            if current_price <= 0:
                logger.error(f"Invalid price for {symbol}: {current_price}")
                return False
            
            # Calculate position size
            max_position_size = self.max_position_size
            position_size = max_position_size * decision['position_size']
            
            # Calculate quantity
            quantity = int(position_size / current_price)
            
            if quantity <= 0:
                logger.warning(f"Calculated quantity is {quantity}, skipping trade")
                return False
            
            # Calculate stop loss and take profit
            stop_loss = current_price * 0.95  # 5% stop loss
            take_profit = current_price * 1.1  # 10% take profit
            
            try:
                # Check if we already have a position in this symbol
                existing_position = None
                try:
                    existing_position = self.alpaca.get_position(symbol)
                    logger.info(f"Found existing position in {symbol}: {existing_position.qty} shares")
                except:
                    pass
                
                if existing_position:
                    logger.warning(f"Already have a position in {symbol}, skipping trade")
                    return False
                
                # Submit order to Alpaca
                logger.info(f"Submitting market order for {quantity} shares of {symbol}")
                
                # Check if we're in paper trading mode
                if self.trading_mode == "paper":
                    # Submit order to Alpaca
                    order = self.alpaca.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    logger.info(f"Order submitted: {order.id} for {order.qty} shares of {symbol} at market")
                    
                    # Store order in Redis
                    redis_client.set(f"alpaca:orders:{order.id}", {
                        'id': order.id,
                        'symbol': symbol,
                        'side': 'buy',
                        'qty': float(order.qty),
                        'type': 'market',
                        'status': order.status,
                        'created_at': order.created_at
                    })
                else:
                    # In live mode, log but don't actually submit
                    logger.info(f"SIMULATION: Would submit market order for {quantity} shares of {symbol}")
                    
                    # Create a simulated order ID
                    import uuid
                    order_id = f"sim-{uuid.uuid4()}"
                    
                    # Simulate order response
                    order = {
                        'id': order_id,
                        'symbol': symbol,
                        'side': 'buy',
                        'qty': quantity,
                        'type': 'market',
                        'status': 'filled',  # Assume immediately filled for simulation
                        'created_at': datetime.now().isoformat()
                    }
                
                # Create position data (we'll update this when the order is filled)
                position_data = {
                    'symbol': symbol,
                    'entry_price': current_price,  # Estimated entry price
                    'entry_time': datetime.now().isoformat(),
                    'quantity': quantity,
                    'side': 'long',
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'trailing_stop': 2.0,  # 2% trailing stop
                    'max_time': 6.0,  # 6 hours max hold time
                    'unrealized_pnl': 0.0,
                    'unrealized_pnl_pct': 0.0,
                    'order_id': order.id if hasattr(order, 'id') else order['id'],
                    'status': 'pending',  # Will be updated to 'active' when filled
                    'reason': decision.get('reasoning', ''),
                    'key_factors': decision.get('key_factors', [])
                }
                
                # Save to Redis
                redis_client.set_active_position(symbol, position_data)
                
                # Create signal
                redis_client.add_trading_signal(symbol, 'entry', {
                    'price': current_price,
                    'quantity': quantity,
                    'order_id': order.id if hasattr(order, 'id') else order['id'],
                    'reason': decision.get('reasoning', ''),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update trading stats
                self.trading_stats['trades_executed'] += 1
                
                logger.info(f"Trade executed for {symbol}: {quantity} shares at ~${current_price:.2f}")
                logger.info(f"Stop loss: ${stop_loss:.2f}, Take profit: ${take_profit:.2f}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
                
                # Update trading stats
                self.trading_stats['failed_trades'] += 1
                
                return False
                
        finally:
            # Mark symbol as no longer being processed
            self.symbols_processing.discard(symbol)
    
    async def process_exit_signals(self):
        """Process exit signals and execute exit orders."""
        logger.debug("Processing exit signals")
        
        # Get all exit signals
        signals = redis_client.get_all_trading_signals('exit')
        
        if not signals:
            return
        
        logger.info(f"Processing {len(signals)} exit signals")
        
        # Process each signal
        for symbol, signal in signals.items():
            # Skip if already processing
            if symbol in self.symbols_processing:
                continue
            
            # Mark as processing
            self.symbols_processing.add(symbol)
            
            try:
                # Get position data
                position = redis_client.get_active_position(symbol)
                
                if not position:
                    logger.warning(f"Exit signal for {symbol} but no active position found")
                    redis_client.clear_trading_signal(symbol, 'exit')
                    continue
                
                # Get exit size
                exit_size = signal.get('size', 1.0)  # Default to full exit
                
                # Execute exit
                await self.execute_exit(symbol, position, exit_size, signal.get('reason', 'exit_signal'))
                
                # Clear the signal after processing
                redis_client.clear_trading_signal(symbol, 'exit')
                
            except Exception as e:
                logger.error(f"Error processing exit signal for {symbol}: {e}")
            finally:
                # Mark as no longer processing
                self.symbols_processing.discard(symbol)
    
    async def execute_exit(self, symbol: str, position: Dict[str, Any], exit_size: float, reason: str):
        """
        Execute an exit order using Alpaca.
        
        Args:
            symbol: Stock symbol
            position: Position data
            exit_size: Size of exit (0.0-1.0, percent of position)
            reason: Reason for exit
            
        Returns:
            True if exit executed, False otherwise
        """
        logger.info(f"Executing exit for {symbol} ({exit_size*100:.0f}% of position): {reason}")
        
        # Validate exit size
        if exit_size <= 0 or exit_size > 1.0:
            logger.error(f"Invalid exit size for {symbol}: {exit_size}")
            return False
        
        try:
            # Get position quantity
            quantity = position.get('quantity', 0)
            
            # Calculate exit quantity
            exit_quantity = int(quantity * exit_size)
            if exit_quantity <= 0:
                logger.warning(f"Calculated exit quantity is {exit_quantity}, skipping exit")
                return False
            
            # Submit order to Alpaca
            if self.trading_mode == "paper":
                # Submit order to Alpaca
                logger.info(f"Submitting market sell order for {exit_quantity} shares of {symbol}")
                
                order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=exit_quantity,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                logger.info(f"Exit order submitted: {order.id} for {order.qty} shares of {symbol} at market")
                
                # Store order in Redis
                redis_client.set(f"alpaca:orders:{order.id}", {
                    'id': order.id,
                    'symbol': symbol,
                    'side': 'sell',
                    'qty': float(order.qty),
                    'type': 'market',
                    'status': order.status,
                    'created_at': order.created_at
                })
            else:
                # In live mode, log but don't actually submit
                logger.info(f"SIMULATION: Would submit market sell order for {exit_quantity} shares of {symbol}")
                
                # Create a simulated order ID
                import uuid
                order_id = f"sim-{uuid.uuid4()}"
                
                # Simulate order response
                order = {
                    'id': order_id,
                    'symbol': symbol,
                    'side': 'sell',
                    'qty': exit_quantity,
                    'type': 'market',
                    'status': 'filled',  # Assume immediately filled for simulation
                    'created_at': datetime.now().isoformat()
                }
            
            # Update position in Redis
            if exit_size >= 1.0 or exit_quantity >= quantity:
                # Full exit
                logger.info(f"Full exit executed for {symbol}: {quantity} shares")
                
                # Move to closed positions
                position['status'] = 'closed'
                position['close_time'] = datetime.now().isoformat()
                position['close_reason'] = reason
                position['exit_order_id'] = order.id if hasattr(order, 'id') else order['id']
                
                # Calculate P&L
                current_price = position.get('current_price', 0)
                entry_price = position.get('entry_price', 0)
                if current_price > 0 and entry_price > 0:
                    realized_pnl = (current_price - entry_price) * quantity
                    realized_pnl_pct = (current_price / entry_price - 1) * 100
                    
                    position['realized_pnl'] = realized_pnl
                    position['realized_pnl_pct'] = realized_pnl_pct
                    
                    # Update trading stats
                    if realized_pnl > 0:
                        self.trading_stats['successful_trades'] += 1
                        self.trading_stats['total_profit'] += realized_pnl
                    else:
                        self.trading_stats['total_loss'] += abs(realized_pnl)
                    
                    # Update win rate
                    total_trades = self.trading_stats['successful_trades'] + self.trading_stats['failed_trades']
                    if total_trades > 0:
                        self.trading_stats['win_rate'] = self.trading_stats['successful_trades'] / total_trades
                
                # Save to closed positions
                redis_client.set_closed_position(symbol, position)
                
                # Remove from active positions
                redis_client.delete_active_position(symbol)
                
            else:
                # Partial exit
                remaining_quantity = quantity - exit_quantity
                
                # Update position data
                position['quantity'] = remaining_quantity
                position['partial_exits'] = position.get('partial_exits', []) + [{
                    'time': datetime.now().isoformat(),
                    'quantity': exit_quantity,
                    'reason': reason,
                    'order_id': order.id if hasattr(order, 'id') else order['id']
                }]
                
                # Save to Redis
                redis_client.set_active_position(symbol, position)
                
                logger.info(f"Partial exit executed for {symbol}: {exit_quantity} shares")
                logger.info(f"Remaining position: {remaining_quantity} shares")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing exit for {symbol}: {e}")
            return False
    
    async def close_all_positions(self):
        """Close all open positions (used at market close)."""
        logger.info("Closing all positions")
        
        # Get active positions
        positions = redis_client.get_all_active_positions()
        
        if not positions:
            logger.info("No positions to close")
            return True
        
        logger.info(f"Closing {len(positions)} positions")
        
        # Process each position
        success = True
        for symbol, position in positions.items():
            try:
                # Execute full exit
                result = await self.execute_exit(symbol, position, 1.0, 'market_close')
                if not result:
                    success = False
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")
                success = False
        
        return success
    
    async def check_order_updates(self):
        """Check for updates on pending orders."""
        logger.debug("Checking order updates")
        
        try:
            # Get recent orders from Alpaca (last 100)
            orders = self.alpaca.list_orders(status='all', limit=100)
            
            # Process filled orders
            for order in orders:
                if order.status == 'filled':
                    # Check if this is a buy or sell order
                    if order.side == 'buy':
                        # This is an entry order
                        await self.process_filled_entry(order)
                    else:
                        # This is an exit order
                        await self.process_filled_exit(order)
            
            # Process canceled or rejected orders
            for order in orders:
                if order.status in ['canceled', 'rejected', 'expired']:
                    # Log the failure
                    logger.warning(f"Order {order.id} for {order.symbol} {order.side} {order.qty} was {order.status}")
                    
                    # Update Redis with order status
                    redis_client.set(f"alpaca:orders:{order.id}", {
                        'id': order.id,
                        'symbol': order.symbol,
                        'side': order.side,
                        'qty': float(order.qty),
                        'type': order.type,
                        'status': order.status,
                        'created_at': order.created_at,
                        'updated_at': order.updated_at
                    })
                    
                    # If entry order was canceled, clean up pending position
                    if order.side == 'buy':
                        position = redis_client.get_active_position(order.symbol)
                        if position and position.get('status') == 'pending' and position.get('order_id') == order.id:
                            # Delete the pending position
                            redis_client.delete_active_position(order.symbol)
                            logger.info(f"Deleted pending position for {order.symbol} due to {order.status} order")
        
        except Exception as e:
            logger.error(f"Error checking order updates: {e}")
    
    async def process_filled_entry(self, order):
        """
        Process a filled entry order.
        
        Args:
            order: Alpaca order object
        """
        symbol = order.symbol
        logger.info(f"Processing filled entry order for {symbol}: {order.id}")
        
        # Check if we already processed this order
        processed_orders = redis_client.get("alpaca:processed_orders") or []
        if order.id in processed_orders:
            logger.debug(f"Order {order.id} already processed, skipping")
            return
        
        # Get position data
        position = redis_client.get_active_position(symbol)
        
        if position and position.get('order_id') == order.id:
            # Update position with fill information
            position['status'] = 'active'
            position['entry_price'] = float(order.filled_avg_price)
            position['quantity'] = float(order.filled_qty)
            position['filled_time'] = order.filled_at
            
            # Calculate stop loss and take profit based on actual fill price
            fill_price = float(order.filled_avg_price)
            position['stop_loss'] = fill_price * 0.95  # 5% stop loss
            position['take_profit'] = fill_price * 1.1  # 10% take profit
            
            # Save updated position
            redis_client.set_active_position(symbol, position)
            
            logger.info(f"Updated position for {symbol}: {position['quantity']} shares @ ${position['entry_price']:.2f}")
            
            # Set up trailing stop order if supported
            if self.trading_mode == "paper" and hasattr(self.alpaca, 'submit_trailing_stop_order'):
                try:
                    # Calculate trailing stop amount
                    trail_percent = position.get('trailing_stop', 2.0)
                    
                    # Submit trailing stop order
                    stop_order = self.alpaca.submit_trailing_stop_order(
                        symbol=symbol,
                        qty=position['quantity'],
                        side='sell',
                        trail_percent=trail_percent,
                        time_in_force='gtc'
                    )
                    
                    # Store stop order ID in position
                    position['stop_order_id'] = stop_order.id
                    redis_client.set_active_position(symbol, position)
                    
                    logger.info(f"Set trailing stop for {symbol} at {trail_percent}%")
                except Exception as e:
                    logger.error(f"Error setting trailing stop for {symbol}: {e}")
        else:
            # This is a new position we don't have tracked
            logger.warning(f"Filled order {order.id} for {symbol} doesn't match any pending position")
            
            # Create new position data
            position_data = {
                'symbol': symbol,
                'entry_price': float(order.filled_avg_price),
                'entry_time': order.filled_at,
                'quantity': float(order.filled_qty),
                'side': 'long',
                'status': 'active',
                'order_id': order.id,
                'stop_loss': float(order.filled_avg_price) * 0.95,  # 5% stop loss
                'take_profit': float(order.filled_avg_price) * 1.1,  # 10% take profit
                'trailing_stop': 2.0,  # 2% trailing stop
                'max_time': 6.0,  # 6 hours max hold time
                'unrealized_pnl': 0.0,
                'unrealized_pnl_pct': 0.0,
                'source': 'alpaca_fill',
                'filled_time': order.filled_at
            }
            
            # Save to Redis
            redis_client.set_active_position(symbol, position_data)
            
            logger.info(f"Created new position from fill for {symbol}: {position_data['quantity']} shares @ ${position_data['entry_price']:.2f}")
        
        # Mark order as processed
        processed_orders.append(order.id)
        redis_client.set("alpaca:processed_orders", processed_orders)
    
    async def process_filled_exit(self, order):
        """
        Process a filled exit order.
        
        Args:
            order: Alpaca order object
        """
        symbol = order.symbol
        logger.info(f"Processing filled exit order for {symbol}: {order.id}")
        
        # Check if we already processed this order
        processed_orders = redis_client.get("alpaca:processed_orders") or []
        if order.id in processed_orders:
            logger.debug(f"Order {order.id} already processed, skipping")
            return
        
        # Get position data
        position = redis_client.get_active_position(symbol)
        
        if position:
            # Calculate how much of the position was closed
            exit_quantity = float(order.filled_qty)
            current_quantity = position.get('quantity', 0)
            
            if exit_quantity >= current_quantity:
                # Full exit
                logger.info(f"Full exit filled for {symbol}: {exit_quantity} shares @ ${float(order.filled_avg_price):.2f}")
                
                # Update position with exit information
                position['status'] = 'closed'
                position['close_price'] = float(order.filled_avg_price)
                position['close_time'] = order.filled_at
                position['exit_order_id'] = order.id
                
                # Calculate realized P&L
                entry_price = position.get('entry_price', 0)
                if entry_price > 0:
                    realized_pnl = (float(order.filled_avg_price) - entry_price) * exit_quantity
                    realized_pnl_pct = (float(order.filled_avg_price) / entry_price - 1) * 100
                    
                    position['realized_pnl'] = realized_pnl
                    position['realized_pnl_pct'] = realized_pnl_pct
                    
                    # Update trading stats
                    if realized_pnl > 0:
                        self.trading_stats['successful_trades'] += 1
                        self.trading_stats['total_profit'] += realized_pnl
                    else:
                        self.trading_stats['total_loss'] += abs(realized_pnl)
                    
                    # Update win rate
                    total_trades = self.trading_stats['successful_trades'] + self.trading_stats['failed_trades']
                    if total_trades > 0:
                        self.trading_stats['win_rate'] = self.trading_stats['successful_trades'] / total_trades
                
                # Move to closed positions
                redis_client.set_closed_position(symbol, position)
                
                # Remove from active positions
                redis_client.delete_active_position(symbol)
            else:
                # Partial exit
                remaining_quantity = current_quantity - exit_quantity
                
                logger.info(f"Partial exit filled for {symbol}: {exit_quantity} shares @ ${float(order.filled_avg_price):.2f}")
                logger.info(f"Remaining position: {remaining_quantity} shares")
                
                # Update position data
                position['quantity'] = remaining_quantity
                position['partial_exits'] = position.get('partial_exits', []) + [{
                    'time': order.filled_at,
                    'quantity': exit_quantity,
                    'price': float(order.filled_avg_price),
                    'order_id': order.id
                }]
                
                # Save updated position
                redis_client.set_active_position(symbol, position)
        else:
            logger.warning(f"Filled exit order {order.id} for {symbol} but no matching position found")
        
        # Mark order as processed
        processed_orders.append(order.id)
        redis_client.set("alpaca:processed_orders", processed_orders)
    
    async def get_account_info(self):
        """
        Get current Alpaca account information.
        
        Returns:
            Dictionary with account information
        """
        try:
            account = self.alpaca.get_account()
            
            # Create account info dictionary
            account_info = {
                'id': account.id,
                'status': account.status,
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_equity': float(account.last_equity),
                'last_maintenance_margin': float(account.last_maintenance_margin),
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in Redis
            redis_client.set("alpaca:account", account_info)
            
            return account_info
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    async def get_trading_stats(self):
        """
        Get trading statistics.
        
        Returns:
            Dictionary with trading statistics
        """
        # Get account info for latest equity
        account_info = await self.get_account_info()
        
        # Get all closed positions
        closed_positions = redis_client.get_all_closed_positions()
        
        # Calculate additional stats
        total_trades = len(closed_positions)
        profitable_trades = sum(1 for p in closed_positions.values() if p.get('realized_pnl', 0) > 0)
        total_profit = sum(p.get('realized_pnl', 0) for p in closed_positions.values() if p.get('realized_pnl', 0) > 0)
        total_loss = sum(abs(p.get('realized_pnl', 0)) for p in closed_positions.values() if p.get('realized_pnl', 0) < 0)
        
        # Avoid division by zero
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Create stats dictionary
        stats = {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'unprofitable_trades': total_trades - profitable_trades,
            'win_rate': win_rate * 100,  # As percentage
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_pnl': total_profit - total_loss,
            'profit_factor': profit_factor,
            'current_equity': account_info.get('equity', 0),
            'active_positions': len(redis_client.get_all_active_positions()),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in Redis
        redis_client.set("trading:stats", stats)
        
        return stats

# Create global instance
trade_executor = TradeExecutor()