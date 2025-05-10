"""
Position monitoring module for the trading system.

Responsibilities:
- Monitor active positions for exit conditions
- Evaluate risk management rules
- Generate exit signals
- Optimize position management
"""
import asyncio
from datetime import datetime
from typing import Dict, Optional, Any

import pandas as pd

from src.config.settings import settings
from src.data_sources.polygon import PolygonAPI
from src.models.exit_optimization import exit_optimization_model
from src.core.data_pipeline import data_pipeline
from src.llm.router import openrouter_client
from src.llm.prompts import PromptTemplates
from src.llm.parsing import parse_exit_decision
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("position_monitor")

class PositionMonitor:
    """
    Position monitoring for managing active positions.
    
    Responsibilities:
    - Monitor active positions
    - Check exit conditions (stop loss, take profit, trailing stop)
    - Optimize position management
    - Generate exit signals
    """
    
    def __init__(self):
        """Initialize the position monitor."""
        self.polygon_client = PolygonAPI()
        
        # Set default risk management parameters
        self.default_stop_loss_pct = 5.0  # Default 5% stop loss
        self.default_take_profit_pct = 10.0  # Default 10% take profit
        self.default_trailing_stop_pct = 2.0  # Default 2% trailing stop
        self.default_max_holding_time = 6.0  # Default 6 hours max holding time
        
        # Maximum daily loss limit
        self.max_daily_loss = settings.trading.max_daily_risk
        
        # Track monitoring state
        self.positions_under_review = set()
        
        logger.info("Position monitor initialized")
    
    async def initialize(self):
        """Initialize the position monitor."""
        logger.info("Initializing position monitor")
        
        # Validate existing positions
        await self.validate_positions()
        
        logger.info("Position monitor initialization complete")
    
    async def validate_positions(self):
        """Validate existing positions and update their status."""
        logger.info("Validating existing positions")
        
        # Get all active positions
        positions = redis_client.get_all_active_positions()
        
        if not positions:
            logger.info("No active positions to validate")
            return True
        
        # Process each position
        for symbol, position_data in positions.items():
            try:
                # Get current price
                price_data = await data_pipeline.get_stock_data(symbol, 'snapshot')
                
                if not price_data or 'price' not in price_data:
                    logger.warning(f"Could not get price data for {symbol}, position may be stale")
                    continue
                
                # Update position P&L
                current_price = price_data.get('price', {}).get('last', 0)
                redis_client.update_position_pnl(symbol, current_price)
                
                logger.info(f"Validated position: {symbol} at ${current_price:.2f}")
            except Exception as e:
                logger.error(f"Error validating position {symbol}: {e}")
        
        return True
    
    async def monitor_positions(self):
        """Monitor all active positions."""
        logger.debug("Monitoring positions")
        
        # Get active positions
        positions = redis_client.get_all_active_positions()
        
        if not positions:
            return
        
        logger.debug(f"Monitoring {len(positions)} active positions")
        
        # Check if market is open
        market_context = await data_pipeline.get_market_context()
        market_state = market_context.get('state', 'unknown')
        time_until_close = market_context.get('time_until_close', 0)
        
        # Process each position
        for symbol, position in positions.items():
            # Skip positions already under review
            if symbol in self.positions_under_review:
                continue
            
            # Mark position as under review
            self.positions_under_review.add(symbol)
            
            try:
                # Monitor this position in the background
                asyncio.create_task(self.monitor_position(symbol, position, market_state, time_until_close))
            except Exception as e:
                logger.error(f"Error starting position monitoring for {symbol}: {e}")
                self.positions_under_review.discard(symbol)
    
    async def monitor_position(self, symbol: str, position: Dict[str, Any], market_state: str, time_until_close: float):
        """
        Monitor a single position.
        
        Args:
            symbol: Stock symbol
            position: Position data
            market_state: Current market state
            time_until_close: Hours until market close
        """
        try:
            # Get current price data
            price_data = await data_pipeline.get_stock_data(symbol, 'snapshot')
            
            if not price_data or 'price' not in price_data:
                logger.warning(f"Could not get price data for {symbol}, skipping monitoring")
                return
            
            # Get OHLCV data
            intraday_data = await data_pipeline.get_stock_data(symbol, 'intraday')
            
            if intraday_data is None or (isinstance(intraday_data, pd.DataFrame) and intraday_data.empty):
                logger.warning(f"Could not get OHLCV data for {symbol}, skipping monitoring")
                return
            
            # Update position P&L
            current_price = price_data.get('price', {}).get('last', 0)
            redis_client.update_position_pnl(symbol, current_price)
            
            # Get updated position data
            position = redis_client.get_active_position(symbol)
            
            # Check for forced exit conditions
            forced_exit = await self.check_forced_exit_conditions(symbol, position, market_state, time_until_close)
            
            if forced_exit:
                logger.info(f"Forced exit triggered for {symbol}: {forced_exit['reason']}")
                
                # Create exit signal
                redis_client.add_trading_signal(symbol, 'exit', {
                    'reason': forced_exit['reason'],
                    'price': current_price,
                    'size': 1.0,  # Full exit
                    'timestamp': datetime.now().isoformat()
                })
                
                return
            
            # Evaluate exit conditions using ML model
            exit_recommendation = exit_optimization_model.evaluate_exit_conditions(
                intraday_data, position, confidence_threshold=0.6
            )
            
            # Check if any exit condition is triggered from ML model
            ml_exit_triggered = (
                exit_recommendation.get('exit', False) or
                exit_recommendation.get('stop_loss_triggered', False) or
                exit_recommendation.get('take_profit_triggered', False) or
                exit_recommendation.get('trailing_stop_triggered', False) or
                exit_recommendation.get('time_stop_triggered', False)
            )
            
            if ml_exit_triggered:
                logger.info(f"ML exit signal triggered for {symbol}: {exit_recommendation.get('reason', 'unknown')}")
                
                # Get market context
                market_context = await data_pipeline.get_market_context()
                
                # Get technical indicators
                indicators = await data_pipeline.get_technical_indicators(symbol)
                
                # Create current data
                current_data = {
                    'symbol': symbol,
                    'price': price_data.get('price', {}),
                    'indicators': indicators or {}
                }
                
                # Get exit decision from LLM
                exit_decision = await self.get_exit_decision(
                    position, current_data, market_context, exit_recommendation
                )
                
                if exit_decision['decision'] == 'exit':
                    logger.info(f"LLM confirmed exit for {symbol} with size {exit_decision['exit_size']}")
                    
                    # Create exit signal
                    redis_client.add_trading_signal(symbol, 'exit', {
                        'reason': exit_decision['reasoning'],
                        'price': current_price,
                        'size': exit_decision['exit_size'],
                        'confidence': exit_decision['confidence'],
                        'key_factors': exit_decision['key_factors'],
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    logger.debug(f"LLM recommended holding {symbol}: {exit_decision['reasoning']}")
        except Exception as e:
            logger.error(f"Error monitoring position {symbol}: {e}")
        finally:
            # Mark position as no longer under review
            self.positions_under_review.discard(symbol)
    
    async def check_forced_exit_conditions(
        self, 
        symbol: str, 
        position: Dict[str, Any], 
        market_state: str,
        time_until_close: float
    ) -> Optional[Dict[str, Any]]:
        """
        Check for conditions that force an immediate exit.
        
        Args:
            symbol: Stock symbol
            position: Position data
            market_state: Current market state
            time_until_close: Hours until market close
            
        Returns:
            Exit reason if exit forced, None otherwise
        """
        # Check for market close (always exit all positions before market close)
        if market_state == 'open' and time_until_close < 0.1:  # Less than 6 minutes until close
            return {
                'reason': 'market_close',
                'description': 'Market closing soon, exiting all positions'
            }
        
        # Get position details
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', 0)
        unrealized_pnl = position.get('unrealized_pnl', 0)
        unrealized_pnl_pct = position.get('unrealized_pnl_pct', 0)
        stop_loss = position.get('stop_loss', 0)
        take_profit = position.get('take_profit', 0)
        trailing_stop = position.get('trailing_stop', 0)
        max_time = position.get('max_time', 0)  # In hours
        side = position.get('side', 'long')
        entry_time_str = position.get('entry_time', '')
        
        # Check for stop loss hit
        if stop_loss > 0 and current_price <= stop_loss and side == 'long':
            return {
                'reason': 'stop_loss',
                'description': f'Stop loss triggered at ${current_price:.2f} (stop: ${stop_loss:.2f})'
            }
        
        # Check for take profit hit
        if take_profit > 0 and current_price >= take_profit and side == 'long':
            return {
                'reason': 'take_profit',
                'description': f'Take profit triggered at ${current_price:.2f} (target: ${take_profit:.2f})'
            }
        
        # Check trailing stop
        if trailing_stop > 0 and side == 'long':
            # Calculate highest price since entry
            high_since_entry = position.get('high_since_entry', entry_price)
            
            # Update highest price if current price is higher
            if current_price > high_since_entry:
                position['high_since_entry'] = current_price
                redis_client.set_active_position(symbol, position)
                high_since_entry = current_price
            
            # Calculate trailing stop price
            trail_price = high_since_entry * (1 - trailing_stop / 100)
            
            if current_price <= trail_price:
                return {
                    'reason': 'trailing_stop',
                    'description': f'Trailing stop triggered at ${current_price:.2f} (trail: ${trail_price:.2f})'
                }
        
        # Check maximum hold time
        if max_time > 0 and entry_time_str:
            try:
                entry_time = pd.Timestamp(entry_time_str)
                current_time = pd.Timestamp.now()
                hours_held = (current_time - entry_time).total_seconds() / 3600
                
                if hours_held >= max_time:
                    return {
                        'reason': 'time_limit',
                        'description': f'Maximum holding time reached ({hours_held:.1f} hours)'
                    }
            except Exception as e:
                logger.error(f"Error calculating holding time for {symbol}: {e}")
        
        # Check for extreme loss (circuit breaker)
        max_loss_pct = -15.0  # Example: -15% max loss
        if unrealized_pnl_pct <= max_loss_pct:
            return {
                'reason': 'circuit_breaker',
                'description': f'Circuit breaker triggered at {unrealized_pnl_pct:.1f}% loss'
            }
        
        # No forced exit condition met
        return None
    
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
            position_data: Position data
            current_data: Current market data
            market_context: Market context
            exit_signals: Exit signals from monitoring
            
        Returns:
            Exit decision
        """
        logger.debug(f"Getting exit decision for {position_data['symbol']}")
        
        # Create messages for LLM
        messages = PromptTemplates.create_exit_decision_prompt(
            position_data,
            current_data,
            market_context,
            exit_signals
        )
        
        # Get decision from LLM
        response = await openrouter_client.chat_completion(messages)
        
        # Parse decision
        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        decision = parse_exit_decision(content)
        
        # Log decision
        if decision['decision'] == 'exit':
            logger.info(f"Decision: EXIT {position_data['symbol']} with {decision['exit_size']*100:.0f}% of position")
            logger.info(f"Reason: {decision['reasoning']}")
        else:
            logger.debug(f"Decision: HOLD {position_data['symbol']}")
            logger.debug(f"Reason: {decision['reasoning']}")
        
        return decision
    
    async def calculate_optimal_exits(self):
        """Calculate optimal exit parameters for all positions."""
        logger.info("Calculating optimal exit parameters")
        
        # Get active positions
        positions = redis_client.get_all_active_positions()
        
        if not positions:
            return
        
        # Process each position
        for symbol, position in positions.items():
            try:
                # Get OHLCV data
                intraday_data = await data_pipeline.get_stock_data(symbol, 'intraday')
                
                if intraday_data is None or (isinstance(intraday_data, pd.DataFrame) and intraday_data.empty):
                    continue
                
                # Calculate ATR (Average True Range)
                df = intraday_data.copy()
                high = df['high']
                low = df['low']
                close = df['close'].shift(1)
                
                tr1 = high - low
                tr2 = abs(high - close)
                tr3 = abs(low - close)
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
                
                # Current price
                current_price = df['close'].iloc[-1]
                
                # Entry price
                entry_price = position.get('entry_price', current_price)
                
                # Calculate optimal stop loss (based on ATR)
                atr_multiplier = 2.0  # Adjust based on risk tolerance
                optimal_stop = entry_price - (atr * atr_multiplier)
                
                # Calculate optimal take profit (based on risk-reward ratio)
                risk = entry_price - optimal_stop
                reward_ratio = 2.0  # Adjust based on strategy
                optimal_target = entry_price + (risk * reward_ratio)
                
                # Update position with optimal parameters
                position['optimal_stop'] = max(optimal_stop, position.get('stop_loss', 0))
                position['optimal_target'] = optimal_target
                position['atr'] = atr
                
                # Update position in Redis
                redis_client.set_active_position(symbol, position)
                
                logger.debug(f"Calculated optimal exits for {symbol}: stop=${optimal_stop:.2f}, target=${optimal_target:.2f}")
            except Exception as e:
                logger.error(f"Error calculating optimal exits for {symbol}: {e}")
    
    async def adjust_position_sizes(self):
        """Adjust position sizes based on performance and risk."""
        logger.info("Adjusting position sizes")
        
        # Get active positions
        positions = redis_client.get_all_active_positions()
        
        if not positions:
            return
        
        # Get total portfolio value (simplified)
        total_value = sum(
            position.get('quantity', 0) * position.get('current_price', 0)
            for position in positions.values()
        )
        
        # Calculate daily P&L
        daily_pnl = sum(position.get('unrealized_pnl', 0) for position in positions.values())
        daily_pnl_pct = (daily_pnl / total_value) * 100 if total_value else 0
        
        # Determine overall risk level based on performance
        if daily_pnl_pct <= -5:
            # Significant loss, reduce risk
            risk_level = 'low'
        elif daily_pnl_pct >= 5:
            # Significant gain, can take more risk
            risk_level = 'high'
        else:
            # Normal range, maintain risk
            risk_level = 'medium'
        
        logger.info(f"Current risk level: {risk_level} (daily P&L: {daily_pnl_pct:.2f}%)")
        
        # Store risk level
        redis_client.set("portfolio:risk_level", risk_level)
    
    async def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Dictionary with portfolio state
        """
        # Get active positions
        positions = redis_client.get_all_active_positions()
        
        # Count positions
        position_count = len(positions)
        
        # Calculate P&L
        daily_pnl = sum(p.get('unrealized_pnl', 0) for p in positions.values())
        
        # Starting capital (from settings)
        starting_capital = 5000.0
        used_capital = sum(p.get('entry_price', 0) * p.get('quantity', 0) for p in positions.values())
        
        # Available capital
        available_capital = starting_capital - used_capital
        
        # Daily P&L percentage
        daily_pnl_pct = (daily_pnl / starting_capital) * 100 if starting_capital else 0
        
        # Risk remaining
        max_daily_risk = settings.trading.max_daily_risk
        risk_remaining = max_daily_risk - abs(min(0, daily_pnl))
        
        # Create portfolio state
        state = {
            'position_count': position_count,
            'max_positions': settings.trading.max_positions,
            'available_capital': available_capital,
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'risk_remaining': risk_remaining,
            'risk_level': redis_client.get("portfolio:risk_level") or 'medium',
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache portfolio state
        redis_client.set("portfolio:state", state)
        
        return state
    
    async def update_position_stats(self):
        """Update position statistics for all active positions."""
        logger.debug("Updating position statistics")
        
        # Get active positions
        positions = redis_client.get_all_active_positions()
        
        if not positions:
            return
        
        # Process each position
        for symbol, position in positions.items():
            try:
                # Get current price data
                price_data = await data_pipeline.get_stock_data(symbol, 'snapshot')
                
                if not price_data or 'price' not in price_data:
                    continue
                
                # Current price
                current_price = price_data.get('price', {}).get('last', 0)
                
                # Update P&L
                redis_client.update_position_pnl(symbol, current_price)
                
                # Get updated position
                position = redis_client.get_active_position(symbol)
                
                # Entry time
                entry_time_str = position.get('entry_time', '')
                
                try:
                    entry_time = pd.Timestamp(entry_time_str)
                    current_time = pd.Timestamp.now()
                    
                    # Update time in trade
                    time_in_trade = (current_time - entry_time).total_seconds() / 3600  # hours
                    position['time_in_trade'] = time_in_trade
                    
                    # Update last update time
                    position['last_update'] = current_time.isoformat()
                    
                    # Save updated position
                    redis_client.set_active_position(symbol, position)
                except Exception as e:
                    logger.error(f"Error updating time stats for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Error updating position stats for {symbol}: {e}")

# Create global instance
position_monitor = PositionMonitor()