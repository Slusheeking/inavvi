#!/usr/bin/env python
"""
Standalone script for running the trading system.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.data_sources.polygon import PolygonAPI
from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.yahoo_finance import yahoo_finance_client
from src.models.pattern_recognition import pattern_recognition_model
from src.models.ranking_model import ranking_model
from src.models.sentiment import sentiment_model
from src.models.exit_optimization import exit_optimization_model
from src.llm.router import openrouter_client
from src.llm.prompts import PromptTemplates
from src.llm.parsing import parse_trade_decision, parse_exit_decision
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("trade")

async def initialize_system():
    """Initialize the trading system."""
    logger.info("Initializing trading system...")
    
    # Create directories if they don't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Check if required models exist
    required_models = [
        settings.model.pattern_model_path,
        settings.model.ranking_model_path,
        settings.model.sentiment_model_path,
        settings.model.exit_model_path
    ]
    
    missing_models = [model for model in required_models if not os.path.exists(model)]
    if missing_models:
        logger.warning(f"Missing model files: {missing_models}")
        logger.info("Some models will operate in fallback mode")
    
    # Initialize Redis
    system_state = {
        "state": "initializing",
        "timestamp": datetime.now().isoformat()
    }
    redis_client.set_system_state(system_state)
    
    logger.info("System initialized successfully")
    return True

async def run_market_scan():
    """Run a market scan to identify trading opportunities."""
    logger.info("Running market scan...")
    
    # Create Polygon client
    polygon_client = PolygonAPI()
    
    # Get market status
    market_status = await polygon_client.get_market_status()
    redis_client.set("market:status", market_status, expiry=300)  # 5 min expiry
    
    # Get pre-market movers if before market open
    now = datetime.now()
    market_open_time = datetime.strptime(f"{now.strftime('%Y-%m-%d')} {settings.timing.market_open}", 
                                         "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
    
    if now < market_open_time:
        logger.info("Getting pre-market movers...")
        pre_market_movers = await polygon_client.get_premarket_movers(limit=20)
        if pre_market_movers:
            redis_client.set("market:pre_market_movers", pre_market_movers)
            
            # Use as initial watchlist
            watchlist = [item['symbol'] for item in pre_market_movers]
            redis_client.set_watchlist(watchlist[:settings.trading.watchlist_size])
            
            logger.info(f"Initial watchlist: {watchlist[:5]}...")
    
    # Get gainers and losers
    gainers_losers = await polygon_client.get_gainers_losers(limit=20)
    if gainers_losers:
        redis_client.set("market:gainers_losers", gainers_losers, expiry=300)
        
        # Update watchlist if not already set
        if not redis_client.get_watchlist():
            watchlist = [item['symbol'] for item in gainers_losers['gainers']]
            watchlist.extend([item['symbol'] for item in gainers_losers['losers']])
            redis_client.set_watchlist(watchlist[:settings.trading.watchlist_size])
            
            logger.info(f"Initial watchlist: {watchlist[:5]}...")
    
    # Get sector performance
    sector_performance = await alpha_vantage_client.get_sector_performance()
    if sector_performance:
        redis_client.set("market:sectors", sector_performance, expiry=3600)
    
    # Get economic indicators
    treasury_yield = await alpha_vantage_client.get_treasury_yield(interval='daily', maturity='10year')
    if treasury_yield is not None:
        redis_client.set("economic:treasury_yield:10year", treasury_yield)
    
    logger.info("Market scan completed")
    return True

async def analyze_watchlist():
    """Analyze the current watchlist to find trading opportunities."""
    logger.info("Analyzing watchlist...")
    
    # Get current watchlist
    watchlist = redis_client.get_watchlist()
    
    if not watchlist:
        logger.warning("Watchlist is empty")
        return False
    
    logger.info(f"Analyzing {len(watchlist)} symbols...")
    
    # Create Polygon client
    polygon_client = PolygonAPI()
    
    # Process each symbol
    candidates = []
    
    for symbol in watchlist:
        try:
            # Get stock data
            snapshot = await polygon_client.get_stock_snapshot(symbol)
            intraday_data = await polygon_client.get_intraday_bars(symbol, minutes=1, days=1)
            
            if not snapshot or intraday_data is None or intraday_data.empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue
            
            # Run pattern recognition
            pattern_name, pattern_confidence = pattern_recognition_model.predict_pattern(intraday_data)
            
            # Get ranking score
            ranking_score = ranking_model.predict(intraday_data)
            
            # Get sentiment
            news_items = await alpha_vantage_client.get_symbol_news(symbol, limit=5)
            sentiment_data = {}
            
            if news_items and sentiment_model:
                # Analyze sentiment
                analyzed_news = sentiment_model.analyze_news_items(news_items)
                overall_sentiment = sentiment_model.get_overall_sentiment(analyzed_news)
                sentiment_data = {
                    'news': analyzed_news,
                    'overall_score': overall_sentiment.get('overall_score', 0),
                    'positive': overall_sentiment.get('positive', 0),
                    'neutral': overall_sentiment.get('neutral', 0),
                    'negative': overall_sentiment.get('negative', 0)
                }
            
            # Create candidate data
            candidate = {
                'symbol': symbol,
                'price': snapshot.get('price', {}),
                'pattern': {
                    'name': pattern_name,
                    'confidence': pattern_confidence
                },
                'ranking_score': ranking_score,
                'sentiment': sentiment_data,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to candidates
            candidates.append(candidate)
            
            # Add to Redis
            redis_client.add_candidate_score(
                symbol, 
                ranking_score, 
                {
                    'price': snapshot.get('price', {}).get('last', 0),
                    'pattern': pattern_name,
                    'pattern_confidence': pattern_confidence,
                    'sentiment': sentiment_data.get('overall_score', 0),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Short pause to avoid rate limits
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
    
    # Get top candidates
    top_candidates = redis_client.get_ranked_candidates()
    
    if top_candidates:
        logger.info(f"Top candidates: {[c['symbol'] for c in top_candidates[:3]]}")
    else:
        logger.warning("No viable candidates found")
    
    return bool(top_candidates)

async def get_market_context():
    """Get current market context."""
    logger.info("Getting market context...")
    
    # Get market status
    market_status = redis_client.get("market:status") or {}
    
    # Get sector performance
    sector_performance = redis_client.get("market:sectors") or {}
    
    # Get VIX data (from cached symbols)
    vix_data = redis_client.get_stock_data("VIX") or {}
    vix_value = vix_data.get('price', {}).get('last', 0)
    
    # Calculate time until market close
    now = datetime.now()
    market_close_time = datetime.strptime(f"{now.strftime('%Y-%m-%d')} {settings.timing.market_close}", 
                                         "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
    time_until_close = (market_close_time - now).total_seconds() / 3600  # hours
    time_until_close = max(0, time_until_close)
    
    # Create market context
    context = {
        'state': market_status.get('market', 'unknown'),
        'sector_performance': sector_performance.get('Rank 1 Day', {}).get('Information Technology', 0),
        'vix': vix_value,
        'breadth': 0.5,  # Placeholder
        'time_until_close': time_until_close
    }
    
    return context

async def get_portfolio_state():
    """Get current portfolio state."""
    logger.info("Getting portfolio state...")
    
    # Get active positions
    positions = redis_client.get_all_active_positions()
    
    # Count positions
    position_count = len(positions)
    
    # Calculate P&L
    daily_pnl = sum(p.get('unrealized_pnl', 0) for p in positions.values())
    
    # Starting capital (for demo)
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
        'risk_remaining': risk_remaining
    }
    
    return state

async def make_trade_decision(candidate: dict):
    """
    Make a trade decision for a candidate.
    
    Args:
        candidate: Candidate data
        
    Returns:
        Trade decision
    """
    logger.info(f"Making trade decision for {candidate['symbol']}...")
    
    # Get market context
    market_context = await get_market_context()
    
    # Get portfolio state
    portfolio_state = await get_portfolio_state()
    
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
    
    return decision

async def execute_trade(symbol: str, decision: dict):
    """
    Execute a trade based on decision.
    
    Args:
        symbol: Stock symbol
        decision: Trade decision
        
    Returns:
        True if trade executed, False otherwise
    """
    logger.info(f"Executing trade for {symbol}...")
    
    # Get current price
    price_data = redis_client.get_stock_data(symbol, 'price') or {}
    current_price = price_data.get('last', 0)
    
    if current_price <= 0:
        logger.error(f"Invalid price for {symbol}: {current_price}")
        return False
    
    # Calculate position size
    max_position_size = settings.trading.max_position_size
    position_size = max_position_size * decision['position_size']
    
    # Calculate quantity
    quantity = int(position_size / current_price)
    
    if quantity <= 0:
        logger.warning(f"Calculated quantity is {quantity}, skipping trade")
        return False
    
    # Calculate stop loss and take profit
    stop_loss = current_price * 0.95  # 5% stop loss
    take_profit = current_price * 1.1  # 10% take profit
    
    # In a real implementation, execute trade with Alpaca
    # Here we'll just simulate it
    
    # Create position data
    position_data = {
        'symbol': symbol,
        'entry_price': current_price,
        'entry_time': datetime.now().isoformat(),
        'quantity': quantity,
        'side': 'long',
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'trailing_stop': 2.0,  # 2% trailing stop
        'max_time': 6.0,  # 6 hours max hold time
        'unrealized_pnl': 0.0,
        'unrealized_pnl_pct': 0.0,
        'reason': decision['reasoning'],
        'key_factors': decision['key_factors']
    }
    
    # Save to Redis
    redis_client.set_active_position(symbol, position_data)
    
    logger.info(f"Trade executed for {symbol}: {quantity} shares at ${current_price:.2f}")
    logger.info(f"Stop loss: ${stop_loss:.2f}, Take profit: ${take_profit:.2f}")
    
    return True

async def monitor_positions():
    """Monitor active positions and make exit decisions."""
    logger.info("Monitoring positions...")
    
    # Get active positions
    positions = redis_client.get_all_active_positions()
    
    if not positions:
        logger.info("No active positions to monitor")
        return True
    
    logger.info(f"Monitoring {len(positions)} positions...")
    
    # Create Polygon client
    polygon_client = PolygonAPI()
    
    # Process each position
    for symbol, position_data in positions.items():
        try:
            # Get current data
            snapshot = await polygon_client.get_stock_snapshot(symbol)
            intraday_data = await polygon_client.get_intraday_bars(symbol, minutes=1, days=1)
            
            if not snapshot or intraday_data is None or intraday_data.empty:
                logger.warning(f"No data for position {symbol}, skipping")
                continue
            
            # Update position P&L
            current_price = snapshot.get('price', {}).get('last', 0)
            redis_client.update_position_pnl(symbol, current_price)
            
            # Get updated position data
            position_data = redis_client.get_active_position(symbol)
            
            # Evaluate exit conditions
            exit_recommendation = exit_optimization_model.evaluate_exit_conditions(
                intraday_data, position_data, confidence_threshold=0.6
            )
            
            # Check if any exit condition is triggered
            exit_triggered = (
                exit_recommendation['exit'] or
                exit_recommendation['stop_loss_triggered'] or
                exit_recommendation['take_profit_triggered'] or
                exit_recommendation['trailing_stop_triggered'] or
                exit_recommendation['time_stop_triggered']
            )
            
            if exit_triggered:
                # Get market context
                market_context = await get_market_context()
                
                # Create current data
                current_data = {
                    'symbol': symbol,
                    'price': snapshot.get('price', {}),
                    'indicators': {
                        'rsi_14': 50,  # Placeholder
                        'macd_histogram': 0,  # Placeholder
                        'bb_position': 0.5  # Placeholder
                    }
                }
                
                # Get exit decision from LLM
                exit_decision = await get_exit_decision(
                    position_data, current_data, market_context, exit_recommendation
                )
                
                if exit_decision['decision'] == 'exit':
                    await execute_exit(symbol, position_data, exit_decision)
            
            # Short pause to avoid rate limits
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error monitoring position {symbol}: {e}")
    
    return True

async def get_exit_decision(
    position_data: dict,
    current_data: dict,
    market_context: dict,
    exit_signals: dict
):
    """
    Get exit decision from LLM.
    
    Args:
        position_data: Position data
        current_data: Current market data
        market_context: Market context
        exit_signals: Exit signals
        
    Returns:
        Exit decision
    """
    logger.info(f"Getting exit decision for {position_data['symbol']}...")
    
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
        logger.info(f"Decision: HOLD {position_data['symbol']}")
        logger.info(f"Reason: {decision['reasoning']}")
    
    return decision

async def execute_exit(symbol: str, position_data: dict, decision: dict):
    """
    Execute an exit based on decision.
    
    Args:
        symbol: Stock symbol
        position_data: Position data
        decision: Exit decision
        
    Returns:
        True if exit executed, False otherwise
    """
    logger.info(f"Executing exit for {symbol}...")
    
    # Get current price
    price_data = redis_client.get_stock_data(symbol, 'price') or {}
    current_price = price_data.get('last', 0)
    
    if current_price <= 0:
        logger.error(f"Invalid price for {symbol}: {current_price}")
        return False
    
    # Get position quantity
    quantity = position_data.get('quantity', 0)
    
    # Calculate exit quantity
    exit_quantity = int(quantity * decision['exit_size'])
    
    if exit_quantity <= 0:
        logger.warning(f"Calculated exit quantity is {exit_quantity}, skipping exit")
        return False
    
    # In a real implementation, execute exit with Alpaca
    # Here we'll just simulate it
    
    if exit_quantity >= quantity:
        # Full exit
        logger.info(f"Full exit executed for {symbol}: {quantity} shares at ${current_price:.2f}")
        redis_client.delete_active_position(symbol)
    else:
        # Partial exit
        remaining_quantity = quantity - exit_quantity
        
        # Update position data
        position_data['quantity'] = remaining_quantity
        
        # Save to Redis
        redis_client.set_active_position(symbol, position_data)
        
        logger.info(f"Partial exit executed for {symbol}: {exit_quantity} shares at ${current_price:.2f}")
        logger.info(f"Remaining position: {remaining_quantity} shares")
    
    return True

async def trading_loop():
    """Main trading loop."""
    logger.info("Starting trading loop...")
    
    # Initialize system
    initialized = await initialize_system()
    if not initialized:
        logger.error("System initialization failed")
        return False
    
    # Set system state to running
    redis_client.update_system_state(state="running")
    
    try:
        # Initial market scan
        await run_market_scan()
        
        # Main loop
        while True:
            # Check if market is open
            now = datetime.now()
            market_open_time = datetime.strptime(f"{now.strftime('%Y-%m-%d')} {settings.timing.market_open}", 
                                               "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
            market_close_time = datetime.strptime(f"{now.strftime('%Y-%m-%d')} {settings.timing.market_close}", 
                                                "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
            
            if now < market_open_time:
                # Pre-market
                logger.info("Market not open yet, waiting...")
                wait_seconds = (market_open_time - now).total_seconds()
                await asyncio.sleep(min(wait_seconds, 300))  # Wait at most 5 minutes
                continue
            
            if now > market_close_time:
                # Post-market
                logger.info("Market closed, closing all positions...")
                
                # Close all positions
                positions = redis_client.get_all_active_positions()
                for symbol, position_data in positions.items():
                    exit_decision = {
                        'decision': 'exit',
                        'exit_size': 1.0,
                        'confidence': 1.0,
                        'reasoning': 'Market close',
                        'key_factors': ['market_close']
                    }
                    await execute_exit(symbol, position_data, exit_decision)
                
                # Wait until next day
                tomorrow = now + timedelta(days=1)
                tomorrow_open = datetime.strptime(f"{tomorrow.strftime('%Y-%m-%d')} {settings.timing.market_open}", 
                                               "%Y-%m-%d %H:%M").replace(tzinfo=now.tzinfo)
                wait_seconds = (tomorrow_open - now).total_seconds()
                logger.info(f"Waiting until next market open: {tomorrow_open}")
                await asyncio.sleep(min(wait_seconds, 3600))  # Wait at most 1 hour
                continue
            
            # Market is open
            
            # Monitor existing positions
            await monitor_positions()
            
            # Check if we can open new positions
            positions = redis_client.get_all_active_positions()
            if len(positions) >= settings.trading.max_positions:
                logger.info(f"Maximum positions reached ({len(positions)}), skipping new trades")
                await asyncio.sleep(60)  # Check again in 1 minute
                continue
            
            # Analyze watchlist for new opportunities
            has_candidates = await analyze_watchlist()
            
            if has_candidates:
                # Get top candidate
                candidates = redis_client.get_ranked_candidates()
                top_candidate = candidates[0]
                
                # Make trade decision
                decision = await make_trade_decision(top_candidate)
                
                if decision['decision'] == 'trade' and decision['position_size'] > 0:
                    # Execute trade
                    await execute_trade(top_candidate['symbol'], decision)
            
            # Wait before next iteration
            await asyncio.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("Trading loop interrupted by user")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}")
        redis_client.update_system_state(state="error", error=str(e))
    finally:
        # Set system state to stopped
        redis_client.update_system_state(state="stopped")
        
        # Close client connections
        await alpha_vantage_client.close()
        await openrouter_client.close()
        
        logger.info("Trading loop stopped")
    
    return True

async def backtest_mode():
    """Run system in backtest mode."""
    logger.info("Starting backtest mode...")
    
    # TODO: Implement backtesting logic
    
    logger.info("Backtesting not implemented yet")
    return False

async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Day Trading System")
    parser.add_argument("--backtest", action="store_true", help="Run in backtest mode")
    parser.add_argument("--initialize", action="store_true", help="Initialize system and exit")
    args = parser.parse_args()
    
    if args.initialize:
        # Just initialize the system
        await initialize_system()
        return
    
    if args.backtest:
        # Run in backtest mode
        await backtest_mode()
        return
    
    # Run trading loop
    await trading_loop()

if __name__ == "__main__":
    # Run main function
    asyncio.run(main())