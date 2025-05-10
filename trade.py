#!/usr/bin/env python
"""
Standalone script for running the trading system.
"""

import argparse
import asyncio
import os
import time
from datetime import datetime

import alpaca_trade_api as tradeapi
import pandas as pd

from src.config.settings import settings
from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.polygon import PolygonAPI
from src.llm.parsing import parse_exit_decision, parse_trade_decision
from src.llm.prompts import PromptTemplates
from src.llm.router import openrouter_client
from src.models.exit_optimization import exit_optimization_model
from src.models.pattern_recognition import pattern_recognition_model
from src.models.ranking_model import ranking_model
from src.models.sentiment import sentiment_model
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("trade")

# Initialize Alpaca API
alpaca_api = tradeapi.REST(
    key_id=settings.api.alpaca_api_key,
    secret_key=settings.api.alpaca_api_secret,
    base_url=settings.api.alpaca_base_url,
)


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
        settings.model.exit_model_path,
    ]

    missing_models = [model for model in required_models if not os.path.exists(model)]
    if missing_models:
        logger.warning(f"Missing model files: {missing_models}")
        logger.info("Some models will operate in fallback mode")

    # Initialize Redis
    system_state = {"state": "initializing", "timestamp": datetime.now().isoformat()}
    redis_client.set_system_state(system_state)

    # Verify Alpaca API connection
    try:
        account = alpaca_api.get_account()
        logger.info(f"Connected to Alpaca API. Account status: {account.status}")
        logger.info(f"Buying power: ${float(account.buying_power):.2f}")
        logger.info(f"Cash: ${float(account.cash):.2f}")

        # Store account info in Redis
        redis_client.set(
            "account:info",
            {
                "id": account.id,
                "status": account.status,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "daytrade_count": int(account.daytrade_count),
                "trading_blocked": account.trading_blocked,
                "updated_at": datetime.now().isoformat(),
            },
        )

        # Check if market is open
        clock = alpaca_api.get_clock()
        if clock.is_open:
            logger.info("Market is open")
            redis_client.set("market:state", "open")
        else:
            next_open = clock.next_open.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Market is closed. Next open: {next_open}")
            redis_client.set("market:state", "closed")

        # Get current positions
        positions = alpaca_api.list_positions()
        logger.info(f"Current positions: {len(positions)}")

        # Store positions in Redis
        for position in positions:
            redis_client.set_active_position(
                position.symbol,
                {
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price),
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pnl": float(position.unrealized_pl),
                    "unrealized_pnl_pct": float(position.unrealized_plpc) * 100,
                    "side": position.side,
                    "entry_time": position.lastday_price_timestamp
                    if hasattr(position, "lastday_price_timestamp")
                    else datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                },
            )

    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {e}")
        return False

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

    # Check if market is open via Alpaca (more reliable)
    try:
        clock = alpaca_api.get_clock()
        market_open = clock.is_open

        # Update Redis with market state
        redis_client.set("market:state", "open" if market_open else "closed")

        # If market closed, log and return
        if not market_open:
            next_open = clock.next_open.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Market is closed. Next open: {next_open}")
            return False
    except Exception as e:
        logger.error(f"Error checking market status via Alpaca: {e}")
        # Continue with polygon data as fallback

    # Get pre-market movers if before market open
    now = datetime.now()
    market_open_time = datetime.strptime(
        f"{now.strftime('%Y-%m-%d')} {settings.timing.market_open}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=now.tzinfo)

    if now < market_open_time:
        logger.info("Getting pre-market movers...")
        pre_market_movers = await polygon_client.get_premarket_movers(limit=20)
        if pre_market_movers:
            redis_client.set("market:pre_market_movers", pre_market_movers)

            # Use as initial watchlist
            watchlist = [item["symbol"] for item in pre_market_movers]
            redis_client.set_watchlist(watchlist[: settings.trading.watchlist_size])

            logger.info(f"Initial watchlist: {watchlist[:5]}...")

    # Get gainers and losers
    gainers_losers = await polygon_client.get_gainers_losers(limit=20)
    if gainers_losers:
        redis_client.set("market:gainers_losers", gainers_losers, expiry=300)

        # Update watchlist if not already set
        if not redis_client.get_watchlist():
            watchlist = [item["symbol"] for item in gainers_losers["gainers"]]
            watchlist.extend([item["symbol"] for item in gainers_losers["losers"]])
            redis_client.set_watchlist(watchlist[: settings.trading.watchlist_size])

            logger.info(f"Initial watchlist: {watchlist[:5]}...")

    # Get sector performance
    sector_performance = await alpha_vantage_client.get_sector_performance()
    if sector_performance:
        redis_client.set("market:sectors", sector_performance, expiry=3600)

    # Get economic indicators
    treasury_yield = await alpha_vantage_client.get_treasury_yield(
        interval="daily", maturity="10year"
    )
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

            if not snapshot or not isinstance(intraday_data, pd.DataFrame) or intraday_data.empty:
                logger.warning(f"No data for {symbol}, skipping")
                continue

            # Run pattern recognition
            pattern_name, pattern_confidence = pattern_recognition_model.predict_pattern(
                intraday_data
            )

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
                    "news": analyzed_news,
                    "overall_score": overall_sentiment.get("overall_score", 0),
                    "positive": overall_sentiment.get("positive", 0),
                    "neutral": overall_sentiment.get("neutral", 0),
                    "negative": overall_sentiment.get("negative", 0),
                }

            # Create candidate data
            candidate = {
                "symbol": symbol,
                "price": snapshot.get("price", {}),
                "pattern": {"name": pattern_name, "confidence": pattern_confidence},
                "ranking_score": ranking_score,
                "sentiment": sentiment_data,
                "timestamp": datetime.now().isoformat(),
            }

            # Add to candidates
            candidates.append(candidate)

            # Add to Redis
            redis_client.add_candidate_score(
                symbol,
                ranking_score,
                {
                    "price": snapshot.get("price", {}).get("last", 0),
                    "pattern": pattern_name,
                    "pattern_confidence": pattern_confidence,
                    "sentiment": sentiment_data.get("overall_score", 0),
                    "timestamp": datetime.now().isoformat(),
                },
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


async def make_trade_decisions():
    """Make trading decisions for top candidates."""
    logger.info("Making trade decisions...")

    # Get top candidates
    candidates = redis_client.get_ranked_candidates()

    if not candidates:
        logger.warning("No candidates available for trading")
        return False

    # Limit to top N candidates
    top_candidates = candidates[: settings.trading.candidate_size]

    logger.info(f"Evaluating top {len(top_candidates)} candidates")

    # Get market context
    market_context = get_market_context()

    # Get portfolio state
    portfolio_state = get_portfolio_state()

    # Check if we can take new positions
    active_positions = redis_client.get_all_active_positions()
    if len(active_positions) >= settings.trading.max_positions:
        logger.info(
            f"Maximum positions reached ({len(active_positions)}/{settings.trading.max_positions})"
        )
        return False

    # Process each candidate
    for candidate in top_candidates:
        symbol = candidate["symbol"]

        # Skip if already have a position
        if symbol in active_positions:
            logger.info(f"Already have a position in {symbol}, skipping")
            continue

        try:
            # Get stock data for LLM
            stock_data = get_stock_data_for_llm(candidate)

            # Get LLM trade decision
            messages = PromptTemplates.create_trade_decision_prompt(
                stock_data, market_context, portfolio_state
            )

            response = await openrouter_client.chat_completion(messages)
            response_content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            # Parse decision
            decision = parse_trade_decision(response_content)

            # If decision is to trade, execute
            if decision["decision"] == "trade" and decision["position_size"] > 0:
                # Calculate position size
                position_size = calculate_position_size(
                    decision["position_size"], portfolio_state, stock_data["price"]["last"]
                )

                # Execute trade
                success = await execute_trade(
                    symbol, "buy", position_size, stock_data["price"]["last"], decision
                )

                if success:
                    logger.info(f"Successfully executed trade for {symbol}")
                    # One trade per run is enough
                    return True
            else:
                logger.info(f"Decision for {symbol}: no trade. Reason: {decision['reasoning']}")

        except Exception as e:
            logger.error(f"Error making trade decision for {symbol}: {e}")

    logger.info("No trades executed this run")
    return False


async def monitor_positions():
    """Monitor active positions for exit signals."""
    logger.info("Monitoring positions...")

    # Get active positions
    positions = redis_client.get_all_active_positions()

    if not positions:
        logger.info("No active positions to monitor")
        return False

    logger.info(f"Monitoring {len(positions)} active positions")

    # Create Polygon client
    polygon_client = PolygonAPI()

    # Get market context
    market_context = get_market_context()

    # Process each position
    for symbol, position_data in positions.items():
        try:
            # Get current data
            snapshot = await polygon_client.get_stock_snapshot(symbol)
            intraday_data = await polygon_client.get_intraday_bars(symbol, minutes=1, days=1)

            if not snapshot or not isinstance(intraday_data, pd.DataFrame) or intraday_data.empty:
                logger.warning(f"No data for {symbol}, skipping position monitoring")
                continue

            # Update position P&L
            current_price = snapshot.get("price", {}).get("last", 0)
            redis_client.update_position_pnl(symbol, current_price)

            # Get updated position data
            position_data = redis_client.get_active_position(symbol)

            # Check exit signals from ML model
            exit_signals = exit_optimization_model.evaluate_exit_conditions(
                intraday_data, position_data
            )

            # Prepare data for LLM
            current_data = get_stock_data_for_llm(
                {
                    "symbol": symbol,
                    "price": snapshot.get("price", {}),
                    "indicators": calculate_indicators(intraday_data),
                }
            )

            # Get LLM exit decision
            messages = PromptTemplates.create_exit_decision_prompt(
                position_data, current_data, market_context, exit_signals
            )

            response = await openrouter_client.chat_completion(messages)
            response_content = (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )

            # Parse decision
            decision = parse_exit_decision(response_content)

            # Determine if we should exit
            should_exit = decision["decision"] == "exit" and decision["exit_size"] > 0
            should_exit = should_exit or exit_signals.get("stop_loss_triggered", False)
            should_exit = should_exit or exit_signals.get("take_profit_triggered", False)
            should_exit = should_exit or exit_signals.get("trailing_stop_triggered", False)
            should_exit = should_exit or exit_signals.get("time_stop_triggered", False)

            # If decision is to exit, execute
            if should_exit:
                # Calculate exit size
                exit_size = decision["exit_size"]

                # If stop loss or take profit triggered, exit full position
                if (
                    exit_signals.get("stop_loss_triggered", False)
                    or exit_signals.get("take_profit_triggered", False)
                    or exit_signals.get("trailing_stop_triggered", False)
                    or exit_signals.get("time_stop_triggered", False)
                ):
                    exit_size = 1.0

                # Calculate quantity to exit
                quantity = position_data["quantity"] * exit_size

                # Exit reason
                if exit_signals.get("stop_loss_triggered", False):
                    reason = "stop_loss_triggered"
                elif exit_signals.get("take_profit_triggered", False):
                    reason = "take_profit_triggered"
                elif exit_signals.get("trailing_stop_triggered", False):
                    reason = "trailing_stop_triggered"
                elif exit_signals.get("time_stop_triggered", False):
                    reason = "time_stop_triggered"
                else:
                    reason = decision["reasoning"]

                # Execute exit
                success = await execute_exit(symbol, "sell", quantity, current_price, reason)

                if success:
                    logger.info(f"Successfully exited position for {symbol}")

                    # If full exit, remove from active positions
                    if exit_size >= 0.99:  # Allow for small rounding errors
                        redis_client.delete_active_position(symbol)
            else:
                logger.info(f"Decision for {symbol}: hold position")

        except Exception as e:
            logger.error(f"Error monitoring position for {symbol}: {e}")

    return True


async def execute_trade(symbol, side, quantity, price, decision):
    """
    Execute a trade via Alpaca.

    Args:
        symbol: Stock symbol
        side: Trade side ('buy' or 'sell')
        quantity: Quantity to trade
        price: Current price
        decision: Trade decision dictionary

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Executing {side} order for {quantity} shares of {symbol} at ${price:.2f}")

    # Calculate stop loss and take profit
    entry_price = price
    stop_loss = entry_price * (1 - settings.trading.default_stop_loss_pct / 100)
    take_profit = entry_price * (1 + settings.trading.default_take_profit_pct / 100)

    try:
        # Submit order
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type="market",
            time_in_force="day",
            client_order_id=f"t_{int(time.time())}",
        )

        logger.info(f"Order submitted: ID {order.id}, Status: {order.status}")

        # Store order in Redis
        redis_client.set(
            f"orders:{order.id}",
            {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": symbol,
                "side": side,
                "quantity": float(order.qty),
                "type": order.type,
                "status": order.status,
                "submitted_at": datetime.now().isoformat(),
                "decision": decision,
            },
        )

        # Poll for order completion
        filled = await wait_for_order_fill(order.id)

        if filled:
            # Get filled price
            order = alpaca_api.get_order(order.id)
            filled_price = float(order.filled_avg_price)

            logger.info(f"Order filled at ${filled_price:.2f}")

            # Create position entry
            position_data = {
                "symbol": symbol,
                "entry_price": filled_price,
                "entry_time": datetime.now().isoformat(),
                "quantity": float(order.qty),
                "side": side,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop": settings.trading.default_trailing_stop_pct,
                "max_time": 4.0,  # 4 hours max hold time
                "order_id": order.id,
                "decision": decision,
                "updated_at": datetime.now().isoformat(),
            }

            # Store in Redis
            redis_client.set_active_position(symbol, position_data)

            return True
        else:
            logger.warning(f"Order for {symbol} did not fill within timeout")
            return False

    except Exception as e:
        logger.error(f"Error executing trade for {symbol}: {e}")
        return False


async def execute_exit(symbol, side, quantity, price, reason):
    """
    Execute an exit trade via Alpaca.

    Args:
        symbol: Stock symbol
        side: Trade side ('buy' or 'sell')
        quantity: Quantity to trade
        price: Current price
        reason: Exit reason

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Executing {side} order to exit {quantity} shares of {symbol} at ${price:.2f}")
    logger.info(f"Exit reason: {reason}")

    try:
        # Submit order
        order = alpaca_api.submit_order(
            symbol=symbol,
            qty=quantity,
            side=side,
            type="market",
            time_in_force="day",
            client_order_id=f"e_{int(time.time())}",
        )

        logger.info(f"Exit order submitted: ID {order.id}, Status: {order.status}")

        # Store order in Redis
        redis_client.set(
            f"orders:{order.id}",
            {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": symbol,
                "side": side,
                "quantity": float(order.qty),
                "type": order.type,
                "status": order.status,
                "submitted_at": datetime.now().isoformat(),
                "reason": reason,
            },
        )

        # Poll for order completion
        filled = await wait_for_order_fill(order.id)

        if filled:
            # Get filled price
            order = alpaca_api.get_order(order.id)
            filled_price = float(order.filled_avg_price)

            logger.info(f"Exit order filled at ${filled_price:.2f}")

            # Get original position
            position = redis_client.get_active_position(symbol)

            # Calculate P&L
            if position:
                entry_price = position.get("entry_price", 0)
                pnl = (filled_price - entry_price) * float(order.qty)
                pnl_pct = (filled_price / entry_price - 1) * 100 if entry_price else 0

                logger.info(f"P&L for {symbol}: ${pnl:.2f} ({pnl_pct:.2f}%)")

                # Store trade in history
                trade_history = {
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "exit_price": filled_price,
                    "quantity": float(order.qty),
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "entry_time": position.get("entry_time", ""),
                    "exit_time": datetime.now().isoformat(),
                    "reason": reason,
                }

                # Store in Redis
                redis_client.set(f"history:trades:{int(time.time())}", trade_history)

                # Update position if partial exit
                remaining_quantity = position.get("quantity", 0) - float(order.qty)

                if remaining_quantity > 0.01:  # Small threshold for rounding errors
                    position["quantity"] = remaining_quantity
                    position["updated_at"] = datetime.now().isoformat()
                    redis_client.set_active_position(symbol, position)
                    logger.info(
                        f"Updated position for {symbol}: {remaining_quantity} shares remaining"
                    )
                else:
                    # Remove position if fully exited
                    redis_client.delete_active_position(symbol)
                    logger.info(f"Removed position for {symbol} (fully exited)")

            return True
        else:
            logger.warning(f"Exit order for {symbol} did not fill within timeout")
            return False

    except Exception as e:
        logger.error(f"Error executing exit for {symbol}: {e}")
        return False


async def wait_for_order_fill(order_id, timeout=60):
    """
    Wait for an order to be filled.

    Args:
        order_id: Alpaca order ID
        timeout: Timeout in seconds

    Returns:
        True if filled, False if still open after timeout
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            order = alpaca_api.get_order(order_id)

            if order.status == "filled":
                return True

            if order.status in ["canceled", "expired", "rejected", "suspended"]:
                logger.warning(f"Order {order_id} status: {order.status}")
                return False

            # Wait before checking again
            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Error checking order status: {e}")
            return False

    return False


def calculate_position_size(size_factor, portfolio_state, current_price):
    """
    Calculate the position size in shares.

    Args:
        size_factor: Size factor (0.0 to 1.0)
        portfolio_state: Portfolio state dictionary
        current_price: Current stock price

    Returns:
        Position size in shares
    """
    # Get maximum position size
    max_position_size = min(
        settings.trading.max_position_size,
        portfolio_state["available_capital"] * 0.9,  # Leave some buffer
    )

    # Calculate dollar amount
    dollar_amount = max_position_size * size_factor

    # Calculate shares (round down)
    shares = int(dollar_amount / current_price)

    # Ensure minimum position size
    if shares < 1:
        shares = 1

    return shares


def get_market_context():
    """
    Get current market context.

    Returns:
        Market context dictionary
    """
    # Get market status
    market_status = redis_client.get("market:status") or {}
    market_state = market_status.get("market", "unknown")

    # Get sector performance
    sectors = redis_client.get("market:sectors") or {}
    sector_perf = sectors.get("Rank A: Real-Time Performance", {}).get("Information Technology", 0)

    # Get time until market close
    now = datetime.now()
    market_close_time = datetime.strptime(
        f"{now.strftime('%Y-%m-%d')} {settings.timing.market_close}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=now.tzinfo)
    time_until_close = (market_close_time - now).total_seconds() / 3600  # Hours
    time_until_close = max(0, time_until_close)

    # Assemble context
    context = {
        "state": market_state,
        "sector_performance": sector_perf,
        "vix": 15.0,  # Placeholder, would get from data source
        "breadth": 0.65,  # Placeholder, would calculate from market data
        "time_until_close": time_until_close,
    }

    return context


def get_portfolio_state():
    """
    Get current portfolio state.

    Returns:
        Portfolio state dictionary
    """
    # Get account info
    account_info = redis_client.get("account:info") or {}

    # Get positions
    positions = redis_client.get_all_active_positions()

    # Calculate daily P&L
    daily_pnl = sum([pos.get("unrealized_pnl", 0) for pos in positions.values()])
    daily_pnl_pct = daily_pnl / float(account_info.get("equity", 1)) * 100

    # Calculate available capital
    available_capital = float(account_info.get("buying_power", 0)) / 4  # For day trading

    # Calculate risk remaining
    max_daily_risk = settings.trading.max_daily_risk
    risk_remaining = max_daily_risk + daily_pnl if daily_pnl < 0 else max_daily_risk

    # Assemble state
    state = {
        "position_count": len(positions),
        "max_positions": settings.trading.max_positions,
        "available_capital": available_capital,
        "daily_pnl": daily_pnl,
        "daily_pnl_pct": daily_pnl_pct,
        "risk_remaining": risk_remaining,
    }

    return state


def get_stock_data_for_llm(candidate):
    """
    Prepare stock data for LLM consumption.

    Args:
        candidate: Candidate dictionary

    Returns:
        Stock data dictionary formatted for LLM
    """
    return {
        "symbol": candidate["symbol"],
        "price": candidate.get("price", {}),
        "indicators": candidate.get("indicators", {}),
        "pattern": candidate.get("pattern", {}),
        "sentiment": candidate.get("sentiment", {}),
        "news": candidate.get("sentiment", {}).get("news", []),
    }


def calculate_indicators(ohlcv_data):
    """
    Calculate technical indicators for a stock.

    Args:
        ohlcv_data: OHLCV DataFrame

    Returns:
        Dictionary of indicators
    """
    if not isinstance(ohlcv_data, pd.DataFrame) or ohlcv_data.empty:
        return {}

    try:
        # Calculate RSI
        delta = ohlcv_data["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        rsi_14 = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        bb_middle = ohlcv_data["close"].rolling(20).mean()
        bb_std = ohlcv_data["close"].rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std

        bb_position = (ohlcv_data["close"].iloc[-1] - bb_lower.iloc[-1]) / (
            bb_upper.iloc[-1] - bb_lower.iloc[-1]
        )

        # Calculate MACD
        ema_12 = ohlcv_data["close"].ewm(span=12, adjust=False).mean()
        ema_26 = ohlcv_data["close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_histogram = macd - macd_signal

        # Assemble indicators
        indicators = {
            "rsi_14": float(rsi_14.iloc[-1]),
            "bb_upper": float(bb_upper.iloc[-1]),
            "bb_lower": float(bb_lower.iloc[-1]),
            "bb_middle": float(bb_middle.iloc[-1]),
            "bb_position": float(bb_position),
            "macd": float(macd.iloc[-1]),
            "macd_signal": float(macd_signal.iloc[-1]),
            "macd_histogram": float(macd_histogram.iloc[-1]),
        }

        return indicators
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {}


async def close_all_positions():
    """Close all open positions at the end of the day."""
    logger.info("Closing all positions for end of day...")

    # Get active positions
    positions = redis_client.get_all_active_positions()

    if not positions:
        logger.info("No active positions to close")
        return True

    logger.info(f"Closing {len(positions)} active positions")

    # Process each position
    for symbol, position_data in positions.items():
        try:
            # Get current price
            price_data = redis_client.get_stock_data(symbol, "price")

            if not price_data or "price" not in price_data:
                # Use last known price from position data
                current_price = position_data.get("current_price", 0)
            else:
                current_price = price_data.get("price", {}).get("last", 0)

            # Exit position
            quantity = position_data["quantity"]

            # Execute exit
            success = await execute_exit(symbol, "sell", quantity, current_price, "end_of_day")

            if success:
                logger.info(f"Successfully closed position for {symbol}")
            else:
                logger.warning(f"Failed to close position for {symbol}")

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    return True


async def run_trading_cycle():
    """Run a complete trading cycle."""
    logger.info("Starting trading cycle...")

    # Initialize system
    initialized = await initialize_system()

    if not initialized:
        logger.error("System initialization failed")
        return False

    # Update system state
    redis_client.update_system_state(state="running", timestamp=datetime.now().isoformat())

    # Check market status
    try:
        clock = alpaca_api.get_clock()
        if not clock.is_open:
            next_open = clock.next_open.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Market is closed. Next open: {next_open}")
            return False
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        # Continue without market status check

    # Run market scan
    await run_market_scan()

    # Analyze watchlist
    await analyze_watchlist()

    # Monitor existing positions
    await monitor_positions()

    # Make trade decisions
    await make_trade_decisions()

    # Check if market close is approaching
    now = datetime.now()
    market_close_time = datetime.strptime(
        f"{now.strftime('%Y-%m-%d')} {settings.timing.market_close}", "%Y-%m-%d %H:%M"
    ).replace(tzinfo=now.tzinfo)

    # If within 10 minutes of market close, close all positions
    if (market_close_time - now).total_seconds() < 600:  # 10 minutes
        logger.info("Market close approaching, closing all positions")
        await close_all_positions()

    # Update system state
    redis_client.update_system_state(state="idle", timestamp=datetime.now().isoformat())

    logger.info("Trading cycle completed")
    return True


async def main_loop():
    """Main trading loop."""
    logger.info("Starting main trading loop...")

    try:
        # Run initial system setup
        initialized = await initialize_system()

        if not initialized:
            logger.error("System initialization failed")
            return

        # Main loop
        while True:
            try:
                # Run trading cycle
                await run_trading_cycle()

                # Wait before next cycle
                logger.info("Waiting for next cycle...")
                await asyncio.sleep(60)  # 1-minute cycle

            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(60)  # Wait before retry

    except KeyboardInterrupt:
        logger.info("Manual interrupt received, shutting down...")

        # Close all positions on shutdown
        await close_all_positions()

        # Close API clients
        await alpha_vantage_client.close()
        if hasattr(openrouter_client, "close") and callable(openrouter_client.close):
            await openrouter_client.close()

    except Exception as e:
        logger.error(f"Unhandled exception in main loop: {e}")

    finally:
        # Final cleanup
        logger.info("Trading system shutdown complete")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Day Trading System")

    parser.add_argument("--cycle", action="store_true", help="Run a single trading cycle")
    parser.add_argument("--scan", action="store_true", help="Run market scan only")
    parser.add_argument("--analyze", action="store_true", help="Analyze watchlist only")
    parser.add_argument("--monitor", action="store_true", help="Monitor positions only")
    parser.add_argument("--trade", action="store_true", help="Make trade decisions only")
    parser.add_argument("--close", action="store_true", help="Close all positions")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Run based on arguments
    if args.cycle:
        asyncio.run(run_trading_cycle())
    elif args.scan:
        asyncio.run(run_market_scan())
    elif args.analyze:
        asyncio.run(analyze_watchlist())
    elif args.monitor:
        asyncio.run(monitor_positions())
    elif args.trade:
        asyncio.run(make_trade_decisions())
    elif args.close:
        asyncio.run(close_all_positions())
    else:
        # Run main loop
        asyncio.run(main_loop())
