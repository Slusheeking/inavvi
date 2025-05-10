"""
Main entry point for the trading system.
"""

import asyncio
import signal
import sys
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.config.settings import settings
from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.polygon import PolygonAPI
from src.data_sources.yahoo_finance import yahoo_finance_client
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("main")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name, version=settings.version, description="Day Trading System API"
)

# Create Polygon API client
polygon_client = PolygonAPI()

# Flag to indicate if system is running
is_running = False

# Store active websocket connections
active_connections: List[WebSocket] = []

# Schedule tasks
scheduled_tasks = []

# ---------- System Startup and Shutdown ----------


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info(f"Starting {settings.app_name} v{settings.version}")

    # Set system state
    await set_system_state("initializing")

    # Schedule background tasks
    schedule_background_tasks()

    # Signal handler for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal_handler)

    # Start the system
    await start_system()

    logger.info(f"{settings.app_name} started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down system")

    # Stop the system
    await stop_system()

    # Close client connections
    await close_clients()

    logger.info("System shutdown complete")


def signal_handler(sig, frame):
    """Handle termination signals."""
    logger.info(f"Received signal {sig}, shutting down")
    asyncio.create_task(stop_system())
    sys.exit(0)


async def close_clients():
    """Close API client connections."""
    # Close Alpha Vantage session
    await alpha_vantage_client.close()

    # Close Polygon WebSocket connection
    await polygon_client.disconnect_websocket()

    logger.info("All API client connections closed")


# ---------- System Control ----------


async def start_system():
    """Start the trading system."""
    global is_running

    try:
        logger.info("Starting trading system")

        # Set system state
        await set_system_state("starting")

        # Initialize data sources
        await initialize_data_sources()

        # Connect to market data
        await connect_market_data()

        # Update system state
        is_running = True
        await set_system_state("running")

        # Broadcast system status
        await broadcast_to_websockets({"event": "system_status", "status": "running"})

        logger.info("Trading system started successfully")
        return True
    except Exception as e:
        logger.error(f"Error starting system: {e}")
        await set_system_state("error", {"error": str(e)})
        return False


async def stop_system():
    """Stop the trading system."""
    global is_running

    try:
        logger.info("Stopping trading system")

        # Set system state
        await set_system_state("stopping")

        # Cancel all scheduled tasks
        for task in scheduled_tasks:
            if not task.done():
                task.cancel()

        # Close client connections
        await close_clients()

        # Update system state
        is_running = False
        await set_system_state("stopped")

        # Broadcast system status
        await broadcast_to_websockets({"event": "system_status", "status": "stopped"})

        logger.info("Trading system stopped successfully")
        return True
    except Exception as e:
        logger.error(f"Error stopping system: {e}")
        await set_system_state("error", {"error": str(e)})
        return False


async def restart_system():
    """Restart the trading system."""
    await stop_system()
    await asyncio.sleep(1)  # Brief pause
    await start_system()


# ---------- System State Management ----------


async def set_system_state(state: str, extra_data: Optional[Dict] = None):
    """
    Set the current system state.

    Args:
        state: State name
        extra_data: Optional additional data
    """
    state_data = {"state": state, "timestamp": datetime.now().isoformat()}

    if extra_data:
        state_data.update(extra_data)

    # Update Redis
    redis_client.update_system_state(**state_data)

    # Log state change
    logger.info(f"System state changed to: {state}")

    # Broadcast to websockets
    await broadcast_to_websockets({"event": "system_state", "data": state_data})


# ---------- Initialization ----------


async def initialize_data_sources():
    """Initialize data sources."""
    logger.info("Initializing data sources")

    # Initialize stock universe
    await initialize_stock_universe()

    # Initialize market data
    await initialize_market_data()

    logger.info("Data sources initialized")


async def initialize_stock_universe():
    """Initialize the stock universe for trading."""
    logger.info("Initializing stock universe")

    # Check if universe already exists in Redis
    universe = redis_client.get("stocks:universe")

    if not universe:
        logger.info("Fetching stock universe from Polygon")
        universe = await polygon_client.get_stock_universe(type="cs", active=True)

        if not universe:
            logger.error("Failed to fetch stock universe")
            return False

    logger.info(f"Stock universe contains {len(universe)} symbols")

    # Filter universe based on criteria (price, volume, etc.)
    filtered_universe = await filter_stock_universe(universe)

    logger.info(f"Filtered stock universe contains {len(filtered_universe)} symbols")

    # Store filtered universe
    redis_client.set("stocks:filtered_universe", filtered_universe)

    return True


async def filter_stock_universe(universe: List[Dict]):
    """
    Filter the stock universe based on trading criteria.

    Args:
        universe: Full stock universe

    Returns:
        Filtered universe
    """
    filtered = []
    count = 0

    # Process in batches
    batch_size = 100
    for i in range(0, min(len(universe), 1000), batch_size):
        batch = universe[i : i + batch_size]

        # Process each symbol in batch
        for stock in batch:
            symbol = stock["symbol"]

            try:
                # Get basic ticker info from Yahoo Finance
                info = await yahoo_finance_client.get_ticker_info(symbol)

                if not info:
                    continue

                # Apply filters
                price = info.get("regularMarketPrice", 0)
                volume = info.get("regularMarketVolume", 0)
                market_cap = info.get("marketCap", 0)

                # Filter criteria (customize as needed)
                if price >= 5 and price <= 100 and volume >= 500000 and market_cap >= 500000000:
                    filtered.append(
                        {
                            "symbol": symbol,
                            "name": info.get("shortName", ""),
                            "price": price,
                            "volume": volume,
                            "market_cap": market_cap,
                            "sector": info.get("sector", ""),
                            "industry": info.get("industry", ""),
                        }
                    )

                    count += 1
                    if count % 10 == 0:
                        logger.info(f"Processed {count} symbols, {len(filtered)} passed filters")

                # Limit to the desired universe size
                if len(filtered) >= settings.trading.initial_universe_size:
                    break
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

            # Add a small delay to avoid rate limits
            await asyncio.sleep(0.1)

        # Early exit if we have enough symbols
        if len(filtered) >= settings.trading.initial_universe_size:
            break

    return filtered


async def initialize_market_data():
    """Initialize market data."""
    logger.info("Initializing market data")

    # Get market status
    market_status = await polygon_client.get_market_status()
    redis_client.set("market:status", market_status)

    # Get sector performance
    sector_performance = await alpha_vantage_client.get_sector_performance()
    if sector_performance:
        redis_client.set("market:sectors", sector_performance)

    # Get economic indicators
    asyncio.create_task(initialize_economic_indicators())

    logger.info("Market data initialized")


async def initialize_economic_indicators():
    """Initialize economic indicators."""
    logger.info("Initializing economic indicators")

    # Get Treasury yields
    yield_10y = await alpha_vantage_client.get_treasury_yield(interval="daily", maturity="10year")
    if yield_10y is not None:
        redis_client.set("economic:treasury_yield:10year", yield_10y)

    # Get inflation data
    inflation = await alpha_vantage_client.get_inflation()
    if inflation is not None:
        redis_client.set("economic:inflation", inflation)

    logger.info("Economic indicators initialized")


async def connect_market_data():
    """Connect to real-time market data."""
    logger.info("Connecting to real-time market data")

    # Connect to Polygon WebSocket
    ws_connected = await polygon_client.connect_websocket()

    if not ws_connected:
        logger.error("Failed to connect to Polygon WebSocket")
        return False

    logger.info("Connected to real-time market data")
    return True


# ---------- Background Tasks ----------


def schedule_background_tasks():
    """Schedule background tasks."""
    logger.info("Scheduling background tasks")

    # Clear any existing tasks
    for task in scheduled_tasks:
        if not task.done():
            task.cancel()

    scheduled_tasks.clear()

    # Schedule tasks
    scheduled_tasks.append(asyncio.create_task(run_market_open_tasks()))
    scheduled_tasks.append(asyncio.create_task(run_market_close_tasks()))
    scheduled_tasks.append(asyncio.create_task(run_periodic_tasks()))

    logger.info(f"Scheduled {len(scheduled_tasks)} background tasks")


async def run_market_open_tasks():
    """Run tasks at market open."""
    while True:
        try:
            # Wait until next market open
            wait_seconds = seconds_until_market_open()

            if wait_seconds > 0:
                logger.info(f"Waiting {wait_seconds} seconds until market open")
                await asyncio.sleep(wait_seconds)

            # Market is open, run tasks
            logger.info("Running market open tasks")

            # Set system state
            await set_system_state("market_open")

            # Initialize pre-market data
            await initialize_pre_market_data()

            # Broadcast market open
            await broadcast_to_websockets({"event": "market_status", "status": "open"})

            # Wait until next day
            await asyncio.sleep(24 * 60 * 60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in market open tasks: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


async def run_market_close_tasks():
    """Run tasks at market close."""
    while True:
        try:
            # Wait until next market close
            wait_seconds = seconds_until_market_close()

            if wait_seconds > 0:
                logger.info(f"Waiting {wait_seconds} seconds until market close")
                await asyncio.sleep(wait_seconds)

            # Market is closed, run tasks
            logger.info("Running market close tasks")

            # Set system state
            await set_system_state("market_closed")

            # Close all positions
            await close_all_positions()

            # Broadcast market close
            await broadcast_to_websockets({"event": "market_status", "status": "closed"})

            # Wait until next day
            await asyncio.sleep(24 * 60 * 60)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in market close tasks: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


async def run_periodic_tasks():
    """Run periodic tasks."""
    while True:
        try:
            # Run every 5 minutes
            logger.debug("Running periodic tasks")

            # Update market data
            asyncio.create_task(update_market_data())

            # Update watchlist
            asyncio.create_task(update_watchlist())

            # Monitor positions
            asyncio.create_task(monitor_positions())

            # Wait for next update
            await asyncio.sleep(5 * 60)  # 5 minutes
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in periodic tasks: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


# ---------- Market Tasks ----------


async def initialize_pre_market_data():
    """Initialize pre-market data."""
    logger.info("Initializing pre-market data")

    # Get pre-market movers
    pre_market_movers = await polygon_client.get_premarket_movers(limit=50)
    if pre_market_movers:
        redis_client.set("market:pre_market_movers", pre_market_movers)

    # Get gainers and losers
    gainers_losers = await polygon_client.get_gainers_losers(limit=50)
    if gainers_losers:
        redis_client.set("market:gainers_losers", gainers_losers)

    # Initialize watchlist
    await initialize_watchlist()

    logger.info("Pre-market data initialized")


async def update_market_data():
    """Update market data."""
    logger.debug("Updating market data")

    # Get market status
    market_status = await polygon_client.get_market_status()
    redis_client.set("market:status", market_status, expiry=300)  # 5 min expiry

    # Get gainers and losers (top 20)
    gainers_losers = await polygon_client.get_gainers_losers(limit=20)
    if gainers_losers:
        redis_client.set("market:gainers_losers", gainers_losers, expiry=300)

    # Get most active stocks
    most_active = await polygon_client.get_most_active(limit=20)
    if most_active:
        redis_client.set("market:most_active", most_active, expiry=300)

    # Broadcast market update
    await broadcast_to_websockets(
        {
            "event": "market_update",
            "data": {
                "status": market_status.get("market", "unknown"),
                "timestamp": datetime.now().isoformat(),
            },
        }
    )


# ---------- Watchlist Management ----------


async def initialize_watchlist():
    """Initialize the trading watchlist."""
    logger.info("Initializing watchlist")

    # Get current watchlist
    watchlist = redis_client.get_watchlist()

    # If watchlist is empty, create a new one
    if not watchlist:
        # Get pre-market movers
        pre_market_movers = redis_client.get("market:pre_market_movers")

        # Get gainers and losers
        gainers_losers = redis_client.get("market:gainers_losers")

        # Combine and deduplicate
        candidates = []

        if pre_market_movers:
            candidates.extend([item["symbol"] for item in pre_market_movers[:30]])

        if gainers_losers:
            candidates.extend([item["symbol"] for item in gainers_losers["gainers"][:15]])
            candidates.extend([item["symbol"] for item in gainers_losers["losers"][:15]])

        # Deduplicate
        candidates = list(set(candidates))

        # Limit to watchlist size
        watchlist = candidates[: settings.trading.watchlist_size]

        # Save watchlist
        redis_client.set_watchlist(watchlist)

    # Subscribe to real-time data for watchlist
    await polygon_client.subscribe_to_symbols(watchlist)

    logger.info(f"Watchlist initialized with {len(watchlist)} symbols")

    # Broadcast watchlist update
    await broadcast_to_websockets(
        {
            "event": "watchlist_update",
            "data": {"watchlist": watchlist, "timestamp": datetime.now().isoformat()},
        }
    )


async def update_watchlist():
    """Update the trading watchlist."""
    logger.debug("Updating watchlist")

    # Get current watchlist
    current_watchlist = redis_client.get_watchlist()

    # Get current candidates
    candidates = redis_client.get_ranked_candidates()

    # Check if we need to update
    if not candidates:
        # Rank current watchlist
        await rank_watchlist()
        candidates = redis_client.get_ranked_candidates()

    # If still no candidates, skip update
    if not candidates:
        logger.warning("No candidates available for watchlist update")
        return

    # Extract top symbols
    top_symbols = [item["symbol"] for item in candidates[: settings.trading.watchlist_size]]

    # Check if watchlist needs updating
    if set(top_symbols) != set(current_watchlist):
        # Unsubscribe from removed symbols
        removed_symbols = [s for s in current_watchlist if s not in top_symbols]
        if removed_symbols:
            await polygon_client.unsubscribe_from_symbols(removed_symbols)

        # Subscribe to new symbols
        new_symbols = [s for s in top_symbols if s not in current_watchlist]
        if new_symbols:
            await polygon_client.subscribe_to_symbols(new_symbols)

        # Update watchlist
        redis_client.set_watchlist(top_symbols)

        logger.info(f"Watchlist updated: added {len(new_symbols)}, removed {len(removed_symbols)}")

        # Broadcast watchlist update
        await broadcast_to_websockets(
            {
                "event": "watchlist_update",
                "data": {
                    "watchlist": top_symbols,
                    "added": new_symbols,
                    "removed": removed_symbols,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )


async def rank_watchlist():
    """Rank stocks in the watchlist based on trading signals."""
    logger.debug("Ranking watchlist")

    # Get current watchlist
    watchlist = redis_client.get_watchlist()

    if not watchlist:
        logger.warning("Empty watchlist, cannot rank")
        return

    # Process each symbol
    for symbol in watchlist:
        try:
            # Get current price data
            price_data = redis_client.get_stock_data(symbol, "price")

            if not price_data:
                # Fetch from Polygon if not in cache
                snapshot = await polygon_client.get_stock_snapshot(symbol)
                if not snapshot:
                    continue
                price_data = snapshot

            # Calculate simple score (placeholder for ML model)
            # In a real system, this would use the ML model for scoring
            score = calculate_simple_score(price_data)

            # Add to candidates
            redis_client.add_candidate_score(
                symbol,
                score,
                {
                    "price": price_data.get("price", {}).get("last", 0),
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.error(f"Error ranking {symbol}: {e}")

    logger.debug("Watchlist ranking completed")


def calculate_simple_score(price_data: Dict) -> float:
    """
    Calculate a simple trading score based on price data.

    Args:
        price_data: Price data dictionary

    Returns:
        Score value (higher is better)
    """
    # Extract price information
    price = price_data.get("price", {})
    last = price.get("last", 0)
    open_price = price.get("open", 0)
    high = price.get("high", 0)
    low = price.get("low", 0)
    volume = price.get("volume", 0)

    # Skip invalid data
    if not all([last, open_price, high, low]):
        return 0

    # Calculate basic metrics
    price_change = last - open_price
    price_change_pct = (price_change / open_price) * 100 if open_price else 0
    range_pct = ((high - low) / low) * 100 if low else 0

    # Simple scoring formula (placeholder)
    # This would be replaced by an ML model in a real system
    score = abs(price_change_pct) * 0.7 + range_pct * 0.3

    # Adjust for volume
    if volume > 1000000:  # High volume
        score *= 1.2

    return score


# ---------- Position Management ----------


async def monitor_positions():
    """Monitor active positions."""
    logger.debug("Monitoring positions")

    # Get active positions
    positions = redis_client.get_all_active_positions()

    if not positions:
        return

    logger.debug(f"Monitoring {len(positions)} active positions")

    # Process each position
    for symbol, position in positions.items():
        try:
            # Get current price
            price_data = redis_client.get_stock_data(symbol, "price")

            if not price_data or "price" not in price_data:
                # Fetch from Polygon if not in cache
                snapshot = await polygon_client.get_stock_snapshot(symbol)
                if not snapshot:
                    continue
                price_data = snapshot

            # Update position P&L
            current_price = price_data.get("price", {}).get("last", 0)
            redis_client.update_position_pnl(symbol, current_price)

            # Check for exit signals
            await check_exit_signals(symbol, position, current_price)
        except Exception as e:
            logger.error(f"Error monitoring position {symbol}: {e}")


async def check_exit_signals(symbol: str, position: Dict, current_price: float):
    """
    Check for exit signals for a position.

    Args:
        symbol: Stock symbol
        position: Position data
        current_price: Current price
    """
    # Get position details
    position.get("entry_price", 0)
    stop_loss = position.get("stop_loss", 0)
    take_profit = position.get("take_profit", 0)
    side = position.get("side", "long")

    # Check stop loss
    if side == "long" and stop_loss > 0 and current_price <= stop_loss:
        logger.info(f"Stop loss triggered for {symbol} at {current_price}")

        # Create exit signal
        redis_client.add_trading_signal(
            symbol,
            "exit",
            {
                "reason": "stop_loss",
                "price": current_price,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Broadcast exit signal
        await broadcast_to_websockets(
            {
                "event": "exit_signal",
                "data": {
                    "symbol": symbol,
                    "reason": "stop_loss",
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )

    # Check take profit
    elif side == "long" and take_profit > 0 and current_price >= take_profit:
        logger.info(f"Take profit triggered for {symbol} at {current_price}")

        # Create exit signal
        redis_client.add_trading_signal(
            symbol,
            "exit",
            {
                "reason": "take_profit",
                "price": current_price,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Broadcast exit signal
        await broadcast_to_websockets(
            {
                "event": "exit_signal",
                "data": {
                    "symbol": symbol,
                    "reason": "take_profit",
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                },
            }
        )


async def close_all_positions():
    """Close all open positions."""
    logger.info("Closing all positions")

    # Get active positions
    positions = redis_client.get_all_active_positions()

    if not positions:
        logger.info("No positions to close")
        return

    logger.info(f"Closing {len(positions)} positions")

    # Process each position
    for symbol, _position in positions.items():
        try:
            # Get current price
            price_data = redis_client.get_stock_data(symbol, "price")

            if not price_data or "price" not in price_data:
                # Fetch from Polygon if not in cache
                snapshot = await polygon_client.get_stock_snapshot(symbol)
                if not snapshot:
                    continue
                price_data = snapshot

            current_price = price_data.get("price", {}).get("last", 0)

            # Create exit signal
            redis_client.add_trading_signal(
                symbol,
                "exit",
                {
                    "reason": "market_close",
                    "price": current_price,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Broadcast exit signal
            await broadcast_to_websockets(
                {
                    "event": "exit_signal",
                    "data": {
                        "symbol": symbol,
                        "reason": "market_close",
                        "price": current_price,
                        "timestamp": datetime.now().isoformat(),
                    },
                }
            )
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")


# ---------- Utility Functions ----------


def seconds_until_market_open():
    """
    Calculate seconds until market open.

    Returns:
        Seconds until next market open (0 if market is open)
    """
    now = datetime.now()

    # Parse market open time
    market_open_str = settings.timing.market_open
    hours, minutes = map(int, market_open_str.split(":"))
    market_open = time(hours, minutes)

    # Check if market is already open
    if now.time() >= market_open:
        # Calculate time until tomorrow's open
        tomorrow = now.date() + timedelta(days=1)
        next_open = datetime.combine(tomorrow, market_open)
    else:
        # Calculate time until today's open
        next_open = datetime.combine(now.date(), market_open)

    # Return seconds until open
    return (next_open - now).total_seconds()


def seconds_until_market_close():
    """
    Calculate seconds until market close.

    Returns:
        Seconds until next market close (0 if market is closed)
    """
    now = datetime.now()

    # Parse market close time
    market_close_str = settings.timing.market_close
    hours, minutes = map(int, market_close_str.split(":"))
    market_close = time(hours, minutes)

    # Check if market is already closed
    if now.time() >= market_close:
        # Calculate time until tomorrow's close
        tomorrow = now.date() + timedelta(days=1)
        next_close = datetime.combine(tomorrow, market_close)
    else:
        # Calculate time until today's close
        next_close = datetime.combine(now.date(), market_close)

    # Return seconds until close
    return (next_close - now).total_seconds()


# ---------- WebSocket Functions ----------


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        # Send initial system state
        system_state = redis_client.get_system_state()
        await websocket.send_json({"event": "system_state", "data": system_state})

        # Send current watchlist
        watchlist = redis_client.get_watchlist()
        await websocket.send_json(
            {
                "event": "watchlist_update",
                "data": {"watchlist": watchlist, "timestamp": datetime.now().isoformat()},
            }
        )

        # Send active positions
        positions = redis_client.get_all_active_positions()
        await websocket.send_json(
            {
                "event": "positions_update",
                "data": {"positions": positions, "timestamp": datetime.now().isoformat()},
            }
        )

        # Listen for client messages
        while True:
            data = await websocket.receive_json()
            await handle_websocket_message(websocket, data)
    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Remove connection
        if websocket in active_connections:
            active_connections.remove(websocket)


async def handle_websocket_message(websocket: WebSocket, data: Dict):
    """
    Handle incoming WebSocket message.

    Args:
        websocket: WebSocket connection
        data: Message data
    """
    try:
        message_type = data.get("type")

        if message_type == "ping":
            # Respond to ping
            await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
        elif message_type == "get_watchlist":
            # Send current watchlist
            watchlist = redis_client.get_watchlist()
            await websocket.send_json(
                {"type": "watchlist", "data": watchlist, "timestamp": datetime.now().isoformat()}
            )
        elif message_type == "get_positions":
            # Send active positions
            positions = redis_client.get_all_active_positions()
            await websocket.send_json(
                {"type": "positions", "data": positions, "timestamp": datetime.now().isoformat()}
            )
        elif message_type == "get_system_state":
            # Send system state
            system_state = redis_client.get_system_state()
            await websocket.send_json(
                {
                    "type": "system_state",
                    "data": system_state,
                    "timestamp": datetime.now().isoformat(),
                }
            )
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await websocket.send_json(
            {"type": "error", "message": str(e), "timestamp": datetime.now().isoformat()}
        )


async def broadcast_to_websockets(message: Dict):
    """
    Broadcast message to all active WebSocket connections.

    Args:
        message: Message to broadcast
    """
    if not active_connections:
        return

    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.error(f"Error broadcasting to WebSocket: {e}")


# ---------- API Routes ----------


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "status": "running" if is_running else "stopped",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status")
async def get_status():
    """Get system status."""
    system_state = redis_client.get_system_state()

    return {
        "status": "running" if is_running else "stopped",
        "state": system_state,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/watchlist")
async def get_watchlist():
    """Get current watchlist."""
    watchlist = redis_client.get_watchlist()

    # Get additional data for each symbol
    watchlist_data = []
    for symbol in watchlist:
        price_data = redis_client.get_stock_data(symbol, "price")

        if price_data:
            watchlist_data.append(
                {
                    "symbol": symbol,
                    "price": price_data.get("price", {}).get("last", 0),
                    "change": price_data.get("price", {}).get("last", 0)
                    - price_data.get("price", {}).get("open", 0),
                    "volume": price_data.get("price", {}).get("volume", 0),
                }
            )

    return {"watchlist": watchlist_data, "timestamp": datetime.now().isoformat()}


@app.get("/positions")
async def get_positions():
    """Get active positions."""
    positions = redis_client.get_all_active_positions()

    return {"positions": positions, "timestamp": datetime.now().isoformat()}


@app.post("/start")
async def start():
    """Start the trading system."""
    if is_running:
        return {
            "status": "already_running",
            "message": "System is already running",
            "timestamp": datetime.now().isoformat(),
        }

    success = await start_system()

    return {
        "status": "started" if success else "error",
        "message": "System started successfully" if success else "Failed to start system",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/stop")
async def stop():
    """Stop the trading system."""
    if not is_running:
        return {
            "status": "already_stopped",
            "message": "System is already stopped",
            "timestamp": datetime.now().isoformat(),
        }

    success = await stop_system()

    return {
        "status": "stopped" if success else "error",
        "message": "System stopped successfully" if success else "Failed to stop system",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/restart")
async def restart():
    """Restart the trading system."""
    await stop_system()
    await asyncio.sleep(1)  # Brief pause
    success = await start_system()

    return {
        "status": "restarted" if success else "error",
        "message": "System restarted successfully" if success else "Failed to restart system",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
