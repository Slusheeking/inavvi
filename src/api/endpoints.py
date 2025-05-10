"""
FastAPI routes for the trading system API.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query
from pydantic import BaseModel

from src.config.settings import settings
from src.core.data_pipeline import data_pipeline
from src.core.screening import stock_screener
from src.core.trade_execution import trade_executor
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("api_endpoints")

# Create API router
router = APIRouter()

# ----- Pydantic Models -----


class SystemStatus(BaseModel):
    """System status model."""

    status: str
    state: Dict[str, Any]
    timestamp: str


class TradeDecision(BaseModel):
    """Trade decision model."""

    decision: str
    position_size: float
    confidence: float
    reasoning: str
    key_factors: List[str]


class ExitDecision(BaseModel):
    """Exit decision model."""

    decision: str
    exit_size: float
    confidence: float
    reasoning: str
    key_factors: List[str]


class SymbolData(BaseModel):
    """Symbol data model."""

    symbol: str
    timeframe: str = "1d"
    bars: int = 100


class TradeRequest(BaseModel):
    """Trade request model."""

    symbol: str
    action: str
    quantity: Optional[float] = None
    position_size: Optional[float] = None
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    reason: Optional[str] = None


class ExitRequest(BaseModel):
    """Exit request model."""

    symbol: str
    exit_size: float = 1.0
    reason: Optional[str] = None


class WatchlistUpdateRequest(BaseModel):
    """Watchlist update request model."""

    action: str  # "add", "remove", "replace"
    symbols: List[str]


# ----- API Routes -----


@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "description": "Trading System API",
        "docs_url": "/docs",
    }


@router.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status."""
    # Get system state from Redis
    system_state = redis_client.get_system_state()

    # Check trading status
    is_running = system_state.get("state", "") == "running"

    return {
        "status": "running" if is_running else "stopped",
        "state": system_state,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/system/start")
async def start_system(background_tasks: BackgroundTasks):
    """Start the trading system."""
    # Get current state
    system_state = redis_client.get_system_state()
    current_state = system_state.get("state", "")

    if current_state == "running":
        return {"message": "System is already running"}

    # Update state to starting
    redis_client.update_system_state(state="starting")

    # Start system in background
    background_tasks.add_task(trade_executor.initialize)

    return {"message": "System starting"}


@router.post("/system/stop")
async def stop_system(background_tasks: BackgroundTasks):
    """Stop the trading system."""
    # Get current state
    system_state = redis_client.get_system_state()
    current_state = system_state.get("state", "")

    if current_state == "stopped":
        return {"message": "System is already stopped"}

    # Update state to stopping
    redis_client.update_system_state(state="stopping")

    # Stop system in background
    # This will close all positions
    background_tasks.add_task(trade_executor.close_all_positions)

    return {"message": "System stopping"}


@router.post("/system/restart")
async def restart_system(background_tasks: BackgroundTasks):
    """Restart the trading system."""
    # Update state to restarting
    redis_client.update_system_state(state="restarting")

    # Stop system, then start it again
    async def restart_task():
        await trade_executor.close_all_positions()  # Close all positions
        await trade_executor.initialize()  # Reinitialize system

    background_tasks.add_task(restart_task)

    return {"message": "System restarting"}


@router.get("/market/status")
async def get_market_status():
    """Get current market status."""
    try:
        # Get market status from data pipeline
        market_context = await data_pipeline.get_market_context()
        return market_context
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/movers")
async def get_market_movers(
    category: str = Query("gainers", enum=["gainers", "losers", "active", "premarket"]),
    limit: int = Query(10, ge=1, le=50),
):
    """Get market movers (gainers, losers, most active)."""
    try:
        movers = {}

        if category == "gainers" or category == "losers":
            # Get gainers and losers
            gainers_losers = await stock_screener.get_market_movers()
            if gainers_losers and category in gainers_losers:
                movers = gainers_losers[category][:limit]
        elif category == "active":
            # Get most active stocks
            active = await stock_screener.get_most_active()
            movers = active[:limit] if active else []
        elif category == "premarket":
            # Get pre-market movers
            premarket = await stock_screener.get_pre_market_movers()
            movers = premarket[:limit] if premarket else []

        return {"category": category, "movers": movers, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting market movers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/sectors")
async def get_sector_performance():
    """Get sector performance."""
    try:
        # Get sector performance from Alpha Vantage
        sector_performance = redis_client.get("market:sectors")

        if not sector_performance:
            # Fetch from API if not in cache
            from src.data_sources.alpha_vantage import alpha_vantage_client

            sector_performance = await alpha_vantage_client.get_sector_performance()
            if sector_performance:
                redis_client.set("market:sectors", sector_performance, expiry=3600)  # 1 hour expiry

        return sector_performance
    except Exception as e:
        logger.error(f"Error getting sector performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist")
async def get_watchlist():
    """Get current watchlist."""
    try:
        # Get watchlist from Redis
        watchlist = redis_client.get_watchlist()

        # Get data for each symbol
        watchlist_data = []
        for symbol in watchlist:
            # Get price data
            price_data = redis_client.get_stock_data(symbol, "price")

            # Get pattern data if available
            pattern_data = {}
            candidates = redis_client.get_ranked_candidates()
            if candidates:
                candidate = next((c for c in candidates if c["symbol"] == symbol), None)
                if candidate:
                    pattern_data = {
                        "name": candidate.get("pattern", "unknown"),
                        "confidence": candidate.get("pattern_confidence", 0),
                    }

            # Create watchlist item
            item = {
                "symbol": symbol,
                "price": price_data.get("price", {}) if price_data else {},
                "pattern": pattern_data,
                "score": next((c.get("score", 0) for c in candidates if c["symbol"] == symbol), 0),
                "timestamp": datetime.now().isoformat(),
            }

            watchlist_data.append(item)

        return {"watchlist": watchlist_data, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist/update")
async def update_watchlist(request: WatchlistUpdateRequest):
    """Update watchlist."""
    try:
        current_watchlist = redis_client.get_watchlist()

        if request.action == "add":
            # Add symbols to watchlist
            new_watchlist = list(set(current_watchlist + request.symbols))
            # Limit to watchlist size
            new_watchlist = new_watchlist[: settings.trading.watchlist_size]

        elif request.action == "remove":
            # Remove symbols from watchlist
            new_watchlist = [s for s in current_watchlist if s not in request.symbols]

        elif request.action == "replace":
            # Replace entire watchlist
            new_watchlist = request.symbols[: settings.trading.watchlist_size]

        else:
            raise HTTPException(status_code=400, detail="Invalid action")

        # Update watchlist in Redis
        redis_client.set_watchlist(new_watchlist)

        # Subscribe to new symbols for real-time data
        from src.data_sources.polygon import polygon_client

        await polygon_client.subscribe_to_symbols(new_watchlist)

        return {
            "message": f"Watchlist updated with action: {request.action}",
            "watchlist": new_watchlist,
        }
    except Exception as e:
        logger.error(f"Error updating watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist/refresh")
async def refresh_watchlist(background_tasks: BackgroundTasks):
    """Refresh watchlist based on market conditions."""
    try:
        # Update in background
        background_tasks.add_task(stock_screener.update_watchlist)

        return {"message": "Watchlist refresh started"}
    except Exception as e:
        logger.error(f"Error refreshing watchlist: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data/{symbol}")
async def get_symbol_data(symbol_data: SymbolData):
    """Get OHLCV data for a symbol."""
    try:
        symbol = symbol_data.symbol
        timeframe = symbol_data.timeframe
        bars = symbol_data.bars

        # Convert timeframe to parameters
        if timeframe == "1m":
            df = await data_pipeline.get_stock_data(symbol, "intraday")

        elif timeframe == "5m":
            # For 5-minute data, we need to use the historical data method
            from src.data_sources.polygon import polygon_client

            df = await polygon_client.get_historical_data(
                symbol=symbol, timeframe="minute", multiplier=5, limit=bars
            )

        elif timeframe == "1h":
            # For hourly data
            from src.data_sources.polygon import polygon_client

            df = await polygon_client.get_historical_data(
                symbol=symbol, timeframe="hour", multiplier=1, limit=bars
            )

        elif timeframe == "1d":
            # For daily data
            df = await data_pipeline.get_stock_data(symbol, "daily")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported timeframe: {timeframe}")

        # Get technical indicators
        indicators = await data_pipeline.get_technical_indicators(symbol)

        if df is not None and not isinstance(df, dict):
            # Convert DataFrame to list of dictionaries
            if hasattr(df, "to_dict"):
                data = df.reset_index().to_dict(orient="records")
            else:
                data = []
        else:
            data = []

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data,
            "indicators": indicators,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting data for {symbol_data.symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions():
    """Get active positions."""
    try:
        # Get positions from Redis
        positions = redis_client.get_all_active_positions()

        return {"positions": positions, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/{symbol}")
async def get_position(symbol: str = Path(..., description="Stock symbol")):
    """Get a specific position."""
    try:
        # Get position from Redis
        position = redis_client.get_active_position(symbol)

        if not position:
            raise HTTPException(status_code=404, detail=f"Position not found for symbol: {symbol}")

        return position
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trade")
async def execute_trade(request: TradeRequest, background_tasks: BackgroundTasks):
    """Execute a trade."""
    try:
        symbol = request.symbol
        action = request.action.lower()

        if action == "buy":
            # Check if system is running
            system_state = redis_client.get_system_state()
            if system_state.get("state", "") != "running":
                raise HTTPException(status_code=400, detail="System is not running")

            # Check if we already have a position in this symbol
            existing_position = redis_client.get_active_position(symbol)
            if existing_position:
                raise HTTPException(status_code=400, detail=f"Position already exists for {symbol}")

            # Create a trade decision
            decision = {
                "decision": "trade",
                "position_size": request.position_size or 0.5,
                "confidence": 0.9,
                "reasoning": request.reason or "Manual trade request",
                "key_factors": ["manual_entry"],
            }

            # Execute trade in background
            background_tasks.add_task(trade_executor.execute_trade, symbol, decision)

            return {"message": f"Buy order submitted for {symbol}", "status": "pending"}

        elif action == "sell" or action == "exit":
            # Check if we have an active position
            position = redis_client.get_active_position(symbol)
            if not position:
                raise HTTPException(status_code=404, detail=f"Position not found for {symbol}")

            # Determine exit size
            exit_size = request.exit_size if hasattr(request, "exit_size") else 1.0

            # Execute exit in background
            background_tasks.add_task(
                trade_executor.execute_exit,
                symbol,
                position,
                exit_size,
                request.reason or "Manual exit",
            )

            return {"message": f"Sell order submitted for {symbol}", "status": "pending"}

        else:
            raise HTTPException(status_code=400, detail=f"Invalid action: {action}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/positions/close-all")
async def close_all_positions(background_tasks: BackgroundTasks):
    """Close all active positions."""
    try:
        # Close all positions in background
        background_tasks.add_task(trade_executor.close_all_positions)

        return {"message": "Closing all positions", "status": "pending"}
    except Exception as e:
        logger.error(f"Error closing all positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/{symbol}")
async def analyze_symbol(symbol: str = Path(..., description="Stock symbol")):
    """Analyze a symbol for trading opportunities."""
    try:
        # Run analysis
        opportunity = await stock_screener.analyze_symbol(symbol)

        if not opportunity:
            return {"message": f"No trading opportunity found for {symbol}"}

        # Get trade decision
        decision = await trade_executor.make_trade_decision(opportunity)

        return {
            "symbol": symbol,
            "opportunity": opportunity,
            "decision": decision,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/history")
async def get_trade_history(limit: int = Query(50, ge=1, le=1000)):
    """Get trade history."""
    try:
        # Get closed positions from Redis
        closed_positions = redis_client.get_all_closed_positions()

        # Convert to list and sort by close time (most recent first)
        trades = []
        for symbol, position in closed_positions.items():
            trades.append(
                {
                    "symbol": symbol,
                    "entry_price": position.get("entry_price", 0),
                    "exit_price": position.get("close_price", 0),
                    "quantity": position.get("quantity", 0),
                    "entry_time": position.get("entry_time", ""),
                    "exit_time": position.get("close_time", ""),
                    "realized_pnl": position.get("realized_pnl", 0),
                    "realized_pnl_pct": position.get("realized_pnl_pct", 0),
                    "reason": position.get("close_reason", ""),
                }
            )

        # Sort by exit time
        trades.sort(key=lambda x: x["exit_time"], reverse=True)

        # Limit results
        trades = trades[:limit]

        return {"trades": trades, "count": len(trades), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/account")
async def get_account_info():
    """Get account information."""
    try:
        # Get account info from Alpaca
        account_info = await trade_executor.get_account_info()

        if not account_info:
            raise HTTPException(status_code=500, detail="Failed to get account information")

        return account_info
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance():
    """Get trading performance statistics."""
    try:
        # Get trading stats
        trading_stats = await trade_executor.get_trading_stats()

        if not trading_stats:
            raise HTTPException(status_code=500, detail="Failed to get trading statistics")

        return trading_stats
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/logs")
async def get_logs(
    level: str = Query("info", enum=["debug", "info", "warning", "error"]),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get system logs."""
    try:
        # Set log level mapping
        level_map = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }

        # Get log level
        level_map.get(level.lower(), logging.INFO)

        # Get log file path
        log_file = os.path.join(settings.logs_dir, "system.log")

        if not os.path.exists(log_file):
            return {"logs": [], "count": 0, "level": level}

        # Read logs from file
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Filter by level and limit
        filtered_logs = []
        for line in reversed(lines):  # Start from the end
            if len(filtered_logs) >= limit:
                break

            # Parse log level from line
            parts = line.split(" - ")
            if len(parts) >= 3:
                line_level = parts[1].strip().lower()

                # Check if level matches
                if (
                    level == "debug"
                    or line_level in ["critical", "error", "warning"]
                    or (level == "info" and line_level == "info")
                ):
                    filtered_logs.append(line.strip())

        return {"logs": filtered_logs, "count": len(filtered_logs), "level": level}
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))
