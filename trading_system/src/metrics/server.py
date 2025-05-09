"""
Metrics server for exposing trading system metrics.
"""
import asyncio
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("metrics_server")

# Create FastAPI app
app = FastAPI(
    title="Trading System Metrics",
    description="API for trading system metrics and control",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# WebSocket connections
active_connections: List[WebSocket] = []

# Models for API responses
class SystemStatus(BaseModel):
    """System status model."""
    state: str
    uptime: float
    positions: int
    daily_pnl: float
    daily_pnl_pct: float
    watchlist_count: int
    timestamp: str

class PositionInfo(BaseModel):
    """Position information model."""
    symbol: str
    entry_price: float
    current_price: float
    quantity: int
    unrealized_pnl: float
    unrealized_pnl_pct: float
    time_in_trade: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class WatchlistItem(BaseModel):
    """Watchlist item model."""
    symbol: str
    price: float
    score: float
    pattern: str
    sentiment: float

# Keep track of system start time
system_start_time = time.time()

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "description": "Trading System Metrics API"
    }

@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get system status."""
    # Get system state
    system_state = redis_client.get_system_state() or {}
    state = system_state.get("state", "unknown")
    
    # Get active positions
    positions = redis_client.get_all_active_positions()
    position_count = len(positions)
    
    # Calculate P&L
    daily_pnl = sum(p.get('unrealized_pnl', 0) for p in positions.values())
    
    # Starting capital (for demo)
    starting_capital = 5000.0
    
    # Daily P&L percentage
    daily_pnl_pct = (daily_pnl / starting_capital) * 100 if starting_capital else 0
    
    # Get watchlist
    watchlist = redis_client.get_watchlist()
    watchlist_count = len(watchlist) if watchlist else 0
    
    # Calculate uptime
    uptime = time.time() - system_start_time
    
    return SystemStatus(
        state=state,
        uptime=uptime,
        positions=position_count,
        daily_pnl=daily_pnl,
        daily_pnl_pct=daily_pnl_pct,
        watchlist_count=watchlist_count,
        timestamp=datetime.now().isoformat()
    )

@app.get("/positions")
async def get_positions():
    """Get active positions."""
    positions = redis_client.get_all_active_positions()
    
    # Convert to list of PositionInfo
    position_list = []
    for symbol, position_data in positions.items():
        try:
            position_list.append(PositionInfo(
                symbol=symbol,
                entry_price=position_data.get('entry_price', 0.0),
                current_price=position_data.get('current_price', 0.0),
                quantity=position_data.get('quantity', 0),
                unrealized_pnl=position_data.get('unrealized_pnl', 0.0),
                unrealized_pnl_pct=position_data.get('unrealized_pnl_pct', 0.0),
                time_in_trade=position_data.get('time_in_trade', 0.0),
                stop_loss=position_data.get('stop_loss'),
                take_profit=position_data.get('take_profit')
            ))
        except Exception as e:
            logger.error(f"Error processing position {symbol}: {e}")
    
    return position_list

@app.get("/watchlist")
async def get_watchlist():
    """Get current watchlist."""
    watchlist = redis_client.get_watchlist()
    
    if not watchlist:
        return []
    
    # Get ranking data
    candidates = redis_client.get_ranked_candidates()
    candidate_map = {c['symbol']: c for c in candidates} if candidates else {}
    
    # Build watchlist items
    items = []
    for symbol in watchlist:
        # Get stock data
        stock_data = redis_client.get_stock_data(symbol, 'price') or {}
        
        # Get candidate data
        candidate = candidate_map.get(symbol, {})
        
        # Create watchlist item
        item = WatchlistItem(
            symbol=symbol,
            price=stock_data.get('last', 0.0),
            score=candidate.get('score', 0.0),
            pattern=candidate.get('pattern', 'unknown'),
            sentiment=candidate.get('sentiment', 0.0)
        )
        
        items.append(item)
    
    return items

@app.get("/candidates")
async def get_candidates():
    """Get trading candidates."""
    candidates = redis_client.get_ranked_candidates()
    return candidates or []

@app.get("/market")
async def get_market_info():
    """Get market information."""
    # Get market status
    market_status = redis_client.get("market:status") or {}
    
    # Get sector performance
    sector_performance = redis_client.get("market:sectors") or {}
    
    # Get most active stocks
    most_active = redis_client.get("market:most_active") or []
    
    # Get gainers and losers
    gainers_losers = redis_client.get("market:gainers_losers") or {}
    
    return {
        "status": market_status,
        "sectors": sector_performance,
        "most_active": most_active,
        "gainers_losers": gainers_losers,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/performance")
async def get_performance():
    """Get system performance metrics."""
    # TODO: Implement performance tracking
    return {
        "trades_today": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/control/start")
async def start_system():
    """Start the trading system."""
    # Get current system state
    system_state = redis_client.get_system_state() or {}
    current_state = system_state.get("state", "unknown")
    
    if current_state == "running":
        return {"status": "already_running", "message": "System is already running"}
    
    # Update system state
    redis_client.update_system_state(state="starting")
    
    # In a real implementation, this would send a message to the main process
    # For now, we'll just update the state
    redis_client.update_system_state(state="running")
    
    return {"status": "started", "message": "System started successfully"}

@app.post("/control/stop")
async def stop_system():
    """Stop the trading system."""
    # Get current system state
    system_state = redis_client.get_system_state() or {}
    current_state = system_state.get("state", "unknown")
    
    if current_state == "stopped":
        return {"status": "already_stopped", "message": "System is already stopped"}
    
    # Update system state
    redis_client.update_system_state(state="stopping")
    
    # In a real implementation, this would send a message to the main process
    # For now, we'll just update the state
    redis_client.update_system_state(state="stopped")
    
    return {"status": "stopped", "message": "System stopped successfully"}

@app.post("/control/restart")
async def restart_system():
    """Restart the trading system."""
    # Stop the system
    await stop_system()
    
    # Wait a moment
    await asyncio.sleep(1)
    
    # Start the system
    return await start_system()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial status
        status = await get_status()
        await websocket.send_json({"event": "status", "data": status.dict()})
        
        # Listen for client messages
        while True:
            data = await websocket.receive_json()
            
            # Handle ping
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            
            # Handle other message types if needed
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

async def broadcast_updates():
    """Broadcast updates to all connected WebSocket clients."""
    while True:
        if active_connections:
            # Get latest status
            status = await get_status()
            
            # Broadcast to all clients
            for connection in active_connections:
                try:
                    await connection.send_json({"event": "status", "data": status.dict()})
                except Exception:
                    # Remove connection if it's broken
                    if connection in active_connections:
                        active_connections.remove(connection)
        
        # Wait before next update
        await asyncio.sleep(5)  # Update every 5 seconds

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup."""
    asyncio.create_task(broadcast_updates())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)