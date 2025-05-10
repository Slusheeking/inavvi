"""
WebSocket handler for real-time updates from the trading system.
"""

# Add the project root to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Set

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("websocket")


class ConnectionManager:
    """
    WebSocket connection manager.

    Handles:
    - Client connections
    - Message broadcasting
    - Subscription management
    - Heartbeat monitoring
    """

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        self.client_info: Dict[WebSocket, Dict[str, Any]] = {}

        # Initialize default subscription channels
        default_channels = [
            "system_status",
            "market_status",
            "watchlist",
            "positions",
            "trades",
            "indicators",
        ]

        for channel in default_channels:
            self.subscriptions[channel] = set()

        logger.info("WebSocket connection manager initialized")

    async def connect(self, websocket: WebSocket):
        """
        Accept a new WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        await websocket.accept()
        self.active_connections.append(websocket)

        # Initialize client info
        self.client_info[websocket] = {
            "connected_at": datetime.now().isoformat(),
            "client_id": f"client_{len(self.active_connections)}",
            "subscriptions": set(),
            "last_ping": datetime.now().timestamp(),
        }

        logger.info(f"Client connected - Total connections: {len(self.active_connections)}")

        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connected",
                "message": "Connected to trading system",
                "client_id": self.client_info[websocket]["client_id"],
                "timestamp": datetime.now().isoformat(),
            },
            websocket,
        )

    def disconnect(self, websocket: WebSocket):
        """
        Handle disconnect of a WebSocket connection.

        Args:
            websocket: WebSocket connection
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

            # Remove from subscriptions
            for channel in self.subscriptions:
                if websocket in self.subscriptions[channel]:
                    self.subscriptions[channel].remove(websocket)

            # Remove client info
            if websocket in self.client_info:
                del self.client_info[websocket]

            logger.info(f"Client disconnected - Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """
        Send a message to a specific client.

        Args:
            message: Message to send
            websocket: WebSocket connection
        """
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending personal message: {e}")
                # Connection might be broken, disconnect
                self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any], channel: str = None):
        """
        Broadcast a message to all subscribed clients.

        Args:
            message: Message to broadcast
            channel: Channel to broadcast to (None for all clients)
        """
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        # Add channel if not present
        if channel and "channel" not in message:
            message["channel"] = channel

        # Determine recipients
        recipients = []
        if channel and channel in self.subscriptions:
            recipients = list(self.subscriptions[channel])
        else:
            recipients = self.active_connections

        if not recipients:
            return

        # Send to all recipients
        for connection in recipients:
            try:
                await self.send_personal_message(message, connection)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                # Connection error, will be handled elsewhere

    def subscribe(self, websocket: WebSocket, channels: List[str]):
        """
        Subscribe a client to channels.

        Args:
            websocket: WebSocket connection
            channels: List of channels to subscribe to
        """
        if websocket not in self.active_connections:
            return

        # Add channels to client subscriptions
        client_subs = self.client_info[websocket]["subscriptions"]

        for channel in channels:
            # Create channel if it doesn't exist
            if channel not in self.subscriptions:
                self.subscriptions[channel] = set()

            # Add client to channel
            self.subscriptions[channel].add(websocket)
            client_subs.add(channel)

        logger.debug(f"Client {self.client_info[websocket]['client_id']} subscribed to {channels}")

    def unsubscribe(self, websocket: WebSocket, channels: List[str]):
        """
        Unsubscribe a client from channels.

        Args:
            websocket: WebSocket connection
            channels: List of channels to unsubscribe from
        """
        if websocket not in self.active_connections:
            return

        # Remove channels from client subscriptions
        client_subs = self.client_info[websocket]["subscriptions"]

        for channel in channels:
            if channel in self.subscriptions and websocket in self.subscriptions[channel]:
                self.subscriptions[channel].remove(websocket)
                client_subs.discard(channel)

        logger.debug(
            f"Client {self.client_info[websocket]['client_id']} unsubscribed from {channels}"
        )

    def update_client_ping(self, websocket: WebSocket):
        """
        Update client's last ping time.

        Args:
            websocket: WebSocket connection
        """
        if websocket in self.client_info:
            self.client_info[websocket]["last_ping"] = datetime.now().timestamp()

    async def handle_message(self, websocket: WebSocket, data: Dict[str, Any]):
        """
        Handle a message from a client.

        Args:
            websocket: WebSocket connection
            data: Message data
        """
        if "type" not in data:
            return

        message_type = data["type"]

        if message_type == "ping":
            # Handle ping
            self.update_client_ping(websocket)
            await self.send_personal_message(
                {"type": "pong", "timestamp": datetime.now().isoformat()}, websocket
            )

        elif message_type == "subscribe":
            # Handle subscription
            if "channels" in data and isinstance(data["channels"], list):
                self.subscribe(websocket, data["channels"])
                await self.send_personal_message(
                    {
                        "type": "subscribed",
                        "channels": list(self.client_info[websocket]["subscriptions"]),
                        "timestamp": datetime.now().isoformat(),
                    },
                    websocket,
                )

        elif message_type == "unsubscribe":
            # Handle unsubscription
            if "channels" in data and isinstance(data["channels"], list):
                self.unsubscribe(websocket, data["channels"])
                await self.send_personal_message(
                    {
                        "type": "unsubscribed",
                        "channels": list(self.client_info[websocket]["subscriptions"]),
                        "timestamp": datetime.now().isoformat(),
                    },
                    websocket,
                )

        elif message_type == "get_subscriptions":
            # Return current subscriptions
            await self.send_personal_message(
                {
                    "type": "subscriptions",
                    "channels": list(self.client_info[websocket]["subscriptions"]),
                    "timestamp": datetime.now().isoformat(),
                },
                websocket,
            )


# Create global connection manager
manager = ConnectionManager()


# Background task to monitor Redis for events
async def redis_subscriber():
    """Subscribe to Redis pubsub for real-time events."""
    logger.info("Starting Redis subscriber")

    try:
        pubsub = redis_client._conn.pubsub()

        # Subscribe to channels
        channels = [
            "events:system_status",
            "events:market_status",
            "events:trade",
            "events:position",
            "events:watchlist",
        ]

        pubsub.subscribe(*channels)

        # Listen for messages
        for message in pubsub.listen():
            if message["type"] == "message":
                channel = message["channel"].decode("utf-8")
                data = json.loads(message["data"].decode("utf-8"))

                # Map Redis channel to WebSocket channel
                ws_channel = channel.replace("events:", "")

                # Broadcast to clients
                await manager.broadcast(data, ws_channel)
    except Exception as e:
        logger.error(f"Error in Redis subscriber: {e}")
    finally:
        logger.info("Redis subscriber stopped")


# Heartbeat task to check client connections
async def heartbeat_monitor():
    """Monitor client connections and send heartbeats."""
    heartbeat_interval = settings.server.ws_ping_interval or 30  # seconds

    while True:
        try:
            current_time = datetime.now().timestamp()

            # Check each connection
            for conn in list(
                manager.active_connections
            ):  # Copy to avoid modification during iteration
                if conn not in manager.client_info:
                    # Connection is in active_connections but not in client_info, remove it
                    manager.disconnect(conn)
                    continue

                last_ping = manager.client_info[conn]["last_ping"]

                # If no ping received in 3x the interval, disconnect
                if current_time - last_ping > heartbeat_interval * 3:
                    logger.warning(f"Client {manager.client_info[conn]['client_id']} timed out")
                    manager.disconnect(conn)
                    continue

                # Send heartbeat
                try:
                    await manager.send_personal_message(
                        {"type": "heartbeat", "timestamp": datetime.now().isoformat()}, conn
                    )
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    manager.disconnect(conn)

            # Wait for next interval
            await asyncio.sleep(heartbeat_interval)

        except Exception as e:
            logger.error(f"Error in heartbeat monitor: {e}")
            await asyncio.sleep(heartbeat_interval)


# Start background tasks
async def start_background_tasks():
    """Start WebSocket background tasks."""
    # Create tasks
    redis_task = asyncio.create_task(redis_subscriber())
    heartbeat_task = asyncio.create_task(heartbeat_monitor())

    # Return tasks for proper shutdown
    return [redis_task, heartbeat_task]


# WebSocket endpoint
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Args:
        websocket: WebSocket connection
    """
    await manager.connect(websocket)

    try:
        # Handle messages
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Handle message
            await manager.handle_message(websocket, data)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Utility functions for broadcasting system events


async def broadcast_system_status(status: str, state: Dict[str, Any]):
    """
    Broadcast system status update.

    Args:
        status: System status
        state: System state
    """
    await manager.broadcast(
        {
            "type": "system_status",
            "status": status,
            "state": state,
            "timestamp": datetime.now().isoformat(),
        },
        "system_status",
    )


async def broadcast_market_status(status: str, data: Dict[str, Any]):
    """
    Broadcast market status update.

    Args:
        status: Market status
        data: Market data
    """
    await manager.broadcast(
        {
            "type": "market_status",
            "status": status,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        },
        "market_status",
    )


async def broadcast_watchlist_update(
    watchlist: List[str], added: List[str] = None, removed: List[str] = None
):
    """
    Broadcast watchlist update.

    Args:
        watchlist: Current watchlist
        added: Added symbols
        removed: Removed symbols
    """
    await manager.broadcast(
        {
            "type": "watchlist_update",
            "watchlist": watchlist,
            "added": added or [],
            "removed": removed or [],
            "timestamp": datetime.now().isoformat(),
        },
        "watchlist",
    )


async def broadcast_position_update(symbol: str, position: Dict[str, Any], action: str = "update"):
    """
    Broadcast position update.

    Args:
        symbol: Stock symbol
        position: Position data
        action: Update action ("update", "open", "close")
    """
    await manager.broadcast(
        {
            "type": "position_update",
            "symbol": symbol,
            "position": position,
            "action": action,
            "timestamp": datetime.now().isoformat(),
        },
        "positions",
    )


async def broadcast_trade_signal(symbol: str, signal_type: str, data: Dict[str, Any]):
    """
    Broadcast trade signal.

    Args:
        symbol: Stock symbol
        signal_type: Signal type ("entry", "exit")
        data: Signal data
    """
    await manager.broadcast(
        {
            "type": "trade_signal",
            "symbol": symbol,
            "signal_type": signal_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        },
        "signals",
    )


async def broadcast_price_update(symbol: str, price_data: Dict[str, Any]):
    """
    Broadcast price update.

    Args:
        symbol: Stock symbol
        price_data: Price data
    """
    await manager.broadcast(
        {
            "type": "price_update",
            "symbol": symbol,
            "price": price_data,
            "timestamp": datetime.now().isoformat(),
        },
        f"price:{symbol}",
    )


async def broadcast_trade_execution(symbol: str, trade_data: Dict[str, Any]):
    """
    Broadcast trade execution.

    Args:
        symbol: Stock symbol
        trade_data: Trade data
    """
    await manager.broadcast(
        {
            "type": "trade_execution",
            "symbol": symbol,
            "trade": trade_data,
            "timestamp": datetime.now().isoformat(),
        },
        "trades",
    )


async def broadcast_error(error_message: str, error_code: str = "system_error"):
    """
    Broadcast error message.

    Args:
        error_message: Error message
        error_code: Error code
    """
    await manager.broadcast(
        {
            "type": "error",
            "message": error_message,
            "code": error_code,
            "timestamp": datetime.now().isoformat(),
        }
    )
