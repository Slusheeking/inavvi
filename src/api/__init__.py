"""
API module for the trading system.

This module contains the REST API endpoints and WebSocket server.
"""

# Add the project root to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.endpoints import router as api_router
from src.api.websocket import ConnectionManager, manager as websocket_manager, websocket_endpoint, start_background_tasks

__all__ = ["api_router", "ConnectionManager", "websocket_manager", "websocket_endpoint", "start_background_tasks"]
