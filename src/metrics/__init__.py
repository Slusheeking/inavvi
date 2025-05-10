"""
Metrics module for the trading system.

This module provides monitoring and metrics collection capabilities:
- System metrics reporting
- Performance measurement
- API for retrieving metrics
"""

# Handle imports differently when run as a script vs imported as a module
if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Import it as part of the metrics package instead.")
else:
    # These imports will work when the module is imported properly
    from .server import (
        app,
        get_status,
        get_positions,
        get_watchlist,
        get_candidates,
        get_market_info,
        get_performance,
        start_system,
        stop_system,
        restart_system,
        SystemStatus,
        PositionInfo,
        WatchlistItem,
    )

    __all__ = [
        "app",
        "get_status",
        "get_positions",
        "get_watchlist",
        "get_candidates",
        "get_market_info",
        "get_performance",
        "start_system",
        "stop_system",
        "restart_system",
        "SystemStatus",
        "PositionInfo",
        "WatchlistItem",
    ]
