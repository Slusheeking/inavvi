"""
Dashboard module for the trading system.

This module provides web-based visualization and control interface.
"""

from .app import create_dashboard_app, app
from .components import (
    create_header,
    create_market_overview,
    create_watchlist_table,
    create_portfolio_summary,
    create_position_card,
    create_positions_grid,
    create_trades_table,
    create_chart_card,
    create_system_logs,
    create_control_panel,
    create_candlestick_chart,
    create_performance_chart,
    create_sector_performance_chart,
    create_watchlist_table_content,
    create_positions_table_content,
    create_trades_table_content,
    create_indices_table_content,
    generate_sample_data,
)

__all__ = [
    "create_dashboard_app",
    "create_header",
    "create_market_overview",
    "create_watchlist_table",
    "create_portfolio_summary",
    "create_position_card",
    "create_positions_grid",
    "create_trades_table",
    "create_chart_card",
    "create_system_logs",
    "create_control_panel",
    "create_candlestick_chart",
    "create_performance_chart",
    "create_sector_performance_chart",
    "create_watchlist_table_content",
    "create_positions_table_content",
    "create_trades_table_content",
    "create_indices_table_content",
    "generate_sample_data",
]
