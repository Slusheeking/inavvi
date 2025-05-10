"""
Dashboard application for trading system monitoring.
"""

import json
from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

# Use absolute imports which work in all contexts
from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client

# Set up logger
logger = setup_logger("dashboard")


# Define configure_app function before it's used
def configure_app(app):
    """
    Configure the Dash application with layout and callbacks.

    Args:
        app: Dash application instance
    """
    # ---------- Page Layout ----------
    app.layout = dbc.Container(
        [
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2(f"{settings.app_name} Dashboard", className="mb-3 mt-3"),
                            html.Div(id="system-status", className="mb-3"),
                        ],
                        width=12,
                    )
                ]
            ),
            # Tabs
            dbc.Tabs(
                [
                    # Overview Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    # System Stats
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader("System Status"),
                                                    dbc.CardBody(id="system-stats-card"),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Market Stats
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader("Market Stats"),
                                                    dbc.CardBody(id="market-stats-card"),
                                                ],
                                                className="mb-3",
                                            ),
                                            # Controls
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader("Controls"),
                                                    dbc.CardBody(
                                                        [
                                                            dbc.Button(
                                                                "Start System",
                                                                id="start-button",
                                                                color="success",
                                                                className="me-2",
                                                            ),
                                                            dbc.Button(
                                                                "Stop System",
                                                                id="stop-button",
                                                                color="danger",
                                                                className="me-2",
                                                            ),
                                                            dbc.Button(
                                                                "Restart System",
                                                                id="restart-button",
                                                                color="warning",
                                                            ),
                                                        ]
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                        ],
                                        width=4,
                                    ),
                                    # Active Positions
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader("Active Positions"),
                                                    dbc.CardBody(id="positions-card"),
                                                ],
                                                className="mb-3 h-100",
                                            ),
                                        ],
                                        width=8,
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    # Watchlist
                                    dbc.Col(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader("Watchlist"),
                                                    dbc.CardBody(id="watchlist-card"),
                                                ],
                                                className="mb-3",
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Overview",
                        tabClassName="text-success",
                    ),
                    # Charts Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("Stock Charts", className="mb-3"),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Symbol"),
                                                    dbc.Input(
                                                        id="chart-symbol-input",
                                                        type="text",
                                                        placeholder="Enter symbol (e.g., AAPL)",
                                                    ),
                                                    dbc.Button(
                                                        "Load Chart",
                                                        id="load-chart-button",
                                                        color="primary",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            dcc.Graph(id="stock-chart", style={"height": "600px"}),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Charts",
                        tabClassName="text-primary",
                    ),
                    # Candidates Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("Trading Candidates", className="mb-3"),
                                            html.Div(id="candidates-table"),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Candidates",
                        tabClassName="text-warning",
                    ),
                    # Logs Tab
                    dbc.Tab(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.H4("System Logs", className="mb-3"),
                                            dbc.InputGroup(
                                                [
                                                    dbc.InputGroupText("Log Level"),
                                                    dbc.Select(
                                                        id="log-level-select",
                                                        options=[
                                                            {"label": "INFO", "value": "INFO"},
                                                            {
                                                                "label": "WARNING",
                                                                "value": "WARNING",
                                                            },
                                                            {"label": "ERROR", "value": "ERROR"},
                                                            {"label": "DEBUG", "value": "DEBUG"},
                                                        ],
                                                        value="INFO",
                                                    ),
                                                    dbc.Button(
                                                        "Refresh Logs",
                                                        id="refresh-logs-button",
                                                        color="primary",
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            html.Div(
                                                id="logs-container",
                                                style={"maxHeight": "600px", "overflow": "auto"},
                                            ),
                                        ],
                                        width=12,
                                    ),
                                ]
                            ),
                        ],
                        label="Logs",
                        tabClassName="text-danger",
                    ),
                ]
            ),
            # Footer
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Hr(),
                            html.P(
                                f"© {datetime.now().year} {settings.app_name} v{settings.version}",
                                className="text-center",
                            ),
                        ],
                        width=12,
                    )
                ]
            ),
            # Refresh Interval
            dcc.Interval(
                id="refresh-interval",
                interval=5000,  # 5 seconds
                n_intervals=0,
            ),
            # Store for WebSocket data
            dcc.Store(id="websocket-data", storage_type="memory"),
            # Store for session data
            dcc.Store(id="session-data", storage_type="session"),
        ],
        fluid=True,
    )


def create_dashboard_app():
    """
    Create and configure the dashboard application.

    Returns:
        dash.Dash: Configured Dash application instance
    """
    # Initialize Dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title=f"{settings.app_name} Dashboard",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    # Configure app layout and callbacks
    configure_app(app)

    return app


# Initialize Dash app for direct run, only if this file is run directly
# This prevents the app from being initialized during imports
app = None
if __name__ == "__main__":
    app = create_dashboard_app()
else:
    # Create a simple app instance for imports
    app = dash.Dash(__name__, suppress_callback_exceptions=True)

# ---------- Callbacks ----------


@callback(
    [
        Output("system-status", "children"),
        Output("system-stats-card", "children"),
        Output("market-stats-card", "children"),
        Output("positions-card", "children"),
        Output("watchlist-card", "children"),
    ],
    [
        Input("refresh-interval", "n_intervals"),
    ],
)
def update_dashboard(n_intervals):
    """Update dashboard components."""
    try:
        # Get system state
        system_state = redis_client.get_system_state()
        state = system_state.get("state", "unknown")

        # System status component
        if state == "running":
            status_component = dbc.Alert("System is running", color="success")
        elif state == "stopped":
            status_component = dbc.Alert("System is stopped", color="danger")
        elif state == "error":
            status_component = dbc.Alert(
                f"System error: {system_state.get('error', 'Unknown error')}", color="danger"
            )
        else:
            status_component = dbc.Alert(f"System state: {state}", color="warning")

        # System stats component
        system_stats = [
            html.P(f"State: {state}"),
            html.P(f"Last update: {system_state.get('timestamp', 'Unknown')}"),
        ]

        # Get market status
        market_status = redis_client.get("market:status")

        # Market stats component
        if market_status:
            market_state = market_status.get("market", "unknown")
            server_time = market_status.get("server_time", "Unknown")

            market_stats = [
                html.P(f"Market: {market_state}"),
                html.P(f"Server time: {server_time}"),
            ]

            # Add exchange info if available
            exchanges = market_status.get("exchanges", {})
            if exchanges:
                exchange_list = []
                for exchange_name, exchange_data in exchanges.items():
                    status = exchange_data.get("status", "unknown")
                    color = (
                        "success"
                        if status == "open"
                        else "danger"
                        if status == "closed"
                        else "warning"
                    )
                    exchange_list.append(
                        html.Li(f"{exchange_name}: ", html.Span(status, className=f"text-{color}"))
                    )

                if exchange_list:
                    market_stats.append(html.P("Exchanges:"))
                    market_stats.append(html.Ul(exchange_list))
        else:
            market_stats = [html.P("Market status not available")]

        # Get active positions
        positions = redis_client.get_all_active_positions()

        # Positions component
        if positions:
            position_rows = []
            for symbol, position_data in positions.items():
                entry_price = position_data.get("entry_price", 0)
                current_price = position_data.get("current_price", 0)
                quantity = position_data.get("quantity", 0)
                side = position_data.get("side", "long")
                pnl = position_data.get("unrealized_pnl", 0)
                pnl_pct = position_data.get("unrealized_pnl_pct", 0)

                # Determine text color based on P&L
                pnl_color = "success" if pnl >= 0 else "danger"

                # Create position card
                position_card = dbc.Card(
                    [
                        dbc.CardHeader(symbol, className="fw-bold"),
                        dbc.CardBody(
                            [
                                html.P(
                                    [
                                        html.Span("Side: "),
                                        html.Span(side.upper(), className="fw-bold text-info"),
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Span("Quantity: "),
                                        html.Span(f"{quantity:,}", className="fw-bold"),
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Span("Entry: "),
                                        html.Span(f"${entry_price:.2f}", className="fw-bold"),
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Span("Current: "),
                                        html.Span(f"${current_price:.2f}", className="fw-bold"),
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Span("P&L: "),
                                        html.Span(
                                            f"${pnl:.2f} ({pnl_pct:.2f}%)",
                                            className=f"fw-bold text-{pnl_color}",
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    className="mb-2",
                )

                position_rows.append(position_card)

            if position_rows:
                positions_component = position_rows
            else:
                positions_component = html.P("No active positions")
        else:
            positions_component = html.P("No active positions")

        # Get watchlist
        watchlist = redis_client.get_watchlist()

        # Watchlist component
        if watchlist:
            # Create table header
            table_header = [
                html.Thead(
                    html.Tr(
                        [
                            html.Th("Symbol"),
                            html.Th("Last Price"),
                            html.Th("Change"),
                            html.Th("Volume"),
                            html.Th("Actions"),
                        ]
                    )
                )
            ]

            # Create table rows
            table_rows = []
            for symbol in watchlist:
                # Get price data
                price_data = redis_client.get_stock_data(symbol, "price")

                if price_data and "price" in price_data:
                    price = price_data["price"]
                    last = price.get("last", 0)
                    open_price = price.get("open", 0)
                    change = last - open_price
                    change_pct = (change / open_price * 100) if open_price else 0
                    volume = price.get("volume", 0)

                    # Determine text color based on change
                    change_color = "success" if change >= 0 else "danger"

                    # Create row
                    row = html.Tr(
                        [
                            html.Td(symbol, className="fw-bold"),
                            html.Td(f"${last:.2f}"),
                            html.Td(
                                [
                                    html.Span(
                                        f"{change:.2f} ({change_pct:.2f}%)",
                                        className=f"text-{change_color}",
                                    ),
                                ]
                            ),
                            html.Td(f"{volume:,}"),
                            html.Td(
                                [
                                    dbc.Button(
                                        "Chart",
                                        color="primary",
                                        size="sm",
                                        className="me-1",
                                        id={"type": "chart-button", "index": symbol},
                                    ),
                                    dbc.Button(
                                        "Trade",
                                        color="success",
                                        size="sm",
                                        id={"type": "trade-button", "index": symbol},
                                    ),
                                ]
                            ),
                        ]
                    )

                    table_rows.append(row)

            # Create table
            if table_rows:
                watchlist_component = dbc.Table(
                    table_header + [html.Tbody(table_rows)],
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                )
            else:
                watchlist_component = html.P("No watchlist data available")
        else:
            watchlist_component = html.P("Watchlist is empty")

        return (
            status_component,
            system_stats,
            market_stats,
            positions_component,
            watchlist_component,
        )
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
        return (
            dbc.Alert("Error updating dashboard", color="danger"),
            html.P(f"Error: {str(e)}"),
            html.P("Market data not available"),
            html.P("Position data not available"),
            html.P("Watchlist data not available"),
        )


@callback(Output("candidates-table", "children"), [Input("refresh-interval", "n_intervals")])
def update_candidates_table(n_intervals):
    """Update candidates table."""
    try:
        # Get ranked candidates
        candidates = redis_client.get_ranked_candidates()

        if not candidates:
            return html.P("No trading candidates available")

        # Create table header
        table_header = [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Rank"),
                        html.Th("Symbol"),
                        html.Th("Score"),
                        html.Th("Price"),
                        html.Th("Last Update"),
                        html.Th("Actions"),
                    ]
                )
            )
        ]

        # Create table rows
        table_rows = []
        for i, candidate in enumerate(candidates[:20]):  # Show top 20
            symbol = candidate.get("symbol", "")
            score = candidate.get("score", 0)
            price = candidate.get("price", 0)
            timestamp = candidate.get("timestamp", "")

            # Create row
            row = html.Tr(
                [
                    html.Td(i + 1, className="fw-bold"),
                    html.Td(symbol, className="fw-bold"),
                    html.Td(f"{score:.2f}"),
                    html.Td(f"${price:.2f}"),
                    html.Td(timestamp),
                    html.Td(
                        [
                            dbc.Button(
                                "Chart",
                                color="primary",
                                size="sm",
                                className="me-1",
                                id={"type": "chart-button", "index": symbol},
                            ),
                            dbc.Button(
                                "Trade",
                                color="success",
                                size="sm",
                                id={"type": "trade-button", "index": symbol},
                            ),
                        ]
                    ),
                ]
            )

            table_rows.append(row)

        # Create table
        if table_rows:
            return dbc.Table(
                table_header + [html.Tbody(table_rows)],
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
            )
        else:
            return html.P("No candidate data available")
    except Exception as e:
        logger.error(f"Error updating candidates table: {e}")
        return html.P(f"Error: {str(e)}")


@callback(
    Output("logs-container", "children"),
    [
        Input("refresh-logs-button", "n_clicks"),
        Input("log-level-select", "value"),
    ],
    prevent_initial_call=True,
)
def update_logs(n_clicks, log_level):
    """Update logs container."""
    if not n_clicks:
        raise PreventUpdate

    try:
        # Read logs from file
        log_file = f"{settings.logging.log_dir}/trading_system.log"

        try:
            with open(log_file, "r") as f:
                log_lines = f.readlines()
        except FileNotFoundError:
            return html.P("Log file not found")

        # Filter logs by level
        filtered_logs = []
        for line in log_lines:
            if f"| {log_level}" in line:
                filtered_logs.append(line)

        # Limit to last 100 lines
        filtered_logs = filtered_logs[-100:]

        # Determine log colors
        log_colors = {
            "DEBUG": "text-info",
            "INFO": "text-light",
            "WARNING": "text-warning",
            "ERROR": "text-danger",
            "CRITICAL": "bg-danger text-white",
        }

        # Format logs
        formatted_logs = []
        for line in filtered_logs:
            # Determine log level
            level = None
            for lvl in log_colors.keys():
                if f"| {lvl}" in line:
                    level = lvl
                    break

            # Apply color
            if level:
                color_class = log_colors.get(level, "text-light")
                formatted_logs.append(html.Pre(line, className=color_class))
            else:
                formatted_logs.append(html.Pre(line))

        return formatted_logs
    except Exception as e:
        logger.error(f"Error updating logs: {e}")
        return html.P(f"Error: {str(e)}")


@callback(
    Output("stock-chart", "figure"),
    [
        Input("load-chart-button", "n_clicks"),
        Input({"type": "chart-button", "index": dash.ALL}, "n_clicks"),
    ],
    [
        State("chart-symbol-input", "value"),
        State({"type": "chart-button", "index": dash.ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def update_stock_chart(n_clicks, chart_buttons, symbol_input, button_ids):
    """Update stock chart."""
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    try:
        # Determine which button was clicked
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]

        if button_id == "load-chart-button":
            symbol = symbol_input
        else:
            # Extract symbol from button id
            button_data = json.loads(button_id)
            symbol = button_data["index"]

        if not symbol:
            raise PreventUpdate

        # Fetch historical data
        ticker_data = None

        # Try to get from Redis cache first
        redis_key = f"stocks:history:{symbol}:1mo:1d"
        ticker_data = redis_client.get(redis_key)

        if ticker_data is None or ticker_data.empty:
            # Data not in cache, show empty chart
            fig = go.Figure()
            fig.update_layout(
                title=f"Chart for {symbol} (No data available)",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
            )
            return fig

        # Create OHLC chart
        fig = go.Figure()

        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=ticker_data.index,
                open=ticker_data["Open"],
                high=ticker_data["High"],
                low=ticker_data["Low"],
                close=ticker_data["Close"],
                name="OHLC",
            )
        )

        # Add volume trace as bar chart on secondary y-axis
        fig.add_trace(
            go.Bar(
                x=ticker_data.index,
                y=ticker_data["Volume"],
                name="Volume",
                yaxis="y2",
                opacity=0.3,
            )
        )

        # Set up the layout
        fig.update_layout(
            title=f"Chart for {symbol}",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )

        return fig
    except Exception as e:
        logger.error(f"Error updating stock chart: {e}")

        # Create empty chart with error message
        fig = go.Figure()
        fig.update_layout(
            title=f"Error: {str(e)}",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_dark",
        )

        return fig


@callback(
    Output("system-status", "children", allow_duplicate=True),
    [
        Input("start-button", "n_clicks"),
        Input("stop-button", "n_clicks"),
        Input("restart-button", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def handle_system_controls(start_clicks, stop_clicks, restart_clicks):
    """Handle system control buttons."""
    ctx = dash.callback_context

    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    try:
        if button_id == "start-button":
            # Call API to start system
            import requests

            response = requests.post("http://localhost:8000/start")
            response_data = response.json()

            status = response_data.get("status", "error")
            message = response_data.get("message", "Unknown error")

            if status == "started":
                return dbc.Alert(message, color="success")
            else:
                return dbc.Alert(f"Error: {message}", color="danger")

        elif button_id == "stop-button":
            # Call API to stop system
            import requests

            response = requests.post("http://localhost:8000/stop")
            response_data = response.json()

            status = response_data.get("status", "error")
            message = response_data.get("message", "Unknown error")

            if status == "stopped":
                return dbc.Alert(message, color="warning")
            else:
                return dbc.Alert(f"Error: {message}", color="danger")

        elif button_id == "restart-button":
            # Call API to restart system
            import requests

            response = requests.post("http://localhost:8000/restart")
            response_data = response.json()

            status = response_data.get("status", "error")
            message = response_data.get("message", "Unknown error")

            if status == "restarted":
                return dbc.Alert(message, color="success")
            else:
                return dbc.Alert(f"Error: {message}", color="danger")
    except Exception as e:
        logger.error(f"Error handling system controls: {e}")
        return dbc.Alert(f"Error: {str(e)}", color="danger")


# ---------- Run Server ----------

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
