"""
Dashboard components for the trading system.

This module contains the UI components used in the dashboard:
- Market overview
- Watchlist table
- Portfolio view
- Position cards
- Performance charts
- System status indicators
"""
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

from src.config.settings import settings

# Color schemes
COLORS = {
    'background': '#1E1E1E',
    'card_background': '#2E2E2E',
    'text': '#FFFFFF',
    'profit': '#00CC00',
    'loss': '#FF4444',
    'neutral': '#AAAAAA',
    'highlight': '#007BFF',
    'warning': '#FFD700',
    'danger': '#FF6B6B',
    'chart': {
        'candle_up': '#00CC00',
        'candle_down': '#FF4444',
        'volume_up': 'rgba(0, 204, 0, 0.3)',
        'volume_down': 'rgba(255, 68, 68, 0.3)',
        'ma_20': '#1E90FF',
        'ma_50': '#FF8C00',
        'ma_200': '#FF00FF',
        'grid': '#333333',
        'legend': '#CCCCCC'
    }
}

def create_header():
    """
    Create dashboard header with logo and system status.
    
    Returns:
        Header component
    """
    return html.Div([
        dbc.Row([
            dbc.Col(html.H1("Day Trading System", className="display-4 text-white"), width=6),
            dbc.Col([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Span("Status: ", className="font-weight-bold"),
                        html.Span("Running", id="system-status", 
                                 className="badge bg-success")
                    ]), width=6),
                    dbc.Col(html.Div([
                        html.Span("Mode: ", className="font-weight-bold"),
                        html.Span(settings.trading.mode.capitalize(), id="trading-mode",
                                 className="badge bg-info")
                    ]), width=6)
                ]),
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Span("Market: ", className="font-weight-bold"),
                        html.Span("Open", id="market-status", 
                                 className="badge bg-success")
                    ]), width=6),
                    dbc.Col(html.Div([
                        html.Span("Last Update: ", className="font-weight-bold"),
                        html.Span(datetime.now().strftime("%H:%M:%S"), id="last-update",
                                 className="text-muted small")
                    ]), width=6)
                ])
            ], width=6)
        ]),
        html.Hr(className="bg-secondary")
    ], className="mt-3 mb-4")

def create_market_overview():
    """
    Create market overview component with major indices and sectors.
    
    Returns:
        Market overview component
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Market Overview", className="text-white")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Major Indices", className="text-white-50"),
                    html.Div(id="indices-table", className="table-responsive")
                ], width=6),
                dbc.Col([
                    html.H6("Sector Performance", className="text-white-50"),
                    dcc.Graph(
                        id="sector-performance-chart",
                        config={'displayModeBar': False},
                        style={"height": "200px"}
                    )
                ], width=6)
            ])
        ])
    ], className="mb-4", style={"backgroundColor": COLORS['card_background']})

def create_watchlist_table():
    """
    Create watchlist table component with sorting and selection.
    
    Returns:
        Watchlist table component
    """
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H5("Watchlist", className="text-white"), width=6),
                dbc.Col([
                    dbc.Button("Refresh", id="refresh-watchlist-btn", color="primary", size="sm", className="mr-2"),
                    dbc.Button("Analyze Selected", id="analyze-selected-btn", color="success", size="sm", disabled=True)
                ], width=6, className="text-right")
            ])
        ]),
        dbc.CardBody([
            html.Div(id="watchlist-table", className="table-responsive")
        ])
    ], className="mb-4", style={"backgroundColor": COLORS['card_background']})

def create_portfolio_summary():
    """
    Create portfolio summary component with key metrics.
    
    Returns:
        Portfolio summary component
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Portfolio", className="text-white")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Cash Available", className="text-white-50"),
                        html.H3("$5,000.00", id="available-cash", className="text-white")
                    ], className="text-center mb-3")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H6("Today's P&L", className="text-white-50"),
                        html.H3("$0.00", id="daily-pnl", className="text-white")
                    ], className="text-center mb-3")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H6("Active Positions", className="text-white-50"),
                        html.H3("0 / 3", id="position-count", className="text-white")
                    ], className="text-center mb-3")
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.H6("Win Rate", className="text-white-50"),
                        html.H3("0.0%", id="win-rate", className="text-white")
                    ], className="text-center mb-3")
                ], width=3)
            ]),
            html.Hr(className="bg-secondary"),
            dbc.Row([
                dbc.Col([
                    html.H6("Recent Performance", className="text-white-50"),
                    dcc.Graph(
                        id="performance-chart",
                        config={'displayModeBar': False},
                        style={"height": "200px"}
                    )
                ])
            ])
        ])
    ], className="mb-4", style={"backgroundColor": COLORS['card_background']})

def create_position_card(position):
    """
    Create a card for a single position.
    
    Args:
        position: Position data
        
    Returns:
        Position card component
    """
    # Extract position details
    symbol = position.get('symbol', 'UNKNOWN')
    entry_price = position.get('entry_price', 0)
    current_price = position.get('current_price', 0)
    quantity = position.get('quantity', 0)
    
    # Calculate P&L
    unrealized_pnl = position.get('unrealized_pnl', 0)
    unrealized_pnl_pct = position.get('unrealized_pnl_pct', 0)
    
    # Set color based on P&L
    pnl_color = COLORS['profit'] if unrealized_pnl >= 0 else COLORS['loss']
    
    # Format data
    entry_price_str = f"${entry_price:.2f}"
    current_price_str = f"${current_price:.2f}"
    quantity_str = f"{quantity} shares"
    pnl_str = f"${unrealized_pnl:.2f}"
    pnl_pct_str = f"{unrealized_pnl_pct:.2f}%"
    
    # Create card
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H5(symbol, className="text-white"), width=6),
                dbc.Col([
                    html.Span(pnl_str, style={"color": pnl_color, "fontWeight": "bold", "marginRight": "8px"}),
                    html.Span(pnl_pct_str, style={"color": pnl_color})
                ], width=6, className="text-right")
            ])
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Entry: ", className="text-white-50"),
                        html.Span(entry_price_str, className="text-white")
                    ])
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("Current: ", className="text-white-50"),
                        html.Span(current_price_str, className="text-white")
                    ])
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Quantity: ", className="text-white-50"),
                        html.Span(quantity_str, className="text-white")
                    ])
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.Span("Side: ", className="text-white-50"),
                        html.Span("Long", className="text-white")
                    ])
                ], width=6)
            ]),
            html.Hr(className="bg-secondary"),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Exit Position", id=f"exit-btn-{symbol}", color="danger", size="sm", className="w-100")
                ], width=12)
            ])
        ])
    ], className="mb-3", style={"backgroundColor": COLORS['card_background']})

def create_positions_grid():
    """
    Create positions grid with all active positions.
    
    Returns:
        Positions grid component
    """
    return html.Div([
        html.H5("Active Positions", className="text-white mb-3"),
        html.Div(id="positions-grid", className="row")
    ], className="mb-4")

def create_trades_table():
    """
    Create trades history table.
    
    Returns:
        Trades table component
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Trade History", className="text-white")),
        dbc.CardBody([
            html.Div(id="trades-table", className="table-responsive")
        ])
    ], className="mb-4", style={"backgroundColor": COLORS['card_background']})

def create_chart_card():
    """
    Create chart card for selected symbol.
    
    Returns:
        Chart card component
    """
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H5("Chart", id="chart-title", className="text-white"), width=6),
                dbc.Col([
                    dbc.Select(
                        id="timeframe-selector",
                        options=[
                            {"label": "1 Minute", "value": "1m"},
                            {"label": "5 Minutes", "value": "5m"},
                            {"label": "15 Minutes", "value": "15m"},
                            {"label": "1 Hour", "value": "1h"},
                            {"label": "1 Day", "value": "1d"}
                        ],
                        value="5m",
                        className="form-select-sm"
                    )
                ], width=6, className="text-right")
            ])
        ]),
        dbc.CardBody([
            dcc.Graph(
                id="price-chart",
                config={'displayModeBar': True},
                style={"height": "500px"}
            )
        ])
    ], className="mb-4", style={"backgroundColor": COLORS['card_background']})

def create_system_logs():
    """
    Create system logs viewer.
    
    Returns:
        System logs component
    """
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H5("System Logs", className="text-white"), width=6),
                dbc.Col([
                    dbc.Select(
                        id="log-level-selector",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Info", "value": "info"},
                            {"label": "Warning", "value": "warning"},
                            {"label": "Error", "value": "error"}
                        ],
                        value="info",
                        className="form-select-sm"
                    )
                ], width=6, className="text-right")
            ])
        ]),
        dbc.CardBody([
            html.Pre(id="log-content", className="text-white-50 small", style={"height": "300px", "overflow": "auto"})
        ])
    ], className="mb-4", style={"backgroundColor": COLORS['card_background']})

def create_control_panel():
    """
    Create control panel with system controls.
    
    Returns:
        Control panel component
    """
    return dbc.Card([
        dbc.CardHeader(html.H5("Control Panel", className="text-white")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Button("Start Trading", id="start-btn", color="success", className="w-100 mb-2"),
                    dbc.Button("Stop Trading", id="stop-btn", color="danger", className="w-100 mb-2"),
                    dbc.Button("Update Watchlist", id="update-watchlist-btn", color="primary", className="w-100 mb-2"),
                    dbc.Button("Close All Positions", id="close-all-btn", color="warning", className="w-100")
                ], width=6),
                dbc.Col([
                    html.Div([
                        html.H6("System Information", className="text-white-50"),
                        html.Div([
                            html.Div([
                                html.Span("Trading Mode: ", className="text-white-50"),
                                html.Span(settings.trading.mode.capitalize(), className="text-white")
                            ]),
                            html.Div([
                                html.Span("Max Positions: ", className="text-white-50"),
                                html.Span(str(settings.trading.max_positions), className="text-white")
                            ]),
                            html.Div([
                                html.Span("Max Position Size: ", className="text-white-50"),
                                html.Span(f"${settings.trading.max_position_size}", className="text-white")
                            ]),
                            html.Div([
                                html.Span("Max Daily Risk: ", className="text-white-50"),
                                html.Span(f"${settings.trading.max_daily_risk}", className="text-white")
                            ]),
                            html.Div([
                                html.Span("Watchlist Size: ", className="text-white-50"),
                                html.Span(str(settings.trading.watchlist_size), className="text-white")
                            ])
                        ])
                    ])
                ], width=6)
            ])
        ])
    ], className="mb-4", style={"backgroundColor": COLORS['card_background']})

def create_candlestick_chart(df, title="Price Chart"):
    """
    Create a candlestick chart with volume and indicators.
    
    Args:
        df: DataFrame with OHLCV data
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Create subplot with 2 rows (price and volume)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.02, 
                        row_heights=[0.8, 0.2])
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price",
        increasing_line_color=COLORS['chart']['candle_up'],
        decreasing_line_color=COLORS['chart']['candle_down']
    ), row=1, col=1)
    
    # Add volume bars
    colors = [COLORS['chart']['volume_up'] if row['close'] >= row['open'] 
             else COLORS['chart']['volume_down'] for _, row in df.iterrows()]
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name="Volume",
        marker_color=colors
    ), row=2, col=1)
    
    # Add moving averages if we have enough data
    if len(df) >= 20:
        df['MA20'] = df['close'].rolling(window=20).mean()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA20'],
            name="20 MA",
            line=dict(color=COLORS['chart']['ma_20'], width=1)
        ), row=1, col=1)
    
    if len(df) >= 50:
        df['MA50'] = df['close'].rolling(window=50).mean()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MA50'],
            name="50 MA",
            line=dict(color=COLORS['chart']['ma_50'], width=1)
        ), row=1, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color=COLORS['chart']['legend'])
        )
    )
    
    # Update Y-axis
    fig.update_yaxes(
        title_text="Price",
        gridcolor=COLORS['chart']['grid'],
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Volume",
        gridcolor=COLORS['chart']['grid'],
        row=2, col=1
    )
    
    # Update X-axis
    fig.update_xaxes(
        gridcolor=COLORS['chart']['grid'],
        row=1, col=1
    )
    fig.update_xaxes(
        gridcolor=COLORS['chart']['grid'],
        row=2, col=1
    )
    
    return fig

def create_performance_chart(data, title="Performance"):
    """
    Create a performance chart showing daily P&L.
    
    Args:
        data: DataFrame with performance data
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add P&L line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['pnl'],
        mode='lines',
        name="P&L",
        line=dict(
            color=COLORS['highlight'],
            width=2
        ),
        fill='tozeroy',
        fillcolor='rgba(0, 123, 255, 0.1)'
    ))
    
    # Add equity line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['equity'],
        mode='lines',
        name="Equity",
        line=dict(
            color=COLORS['profit'],
            width=2
        )
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=200,
        margin=dict(l=50, r=50, t=50, b=30),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color=COLORS['chart']['legend'])
        )
    )
    
    # Update Y-axis
    fig.update_yaxes(
        title_text="USD",
        gridcolor=COLORS['chart']['grid']
    )
    
    # Update X-axis
    fig.update_xaxes(
        gridcolor=COLORS['chart']['grid']
    )
    
    return fig

def create_sector_performance_chart(data, title="Sector Performance"):
    """
    Create a sector performance chart as a horizontal bar chart.
    
    Args:
        data: Dictionary with sector performance data
        title: Chart title
        
    Returns:
        Plotly figure
    """
    # Convert data to dataframe
    sectors = []
    performances = []
    
    # Extract data from the standard Alpha Vantage format
    if 'Rank A: Real-Time Performance' in data:
        for sector, perf in data['Rank A: Real-Time Performance'].items():
            sectors.append(sector)
            performances.append(float(perf))
    else:
        # Handle alternate format or empty data
        sectors = ['Technology', 'Healthcare', 'Financials', 'Consumer', 'Energy']
        performances = [0, 0, 0, 0, 0]
    
    # Create dataframe
    df = pd.DataFrame({
        'Sector': sectors,
        'Performance': performances
    })
    
    # Sort by performance
    df = df.sort_values('Performance')
    
    # Create colors based on performance
    colors = [COLORS['profit'] if p >= 0 else COLORS['loss'] for p in df['Performance']]
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=df['Performance'],
        y=df['Sector'],
        orientation='h',
        marker_color=colors,
        text=[f"{p:.2f}%" for p in df['Performance']],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        template="plotly_dark",
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        height=200,
        margin=dict(l=50, r=50, t=50, b=30),
        showlegend=False
    )
    
    # Update X-axis
    fig.update_xaxes(
        title_text="Performance (%)",
        gridcolor=COLORS['chart']['grid']
    )
    
    # Update Y-axis
    fig.update_yaxes(
        gridcolor=COLORS['chart']['grid']
    )
    
    return fig

def create_watchlist_table_content(data):
    """
    Create a watchlist table from data.
    
    Args:
        data: List of dictionaries with watchlist data
        
    Returns:
        HTML table
    """
    # Create header
    header = html.Thead(html.Tr([
        html.Th("Symbol", className="text-white"),
        html.Th("Price", className="text-white text-right"),
        html.Th("Change", className="text-white text-right"),
        html.Th("Volume", className="text-white text-right"),
        html.Th("Pattern", className="text-white"),
        html.Th("Score", className="text-white text-right"),
        html.Th("Actions", className="text-white text-center")
    ]))
    
    # Create rows
    rows = []
    for item in data:
        symbol = item.get('symbol', '')
        price = item.get('price', {}).get('last', 0)
        price_str = f"${price:.2f}"
        
        # Calculate change
        open_price = item.get('price', {}).get('open', 0)
        if open_price > 0:
            change = price - open_price
            change_pct = (change / open_price) * 100
            change_str = f"{change_pct:.2f}%"
            change_class = "text-success" if change >= 0 else "text-danger"
        else:
            change_str = "0.00%"
            change_class = ""
        
        # Format volume
        volume = item.get('price', {}).get('volume', 0)
        volume_str = f"{volume:,}"
        
        # Pattern and score
        pattern = item.get('pattern', {}).get('name', 'No Pattern')
        score = item.get('score', 0)
        
        # Create row
        row = html.Tr([
            html.Td(html.A(symbol, href="#", id=f"symbol-link-{symbol}", className="text-white")),
            html.Td(price_str, className="text-right"),
            html.Td(change_str, className=f"text-right {change_class}"),
            html.Td(volume_str, className="text-right"),
            html.Td(pattern),
            html.Td(f"{score:.2f}", className="text-right"),
            html.Td([
                dbc.Button("Trade", id=f"trade-btn-{symbol}", color="success", size="sm", className="mr-1"),
                dbc.Button("Chart", id=f"chart-btn-{symbol}", color="primary", size="sm")
            ], className="text-center")
        ])
        rows.append(row)
    
    # Create table body
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table([header, body], bordered=False, dark=True, hover=True, responsive=True, striped=True)
    
    return table

def create_positions_table_content(data):
    """
    Create a positions table from data.
    
    Args:
        data: Dictionary with position data
        
    Returns:
        HTML table
    """
    # Create header
    header = html.Thead(html.Tr([
        html.Th("Symbol", className="text-white"),
        html.Th("Entry", className="text-white text-right"),
        html.Th("Current", className="text-white text-right"),
        html.Th("Quantity", className="text-white text-right"),
        html.Th("P&L", className="text-white text-right"),
        html.Th("P&L %", className="text-white text-right"),
        html.Th("Actions", className="text-white text-center")
    ]))
    
    # Create rows
    rows = []
    for symbol, position in data.items():
        # Extract data
        entry_price = position.get('entry_price', 0)
        current_price = position.get('current_price', 0)
        quantity = position.get('quantity', 0)
        unrealized_pnl = position.get('unrealized_pnl', 0)
        unrealized_pnl_pct = position.get('unrealized_pnl_pct', 0)
        
        # Format data
        entry_price_str = f"${entry_price:.2f}"
        current_price_str = f"${current_price:.2f}"
        quantity_str = f"{quantity:,}"
        pnl_str = f"${unrealized_pnl:.2f}"
        pnl_pct_str = f"{unrealized_pnl_pct:.2f}%"
        
        # Set color for P&L
        pnl_class = "text-success" if unrealized_pnl >= 0 else "text-danger"
        
        # Create row
        row = html.Tr([
            html.Td(html.A(symbol, href="#", id=f"position-link-{symbol}", className="text-white")),
            html.Td(entry_price_str, className="text-right"),
            html.Td(current_price_str, className="text-right"),
            html.Td(quantity_str, className="text-right"),
            html.Td(pnl_str, className=f"text-right {pnl_class}"),
            html.Td(pnl_pct_str, className=f"text-right {pnl_class}"),
            html.Td([
                dbc.Button("Exit", id=f"exit-pos-btn-{symbol}", color="danger", size="sm", className="mr-1"),
                dbc.Button("Chart", id=f"chart-pos-btn-{symbol}", color="primary", size="sm")
            ], className="text-center")
        ])
        rows.append(row)
    
    # Create table body
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table([header, body], bordered=False, dark=True, hover=True, responsive=True, striped=True)
    
    return table

def create_trades_table_content(data):
    """
    Create a trades history table from data.
    
    Args:
        data: List of dictionaries with trade data
        
    Returns:
        HTML table
    """
    # Create header
    header = html.Thead(html.Tr([
        html.Th("Date/Time", className="text-white"),
        html.Th("Symbol", className="text-white"),
        html.Th("Side", className="text-white"),
        html.Th("Price", className="text-white text-right"),
        html.Th("Quantity", className="text-white text-right"),
        html.Th("P&L", className="text-white text-right"),
        html.Th("Reason", className="text-white")
    ]))
    
    # Create rows
    rows = []
    for trade in data:
        # Extract data
        symbol = trade.get('symbol', '')
        side = trade.get('side', '')
        price = trade.get('price', 0)
        quantity = trade.get('quantity', 0)
        pnl = trade.get('realized_pnl', 0)
        reason = trade.get('reason', '')
        timestamp = trade.get('timestamp', '')
        
        # Format data
        try:
            dt = datetime.fromisoformat(timestamp)
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            timestamp_str = timestamp
        
        price_str = f"${price:.2f}"
        quantity_str = f"{quantity:,}"
        pnl_str = f"${pnl:.2f}" if pnl != 0 else "-"
        
        # Set color for P&L and side
        pnl_class = "text-success" if pnl > 0 else "text-danger" if pnl < 0 else ""
        side_class = "text-success" if side.lower() == 'buy' else "text-danger" if side.lower() == 'sell' else ""
        
        # Create row
        row = html.Tr([
            html.Td(timestamp_str),
            html.Td(symbol),
            html.Td(side.capitalize(), className=side_class),
            html.Td(price_str, className="text-right"),
            html.Td(quantity_str, className="text-right"),
            html.Td(pnl_str, className=f"text-right {pnl_class}"),
            html.Td(reason)
        ])
        rows.append(row)
    
    # Create table body
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table([header, body], bordered=False, dark=True, hover=True, responsive=True, striped=True)
    
    return table

def create_indices_table_content(data):
    """
    Create an indices table from data.
    
    Args:
        data: Dictionary with indices data
        
    Returns:
        HTML table
    """
    # Create header
    header = html.Thead(html.Tr([
        html.Th("Index", className="text-white"),
        html.Th("Price", className="text-white text-right"),
        html.Th("Change", className="text-white text-right"),
        html.Th("Change %", className="text-white text-right")
    ]))
    
    # Create rows
    rows = []
    for index_name, index_data in data.items():
        # Extract data
        price = index_data.get('price', 0)
        change = index_data.get('change', 0)
        change_pct = index_data.get('change_pct', 0)
        
        # Format data
        price_str = f"{price:.2f}"
        change_str = f"{change:.2f}"
        change_pct_str = f"{change_pct:.2f}%"
        
        # Set color for change
        change_class = "text-success" if change >= 0 else "text-danger"
        
        # Create row
        row = html.Tr([
            html.Td(index_name),
            html.Td(price_str, className="text-right"),
            html.Td(change_str, className=f"text-right {change_class}"),
            html.Td(change_pct_str, className=f"text-right {change_class}")
        ])
        rows.append(row)
    
    # Create table body
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table([header, body], bordered=False, dark=True, hover=True, responsive=True, striped=True, size="sm")
    
    return table

def generate_sample_data():
    """
    Generate sample data for dashboard testing.
    
    Returns:
        Dictionary with sample data
    """
    # Generate sample watchlist
    watchlist = []
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']:
        price = np.random.uniform(100, 300)
        open_price = price * np.random.uniform(0.95, 1.05)
        high = max(price, open_price) * np.random.uniform(1.0, 1.05)
        low = min(price, open_price) * np.random.uniform(0.95, 1.0)
        volume = np.random.randint(500000, 5000000)
        
        watchlist.append({
            'symbol': symbol,
            'price': {
                'last': price,
                'open': open_price,
                'high': high,
                'low': low,
                'volume': volume
            },
            'pattern': {
                'name': np.random.choice(['breakout', 'reversal', 'continuation', 'no_pattern']),
                'confidence': np.random.uniform(0.6, 0.9)
            },
            'score': np.random.uniform(0.5, 0.9),
            'timestamp': datetime.now().isoformat()
        })
    
    # Generate sample positions
    positions = {}
    for symbol in ['AAPL', 'MSFT']:
        entry_price = np.random.uniform(100, 300)
        current_price = entry_price * np.random.uniform(0.9, 1.1)
        quantity = np.random.randint(10, 50)
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = (current_price / entry_price - 1) * 100
        
        positions[symbol] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'entry_time': (datetime.now() - timedelta(hours=np.random.randint(1, 5))).isoformat(),
            'current_price': current_price,
            'quantity': quantity,
            'side': 'long',
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'status': 'active'
        }
    
    # Generate sample trades
    trades = []
    for i in range(10):
        symbol = np.random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
        side = np.random.choice(['buy', 'sell'])
        price = np.random.uniform(100, 300)
        quantity = np.random.randint(10, 50)
        pnl = np.random.uniform(-100, 100) if side == 'sell' else 0
        reason = np.random.choice(['market_open', 'take_profit', 'stop_loss', 'trailing_stop', 'exit_signal'])
        timestamp = (datetime.now() - timedelta(hours=np.random.randint(1, 24))).isoformat()
        
        trades.append({
            'symbol': symbol,
            'side': side,
            'price': price,
            'quantity': quantity,
            'realized_pnl': pnl,
            'reason': reason,
            'timestamp': timestamp
        })
    
    # Sort trades by timestamp (newest first)
    trades.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Generate sample indices
    indices = {
        'S&P 500': {
            'price': 4200 + np.random.uniform(-50, 50),
            'change': np.random.uniform(-20, 20),
            'change_pct': np.random.uniform(-0.5, 0.5)
        },
        'Dow Jones': {
            'price': 33000 + np.random.uniform(-300, 300),
            'change': np.random.uniform(-100, 100),
            'change_pct': np.random.uniform(-0.5, 0.5)
        },
        'Nasdaq': {
            'price': 13000 + np.random.uniform(-100, 100),
            'change': np.random.uniform(-50, 50),
            'change_pct': np.random.uniform(-0.5, 0.5)
        },
        'Russell 2000': {
            'price': 2000 + np.random.uniform(-20, 20),
            'change': np.random.uniform(-10, 10),
            'change_pct': np.random.uniform(-0.5, 0.5)
        }
    }
    
    # Generate sample sector performance
    sectors = ['Information Technology', 'Healthcare', 'Financials', 'Consumer Discretionary', 
              'Communication Services', 'Industrials', 'Materials', 'Energy', 
              'Consumer Staples', 'Utilities', 'Real Estate']
    
    sector_performance = {
        'Rank A: Real-Time Performance': {
            sector: f"{np.random.uniform(-2.0, 2.0):.2f}" for sector in sectors
        }
    }
    
    # Generate sample performance data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    performance_data = pd.DataFrame({
        'equity': [5000 + np.random.uniform(-100, 500) for _ in range(30)],
        'pnl': [np.random.uniform(-100, 100) for _ in range(30)]
    }, index=dates)
    
    # Create sample OHLCV data for chart
    chart_dates = pd.date_range(end=datetime.now(), periods=100, freq='5min')
    price = 150.0
    chart_data = []
    
    for _ in range(len(chart_dates)):
        open_price = price * np.random.uniform(0.998, 1.002)
        close = price * np.random.uniform(0.998, 1.002)
        high = max(open_price, close) * np.random.uniform(1.0, 1.003)
        low = min(open_price, close) * np.random.uniform(0.997, 1.0)
        volume = np.random.randint(10000, 100000)
        
        chart_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        price = close  # Update price for next iteration
    
    chart_df = pd.DataFrame(chart_data, index=chart_dates)
    
    # Return all sample data
    return {
        'watchlist': watchlist,
        'positions': positions,
        'trades': trades,
        'indices': indices,
        'sector_performance': sector_performance,
        'performance_data': performance_data,
        'chart_data': chart_df
    }