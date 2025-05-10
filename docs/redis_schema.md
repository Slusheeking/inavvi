# Redis Schema for Trading System

This document outlines the Redis data structure for the trading system, focusing on efficient data storage and retrieval for real-time trading operations.

## Redis Configuration

Redis has been configured for production use with the following settings:

- **Memory Management**:
  - `maxmemory`: 2GB
  - `maxmemory-policy`: volatile-lru (evicts keys with expiration set when memory limit is reached)

- **Persistence**:
  - `appendonly`: yes (AOF persistence enabled for better durability)
  - `appendfsync`: everysec (good balance between performance and durability)

- **Network**:
  - Listening on: 127.0.0.1:6379 (localhost only for security)
  - Protected mode: enabled

## Data Structure

### 1. Real-time Data

#### Stock Price Data
```
stocks:price:{symbol}
```
Example:
```json
{
  "price": {
    "last": 150.25,
    "open": 149.80,
    "high": 151.20,
    "low": 149.50,
    "volume": 25000000
  },
  "timestamp": "2025-05-09T08:15:30.123Z"
}
```

#### Technical Indicators
```
stocks:indicators:{symbol}
```
Example:
```json
{
  "rsi_14": 65.75,
  "macd": 0.5,
  "macd_signal": 0.3,
  "bollinger_upper": 152.50,
  "bollinger_lower": 148.50,
  "bollinger_width": 4.0,
  "timestamp": "2025-05-09T08:15:30.123Z"
}
```

#### News & Sentiment
```
stocks:sentiment:{symbol}
```
Example:
```json
{
  "sentiment_score": 0.75,
  "sentiment_label": "bullish",
  "news_count": 3,
  "latest_news": [
    {
      "headline": "Company XYZ Reports Strong Earnings",
      "source": "Financial Times",
      "timestamp": "2025-05-09T07:30:00.000Z",
      "url": "https://example.com/news/xyz-earnings"
    }
  ],
  "timestamp": "2025-05-09T08:15:30.123Z"
}
```

### 2. Watchlist & Screening

#### Current Watchlist
```
watchlist:current
```
Example:
```json
["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
```

#### Ranked Candidates
```
candidates:ranked
```
Example:
```json
[
  {
    "symbol": "AAPL",
    "score": 0.95,
    "price": 150.25,
    "signals": ["breakout", "volume_surge"],
    "timestamp": "2025-05-09T08:15:30.123Z"
  },
  {
    "symbol": "MSFT",
    "score": 0.92,
    "price": 290.10,
    "signals": ["support_bounce", "macd_cross"],
    "timestamp": "2025-05-09T08:15:30.123Z"
  }
]
```

### 3. Position Management

#### Active Positions
```
positions:active:{symbol}
```
Example:
```json
{
  "symbol": "AAPL",
  "entry_price": 150.0,
  "current_price": 152.5,
  "quantity": 10,
  "side": "long",
  "entry_time": "2025-05-09T09:45:32.000Z",
  "unrealized_pnl": 25.0,
  "unrealized_pnl_pct": 1.67,
  "stop_loss": 145.0,
  "take_profit": 160.0,
  "last_update": "2025-05-09T10:15:30.123Z"
}
```

#### Position Monitoring Data
```
positions:monitoring:{symbol}
```
Example:
```json
{
  "technical_signals": {
    "rsi_1m": {"value": 78.5, "signal": "overbought"},
    "macd_5m": {"value": 0.15, "signal": "weakening"},
    "support_resistance": {"next_level": 26.50, "type": "resistance"}
  },
  "profit_target": {"value": 26.75, "distance": "1.9%"},
  "stop_loss": {"value": 25.25, "distance": "3.8%"},
  "time_in_trade": "00:27:15",
  "timestamp": "2025-05-09T10:15:30.123Z"
}
```

#### Trading Signals
```
signals:{signal_type}:{symbol}
```
Example (entry signal):
```json
{
  "reason": "breakout",
  "price": 150.25,
  "timestamp": "2025-05-09T09:45:30.123Z",
  "indicators": {
    "macd": 0.5,
    "rsi": 65
  },
  "confidence": 0.85
}
```

Example (exit signal):
```json
{
  "reason": "target_reached",
  "price": 160.0,
  "timestamp": "2025-05-09T11:30:30.123Z",
  "indicators": {
    "macd": 0.2,
    "rsi": 75
  },
  "confidence": 0.9
}
```

### 4. System State

#### System State
```
system:state
```
Example:
```json
{
  "state": "running",
  "market_status": "open",
  "active_positions_count": 2,
  "watchlist_count": 15,
  "last_scan_time": "2025-05-09T10:00:00.000Z",
  "timestamp": "2025-05-09T10:15:30.123Z"
}
```

#### Performance Metrics
```
metrics:performance
```
Example:
```json
{
  "daily_pnl": 125.50,
  "daily_pnl_pct": 2.51,
  "win_count": 3,
  "loss_count": 1,
  "win_rate": 0.75,
  "avg_win": 50.0,
  "avg_loss": -25.0,
  "largest_win": 75.0,
  "largest_loss": -25.0,
  "timestamp": "2025-05-09T16:00:00.000Z"
}
```

### 5. Context Data

#### Market Context
```
context:market
```
Example:
```json
{
  "sector_performance": {
    "technology": 0.5,
    "healthcare": -0.2,
    "financials": 0.3,
    "consumer": 0.1,
    "energy": -0.4
  },
  "market_breadth": {
    "advancing": 1250,
    "declining": 1750,
    "unchanged": 500
  },
  "vix": 18.5,
  "market_regime": "bullish_consolidation",
  "timestamp": "2025-05-09T10:15:30.123Z"
}
```

## Data Flow for Position Monitoring

For efficient position monitoring as described in the requirements, the data flow should be:

1. **Real-time Data Collection**:
   - Update `stocks:price:{symbol}` with latest price data
   - Update `stocks:indicators:{symbol}` with calculated indicators
   - Update `stocks:sentiment:{symbol}` with news and sentiment data

2. **Position State Management**:
   - Update `positions:active:{symbol}` with current position details
   - Update `positions:monitoring:{symbol}` with monitoring metrics
   - Set `signals:{signal_type}:{symbol}` when signals are detected

3. **LLM Integration**:
   - Combine data from multiple Redis keys to create the structured data package for the Trade LLM
   - Store LLM decisions and reasoning for audit and improvement

## Performance Considerations

1. **Data Expiry**:
   - Set appropriate TTL for time-sensitive data (e.g., intraday price data)
   - Use Redis expiry for automatic cleanup

2. **Memory Optimization**:
   - Store only necessary data fields
   - Use efficient serialization formats
   - Leverage Redis data structures (hashes, sorted sets) for complex data

3. **Access Patterns**:
   - Use Redis pipelining for batch operations
   - Implement Redis pub/sub for real-time updates
   - Consider Redis Streams for time-series data

## Monitoring and Maintenance

1. **Regular Backups**:
   - AOF persistence is enabled for durability
   - Consider scheduled RDB snapshots for additional backup

2. **Performance Monitoring**:
   - Monitor memory usage
   - Track keyspace hits/misses
   - Observe command latency

3. **Data Cleanup**:
   - Implement daily cleanup routines
   - Archive historical data to long-term storage
