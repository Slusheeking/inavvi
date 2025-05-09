# Database Architecture: Redis and TimescaleDB

This document outlines how Redis and TimescaleDB are used in the trading system, explaining their complementary roles and integration.

## Overview

The trading system uses a dual-database architecture:

1. **Redis**: In-memory database for real-time data, caching, and ephemeral storage
2. **TimescaleDB**: Time-series optimized PostgreSQL database for historical data and persistent storage

This architecture leverages the strengths of each database:
- Redis provides ultra-fast access to real-time data needed for trading decisions
- TimescaleDB provides robust storage and querying capabilities for historical data and analytics

## Redis Usage

### Configuration

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

### Primary Use Cases

1. **Real-time Market Data**
   - Current price data for active symbols
   - Technical indicators
   - News and sentiment data
   - Market context (sector performance, market breadth)

2. **Trading Operations**
   - Active positions tracking
   - Watchlists and screened candidates
   - Trading signals
   - System state

3. **Caching Layer**
   - Frequently accessed data
   - Computation results
   - Dashboard data

### Key Data Structures

- **Stock Data**: `stocks:price:{symbol}`, `stocks:indicators:{symbol}`, `stocks:sentiment:{symbol}`
- **Watchlists**: `watchlist:current`, `candidates:ranked`
- **Positions**: `positions:active:{symbol}`, `positions:monitoring:{symbol}`
- **Signals**: `signals:{signal_type}:{symbol}`
- **System State**: `system:state`, `metrics:performance`

### Data Flow

1. **Data Ingestion**:
   - Market data from external APIs → Redis
   - Trading signals from analysis → Redis
   - System state updates → Redis

2. **Data Access**:
   - Trading algorithms read from Redis
   - Dashboard reads from Redis
   - Position monitor reads from Redis

## TimescaleDB Usage

### Configuration

TimescaleDB is configured as a time-series optimized PostgreSQL database with hypertables for efficient time-series data management.

### Primary Use Cases

1. **Historical Data Storage**
   - Price history (OHLCV data)
   - Trade execution records
   - System metrics

2. **Analytics and Reporting**
   - Performance analysis
   - Backtesting
   - Compliance and audit

### Key Tables

1. **price_data**
   - Hypertable partitioned by timestamp
   - Stores OHLCV data with symbol and source
   - Primary key: (symbol, timestamp)

2. **trade_data**
   - Hypertable partitioned by timestamp
   - Stores executed trades with order details
   - Primary key: (order_id, timestamp)

3. **metrics_data**
   - Hypertable partitioned by timestamp
   - Stores system metrics
   - Primary key: (metric_name, timestamp)

### Data Flow

1. **Data Ingestion**:
   - Price data → TimescaleDB for historical records
   - Executed trades → TimescaleDB for record-keeping
   - System metrics → TimescaleDB for performance tracking

2. **Data Access**:
   - Analytics queries against historical data
   - Reporting and visualization
   - Backtesting algorithms

## Integration Between Redis and TimescaleDB

### Data Lifecycle

1. **Real-time Data Flow**:
   - Live market data enters the system → stored in Redis
   - Trading algorithms use Redis data for decision-making
   - Executed trades stored in Redis for active monitoring

2. **Archival Process**:
   - Periodically, data is moved from Redis to TimescaleDB
   - Older data in Redis may be evicted based on LRU policy
   - Critical data is persisted to TimescaleDB before removal from Redis

3. **Historical Analysis**:
   - Recent data queried from Redis for speed
   - Historical data queried from TimescaleDB
   - Combined results when needed

### Specific Integration Points

1. **Price Data**:
   - Real-time: Redis `stocks:price:{symbol}`
   - Historical: TimescaleDB `price_data` table

2. **Trade Execution**:
   - Active trades: Redis `positions:active:{symbol}`
   - Trade history: TimescaleDB `trade_data` table

3. **System Metrics**:
   - Current metrics: Redis `metrics:performance`
   - Historical metrics: TimescaleDB `metrics_data` table

## Performance Considerations

### Redis Optimization

- Use appropriate data structures (hashes for objects, sorted sets for rankings)
- Set TTL for time-sensitive data
- Use Redis pipelining for batch operations
- Implement Redis pub/sub for real-time updates

### TimescaleDB Optimization

- Leverage hypertables for time-series data
- Use appropriate chunk intervals
- Create indexes on frequently queried columns
- Use continuous aggregates for pre-computed views

### Data Synchronization

- Implement background jobs for data archival
- Use transaction batching for efficient writes to TimescaleDB
- Consider write-behind caching patterns

## Monitoring and Maintenance

### Redis Monitoring

- Memory usage and fragmentation
- Hit/miss ratio for cache effectiveness
- Command latency
- Connection count

### TimescaleDB Monitoring

- Query performance
- Index usage
- Disk space usage
- Chunk compression ratio

### Backup Strategy

- Redis: AOF persistence and RDB snapshots
- TimescaleDB: Regular PostgreSQL backups
- Consider point-in-time recovery requirements

## Conclusion

The dual-database architecture with Redis and TimescaleDB provides an optimal solution for the trading system:

- **Redis** handles the real-time, low-latency requirements critical for trading decisions
- **TimescaleDB** provides robust, scalable storage for historical data and analytics

This separation of concerns allows each database to excel at its primary function while providing a complete data management solution for the trading system.
