"""
Database client for TimescaleDB using SQLAlchemy.
"""

import concurrent.futures
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    PrimaryKeyConstraint,
    String,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from src.config.settings import settings
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("db_client")

# Create SQLAlchemy Base
Base = declarative_base()


# Define models
class PriceData(Base):
    """Price data model."""

    __tablename__ = "price_data"

    # Use composite primary key with symbol and timestamp
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)  # Changed to BigInteger for high-volume stocks
    source = Column(String, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("symbol", "timestamp"),
        Index("idx_price_data_timestamp", "timestamp"),
        Index("idx_price_data_symbol", "symbol"),
    )


class TradeData(Base):
    """Trade data model."""

    __tablename__ = "trade_data"

    # Use composite primary key with order_id and timestamp
    order_id = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    strategy = Column(String, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("order_id", "timestamp"),
        Index("idx_trade_data_timestamp", "timestamp"),
        Index("idx_trade_data_symbol", "symbol"),
    )


class MetricsData(Base):
    """System metrics data model."""

    __tablename__ = "metrics_data"

    # Use composite primary key with metric_name and timestamp
    metric_name = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    metric_value = Column(Float, nullable=False)
    tags = Column(String, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("metric_name", "timestamp"),
        Index("idx_metrics_data_timestamp", "timestamp"),
    )


class TimescaleDBClient:
    """Client for interacting with TimescaleDB."""

    def __init__(self):
        """Initialize the TimescaleDB client."""
        self._engine = None
        self._session_factory = None
        self._metadata = MetaData()
        self._initialize_connection()

    def _initialize_connection(self):
        """Initialize the database connection with connection pooling."""
        try:
            # Create engine with connection pooling
            self._engine = create_engine(
                settings.database.timescaledb_url,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_timeout=30,
                pool_recycle=1800,  # Recycle connections after 30 minutes
            )

            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine)

            # Log connection but hide credentials
            db_url_parts = settings.database.timescaledb_url.split("@")
            if len(db_url_parts) > 1:
                safe_url = f"...@{db_url_parts[-1]}"
                logger.info(f"Connected to TimescaleDB at {safe_url}")
            else:
                logger.info("Connected to TimescaleDB")

        except Exception as e:
            logger.error(f"Error connecting to TimescaleDB: {e}")
            raise

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction error: {e}")
            raise
        finally:
            session.close()

    def create_tables(self, drop_existing: bool = False):
        """
        Create all tables and hypertables.

        Args:
            drop_existing: Whether to drop existing tables
        """
        try:
            # Create tables using SQLAlchemy
            if drop_existing:
                Base.metadata.drop_all(self._engine)
                logger.info("Dropped existing tables")

            # Create tables
            Base.metadata.create_all(self._engine)
            logger.info("Created base tables")

            # Create hypertables (TimescaleDB specific)
            self._create_hypertables()
            logger.info("Created hypertables")

            # Create additional indexes for performance
            self._create_additional_indexes()
            logger.info("Created additional indexes")

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def _create_hypertables(self):
        """Create TimescaleDB hypertables for time-series data."""
        with self.session_scope() as session:
            # Create hypertable for price data
            session.execute(
                text(
                    "SELECT create_hypertable('price_data', 'timestamp', "
                    "if_not_exists => TRUE, migrate_data => TRUE, "
                    "chunk_time_interval => INTERVAL '1 day')"
                )
            )

            # Create hypertable for trade data
            session.execute(
                text(
                    "SELECT create_hypertable('trade_data', 'timestamp', "
                    "if_not_exists => TRUE, migrate_data => TRUE, "
                    "chunk_time_interval => INTERVAL '1 day')"
                )
            )

            # Create hypertable for metrics data
            session.execute(
                text(
                    "SELECT create_hypertable('metrics_data', 'timestamp', "
                    "if_not_exists => TRUE, migrate_data => TRUE, "
                    "chunk_time_interval => INTERVAL '1 hour')"
                )
            )

    def _create_additional_indexes(self):
        """Create additional indexes for query performance."""
        with self.session_scope() as session:
            # Composite indexes for common queries
            session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_price_data_symbol_timestamp "
                    "ON price_data (symbol, timestamp DESC)"
                )
            )

            session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_trade_data_symbol_timestamp "
                    "ON trade_data (symbol, timestamp DESC)"
                )
            )

            # Add expression indexes for TimescaleDB time buckets
            session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_price_data_time_bucket_daily "
                    "ON price_data (time_bucket('1 day', timestamp), symbol)"
                )
            )

    def get_session(self) -> Session:
        """Get a new database session."""
        return self._session_factory()

    def store_price_data(self, symbol: str, data: Dict[str, Any]):
        """
        Store price data for a symbol.

        Args:
            symbol: Stock symbol
            data: Price data dictionary
        """
        try:
            with self.session_scope() as session:
                # Create price data entry
                price_data = PriceData(
                    symbol=symbol,
                    timestamp=data.get("timestamp", datetime.now()),
                    open=data.get("open", 0.0),
                    high=data.get("high", 0.0),
                    low=data.get("low", 0.0),
                    close=data.get("close", 0.0),
                    volume=data.get("volume", 0),
                    source=data.get("source", "unknown"),
                )

                # Add to session
                session.add(price_data)
        except Exception as e:
            logger.error(f"Error storing price data for {symbol}: {e}")
            # Handle gracefully instead of re-raising
            return False
        return True

    def store_price_data_batch(self, price_data_list: List[Dict[str, Any]]):
        """
        Store multiple price data entries in a batch.

        Args:
            price_data_list: List of price data dictionaries

        Returns:
            Number of records successfully inserted
        """
        if not price_data_list:
            return 0

        try:
            with self.session_scope() as session:
                # Convert dictionaries to ORM objects
                price_data_objects = []
                for data in price_data_list:
                    price_data = PriceData(
                        symbol=data.get("symbol"),
                        timestamp=data.get("timestamp", datetime.now()),
                        open=data.get("open", 0.0),
                        high=data.get("high", 0.0),
                        low=data.get("low", 0.0),
                        close=data.get("close", 0.0),
                        volume=data.get("volume", 0),
                        source=data.get("source", "unknown"),
                    )
                    price_data_objects.append(price_data)

                # Batch insert
                session.bulk_save_objects(price_data_objects)

                return len(price_data_objects)
        except Exception as e:
            logger.error(f"Error storing batch price data: {e}")
            return 0

    def store_trade_data(self, trade_data: Dict[str, Any]):
        """
        Store trade execution data.

        Args:
            trade_data: Trade data dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_scope() as session:
                # Ensure order_id is present
                order_id = trade_data.get("order_id")
                if not order_id:
                    order_id = f"order_{int(datetime.now().timestamp())}"

                # Create trade data entry
                trade = TradeData(
                    symbol=trade_data.get("symbol"),
                    timestamp=trade_data.get("timestamp", datetime.now()),
                    side=trade_data.get("side"),
                    quantity=trade_data.get("quantity"),
                    price=trade_data.get("price"),
                    order_id=order_id,
                    strategy=trade_data.get("strategy", "unknown"),
                )

                # Add to session
                session.add(trade)
                return True
        except Exception as e:
            logger.error(f"Error storing trade data: {e}")
            return False

    def store_trade_data_batch(self, trade_data_list: List[Dict[str, Any]]):
        """
        Store multiple trade data entries in a batch.

        Args:
            trade_data_list: List of trade data dictionaries

        Returns:
            Number of records successfully inserted
        """
        if not trade_data_list:
            return 0

        try:
            with self.session_scope() as session:
                trade_objects = []
                for data in trade_data_list:
                    # Ensure order_id is present
                    order_id = data.get("order_id")
                    if not order_id:
                        order_id = f"order_{int(datetime.now().timestamp())}_{len(trade_objects)}"

                    trade = TradeData(
                        symbol=data.get("symbol"),
                        timestamp=data.get("timestamp", datetime.now()),
                        side=data.get("side"),
                        quantity=data.get("quantity"),
                        price=data.get("price"),
                        order_id=order_id,
                        strategy=data.get("strategy", "unknown"),
                    )
                    trade_objects.append(trade)

                # Batch insert
                session.bulk_save_objects(trade_objects)
                return len(trade_objects)
        except Exception as e:
            logger.error(f"Error storing batch trade data: {e}")
            return 0

    def get_price_history(
        self, symbol: str, start_time: datetime, end_time: datetime, interval: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get price history for a symbol with optional time bucketing.

        Args:
            symbol: Stock symbol
            start_time: Start time
            end_time: End time
            interval: Optional time interval for aggregation ('1m', '5m', '1h', '1d')

        Returns:
            List of price data dictionaries
        """
        try:
            with self.session_scope() as session:
                if interval:
                    # Convert interval to PostgreSQL interval
                    pg_interval = self._convert_interval(interval)

                    # Use TimescaleDB time_bucket for aggregation
                    query = text(f"""
                        SELECT
                            time_bucket('{pg_interval}', timestamp) AS bucket_time,
                            symbol,
                            first(open, timestamp) AS open,
                            max(high) AS high,
                            min(low) AS low,
                            last(close, timestamp) AS close,
                            sum(volume) AS volume,
                            last(source, timestamp) AS source
                        FROM price_data
                        WHERE symbol = :symbol
                        AND timestamp >= :start_time
                        AND timestamp <= :end_time
                        GROUP BY bucket_time, symbol
                        ORDER BY bucket_time
                    """)

                    result = session.execute(
                        query, {"symbol": symbol, "start_time": start_time, "end_time": end_time}
                    ).fetchall()

                    # Convert to dictionaries
                    return [
                        {
                            "symbol": row.symbol,
                            "timestamp": row.bucket_time,
                            "open": row.open,
                            "high": row.high,
                            "low": row.low,
                            "close": row.close,
                            "volume": row.volume,
                            "source": row.source,
                        }
                        for row in result
                    ]
                else:
                    # Regular query without aggregation
                    query = (
                        select(PriceData)
                        .where(
                            PriceData.symbol == symbol,
                            PriceData.timestamp >= start_time,
                            PriceData.timestamp <= end_time,
                        )
                        .order_by(PriceData.timestamp)
                    )

                    # Execute query
                    result = session.execute(query).scalars().all()

                    # Convert to dictionaries
                    return [
                        {
                            "symbol": item.symbol,
                            "timestamp": item.timestamp,
                            "open": item.open,
                            "high": item.high,
                            "low": item.low,
                            "close": item.close,
                            "volume": item.volume,
                            "source": item.source,
                        }
                        for item in result
                    ]
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return []

    def _convert_interval(self, interval: str) -> str:
        """
        Convert common interval notation to PostgreSQL interval.

        Args:
            interval: Interval string ('1m', '5m', '1h', '1d', etc.)

        Returns:
            PostgreSQL interval string
        """
        interval = interval.lower()

        if interval.endswith("m"):
            minutes = int(interval[:-1])
            return f"{minutes} minute"
        elif interval.endswith("h"):
            hours = int(interval[:-1])
            return f"{hours} hour"
        elif interval.endswith("d"):
            days = int(interval[:-1])
            return f"{days} day"
        else:
            # Default to 1 minute if unrecognized
            logger.warning(f"Unrecognized interval format: {interval}, defaulting to 1 minute")
            return "1 minute"

    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Get trade history with pagination.

        Args:
            symbol: Stock symbol (optional)
            start_time: Start time (optional)
            end_time: End time (optional)
            limit: Maximum number of records to return

        Returns:
            List of trade data dictionaries
        """
        try:
            with self.session_scope() as session:
                # Build query
                query = select(TradeData)

                # Add filters if provided
                if symbol:
                    query = query.where(TradeData.symbol == symbol)

                if start_time:
                    query = query.where(TradeData.timestamp >= start_time)

                if end_time:
                    query = query.where(TradeData.timestamp <= end_time)

                # Order by timestamp (descending to get most recent first)
                query = query.order_by(TradeData.timestamp.desc())

                # Add limit
                query = query.limit(limit)

                # Execute query
                result = session.execute(query).scalars().all()

                # Convert to dictionaries
                return [
                    {
                        "symbol": item.symbol,
                        "timestamp": item.timestamp,
                        "side": item.side,
                        "quantity": item.quantity,
                        "price": item.price,
                        "order_id": item.order_id,
                        "strategy": item.strategy,
                    }
                    for item in result
                ]
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []

    def get_pnl_by_day(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get profit and loss by day.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            List of daily P&L data
        """
        try:
            with self.session_scope() as session:
                query = text("""
                    WITH trade_values AS (
                        SELECT
                            time_bucket('1 day', timestamp) AS trade_day,
                            symbol,
                            CASE
                                WHEN side = 'buy' THEN -1 * quantity * price
                                WHEN side = 'sell' THEN quantity * price
                                ELSE 0
                            END AS trade_value
                        FROM trade_data
                        WHERE timestamp >= :start_date
                        AND timestamp <= :end_date
                    )
                    SELECT
                        trade_day,
                        symbol,
                        SUM(trade_value) AS pnl
                    FROM trade_values
                    GROUP BY trade_day, symbol
                    ORDER BY trade_day, symbol
                """)

                result = session.execute(
                    query, {"start_date": start_date, "end_date": end_date}
                ).fetchall()

                return [
                    {"date": row.trade_day, "symbol": row.symbol, "pnl": row.pnl} for row in result
                ]
        except Exception as e:
            logger.error(f"Error getting P&L by day: {e}")
            return []

    def store_metrics_data(self, metrics_data: Dict[str, Any]):
        """
        Store system metrics data.

        Args:
            metrics_data: Metrics data dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            with self.session_scope() as session:
                # Create metrics data entry
                metrics = MetricsData(
                    metric_name=metrics_data.get("metric_name"),
                    timestamp=metrics_data.get("timestamp", datetime.now()),
                    metric_value=metrics_data.get("metric_value"),
                    tags=metrics_data.get("tags"),
                )

                # Add to session
                session.add(metrics)
                return True
        except Exception as e:
            logger.error(f"Error storing metrics data: {e}")
            return False

    def store_metrics_data_batch(self, metrics_data_list: List[Dict[str, Any]]):
        """
        Store multiple metrics data entries in a batch.

        Args:
            metrics_data_list: List of metrics data dictionaries

        Returns:
            Number of records successfully inserted
        """
        if not metrics_data_list:
            return 0

        try:
            with self.session_scope() as session:
                metrics_objects = []
                for data in metrics_data_list:
                    metrics = MetricsData(
                        metric_name=data.get("metric_name"),
                        timestamp=data.get("timestamp", datetime.now()),
                        metric_value=data.get("metric_value"),
                        tags=data.get("tags"),
                    )
                    metrics_objects.append(metrics)

                # Batch insert
                session.bulk_save_objects(metrics_objects)
                return len(metrics_objects)
        except Exception as e:
            logger.error(f"Error storing batch metrics data: {e}")
            return 0

    def get_metrics_history(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        interval: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get metrics history for a metric with optional aggregation.

        Args:
            metric_name: Metric name
            start_time: Start time
            end_time: End time
            interval: Optional time interval for aggregation

        Returns:
            List of metrics data dictionaries
        """
        try:
            with self.session_scope() as session:
                if interval:
                    # Convert interval to PostgreSQL interval
                    pg_interval = self._convert_interval(interval)

                    # Use TimescaleDB time_bucket for aggregation
                    query = text(f"""
                        SELECT
                            time_bucket('{pg_interval}', timestamp) AS bucket_time,
                            metric_name,
                            AVG(metric_value) AS avg_value,
                            MIN(metric_value) AS min_value,
                            MAX(metric_value) AS max_value,
                            last(tags, timestamp) AS tags
                        FROM metrics_data
                        WHERE metric_name = :metric_name
                        AND timestamp >= :start_time
                        AND timestamp <= :end_time
                        GROUP BY bucket_time, metric_name
                        ORDER BY bucket_time
                    """)

                    result = session.execute(
                        query,
                        {
                            "metric_name": metric_name,
                            "start_time": start_time,
                            "end_time": end_time,
                        },
                    ).fetchall()

                    # Convert to dictionaries
                    return [
                        {
                            "metric_name": row.metric_name,
                            "timestamp": row.bucket_time,
                            "avg_value": row.avg_value,
                            "min_value": row.min_value,
                            "max_value": row.max_value,
                            "tags": row.tags,
                        }
                        for row in result
                    ]
                else:
                    # Regular query without aggregation
                    query = (
                        select(MetricsData)
                        .where(
                            MetricsData.metric_name == metric_name,
                            MetricsData.timestamp >= start_time,
                            MetricsData.timestamp <= end_time,
                        )
                        .order_by(MetricsData.timestamp)
                    )

                    # Execute query
                    result = session.execute(query).scalars().all()

                    # Convert to dictionaries
                    return [
                        {
                            "metric_name": item.metric_name,
                            "timestamp": item.timestamp,
                            "metric_value": item.metric_value,
                            "tags": item.tags,
                        }
                        for item in result
                    ]
        except Exception as e:
            logger.error(f"Error getting metrics history for {metric_name}: {e}")
            return []

    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary of system statistics
        """
        try:
            with self.session_scope() as session:
                # Get table row counts
                price_count = session.execute(select(func.count()).select_from(PriceData)).scalar()

                trade_count = session.execute(select(func.count()).select_from(TradeData)).scalar()

                metrics_count = session.execute(
                    select(func.count()).select_from(MetricsData)
                ).scalar()

                # Get chunk info from TimescaleDB
                chunk_info = session.execute(
                    text("""
                    SELECT
                        hypertable_name,
                        count(*) as chunk_count,
                        pg_size_pretty(sum(total_bytes)) as total_size
                    FROM timescaledb_information.hypertables h
                    JOIN timescaledb_information.chunks c
                        ON h.hypertable_name = c.hypertable_name
                    GROUP BY hypertable_name
                """)
                ).fetchall()

                chunk_stats = {
                    row.hypertable_name: {
                        "chunk_count": row.chunk_count,
                        "total_size": row.total_size,
                    }
                    for row in chunk_info
                }

                return {
                    "row_counts": {
                        "price_data": price_count,
                        "trade_data": trade_count,
                        "metrics_data": metrics_count,
                    },
                    "storage": chunk_stats,
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def cleanup_old_data(self, retention_days: int):
        """
        Clean up old data using TimescaleDB retention policy.

        Args:
            retention_days: Number of days to retain data

        Returns:
            Dictionary with results
        """
        try:
            with self.session_scope() as session:
                # Add retention policy to each hypertable
                session.execute(
                    text(f"""
                    SELECT add_retention_policy(
                        'price_data',
                        INTERVAL '{retention_days} days',
                        if_not_exists => TRUE
                    )
                """)
                )

                session.execute(
                    text(f"""
                    SELECT add_retention_policy(
                        'trade_data',
                        INTERVAL '{retention_days} days',
                        if_not_exists => TRUE
                    )
                """)
                )

                session.execute(
                    text(f"""
                    SELECT add_retention_policy(
                        'metrics_data',
                        INTERVAL '{retention_days} days',
                        if_not_exists => TRUE
                    )
                """)
                )

                logger.info(f"Added retention policy of {retention_days} days to all hypertables")

                return {
                    "status": "success",
                    "retention_days": retention_days,
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Error setting up retention policy: {e}")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}

    def compress_chunks(self, older_than_days: int):
        """
        Compress old chunks using TimescaleDB compression.

        Args:
            older_than_days: Compress chunks older than this many days

        Returns:
            Dictionary with compression results
        """
        try:
            with self.session_scope() as session:
                # Set compression policy
                session.execute(
                    text("""
                    ALTER TABLE price_data
                    SET (timescaledb.compress, timescaledb.compress_segmentby='symbol')
                """)
                )

                session.execute(
                    text("""
                    ALTER TABLE trade_data
                    SET (timescaledb.compress, timescaledb.compress_segmentby='symbol')
                """)
                )

                session.execute(
                    text("""
                    ALTER TABLE metrics_data
                    SET (timescaledb.compress, timescaledb.compress_segmentby='metric_name')
                """)
                )

                # Add compression policy
                session.execute(
                    text(f"""
                    SELECT add_compression_policy(
                        'price_data',
                        INTERVAL '{older_than_days} days',
                        if_not_exists => TRUE
                    )
                """)
                )

                session.execute(
                    text(f"""
                    SELECT add_compression_policy(
                        'trade_data',
                        INTERVAL '{older_than_days} days',
                        if_not_exists => TRUE
                    )
                """)
                )

                session.execute(
                    text(f"""
                    SELECT add_compression_policy(
                        'metrics_data',
                        INTERVAL '{older_than_days} days',
                        if_not_exists => TRUE
                    )
                """)
                )

                logger.info(f"Added compression policy for data older than {older_than_days} days")

                return {
                    "status": "success",
                    "compression_days": older_than_days,
                    "timestamp": datetime.now().isoformat(),
                }
        except Exception as e:
            logger.error(f"Error setting up compression policy: {e}")
            return {"status": "error", "error": str(e), "timestamp": datetime.now().isoformat()}


# Create global client instance
timescaledb_client = TimescaleDBClient()
