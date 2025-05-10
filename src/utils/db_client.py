"""
Database client for TimescaleDB using SQLAlchemy.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, MetaData, select, text, PrimaryKeyConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from src.config.settings import settings
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("db_client")

# Create SQLAlchemy Base
Base = declarative_base()

class TimescaleDBClient:
    """Client for interacting with TimescaleDB."""
    
    def __init__(self):
        """Initialize the TimescaleDB client."""
        self._engine = None
        self._session_factory = None
        self._metadata = MetaData()
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize the database connection."""
        try:
            # Create engine
            self._engine = create_engine(settings.database.timescaledb_url)
            
            # Create session factory
            self._session_factory = sessionmaker(bind=self._engine)
            
            logger.info(f"Connected to TimescaleDB at {settings.database.timescaledb_url.split('@')[1]}")
        except Exception as e:
            logger.error(f"Error connecting to TimescaleDB: {e}")
            raise
    
    def create_tables(self):
        """Create all tables defined in the models using raw SQL without primary keys initially."""
        try:
            with self.get_session() as session:
                # Drop tables if they exist to ensure a clean state
                session.execute(text("DROP TABLE IF EXISTS price_data CASCADE;"))
                session.execute(text("DROP TABLE IF EXISTS trade_data CASCADE;"))
                session.execute(text("DROP TABLE IF EXISTS metrics_data CASCADE;"))
                logger.info("Dropped existing tables (if any)")

                # Create price_data table
                session.execute(text("""
                    CREATE TABLE price_data (
                        symbol VARCHAR NOT NULL,
                        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        open FLOAT NOT NULL,
                        high FLOAT NOT NULL,
                        low FLOAT NOT NULL,
                        close FLOAT NOT NULL,
                        volume INTEGER NOT NULL,
                        source VARCHAR NOT NULL
                    );
                """))

                # Create trade_data table
                session.execute(text("""
                    CREATE TABLE trade_data (
                        order_id VARCHAR NOT NULL,
                        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        symbol VARCHAR NOT NULL,
                        side VARCHAR NOT NULL,
                        quantity FLOAT NOT NULL,
                        price FLOAT NOT NULL,
                        strategy VARCHAR NOT NULL
                    );
                """))

                # Create metrics_data table
                session.execute(text("""
                    CREATE TABLE metrics_data (
                        metric_name VARCHAR NOT NULL,
                        timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        metric_value FLOAT NOT NULL,
                        tags VARCHAR
                    );
                """))

                logger.info("Tables created successfully without primary keys")

            # Create hypertables and add primary keys
            self._create_hypertables()

            logger.info("All tables created successfully with hypertables and primary keys")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def _create_hypertables(self):
        """Create hypertables for time-series data and add primary keys."""
        with self.get_session() as session:
            # --- Debugging: Check indexes before creating hypertable ---
            logger.info("Checking indexes on price_data before creating hypertable...")
            indexes_before = session.execute(text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'price_data';
            """)).fetchall()
            logger.info(f"Indexes on price_data before hypertable creation: {indexes_before}")
            # --- End Debugging ---

            # First, drop any existing primary key constraints
            session.execute(text("ALTER TABLE price_data DROP CONSTRAINT IF EXISTS price_data_pkey;"))
            session.execute(text("ALTER TABLE trade_data DROP CONSTRAINT IF EXISTS trade_data_pkey;"))
            session.execute(text("ALTER TABLE metrics_data DROP CONSTRAINT IF EXISTS metrics_data_pkey;"))
            
            # Add primary key constraint before hypertable creation for price_data
            session.execute(text(
                "ALTER TABLE price_data ADD CONSTRAINT price_data_pkey PRIMARY KEY (symbol, timestamp);"
            ))

            # Create hypertable for price data
            session.execute(text(
                "SELECT create_hypertable('price_data', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE)"
            ))

            # Add primary key constraint before hypertable creation for trade_data
            session.execute(text(
                "ALTER TABLE trade_data ADD CONSTRAINT trade_data_pkey PRIMARY KEY (order_id, timestamp);"
            ))

            # Create hypertable for trade data
            session.execute(text(
                "SELECT create_hypertable('trade_data', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE)"
            ))

            # Add primary key constraint before hypertable creation for metrics_data
            session.execute(text(
                "ALTER TABLE metrics_data ADD CONSTRAINT metrics_data_pkey PRIMARY KEY (metric_name, timestamp);"
            ))

            # Create hypertable for metrics data
            session.execute(text(
                "SELECT create_hypertable('metrics_data', 'timestamp', if_not_exists => TRUE, migrate_data => TRUE)"
            ))

            logger.info("Hypertables and primary keys added successfully")
    
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
            with self.get_session() as session:
                # Create price data entry
                price_data = PriceData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=data.get('open', 0.0),
                    high=data.get('high', 0.0),
                    low=data.get('low', 0.0),
                    close=data.get('close', 0.0),
                    volume=data.get('volume', 0),
                    source=data.get('source', 'unknown')
                )
                
                # Add to session
                session.add(price_data)
                
                # Commit
                session.commit()
        except Exception as e:
            logger.error(f"Error storing price data for {symbol}: {e}")
            raise
    
    def store_trade_data(self, trade_data: Dict[str, Any]):
        """
        Store trade execution data.
        
        Args:
            trade_data: Trade data dictionary
        """
        try:
            with self.get_session() as session:
                # Ensure order_id is present
                order_id = trade_data.get('order_id')
                if not order_id:
                    order_id = f"order_{int(datetime.now().timestamp())}"
                
                # Create trade data entry
                trade = TradeData(
                    symbol=trade_data.get('symbol'),
                    timestamp=datetime.now(),
                    side=trade_data.get('side'),
                    quantity=trade_data.get('quantity'),
                    price=trade_data.get('price'),
                    order_id=order_id,
                    strategy=trade_data.get('strategy', 'unknown')
                )
                
                # Add to session
                session.add(trade)
                
                # Commit
                session.commit()
        except Exception as e:
            logger.error(f"Error storing trade data: {e}")
            raise
    
    def get_price_history(self, symbol: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Get price history for a symbol.
        
        Args:
            symbol: Stock symbol
            start_time: Start time
            end_time: End time
            
        Returns:
            List of price data dictionaries
        """
        try:
            with self.get_session() as session:
                # Query price data
                query = select(PriceData).where(
                    PriceData.symbol == symbol,
                    PriceData.timestamp >= start_time,
                    PriceData.timestamp <= end_time
                ).order_by(PriceData.timestamp)
                
                # Execute query
                result = session.execute(query).scalars().all()
                
                # Convert to dictionaries
                return [
                    {
                        'symbol': item.symbol,
                        'timestamp': item.timestamp,
                        'open': item.open,
                        'high': item.high,
                        'low': item.low,
                        'close': item.close,
                        'volume': item.volume,
                        'source': item.source
                    }
                    for item in result
                ]
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            raise
    
    def get_trade_history(self, symbol: Optional[str] = None, start_time: Optional[datetime] = None, 
                         end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            symbol: Stock symbol (optional)
            start_time: Start time (optional)
            end_time: End time (optional)
            
        Returns:
            List of trade data dictionaries
        """
        try:
            with self.get_session() as session:
                # Build query
                query = select(TradeData)
                
                # Add filters if provided
                if symbol:
                    query = query.where(TradeData.symbol == symbol)
                
                if start_time:
                    query = query.where(TradeData.timestamp >= start_time)
                
                if end_time:
                    query = query.where(TradeData.timestamp <= end_time)
                
                # Order by timestamp
                query = query.order_by(TradeData.timestamp)
                
                # Execute query
                result = session.execute(query).scalars().all()
                
                # Convert to dictionaries
                return [
                    {
                        'symbol': item.symbol,
                        'timestamp': item.timestamp,
                        'side': item.side,
                        'quantity': item.quantity,
                        'price': item.price,
                        'order_id': item.order_id,
                        'strategy': item.strategy
                    }
                    for item in result
                ]
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            raise


# Define models
class PriceData(Base):
    """Price data model."""
    __tablename__ = 'price_data'
    
    """Price data model."""
    __tablename__ = 'price_data'

    # Use composite primary key with symbol and timestamp
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    source = Column(String, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('symbol', 'timestamp'),
        # Explicitly define unique constraint for TimescaleDB - removed in previous attempt
        # UniqueConstraint('symbol', 'timestamp'),
        Index('idx_price_data_timestamp', 'timestamp'), # Add an index on the timestamp column
    )


class TradeData(Base):
    """Trade data model."""
    __tablename__ = 'trade_data'
    
    # Use composite primary key with order_id and timestamp
    order_id = Column(String, primary_key=True, nullable=False)
    timestamp = Column(DateTime, primary_key=True, nullable=False)
    symbol = Column(String, index=True, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    strategy = Column(String, nullable=False)


class MetricsData(Base):
    """System metrics data model."""
    __tablename__ = 'metrics_data'
    
    # Use composite primary key with metric_name and timestamp
    metric_name = Column(String, primary_key=True, nullable=False)
    timestamp = Column(DateTime, primary_key=True, nullable=False)
    metric_value = Column(Float, nullable=False)
    tags = Column(String, nullable=True)


# Create global client instance
timescaledb_client = TimescaleDBClient()
