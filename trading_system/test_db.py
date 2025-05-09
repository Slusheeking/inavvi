import os
from datetime import datetime, timedelta
from src.utils.db_client import timescaledb_client

def test_database():
    """Tests TimescaleDB connection, table creation, and data operations."""
    print("Testing TimescaleDB...")

    try:
        # Ensure tables are created
        print("Creating tables...")
        timescaledb_client.create_tables()
        print("Tables created.")

        # Test storing price data
        print("Storing price data...")
        dummy_price_data = {
            'open': 150.0,
            'high': 155.0,
            'low': 148.0,
            'close': 154.0,
            'volume': 100000,
            'source': 'test_source'
        }
        timescaledb_client.store_price_data("TESTSYM", dummy_price_data)
        print("Price data stored.")

        # Test storing trade data
        print("Storing trade data...")
        dummy_trade_data = {
            'symbol': 'TESTSYM',
            'side': 'buy',
            'quantity': 10.0,
            'price': 152.0,
            'strategy': 'test_strategy'
        }
        timescaledb_client.store_trade_data(dummy_trade_data)
        print("Trade data stored.")

        # Test retrieving price data
        print("Retrieving price data...")
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5) # Look back 5 minutes
        retrieved_price_data = timescaledb_client.get_price_history("TESTSYM", start_time, end_time)
        print(f"Retrieved price data: {retrieved_price_data}")

        # Test retrieving trade data
        print("Retrieving trade data...")
        retrieved_trade_data = timescaledb_client.get_trade_history(symbol="TESTSYM", start_time=start_time, end_time=end_time)
        print(f"Retrieved trade data: {retrieved_trade_data}")

        print("TimescaleDB test completed successfully.")

    except Exception as e:
        print(f"TimescaleDB test failed: {e}")
        # Re-raise the exception to indicate failure
        raise

if __name__ == "__main__":
    # Add the trading_system directory to the Python path
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    test_database()
