#!/usr/bin/env python
"""
Test script for TimescaleDB client.
"""

import os
import random
import sys
import uuid
from datetime import datetime, timedelta

import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.db_client import MetricsData, PriceData, TradeData, timescaledb_client


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
            "open": 150.0,
            "high": 155.0,
            "low": 148.0,
            "close": 154.0,
            "volume": 100000,
            "source": "test_source",
        }
        timescaledb_client.store_price_data("TESTSYM", dummy_price_data)
        print("Price data stored.")

        # Test storing trade data
        print("Storing trade data...")
        dummy_trade_data = {
            "symbol": "TESTSYM",
            "side": "buy",
            "quantity": 10.0,
            "price": 152.0,
            "strategy": "test_strategy",
        }
        timescaledb_client.store_trade_data(dummy_trade_data)
        print("Trade data stored.")

        # Test retrieving price data
        print("Retrieving price data...")
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5)  # Look back 5 minutes
        retrieved_price_data = timescaledb_client.get_price_history("TESTSYM", start_time, end_time)
        print(f"Retrieved price data: {retrieved_price_data}")

        # Test retrieving trade data
        print("Retrieving trade data...")
        retrieved_trade_data = timescaledb_client.get_trade_history(
            symbol="TESTSYM", start_time=start_time, end_time=end_time
        )
        print(f"Retrieved trade data: {retrieved_trade_data}")

        # Test storing metrics data
        print("Storing metrics data...")
        dummy_metrics_data = {
            "metric_name": "test_metric",
            "metric_value": 123.45,
            "tags": "test_tag",
        }
        timescaledb_client.store_metrics_data(dummy_metrics_data)
        print("Metrics data stored.")

        # Test retrieving metrics data
        print("Retrieving metrics data...")
        retrieved_metrics_data = timescaledb_client.get_metrics_history(
            metric_name="test_metric", start_time=start_time, end_time=end_time
        )
        print(f"Retrieved metrics data: {retrieved_metrics_data}")

        print("TimescaleDB test completed successfully.")

    except Exception as e:
        print(f"TimescaleDB test failed: {e}")
        # Re-raise the exception to indicate failure
        raise


def test_batch_operations():
    """Tests batch operations for TimescaleDB client."""
    print("Testing batch operations...")

    try:
        # Generate batch price data
        price_data_batch = []
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

        now = datetime.now()
        for symbol in symbols:
            for i in range(10):  # 10 data points per symbol
                timestamp = now - timedelta(minutes=i)
                price = 100 + random.random() * 50
                price_data_batch.append(
                    {
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "open": price - random.random() * 2,
                        "high": price + random.random() * 2,
                        "low": price - random.random() * 3,
                        "close": price,
                        "volume": int(random.random() * 100000),
                        "source": "batch_test",
                    }
                )

        # Store batch price data
        print(f"Storing batch of {len(price_data_batch)} price records...")
        inserted_count = timescaledb_client.store_price_data_batch(price_data_batch)
        print(f"Inserted {inserted_count} price records.")

        # Generate batch trade data
        trade_data_batch = []
        for symbol in symbols:
            for _ in range(5):  # 5 trades per symbol
                side = "buy" if random.random() > 0.5 else "sell"
                price = 100 + random.random() * 50
                quantity = random.randint(1, 100)
                trade_data_batch.append(
                    {
                        "symbol": symbol,
                        "timestamp": now - timedelta(minutes=random.randint(1, 30)),
                        "side": side,
                        "quantity": quantity,
                        "price": price,
                        "order_id": str(uuid.uuid4()),
                        "strategy": f"strategy_{random.randint(1, 3)}",
                    }
                )

        # Store batch trade data
        print(f"Storing batch of {len(trade_data_batch)} trade records...")
        inserted_count = timescaledb_client.store_trade_data_batch(trade_data_batch)
        print(f"Inserted {inserted_count} trade records.")

        # Generate batch metrics data
        metrics_data_batch = []
        metric_names = ["cpu_usage", "memory_usage", "api_latency", "queue_size", "error_rate"]

        for metric in metric_names:
            for i in range(20):  # 20 data points per metric
                metrics_data_batch.append(
                    {
                        "metric_name": metric,
                        "timestamp": now - timedelta(minutes=i),
                        "metric_value": random.random() * 100,
                        "tags": "component=trading_system,env=test",
                    }
                )

        # Store batch metrics data
        print(f"Storing batch of {len(metrics_data_batch)} metrics records...")
        inserted_count = timescaledb_client.store_metrics_data_batch(metrics_data_batch)
        print(f"Inserted {inserted_count} metrics records.")

        # Test aggregated queries
        print("Testing aggregated queries...")

        # Get price history with time bucketing
        for symbol in symbols[:2]:  # Test first two symbols
            agg_prices = timescaledb_client.get_price_history(
                symbol,
                now - timedelta(days=1),
                now,
                interval="5m",  # 5-minute buckets
            )
            print(f"Retrieved {len(agg_prices)} aggregated price records for {symbol}")

        # Get metrics with time bucketing
        for metric in metric_names[:2]:  # Test first two metrics
            agg_metrics = timescaledb_client.get_metrics_history(
                metric,
                now - timedelta(days=1),
                now,
                interval="5m",  # 5-minute buckets
            )
            print(f"Retrieved {len(agg_metrics)} aggregated metric records for {metric}")

        # Test P&L calculation
        pnl_data = timescaledb_client.get_pnl_by_day(now - timedelta(days=7), now)
        print(f"Retrieved {len(pnl_data)} P&L records")

        # Test system stats
        stats = timescaledb_client.get_system_stats()
        print(f"System stats: {stats}")

        print("Batch operations test completed successfully.")

    except Exception as e:
        print(f"Batch operations test failed: {e}")
        raise


def test_retention_and_compression():
    """Tests retention policy and compression features."""
    print("Testing retention and compression features...")

    try:
        # Set up retention policy (30 days)
        print("Setting up retention policy...")
        result = timescaledb_client.cleanup_old_data(retention_days=30)
        print(f"Retention policy result: {result}")

        # Set up compression (7 days)
        print("Setting up compression policy...")
        result = timescaledb_client.compress_chunks(older_than_days=7)
        print(f"Compression policy result: {result}")

        print("Retention and compression test completed successfully.")

    except Exception as e:
        print(f"Retention and compression test failed: {e}")
        raise


if __name__ == "__main__":
    # Add the trading_system directory to the Python path
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    # Run all tests
    test_database()
    test_batch_operations()
    test_retention_and_compression()

    print("All TimescaleDB tests completed successfully!")
