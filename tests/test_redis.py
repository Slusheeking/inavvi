"""
Redis integration test for the trading system.

This script tests Redis functionality and integration with the trading system.
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.redis_client import redis_client
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("redis_test")

def test_basic_connectivity():
    """Test basic Redis connectivity."""
    logger.info("Testing basic Redis connectivity...")
    
    try:
        # Ping Redis server
        ping_result = redis_client._conn.ping()
        logger.info(f"Redis ping result: {ping_result}")
        
        # Get server info
        logger.info("Redis server info: Connected")
        
        return True
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        return False

def test_basic_operations():
    """Test basic Redis operations."""
    logger.info("Testing basic Redis operations...")
    
    try:
        # Test string operations
        key = "test:string"
        value = "test_value"
        redis_client.set(key, value)
        retrieved = redis_client.get(key)
        assert retrieved == value, f"String value mismatch: {retrieved} != {value}"
        logger.info("String operations: PASS")
        
        # Test integer operations
        key = "test:int"
        value = 12345
        redis_client.set(key, value)
        retrieved = redis_client.get(key)
        assert int(retrieved) == value, f"Integer value mismatch: {retrieved} != {value}"
        logger.info("Integer operations: PASS")
        
        # Test JSON operations
        key = "test:json"
        value = {"name": "Test", "value": 123, "nested": {"key": "value"}}
        redis_client.set(key, value)
        retrieved = redis_client.get(key)
        assert retrieved == value, f"JSON value mismatch: {retrieved} != {value}"
        logger.info("JSON operations: PASS")
        
        # Test expiry
        key = "test:expiry"
        value = "expiring_value"
        redis_client.set(key, value, expiry=1)  # 1 second expiry
        assert redis_client.get(key) == value
        time.sleep(1.5)  # Wait for expiry
        assert redis_client.get(key) is None, "Expiry failed"
        logger.info("Expiry operations: PASS")
        
        # Cleanup
        redis_client.delete("test:string")
        redis_client.delete("test:int")
        redis_client.delete("test:json")
        
        return True
    except Exception as e:
        logger.error(f"Error in basic operations: {e}")
        return False

def test_complex_data():
    """Test complex data types with Redis."""
    logger.info("Testing complex data operations...")
    
    try:
        # Test DataFrame
        key = "test:dataframe"
        df = pd.DataFrame({
            'A': np.random.randn(10),
            'B': np.random.randn(10),
            'C': pd.date_range('2020-01-01', periods=10)
        })
        redis_client.set(key, df)
        retrieved_df = redis_client.get(key)
        assert isinstance(retrieved_df, pd.DataFrame), "Retrieved object is not a DataFrame"
        assert df.shape == retrieved_df.shape, f"DataFrame shape mismatch: {df.shape} != {retrieved_df.shape}"
        logger.info("DataFrame operations: PASS")
        
        # Test NumPy array
        key = "test:numpy"
        arr = np.random.randn(5, 5)
        redis_client.set(key, arr)
        retrieved_arr = redis_client.get(key)
        assert isinstance(retrieved_arr, np.ndarray), "Retrieved object is not a NumPy array"
        assert arr.shape == retrieved_arr.shape, f"NumPy array shape mismatch: {arr.shape} != {retrieved_arr.shape}"
        assert np.allclose(arr, retrieved_arr), "NumPy array values don't match"
        logger.info("NumPy operations: PASS")
        
        # Cleanup
        redis_client.delete("test:dataframe")
        redis_client.delete("test:numpy")
        
        return True
    except Exception as e:
        logger.error(f"Error in complex data operations: {e}")
        return False

def test_trading_operations():
    """Test trading-specific Redis operations."""
    logger.info("Testing trading-specific operations...")
    
    try:
        # Test stock data
        symbol = "AAPL"
        price_data = {
            "price": {
                "last": 150.25,
                "open": 149.80,
                "high": 151.20,
                "low": 149.50,
                "volume": 25000000
            },
            "timestamp": datetime.now().isoformat()
        }
        redis_client.set_stock_data(symbol, price_data, "price")
        retrieved_data = redis_client.get_stock_data(symbol, "price")
        assert retrieved_data["price"]["last"] == price_data["price"]["last"], "Stock price data mismatch"
        logger.info("Stock data operations: PASS")
        
        # Test watchlist
        watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        redis_client.set_watchlist(watchlist)
        retrieved_watchlist = redis_client.get_watchlist()
        # Convert both lists to sets of strings for comparison
        watchlist_set = set(str(s) for s in watchlist)
        retrieved_set = set(str(s) for s in retrieved_watchlist)
        assert retrieved_set == watchlist_set, f"Watchlist mismatch: {retrieved_watchlist} != {watchlist}"
        
        # Test add/remove from watchlist
        redis_client.add_to_watchlist("TSLA")
        assert "TSLA" in redis_client.get_watchlist(), "Add to watchlist failed"
        redis_client.remove_from_watchlist("TSLA")
        assert "TSLA" not in redis_client.get_watchlist(), "Remove from watchlist failed"
        logger.info("Watchlist operations: PASS")
        
        # Test ranked candidates
        candidates = [
            {"symbol": "AAPL", "score": 0.95, "price": 150.25},
            {"symbol": "MSFT", "score": 0.92, "price": 290.10},
            {"symbol": "GOOGL", "score": 0.89, "price": 135.50}
        ]
        redis_client.set_ranked_candidates(candidates)
        retrieved_candidates = redis_client.get_ranked_candidates()
        assert len(retrieved_candidates) == len(candidates), "Candidates length mismatch"
        assert retrieved_candidates[0]["symbol"] == candidates[0]["symbol"], "Candidates order mismatch"
        logger.info("Ranked candidates operations: PASS")
        
        # Test position operations
        position_data = {
            "entry_price": 150.0,
            "quantity": 10,
            "side": "long",
            "stop_loss": 145.0,
            "take_profit": 160.0,
            "timestamp": datetime.now().isoformat()
        }
        redis_client.set_active_position(symbol, position_data)
        retrieved_position = redis_client.get_active_position(symbol)
        assert retrieved_position["entry_price"] == position_data["entry_price"], "Position data mismatch"
        
        # Test all positions
        all_positions = redis_client.get_all_active_positions()
        assert symbol in all_positions, "Missing position in get_all_active_positions"
        
        # Test position P&L update
        redis_client.update_position_pnl(symbol, 155.0)  # $5 higher than entry
        updated_position = redis_client.get_active_position(symbol)
        assert updated_position["unrealized_pnl"] == 50.0, f"P&L calculation error: {updated_position['unrealized_pnl']}"
        
        # Cleanup position
        redis_client.delete_active_position(symbol)
        assert redis_client.get_active_position(symbol) is None, "Position deletion failed"
        logger.info("Position operations: PASS")
        
        # Test system state
        system_state = {"state": "running", "timestamp": datetime.now().isoformat()}
        redis_client.set_system_state(system_state)
        retrieved_state = redis_client.get_system_state()
        assert retrieved_state["state"] == system_state["state"], "System state mismatch"
        
        # Test system state update
        redis_client.update_system_state(market_status="open")
        updated_state = redis_client.get_system_state()
        assert updated_state["market_status"] == "open", "System state update failed"
        logger.info("System state operations: PASS")
        
        # Test trading signals
        signal_data = {
            "reason": "crossover",
            "price": 155.25,
            "timestamp": datetime.now().isoformat(),
            "indicators": {"macd": 0.5, "rsi": 65}
        }
        redis_client.add_trading_signal(symbol, "entry", signal_data)
        retrieved_signal = redis_client.get_trading_signal(symbol, "entry")
        assert retrieved_signal["reason"] == signal_data["reason"], "Trading signal mismatch"
        
        # Clear signal
        redis_client.clear_trading_signal(symbol, "entry")
        assert redis_client.get_trading_signal(symbol, "entry") is None, "Signal clearing failed"
        logger.info("Trading signal operations: PASS")
        
        # Cleanup
        redis_client.delete(f"stocks:price:{symbol}")
        redis_client.set_watchlist([])
        redis_client.set_ranked_candidates([])
        redis_client.set_system_state({})
        
        return True
    except Exception as e:
        logger.error(f"Error in trading operations: {e}")
        return False

def test_performance():
    """Test Redis performance."""
    logger.info("Testing Redis performance...")
    
    try:
        n = 1000  # Number of operations
        
        # Test write performance
        start_time = time.time()
        for i in range(n):
            redis_client.set(f"perf:test:{i}", {"value": i, "data": f"data_{i}"})
        write_time = time.time() - start_time
        write_ops = n / write_time
        logger.info(f"Write performance: {write_ops:.2f} ops/sec ({n} operations in {write_time:.2f} seconds)")
        
        # Test read performance
        start_time = time.time()
        for i in range(n):
            redis_client.get(f"perf:test:{i}")
        read_time = time.time() - start_time
        read_ops = n / read_time
        logger.info(f"Read performance: {read_ops:.2f} ops/sec ({n} operations in {read_time:.2f} seconds)")
        
        # Cleanup
        for i in range(n):
            redis_client.delete(f"perf:test:{i}")
        
        return True
    except Exception as e:
        logger.error(f"Error in performance testing: {e}")
        return False

def main():
    """Run all Redis tests."""
    logger.info("=== Starting Redis Integration Tests ===")
    
    test_results = {
        "Basic Connectivity": test_basic_connectivity(),
        "Basic Operations": test_basic_operations(),
        "Complex Data": test_complex_data(),
        "Trading Operations": test_trading_operations(),
        "Performance": test_performance()
    }
    
    # Print summary
    logger.info("=== Test Results Summary ===")
    all_passed = True
    for test, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("✅ All Redis tests passed successfully!")
        logger.info("Redis is properly set up for the trading system.")
    else:
        logger.error("❌ Some Redis tests failed!")
        logger.error("Please check the logs for details.")
    
    return all_passed

if __name__ == "__main__":
    main()
