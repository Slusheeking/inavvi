import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.models.exit_optimization import exit_optimization_model
# from src.training.data_fetcher import fetch_data # Keep this commented for now, will mock

# Create a simple mock for fetch_data to return a small, consistent dataset
def mock_fetch_data_for_exit_opt_test(symbols=None, timeframe="1d", data_days=5):
    """Mocks fetch_data to return a small DataFrame for testing."""
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data = {
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [99, 100, 101, 102, 103],
        'close': [101, 102, 103, 104, 105],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    df = pd.DataFrame(data, index=dates)
    # The test expects a dictionary of dataframes, keyed by symbol
    return {"AAPL": df}

@patch('src.training.data_fetcher.fetch_data', side_effect=mock_fetch_data_for_exit_opt_test)
def test_exit_optimization_model_evaluate_conditions(mock_fetch):
    """Test the evaluate_exit_conditions method of the exit optimization model."""
    # fetch_data is now mocked by mock_fetch
    data_dict = mock_fetch() 
    sample_ohlcv_data = data_dict["AAPL"]

    position_data = {
        "entry_price": 100.0,
        "entry_time": sample_ohlcv_data.index[0],
        "position_size": 1.0, # Represents 100% of the position
        "stop_loss": 95.0,
        "take_profit": 110.0,
        "trailing_stop": 2.0, # 2% trailing stop
        "max_time": 10 # Max 10 days in trade
    }
    
    # Ensure the model is loaded or initialized (as it's done globally in exit_optimization.py)
    # If the model isn't loaded by default, you might need to call model.load_model(path) here
    # or ensure it has a default initialized state.
    # For this test, we assume exit_optimization_model is a ready-to-use instance.

    result = exit_optimization_model.evaluate_exit_conditions(sample_ohlcv_data, position_data)
    
    assert result is not None
    assert "exit" in result
    assert "size" in result
    assert "reason" in result
    assert "confidence" in result
    assert "prediction" in result
    assert "manual_checks" in result
    assert "risk_metrics" in result
