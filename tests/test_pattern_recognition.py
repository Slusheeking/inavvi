import pytest
import pandas as pd
import numpy as np
from src.models.pattern_recognition import pattern_recognition_model

def test_pattern_recognition_model_loaded():
    """Test that the pattern recognition model is loaded properly"""
    assert pattern_recognition_model is not None
    assert pattern_recognition_model.model is not None

def test_pattern_recognition_predict():
    """Test the pattern recognition model predict functionality"""
    # Create a sample OHLCV dataframe with 30 periods
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range(start='2025-01-01', periods=30)
    
    # Generate realistic stock price data
    base_price = 100.0
    volatility = 0.02
    
    # Generate random returns
    returns = np.random.normal(0, volatility, 30)
    
    # Calculate prices using cumulative returns
    prices = base_price * (1 + np.cumsum(returns))
    
    # Create realistic OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.015, 30)),
        'low': prices * (1 - np.random.uniform(0, 0.015, 30)),
        'close': prices * (1 + np.random.normal(0, 0.008, 30)),
        'volume': np.random.randint(100000, 1000000, 30)
    })
    
    # Set date as index as expected by most functions
    data.set_index('date', inplace=True)
    
    # Test the predict method
    prediction = pattern_recognition_model.predict(data)
    
    # Verify prediction format
    assert isinstance(prediction, dict)
    assert len(prediction) > 0
    assert all(isinstance(k, str) for k in prediction.keys())
    assert all(isinstance(v, float) for v in prediction.values())
    
    # Test the predict_pattern method
    pattern_name, confidence = pattern_recognition_model.predict_pattern(data)
    
    # Verify prediction_pattern results
    assert isinstance(pattern_name, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1.0
