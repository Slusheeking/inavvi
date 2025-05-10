"""
Script to update all models in the trading system.

This script:
1. Fixes the XGBoost model compatibility issue
2. Runs the existing model training pipeline to update all models
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import training functions
from src.training import train_all_models

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("update_models")

async def main():
    """Run the model update process."""
    logger.info("Starting model update process")
    
    # Check if XGBoost model was fixed
    ranking_model_path = Path("models/ranking_model.pkl")
    if not ranking_model_path.exists():
        logger.error("Ranking model file not found. Please run fix_xgboost_model.py first.")
        return
    
    # Run the training pipeline to update all models
    logger.info("Running training pipeline to update all models")
    await train_all_models()
    
    logger.info("Model update process completed")

if __name__ == "__main__":
    asyncio.run(main())
