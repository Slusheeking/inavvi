#!/usr/bin/env python3
"""
Generate Test Dataset for E2E Testing

This script generates a test dataset for the end-to-end test using data from a specific date.
The dataset includes:
- Price data for specified symbols (default: AAPL and MSFT)
- News data
- Market data

Usage:
    python generate_test_dataset.py [options]

Options:
    --date DATE           Date to generate data for (YYYY-MM-DD, default: 7 days ago)
    --time-of-day TOD     Time of day to include (market_open, intraday, power_hour, all, default: all)
    --name NAME           Name for the dataset (default: dataset_last_week)
    --symbols SYMBOLS     Comma-separated list of symbols to include (default: AAPL,MSFT)
    --data-source SOURCE  Data source to use (polygon, alpha_vantage, yahoo_finance, auto, default: auto)
    --no-news             Skip fetching news data
    --no-market-data      Skip fetching market data
    --list                List available datasets
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime, timedelta

from src.training.dataset_generator import DatasetGenerator
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("generate_test_dataset")

async def main():
    """Generate test dataset for E2E testing."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate test dataset for E2E testing")
    parser.add_argument("--date", type=str, help="Date to generate data for (YYYY-MM-DD)")
    parser.add_argument("--time-of-day", type=str, default="all", 
                        choices=["market_open", "intraday", "power_hour", "all"],
                        help="Time of day to include")
    parser.add_argument("--name", type=str, help="Name for the dataset")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT", 
                        help="Comma-separated list of symbols to include")
    parser.add_argument("--data-source", type=str, default="auto",
                        choices=["polygon", "alpha_vantage", "yahoo_finance", "auto"],
                        help="Data source to use")
    parser.add_argument("--no-news", action="store_true", help="Skip fetching news data")
    parser.add_argument("--no-market-data", action="store_true", help="Skip fetching market data")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    # Initialize dataset generator
    dataset_generator = DatasetGenerator()
    
    # List datasets if requested
    if args.list:
        datasets = dataset_generator.list_datasets()
        if datasets:
            print("Available datasets:")
            for dataset in datasets:
                print(f"  - {dataset}")
        else:
            print("No datasets available")
        return
    
    # Set date (default to 7 days ago, but ensure it's a weekday)
    if args.date:
        date = args.date
    else:
        # Get date from 7 days ago
        date_obj = datetime.now() - timedelta(days=7)
        # If it's a weekend, move to the previous Friday
        if date_obj.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            days_to_subtract = date_obj.weekday() - 4  # Move to Friday
            date_obj = date_obj - timedelta(days=days_to_subtract)
        date = date_obj.strftime("%Y-%m-%d")
    
    # Set dataset name (default based on date and time of day)
    if args.name:
        dataset_name = args.name
    else:
        date_str = date.replace("-", "")
        dataset_name = f"dataset_{date_str}_{args.time_of_day}"
    
    # Parse symbols
    symbols = args.symbols.split(",")
    
    # Generate dataset
    logger.info(f"Generating dataset '{dataset_name}' for date {date}")
    logger.info(f"Time of day: {args.time_of_day}")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Data source: {args.data_source}")
    
    # Generate dataset with a date range to ensure we get data
    # Use a 3-day window to increase chances of getting data
    start_date_obj = datetime.strptime(date, "%Y-%m-%d")
    end_date_obj = start_date_obj + timedelta(days=2)
    end_date = end_date_obj.strftime("%Y-%m-%d")
    
    print(f"Fetching data from {date} to {end_date}")
    
    # Generate dataset
    dataset = await dataset_generator.generate_dataset(
        symbols=symbols,
        start_date=date,
        end_date=end_date,  # Use a 3-day window
        time_of_day=args.time_of_day,
        data_source=args.data_source,
        include_news=not args.no_news,
        include_market_data=not args.no_market_data,
        dataset_name=dataset_name
    )
    
    # Print summary
    logger.info(f"Dataset '{dataset_name}' generated successfully")
    
    # Print data counts
    symbol_counts = {}
    for symbol in symbols:
        if symbol in dataset["price_data"]:
            symbol_counts[symbol] = len(dataset["price_data"][symbol])
    
    logger.info("Price data counts:")
    for symbol, count in symbol_counts.items():
        logger.info(f"  - {symbol}: {count} data points")
    
    if not args.no_news:
        logger.info(f"News data: {len(dataset['news_data'])} items")
    
    if not args.no_market_data:
        logger.info("Market data included")
    
    print(f"\nDataset '{dataset_name}' generated successfully!")
    print(f"To use this dataset in the E2E test, run:")
    print(f"USE_TEST_DATASET=true TEST_DATASET_NAME={dataset_name} python -m unittest tests/test_system_e2e.py")

if __name__ == "__main__":
    asyncio.run(main())
