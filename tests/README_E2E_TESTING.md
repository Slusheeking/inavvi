# End-to-End Testing with Generated Datasets

This document explains how to use the dataset generator to create test datasets for end-to-end testing of the trading system.

## Overview

The E2E test (`test_system_e2e.py`) can run in two modes:
1. **Live Data Mode**: Fetches real-time data from APIs (default)
2. **Dataset Mode**: Uses a pre-generated dataset for consistent testing

Using pre-generated datasets offers several advantages:
- Consistent test results across runs
- No dependency on external APIs during testing
- Ability to test with specific market conditions (e.g., bullish/bearish days)
- Faster test execution

## Generating Test Datasets

The `generate_test_dataset.py` script creates test datasets with configurable parameters.

### Basic Usage

```bash
# Generate a dataset with default settings (last week's data for AAPL and MSFT)
./generate_test_dataset.py

# Generate a dataset for a specific date
./generate_test_dataset.py --date 2025-05-01

# Generate a dataset with specific time of day
./generate_test_dataset.py --time-of-day market_open

# Generate a dataset with custom symbols
./generate_test_dataset.py --symbols AAPL,MSFT,GOOGL,AMZN

# List all available datasets
./generate_test_dataset.py --list
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--date DATE` | Date to generate data for (YYYY-MM-DD) | 7 days ago |
| `--time-of-day TOD` | Time of day to include (market_open, intraday, power_hour, all) | all |
| `--name NAME` | Name for the dataset | dataset_YYYYMMDD_timeofday |
| `--symbols SYMBOLS` | Comma-separated list of symbols to include | AAPL,MSFT |
| `--data-source SOURCE` | Data source to use (polygon, alpha_vantage, yahoo_finance, auto) | auto |
| `--no-news` | Skip fetching news data | (include news) |
| `--no-market-data` | Skip fetching market data | (include market data) |
| `--list` | List available datasets | |

### Time of Day Options

- **market_open**: 9:30 AM - 10:30 AM ET (first hour of trading)
- **intraday**: 10:30 AM - 3:00 PM ET (middle of trading day)
- **power_hour**: 3:00 PM - 4:00 PM ET (last hour of trading)
- **all**: All trading hours

## Running E2E Tests with Datasets

To run the E2E test with a generated dataset:

```bash
# Run with a specific dataset
USE_TEST_DATASET=true TEST_DATASET_NAME=dataset_20250504_all python -m unittest tests/test_system_e2e.py

# Run with a specific dataset and date
USE_TEST_DATASET=true TEST_DATASET_NAME=my_custom_dataset TEST_DATASET_DATE=2025-05-01 python -m unittest tests/test_system_e2e.py
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_TEST_DATASET` | Whether to use a test dataset (true/false) | false |
| `TEST_DATASET_NAME` | Name of the dataset to use | dataset_last_week |
| `TEST_DATASET_DATE` | Date to use if dataset doesn't exist | 7 days ago |
| `SKIP_LIVE_TESTS` | Skip tests that require live API access | false |

## Examples

### Generate Datasets for Different Market Conditions

```bash
# Generate dataset for market open on a specific date
./generate_test_dataset.py --date 2025-05-01 --time-of-day market_open --name dataset_bullish_open

# Generate dataset for power hour on a different date
./generate_test_dataset.py --date 2025-05-02 --time-of-day power_hour --name dataset_bearish_close

# Generate dataset with specific symbols
./generate_test_dataset.py --symbols TSLA,NVDA,AMD --name dataset_tech_stocks
```

### Run Tests with Generated Datasets

```bash
# Run test with bullish market open dataset
USE_TEST_DATASET=true TEST_DATASET_NAME=dataset_bullish_open python -m unittest tests/test_system_e2e.py

# Run test with bearish power hour dataset
USE_TEST_DATASET=true TEST_DATASET_NAME=dataset_bearish_close python -m unittest tests/test_system_e2e.py

# Run test with tech stocks dataset
USE_TEST_DATASET=true TEST_DATASET_NAME=dataset_tech_stocks python -m unittest tests/test_system_e2e.py
```

## Dataset Structure

Generated datasets are stored in JSON format in the `data/test_datasets` directory. Each dataset contains:

- **Metadata**: Information about the dataset (date, symbols, etc.)
- **Price Data**: OHLCV data for each symbol
- **News Data**: News articles related to the symbols
- **Market Data**: Market status, sector performance, and economic indicators
- **Labels**: Optional labels for expected outcomes (can be added later)

## Adding Labels to Datasets

You can add labels to existing datasets to specify expected outcomes for testing:

```python
from src.training.dataset_generator import DatasetGenerator

# Initialize dataset generator
dataset_generator = DatasetGenerator()

# Add labels to an existing dataset
dataset_generator.add_labels("dataset_name", {
    "expected_bullish": ["AAPL"],
    "expected_bearish": ["MSFT"],
    "expected_trades": 1,
    "expected_no_trades": 1
})
```

These labels can be used in the E2E test to verify that the system makes the expected decisions.
