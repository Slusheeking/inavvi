"""
Training utilities for ML models.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config.settings import settings
from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.polygon import PolygonAPI
from src.models.exit_optimization import ExitOptimizationModel
from src.models.pattern_recognition import PATTERN_CLASSES, PatternRecognitionModel
from src.models.ranking_model import MultiFactorRankingModel
from src.models.sentiment import FinancialSentimentModel
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("training")


async def download_training_data(
    symbols: List[str], days: int = 100, intraday: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Download historical data for training.

    Args:
        symbols: List of stock symbols
        days: Number of days of data to download
        intraday: Whether to download intraday data

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    logger.info(f"Downloading training data for {len(symbols)} symbols, {days} days of history")

    # Create Polygon client
    polygon_client = PolygonAPI()

    # Dictionary to store results
    data_dict = {}

    # Download data for each symbol
    for symbol in symbols:
        try:
            if intraday:
                # Get intraday data (1-minute bars)
                df = await polygon_client.get_intraday_bars(symbol, minutes=1, days=days)
            else:
                # Get daily data
                df = await polygon_client.get_daily_bars(symbol, days=days)

            if df is not None and not df.empty:
                logger.info(f"Downloaded {len(df)} bars for {symbol}")
                data_dict[symbol] = df
            else:
                logger.warning(f"No data found for {symbol}")
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {e}")

    logger.info(f"Downloaded data for {len(data_dict)} symbols")
    return data_dict


async def get_news_sentiment_data(
    symbols: List[str], days: int = 30
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get news sentiment data for training.

    Args:
        symbols: List of stock symbols
        days: Number of days of news to retrieve

    Returns:
        Dictionary mapping symbols to lists of news items
    """
    logger.info(f"Getting news sentiment data for {len(symbols)} symbols, {days} days of history")

    # Dictionary to store results
    news_dict = {}

    # Get time range
    time_to = datetime.now().strftime("%Y%m%dT%H%M")
    time_from = (datetime.now() - timedelta(days=days)).strftime("%Y%m%dT%H%M")

    # Process in batches to avoid rate limits
    batch_size = 5
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]

        try:
            # Get news for batch
            news_data = await alpha_vantage_client.get_news_sentiment(
                symbols=batch, time_from=time_from, time_to=time_to, limit=100
            )

            if news_data and "feed" in news_data:
                # Process each news item
                for item in news_data["feed"]:
                    # Check which symbols this item is relevant to
                    ticker_sentiments = item.get("ticker_sentiment", [])
                    for ticker_sentiment in ticker_sentiments:
                        ticker = ticker_sentiment.get("ticker")
                        if ticker in symbols:
                            # Add to dictionary
                            if ticker not in news_dict:
                                news_dict[ticker] = []

                            # Extract relevant information
                            news_item = {
                                "title": item.get("title", ""),
                                "summary": item.get("summary", ""),
                                "url": item.get("url", ""),
                                "time_published": item.get("time_published", ""),
                                "relevance_score": float(
                                    ticker_sentiment.get("relevance_score", 0)
                                ),
                                "sentiment_score": float(
                                    ticker_sentiment.get("ticker_sentiment_score", 0)
                                ),
                                "overall_sentiment_score": float(
                                    item.get("overall_sentiment_score", 0)
                                ),
                            }

                            news_dict[ticker].append(news_item)

            # Avoid rate limits
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error getting news for {batch}: {e}")

    # Log results
    total_news = sum(len(items) for items in news_dict.values())
    logger.info(f"Retrieved {total_news} news items for {len(news_dict)} symbols")

    return news_dict


async def generate_pattern_training_data(
    data_dict: Dict[str, pd.DataFrame], patterns_per_symbol: int = 5, lookback: int = 20
) -> Tuple[List[pd.DataFrame], List[int]]:
    """
    Generate training data for pattern recognition model.

    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        patterns_per_symbol: Number of pattern examples to generate per symbol
        lookback: Number of bars for each pattern

    Returns:
        Tuple of (data_samples, labels)
    """
    logger.info(f"Generating pattern training data from {len(data_dict)} symbols")

    # Lists to store results
    data_samples = []
    labels = []

    # Process each symbol
    for symbol, df in data_dict.items():
        if len(df) < lookback + 10:
            logger.warning(f"Not enough data for {symbol}, skipping")
            continue

        try:
            # Number of potential windows
            num_windows = len(df) - lookback

            # Randomly select windows
            for _ in range(patterns_per_symbol):
                # Random start index
                start_idx = np.random.randint(0, num_windows)

                # Extract window
                window = df.iloc[start_idx : start_idx + lookback].copy()

                # For simplicity, assign random pattern labels for demonstration
                # In a real implementation, you would use a more sophisticated method
                pattern_idx = np.random.randint(0, len(PATTERN_CLASSES))

                # Add to results
                data_samples.append(window)
                labels.append(pattern_idx)
        except Exception as e:
            logger.error(f"Error processing {symbol} for pattern data: {e}")

    logger.info(f"Generated {len(data_samples)} pattern samples")
    return data_samples, labels


async def generate_ranking_training_data(
    data_dict: Dict[str, pd.DataFrame],
    news_dict: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    forward_bars: int = 10,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Generate training data for ranking model.

    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        news_dict: Optional dictionary mapping symbols to news items
        forward_bars: Number of bars to look forward for returns

    Returns:
        Tuple of (data_dict_list, target_returns)
    """
    logger.info(f"Generating ranking training data from {len(data_dict)} symbols")

    # Lists to store results
    data_dict_list = []
    target_returns = []

    # Process each symbol
    for symbol, df in data_dict.items():
        if len(df) < 30:  # Need enough data for features and forward returns
            logger.warning(f"Not enough data for {symbol}, skipping")
            continue

        try:
            # Calculate forward returns for each point except the last 'forward_bars'
            for i in range(len(df) - forward_bars):
                # Current point
                current_idx = i
                current_data = df.iloc[: current_idx + 1].copy()

                # Future point
                future_idx = current_idx + forward_bars
                future_price = df.iloc[future_idx]["close"]
                current_price = df.iloc[current_idx]["close"]

                # Calculate forward return
                forward_return = (future_price / current_price - 1) * 100

                # Create data dictionary
                data_item = {
                    "symbol": symbol,
                    "ohlcv": current_data,
                    "timestamp": df.index[current_idx],
                }

                # Add news if available
                if news_dict and symbol in news_dict:
                    # Filter news up to current timestamp
                    current_time = df.index[current_idx]
                    relevant_news = [
                        news
                        for news in news_dict[symbol]
                        if pd.to_datetime(news.get("time_published", "1970-01-01")) <= current_time
                    ]
                    data_item["news"] = relevant_news

                # Add to results
                data_dict_list.append(data_item)
                target_returns.append(forward_return)
        except Exception as e:
            logger.error(f"Error processing {symbol} for ranking data: {e}")

    logger.info(f"Generated {len(data_dict_list)} ranking samples")
    return data_dict_list, target_returns


async def generate_exit_training_data(
    data_dict: Dict[str, pd.DataFrame], episodes_per_symbol: int = 3, max_steps: int = 30
) -> List[Dict[str, Any]]:
    """
    Generate training data for exit optimization model.

    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        episodes_per_symbol: Number of episodes to generate per symbol
        max_steps: Maximum number of steps per episode

    Returns:
        List of episode dictionaries
    """
    logger.info(f"Generating exit training data from {len(data_dict)} symbols")

    # List to store results
    episodes = []

    # Process each symbol
    for symbol, df in data_dict.items():
        if len(df) < 50:  # Need enough data for features
            logger.warning(f"Not enough data for {symbol}, skipping")
            continue

        try:
            # Generate episodes
            for _ in range(episodes_per_symbol):
                # Random entry point
                entry_idx = np.random.randint(20, len(df) - max_steps - 1)
                entry_price = df.iloc[entry_idx]["close"]

                # Initialize episode data
                states = []
                actions = []
                rewards = []

                # Initialize position
                position_size = 1.0  # Start with full position
                highest_price = entry_price
                current_step = 0

                # Simulate episode
                while position_size > 0 and current_step < max_steps:
                    # Current index
                    current_idx = entry_idx + current_step + 1

                    # Extract data for state
                    df.iloc[entry_idx : current_idx + 1].copy()
                    current_price = df.iloc[current_idx]["close"]

                    # Update highest price
                    highest_price = max(highest_price, current_price)

                    # Create position data
                    {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "entry_time": df.index[entry_idx],
                        "quantity": 10,  # Arbitrary for training
                        "stop_loss": entry_price * 0.95,  # 5% stop loss
                        "take_profit": entry_price * 1.1,  # 10% take profit
                        "trailing_stop": 2.0,  # 2% trailing stop
                    }

                    # Create state (features)
                    # In a real implementation, use _extract_features from ExitOptimizationModel
                    profit_pct = (current_price / entry_price - 1) * 100
                    time_in_trade = current_step / max_steps  # Normalize
                    price_to_high = current_price / highest_price - 1

                    # Simplified features for demonstration
                    state = [
                        profit_pct / 20,  # Normalize
                        time_in_trade,
                        (current_price / entry_price - 1),  # price_to_entry
                        price_to_high,
                        0.5,  # Placeholder for other features
                        0.5,
                        0.5,
                        0.0,
                        0.2,
                        1.0,
                    ]

                    # Decide action
                    # For demonstration, use a simple heuristic:
                    # - Hold if small profit/loss
                    # - Partial exit if decent profit
                    # - Full exit if large profit or loss
                    if profit_pct > 5.0 or profit_pct < -3.0:
                        action = 3  # exit_full
                        exit_size = 1.0
                    elif profit_pct > 2.0:
                        action = 2  # exit_half
                        exit_size = 0.5
                    elif profit_pct > 1.0:
                        action = 1  # exit_partial
                        exit_size = 0.33
                    else:
                        action = 0  # hold
                        exit_size = 0.0

                    # Calculate reward
                    # In a real implementation, use a more sophisticated reward function
                    if action == 0:  # hold
                        reward = 0.1  # Small reward for holding
                    else:
                        # Reward based on profit
                        reward = profit_pct * exit_size

                    # Update position
                    position_size -= position_size * exit_size

                    # Add to episode data
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    # Next step
                    current_step += 1

                    # End episode if full exit
                    if action == 3:  # exit_full
                        break

                # Add episode to results
                episode = {
                    "symbol": symbol,
                    "entry_idx": entry_idx,
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                }
                episodes.append(episode)
        except Exception as e:
            logger.error(f"Error processing {symbol} for exit data: {e}")

    logger.info(f"Generated {len(episodes)} exit episodes")
    return episodes


async def generate_sentiment_training_data(
    news_dict: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[str], List[int]]:
    """
    Generate training data for sentiment analysis model.

    Args:
        news_dict: Dictionary mapping symbols to news items

    Returns:
        Tuple of (texts, labels)
    """
    logger.info(f"Generating sentiment training data from {len(news_dict)} symbols")

    # Lists to store results
    texts = []
    labels = []

    # Process each symbol
    for symbol, news_items in news_dict.items():
        for news in news_items:
            try:
                # Extract text
                title = news.get("title", "")
                summary = news.get("summary", "")
                text = f"{title} {summary}".strip()

                if not text:
                    continue

                # Extract sentiment score
                sentiment_score = news.get("sentiment_score", 0)

                # Convert to label
                if sentiment_score > 0.2:
                    label = 2  # positive
                elif sentiment_score < -0.2:
                    label = 0  # negative
                else:
                    label = 1  # neutral

                # Add to results
                texts.append(text)
                labels.append(label)
            except Exception as e:
                logger.error(f"Error processing news item for {symbol}: {e}")

    logger.info(f"Generated {len(texts)} sentiment samples")
    return texts, labels


async def train_pattern_recognition_model():
    """Train the pattern recognition model."""
    logger.info("Training pattern recognition model...")

    try:
        # Get list of symbols from S&P 500
        symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]  # Example symbols

        # Download training data
        data_dict = await download_training_data(symbols, days=60, intraday=True)

        # Generate training data
        data_samples, labels = await generate_pattern_training_data(
            data_dict, patterns_per_symbol=20, lookback=20
        )

        if not data_samples:
            logger.error("No training data generated")
            return

        # Create and train model
        model = PatternRecognitionModel(model_path=None, lookback=20)

        # Train model
        history = model.train(
            train_data=data_samples,
            train_labels=labels,
            epochs=5,  # Use more epochs for real training
            batch_size=32,
            learning_rate=0.001,
        )

        logger.info("Pattern recognition model training completed")
        logger.info(
            f"Final loss: {history['train_loss'][-1]:.4f}, "
            f"Final accuracy: {history['train_acc'][-1]:.4f}"
        )

        # Save model
        model.save_model(settings.model.pattern_model_path)
        logger.info(f"Model saved to {settings.model.pattern_model_path}")
    except Exception as e:
        logger.error(f"Error training pattern recognition model: {e}")


async def train_ranking_model():
    """Train the multi-factor ranking model."""
    logger.info("Training multi-factor ranking model...")

    try:
        # Get list of symbols from S&P 500
        symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]  # Example symbols

        # Download training data
        data_dict = await download_training_data(symbols, days=100, intraday=False)

        # Get news data
        news_dict = await get_news_sentiment_data(symbols, days=30)

        # Generate training data
        data_dict_list, target_returns = await generate_ranking_training_data(
            data_dict, news_dict, forward_bars=5
        )

        if not data_dict_list:
            logger.error("No training data generated")
            return

        # Create and train model
        model = MultiFactorRankingModel(model_path=None)

        # Train model
        metrics = model.train(
            data=data_dict_list, labels=target_returns, test_size=0.2, random_state=42
        )

        logger.info("Multi-factor ranking model training completed")
        logger.info(
            f"Train score: {metrics['train_score']:.4f}, Test score: {metrics['test_score']:.4f}"
        )

        # Save model
        model.save_model(settings.model.ranking_model_path)
        logger.info(f"Model saved to {settings.model.ranking_model_path}")
    except Exception as e:
        logger.error(f"Error training multi-factor ranking model: {e}")


async def train_sentiment_model():
    """Train the sentiment analysis model."""
    logger.info("Training sentiment analysis model...")

    try:
        # Get list of symbols
        symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]  # Example symbols

        # Get news data
        news_dict = await get_news_sentiment_data(symbols, days=60)

        # Generate training data
        texts, labels = await generate_sentiment_training_data(news_dict)

        if not texts:
            logger.error("No training data generated")
            return

        # Create model
        model = FinancialSentimentModel(model_path=None, model_name="ProsusAI/finbert")

        # Fine-tune model
        history = model.fine_tune(
            texts=texts,
            labels=labels,
            epochs=3,  # Use more epochs for real training
            batch_size=16,
            learning_rate=2e-5,
        )

        logger.info("Sentiment analysis model training completed")
        logger.info(f"Final loss: {history['loss'][-1]:.4f}")

        # Save model
        model.save_model(settings.model.sentiment_model_path)
        logger.info(f"Model saved to {settings.model.sentiment_model_path}")
    except Exception as e:
        logger.error(f"Error training sentiment analysis model: {e}")


async def train_exit_optimization_model():
    """Train the exit optimization model."""
    logger.info("Training exit optimization model...")

    try:
        # Get list of symbols
        symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]  # Example symbols

        # Download training data
        data_dict = await download_training_data(symbols, days=60, intraday=True)

        # Generate training data
        episodes = await generate_exit_training_data(
            data_dict, episodes_per_symbol=10, max_steps=30
        )

        if not episodes:
            logger.error("No training data generated")
            return

        # Create model
        model = ExitOptimizationModel(model_path=None)

        # Train model
        metrics = model.train_ppo(
            training_data=episodes,
            epochs=20,  # Use more epochs for real training
            batch_size=16,
            lr=0.0005,
            gamma=0.99,
            eps_clip=0.2,
            k_epochs=4,
        )

        logger.info("Exit optimization model training completed")
        logger.info(f"Final reward: {metrics['epoch_rewards'][-1]:.4f}")

        # Save model
        model.save_model(settings.model.exit_model_path)
        logger.info(f"Model saved to {settings.model.exit_model_path}")
    except Exception as e:
        logger.error(f"Error training exit optimization model: {e}")


async def train_all_models():
    """Train all ML models."""
    logger.info("Starting training for all models...")

    # Create directories if they don't exist
    os.makedirs(settings.models_dir, exist_ok=True)

    # Train models sequentially
    await train_pattern_recognition_model()
    await train_ranking_model()
    await train_sentiment_model()
    await train_exit_optimization_model()

    logger.info("All models trained successfully")


if __name__ == "__main__":
    # Run training
    asyncio.run(train_all_models())
