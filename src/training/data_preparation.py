"""
Data preparation for model training.

Handles:
- Downloading historical data
- Feature engineering
- Dataset creation and preprocessing
- Dataset splitting and normalization
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config.settings import settings
from src.data_sources.polygon import polygon_client
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("data_preparation")


class DataPreparation:
    """
    Data preparation for model training.

    Responsibilities:
    - Download and prepare data for model training
    - Create features for different model types
    - Split data into training and validation sets
    - Normalize and preprocess data
    """

    def __init__(self):
        """Initialize data preparation."""
        # Create data directories if they don't exist
        os.makedirs(os.path.join(settings.data_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(settings.data_dir, "processed"), exist_ok=True)

        # Initialize scalers
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()

        logger.info("Data preparation initialized")

    async def download_training_data(
        self,
        symbols: List[str],
        days: int = 180,
        timeframe: str = "day",
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download historical data for training.

        Args:
            symbols: List of stock symbols
            days: Number of days of historical data
            timeframe: Timeframe for data ('day', 'hour', 'minute')
            force_refresh: Whether to force download even if data exists

        Returns:
            Dictionary mapping symbols to DataFrames with historical data
        """
        logger.info(f"Downloading {timeframe} data for {len(symbols)} symbols ({days} days)")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Format dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Download data for each symbol
        data = {}
        for symbol in symbols:
            # Check if data already exists
            filename = os.path.join(settings.data_dir, "raw", f"{symbol}_{timeframe}_{days}d.csv")

            if os.path.exists(filename) and not force_refresh:
                logger.info(f"Loading existing data for {symbol}")
                try:
                    df = pd.read_csv(filename, index_col="timestamp", parse_dates=True)
                    data[symbol] = df
                    continue
                except Exception as e:
                    logger.error(f"Error loading file for {symbol}: {e}")

            # Download data
            logger.info(f"Downloading data for {symbol}")
            try:
                if timeframe == "day":
                    df = await polygon_client.get_daily_bars(symbol, days=days)
                elif timeframe == "minute":
                    # For minute data, we might need to stitch multiple days
                    # We'll use a simplified approach for now
                    df = await polygon_client.get_intraday_bars(
                        symbol, minutes=1, days=min(days, 5)
                    )
                else:
                    # Use generic historical data method
                    df = await polygon_client.get_historical_data(
                        symbol, timeframe=timeframe, start_date=start_str, end_date=end_str
                    )

                if df is not None and not df.empty:
                    # Save to file
                    df.to_csv(filename)
                    data[symbol] = df
                    logger.info(f"Downloaded {len(df)} rows for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")
            except Exception as e:
                logger.error(f"Error downloading data for {symbol}: {e}")

            # Delay to avoid rate limits
            await asyncio.sleep(0.5)

        logger.info(f"Downloaded data for {len(data)} symbols")
        return data

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators and features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional features
        """
        # Make a copy to avoid modifying the original
        df_features = df.copy()

        # Ensure we have the required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df_features.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return df_features

        # Price and returns features
        df_features["returns"] = df_features["close"].pct_change()
        df_features["log_returns"] = np.log(df_features["close"] / df_features["close"].shift(1))
        df_features["range"] = df_features["high"] - df_features["low"]
        df_features["range_pct"] = df_features["range"] / df_features["close"].shift(1) * 100

        # Moving averages
        for period in [5, 10, 20, 50]:
            df_features[f"sma_{period}"] = df_features["close"].rolling(period).mean()
            df_features[f"ema_{period}"] = (
                df_features["close"].ewm(span=period, adjust=False).mean()
            )

        # Price relative to moving averages
        for period in [5, 10, 20, 50]:
            df_features[f"close_to_sma_{period}"] = (
                df_features["close"] / df_features[f"sma_{period}"] - 1
            )
            df_features[f"close_to_ema_{period}"] = (
                df_features["close"] / df_features[f"ema_{period}"] - 1
            )

        # Volatility indicators
        df_features["volatility_5d"] = df_features["returns"].rolling(5).std() * np.sqrt(252)
        df_features["volatility_10d"] = df_features["returns"].rolling(10).std() * np.sqrt(252)
        df_features["volatility_20d"] = df_features["returns"].rolling(20).std() * np.sqrt(252)

        # Volume indicators
        df_features["volume_sma_5"] = df_features["volume"].rolling(5).mean()
        df_features["volume_sma_10"] = df_features["volume"].rolling(10).mean()
        df_features["volume_ratio_5"] = df_features["volume"] / df_features["volume_sma_5"]
        df_features["volume_ratio_10"] = df_features["volume"] / df_features["volume_sma_10"]

        # RSI (Relative Strength Index)
        delta = df_features["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        df_features["rsi_14"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        for period in [20]:
            middle = df_features["close"].rolling(period).mean()
            stddev = df_features["close"].rolling(period).std()

            df_features[f"bb_upper_{period}"] = middle + 2 * stddev
            df_features[f"bb_middle_{period}"] = middle
            df_features[f"bb_lower_{period}"] = middle - 2 * stddev
            df_features[f"bb_width_{period}"] = (
                df_features[f"bb_upper_{period}"] - df_features[f"bb_lower_{period}"]
            ) / middle
            df_features[f"bb_position_{period}"] = (
                df_features["close"] - df_features[f"bb_lower_{period}"]
            ) / (df_features[f"bb_upper_{period}"] - df_features[f"bb_lower_{period}"])

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df_features["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df_features["close"].ewm(span=26, adjust=False).mean()
        df_features["macd"] = ema_12 - ema_26
        df_features["macd_signal"] = df_features["macd"].ewm(span=9, adjust=False).mean()
        df_features["macd_histogram"] = df_features["macd"] - df_features["macd_signal"]

        # Momentum indicators
        for period in [5, 10, 20]:
            df_features[f"momentum_{period}"] = (
                df_features["close"] / df_features["close"].shift(period) - 1
            )

        # Drop NaN values created by indicators
        df_features.dropna(inplace=True)

        return df_features

    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically for pattern recognition.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with pattern features
        """
        # First create technical features
        df_features = self.create_technical_features(df)

        # Normalize price series for better pattern recognition
        # We use a rolling window to normalize
        window = 20
        for col in ["open", "high", "low", "close"]:
            rolling_min = df_features[col].rolling(window).min()
            rolling_max = df_features[col].rolling(window).max()
            df_features[f"{col}_norm"] = (df_features[col] - rolling_min) / (
                rolling_max - rolling_min + 1e-8
            )

        # Calculate candlestick patterns
        df_features["body"] = abs(df_features["close"] - df_features["open"])
        df_features["upper_shadow"] = df_features["high"] - df_features[["open", "close"]].max(
            axis=1
        )
        df_features["lower_shadow"] = (
            df_features[["open", "close"]].min(axis=1) - df_features["low"]
        )

        df_features["body_pct"] = df_features["body"] / df_features["close"] * 100
        df_features["upper_shadow_pct"] = df_features["upper_shadow"] / df_features["close"] * 100
        df_features["lower_shadow_pct"] = df_features["lower_shadow"] / df_features["close"] * 100

        # Candlestick classification
        df_features["bullish"] = df_features["close"] > df_features["open"]
        df_features["doji"] = df_features["body_pct"] < 0.1

        # Identify potential patterns
        # Trend reversal patterns
        df_features["potential_hammer"] = (
            df_features["lower_shadow_pct"] > 2 * df_features["body_pct"]
        ) & (df_features["upper_shadow_pct"] < df_features["body_pct"])

        df_features["potential_shooting_star"] = (
            df_features["upper_shadow_pct"] > 2 * df_features["body_pct"]
        ) & (df_features["lower_shadow_pct"] < df_features["body_pct"])

        # Trend continuation patterns
        df_features["potential_breakout"] = (
            df_features["close"] > df_features["close"].rolling(20).max().shift(1)
        ) & (df_features["volume"] > df_features["volume"].rolling(20).mean() * 1.5)

        # Drop NaN values
        df_features.dropna(inplace=True)

        return df_features

    def create_target_variables(
        self, df: pd.DataFrame, lookahead_periods: int = 5, threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Create target variables for supervised learning.

        Args:
            df: DataFrame with OHLCV data
            lookahead_periods: Number of periods to look ahead for returns
            threshold: Threshold for classifying significant price moves

        Returns:
            DataFrame with target variables
        """
        # Make a copy to avoid modifying the original
        df_with_targets = df.copy()

        # Calculate future returns
        future_close = df_with_targets["close"].shift(-lookahead_periods)
        df_with_targets[f"future_return_{lookahead_periods}"] = (
            future_close / df_with_targets["close"] - 1
        )

        # Create binary target variables
        df_with_targets[f"target_up_{lookahead_periods}"] = (
            df_with_targets[f"future_return_{lookahead_periods}"] > threshold
        )
        df_with_targets[f"target_down_{lookahead_periods}"] = (
            df_with_targets[f"future_return_{lookahead_periods}"] < -threshold
        )

        # Convert to integer for classification
        df_with_targets[f"target_up_{lookahead_periods}"] = df_with_targets[
            f"target_up_{lookahead_periods}"
        ].astype(int)
        df_with_targets[f"target_down_{lookahead_periods}"] = df_with_targets[
            f"target_down_{lookahead_periods}"
        ].astype(int)

        # Multi-class target (-1 for down, 0 for neutral, 1 for up)
        df_with_targets[f"target_direction_{lookahead_periods}"] = 0
        df_with_targets.loc[
            df_with_targets[f"target_up_{lookahead_periods}"] == 1,
            f"target_direction_{lookahead_periods}",
        ] = 1
        df_with_targets.loc[
            df_with_targets[f"target_down_{lookahead_periods}"] == 1,
            f"target_direction_{lookahead_periods}",
        ] = -1

        # Remove future data (where target is NaN)
        df_with_targets.dropna(subset=[f"future_return_{lookahead_periods}"], inplace=True)

        return df_with_targets

    def prepare_training_dataset(
        self,
        data: Dict[str, pd.DataFrame],
        feature_type: str = "technical",
        lookahead_periods: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple:
        """
        Prepare a complete training dataset.

        Args:
            data: Dictionary of DataFrames with OHLCV data
            feature_type: Type of features to create ('technical' or 'pattern')
            lookahead_periods: Number of periods to look ahead for target
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        logger.info(f"Preparing {feature_type} features for {len(data)} symbols")

        all_features = []
        symbols = []

        # Process each symbol
        for symbol, df in data.items():
            if len(df) < 100:  # Skip symbols with too little data
                logger.warning(f"Skipping {symbol}: insufficient data ({len(df)} rows)")
                continue

            try:
                # Create features based on the specified type
                if feature_type == "technical":
                    df_features = self.create_technical_features(df)
                elif feature_type == "pattern":
                    df_features = self.create_pattern_features(df)
                else:
                    logger.error(f"Unknown feature type: {feature_type}")
                    continue

                # Create target variables
                df_with_targets = self.create_target_variables(df_features, lookahead_periods)

                if len(df_with_targets) > 0:
                    # Add symbol column as a categorical feature
                    df_with_targets["symbol"] = symbol

                    # Store processed data
                    all_features.append(df_with_targets)
                    symbols.append(symbol)

                    logger.info(f"Processed {symbol}: {len(df_with_targets)} samples")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        if not all_features:
            logger.error("No valid features generated")
            return None, None, None, None, None

        # Combine all data
        combined_df = pd.concat(all_features)

        # Save the combined dataset
        os.makedirs(os.path.join(settings.data_dir, "processed"), exist_ok=True)
        combined_df.to_csv(
            os.path.join(settings.data_dir, "processed", f"{feature_type}_features.csv")
        )

        # Get feature names
        target_col = f"target_direction_{lookahead_periods}"
        feature_cols = [
            col
            for col in combined_df.columns
            if col
            not in [
                target_col,
                f"future_return_{lookahead_periods}",
                f"target_up_{lookahead_periods}",
                f"target_down_{lookahead_periods}",
            ]
        ]

        # Handle non-numeric columns (like 'symbol')
        numeric_features = (
            combined_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        )

        # Convert to numpy arrays
        X = combined_df[numeric_features].values
        y = combined_df[target_col].values

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale features
        self.feature_scaler.fit(X_train)
        X_train_scaled = self.feature_scaler.transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)

        logger.info(
            f"Created dataset with {X_train.shape[0]} training and {X_test.shape[0]} testing samples"
        )

        return X_train_scaled, X_test_scaled, y_train, y_test, numeric_features

    def prepare_pattern_dataset(
        self, data: Dict[str, pd.DataFrame], window_size: int = 20
    ) -> Tuple:
        """
        Prepare dataset specifically for pattern recognition.

        Args:
            data: Dictionary of DataFrames with OHLCV data
            window_size: Size of the sliding window for patterns

        Returns:
            Tuple of (X, y, pattern_labels)
        """
        logger.info(f"Preparing pattern dataset with window size {window_size}")

        X_windows = []
        y_labels = []
        symbols = []

        # Define pattern labels (same as in pattern_recognition_model.py)
        pattern_classes = [
            "no_pattern",  # 0: No specific pattern
            "breakout",  # 1: Price breaking out of a consolidation
            "reversal",  # 2: Price reversing its trend
            "continuation",  # 3: Continuation of existing trend
            "flag",  # 4: Bull/bear flag pattern
            "triangle",  # 5: Triangle pattern
            "head_shoulders",  # 6: Head and shoulders pattern
            "double_top",  # 7: Double top pattern
            "double_bottom",  # 8: Double bottom pattern
        ]

        # Process each symbol
        for symbol, df in data.items():
            if len(df) < window_size + 20:  # Need enough data
                continue

            try:
                # Create features for pattern detection
                df_features = self.create_pattern_features(df)

                # Create sliding windows
                for i in range(len(df_features) - window_size):
                    window = df_features.iloc[i : i + window_size]

                    # Extract OHLCV data
                    ohlcv_window = window[["open", "high", "low", "close", "volume"]].values

                    # Detect pattern (simplified logic - in reality would use a more sophisticated approach)
                    # Here we're just identifying some common patterns based on price action

                    # Check for breakout: price breaks above recent resistance
                    if (window["close"].iloc[-1] > window["high"].iloc[:-1].max()) and (
                        window["volume"].iloc[-1] > window["volume"].iloc[:-1].mean() * 1.5
                    ):
                        pattern = 1  # breakout

                    # Check for reversal: price changes direction
                    elif (
                        (
                            window["close"].pct_change().iloc[-3:].mean()
                            * window["close"].pct_change().iloc[-6:-3].mean()
                        )
                        < 0
                    ) and abs(window["close"].pct_change().iloc[-3:].mean()) > 0.02:
                        pattern = 2  # reversal

                    # Check for continuation: price continues in the same direction after a pause
                    elif (
                        (abs(window["close"].pct_change().iloc[-10:-5].mean()) < 0.01)
                        and (abs(window["close"].pct_change().iloc[-5:].mean()) > 0.01)
                        and (
                            (
                                window["close"].pct_change().iloc[-5:].mean()
                                * window["close"].pct_change().iloc[-15:-10].mean()
                            )
                            > 0
                        )
                    ):
                        pattern = 3  # continuation

                    # Check for flag pattern: short consolidation in a trend
                    elif (abs(window["close"].pct_change().iloc[-5:].mean()) < 0.01) and (
                        abs(window["close"].pct_change().iloc[-10:-5].mean()) > 0.02
                    ):
                        pattern = 4  # flag

                    # Check for triangle: decreasing volatility
                    elif (window["bb_width_20"].iloc[-1] < window["bb_width_20"].iloc[-5]) and (
                        window["bb_width_20"].iloc[-5] < window["bb_width_20"].iloc[-10]
                    ):
                        pattern = 5  # triangle

                    # Add other pattern checks here
                    # For simplicity, we're not implementing all pattern detections

                    else:
                        pattern = 0  # no pattern

                    # Store the window and pattern
                    X_windows.append(ohlcv_window)
                    y_labels.append(pattern)
                    symbols.append(symbol)

            except Exception as e:
                logger.error(f"Error processing {symbol} for pattern detection: {e}")

        # Convert to numpy arrays
        X = np.array(X_windows)
        y = np.array(y_labels)

        # Save the dataset
        np.save(os.path.join(settings.data_dir, "processed", "pattern_X.npy"), X)
        np.save(os.path.join(settings.data_dir, "processed", "pattern_y.npy"), y)

        logger.info(f"Created pattern dataset with {len(X)} samples")

        return X, y, pattern_classes

    def prepare_sentiment_dataset(
        self, news_data: List[Dict], labels: Optional[List[int]] = None
    ) -> Tuple:
        """
        Prepare dataset for sentiment analysis.

        Args:
            news_data: List of news articles
            labels: Optional list of sentiment labels (for supervised learning)

        Returns:
            Tuple of (texts, labels)
        """
        logger.info(f"Preparing sentiment dataset with {len(news_data)} articles")

        texts = []
        symbols = []
        timestamps = []

        # Extract text and metadata
        for article in news_data:
            text = article.get("title", "") + " " + article.get("description", "")
            texts.append(text)

            symbols.append(article.get("tickers", ["unknown"])[0])
            timestamps.append(article.get("published_utc", ""))

        # Save the raw texts
        with open(os.path.join(settings.data_dir, "processed", "sentiment_texts.txt"), "w") as f:
            for text in texts:
                f.write(text + "\n")

        # If no labels provided, return just the texts
        if labels is None:
            return texts, None

        # Ensure labels match texts
        if len(labels) != len(texts):
            logger.error(
                f"Number of labels ({len(labels)}) doesn't match number of texts ({len(texts)})"
            )
            return texts, None

        return texts, np.array(labels)

    def prepare_exit_optimization_dataset(self, data: Dict[str, pd.DataFrame]) -> Tuple:
        """
        Prepare dataset for exit optimization using reinforcement learning.

        Args:
            data: Dictionary of DataFrames with OHLCV data

        Returns:
            List of training episodes
        """
        logger.info(f"Preparing exit optimization dataset with {len(data)} symbols")

        training_episodes = []

        # Process each symbol
        for symbol, df in data.items():
            if len(df) < 100:  # Skip symbols with too little data
                continue

            try:
                # Create technical features
                df_features = self.create_technical_features(df)

                # Simulate trades for training
                for i in range(20, len(df_features) - 20):
                    # Simulate entering a position
                    entry_price = df_features["close"].iloc[i]
                    entry_idx = i

                    # Track the trade
                    states = []
                    actions = []
                    rewards = []

                    # Simulate holding the position for up to 20 bars
                    for j in range(1, min(21, len(df_features) - i)):
                        # Current state (feature vector)
                        current_idx = i + j
                        current_price = df_features["close"].iloc[current_idx]

                        # Calculate profit percentage
                        profit_pct = (current_price / entry_price - 1) * 100

                        # Create state features
                        state_features = [
                            profit_pct,  # Current profit percentage
                            j / 20,  # Normalized time in trade
                            current_price / entry_price - 1,  # Price relative to entry
                            current_price / df_features["high"].iloc[entry_idx:current_idx].max()
                            - 1,  # Price relative to high
                            current_price / df_features["low"].iloc[entry_idx:current_idx].min()
                            - 1,  # Price relative to low
                            df_features["rsi_14"].iloc[current_idx] / 100,  # RSI (normalized)
                            df_features["bb_position_20"].iloc[current_idx],  # BB position
                            df_features["macd_histogram"].iloc[current_idx],  # MACD histogram
                            df_features["volatility_5d"].iloc[current_idx],  # Volatility
                            df_features["volume_ratio_5"].iloc[current_idx]
                            / 5,  # Volume ratio (normalized)
                        ]

                        # Determine action
                        # 0: hold, 1: exit 1/3, 2: exit 1/2, 3: exit full

                        # Simple exit strategy for generating training data:
                        # - Take profit at +3%
                        # - Cut loss at -2%
                        # - Scale out if momentum weakening
                        # - Full exit if strong reversal signal

                        if profit_pct >= 3.0:
                            action = 3  # Full exit at good profit
                            reward = 1.0  # Good reward for profitable exit
                        elif profit_pct <= -2.0:
                            action = 3  # Full exit to stop loss
                            reward = -0.5  # Negative reward but not as bad as max loss
                        elif j >= 19:  # Near max holding period
                            action = 3  # Full exit at end of holding period
                            reward = (
                                0.1 if profit_pct > 0 else -0.1
                            )  # Small reward/penalty based on P&L
                        elif profit_pct > 0 and df_features["macd_histogram"].iloc[current_idx] < 0:
                            action = 2  # Exit half on weakening momentum
                            reward = 0.3  # Moderate reward for locking in some profit
                        elif profit_pct > 0 and df_features["rsi_14"].iloc[current_idx] > 70:
                            action = 1  # Exit 1/3 when overbought
                            reward = 0.2  # Small reward for scaling out
                        else:
                            action = 0  # Hold
                            reward = 0.0  # Neutral reward for holding

                        # Store state, action, reward
                        states.append(state_features)
                        actions.append(action)
                        rewards.append(reward)

                        # If we took an exit action, end this episode
                        if action > 0:
                            break

                    # Create episode
                    episode = {
                        "symbol": symbol,
                        "entry_idx": entry_idx,
                        "entry_price": entry_price,
                        "states": states,
                        "actions": actions,
                        "rewards": rewards,
                    }

                    training_episodes.append(episode)

            except Exception as e:
                logger.error(f"Error creating exit optimization data for {symbol}: {e}")

        logger.info(f"Created {len(training_episodes)} training episodes for exit optimization")

        # Save the episodes
        import pickle

        with open(os.path.join(settings.data_dir, "processed", "exit_episodes.pkl"), "wb") as f:
            pickle.dump(training_episodes, f)

        return training_episodes


# Create global instance
data_preparation = DataPreparation()
