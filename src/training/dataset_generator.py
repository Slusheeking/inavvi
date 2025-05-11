"""
Dataset Generator for Testing

This module provides functionality for generating datasets for testing:
- Create datasets with specific dates and times of day
- Use production data sources (Polygon, Alpha Vantage, Yahoo Finance)
- Label data for expected outcomes
- Save and load datasets for consistent testing
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from typing import Dict, List, Tuple, Any, Optional, Union

from src.config.settings import settings
from src.utils.logging import setup_logger
from src.utils.redis_client import redis_client
from src.data_sources.polygon import polygon_client
from src.data_sources.alpha_vantage import alpha_vantage_client
from src.data_sources.yahoo_finance import yahoo_finance_client

# Set up logger
logger = setup_logger("dataset_generator")

class DatasetGenerator:
    """
    Dataset Generator for creating test datasets with specific characteristics.
    
    Features:
    - Generate datasets for specific dates and times
    - Use real data from production data sources
    - Label data for expected outcomes
    - Save and load datasets for consistent testing
    """
    
    def __init__(
        self,
        data_dir: str = None,
        use_polygon: bool = True,
        use_alpha_vantage: bool = True,
        use_yahoo_finance: bool = True,
        use_redis_cache: bool = True
    ):
        """
        Initialize the dataset generator.
        
        Args:
            data_dir: Directory to store datasets
            use_polygon: Whether to use Polygon.io API
            use_alpha_vantage: Whether to use Alpha Vantage API
            use_yahoo_finance: Whether to use Yahoo Finance API
            use_redis_cache: Whether to use Redis for caching data
        """
        # Configuration
        self.data_dir = data_dir or os.path.join(settings.data_dir, "test_datasets")
        self.use_polygon = use_polygon
        self.use_alpha_vantage = use_alpha_vantage
        self.use_yahoo_finance = use_yahoo_finance
        self.use_redis_cache = use_redis_cache
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize clients
        self.polygon = polygon_client
        self.alpha_vantage = alpha_vantage_client
        self.yahoo_finance = yahoo_finance_client
        
        logger.info("Dataset Generator initialized")
        logger.info(f"Data sources: Polygon={use_polygon}, Alpha Vantage={use_alpha_vantage}, Yahoo Finance={use_yahoo_finance}")

    async def generate_dataset(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        time_of_day: str = "all",
        data_source: str = "auto",
        include_news: bool = True,
        include_market_data: bool = True,
        dataset_name: str = None,
        labels: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Generate a dataset for testing.
        
        Args:
            symbols: List of stock symbols to include
            start_date: Start date for the dataset (YYYY-MM-DD or datetime)
            end_date: End date for the dataset (YYYY-MM-DD or datetime, defaults to start_date)
            time_of_day: Time of day to include ('market_open', 'intraday', 'power_hour', 'all')
            data_source: Data source to use ('polygon', 'alpha_vantage', 'yahoo_finance', 'auto')
            include_news: Whether to include news data
            include_market_data: Whether to include market data
            dataset_name: Name for the dataset (defaults to date-based name)
            labels: Optional labels for expected outcomes
            
        Returns:
            Dictionary containing the generated dataset
        """
        # Process dates
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        
        if end_date is None:
            end_date = start_date
        elif isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        # Generate dataset name if not provided
        if dataset_name is None:
            date_str = start_date.strftime("%Y%m%d")
            dataset_name = f"dataset_{date_str}_{time_of_day}"
        
        logger.info(f"Generating dataset '{dataset_name}' for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        
        # Initialize dataset structure
        dataset = {
            "metadata": {
                "name": dataset_name,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "time_of_day": time_of_day,
                "data_source": data_source,
                "symbols": symbols,
                "created_at": datetime.now().isoformat(),
                "includes_news": include_news,
                "includes_market_data": include_market_data
            },
            "price_data": {},
            "news_data": [],
            "market_data": {},
            "labels": labels or {}
        }
        
        # Fetch price data for each symbol
        for symbol in symbols:
            try:
                price_data = await self._fetch_price_data(
                    symbol, 
                    start_date, 
                    end_date, 
                    time_of_day, 
                    data_source
                )
                
                if price_data is not None and not price_data.empty:
                    # Convert DataFrame to dictionary for JSON serialization
                    dataset["price_data"][symbol] = price_data.reset_index().to_dict(orient="records")
                    logger.info(f"Fetched {len(price_data)} price records for {symbol}")
                else:
                    logger.warning(f"No price data found for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching price data for {symbol}: {e}")
        
        # Fetch news data if requested
        if include_news:
            try:
                news_data = await self._fetch_news_data(symbols, start_date, end_date, data_source)
                dataset["news_data"] = news_data
                logger.info(f"Fetched {len(news_data)} news items")
            except Exception as e:
                logger.error(f"Error fetching news data: {e}")
        
        # Fetch market data if requested
        if include_market_data:
            try:
                market_data = await self._fetch_market_data(start_date, end_date, data_source)
                dataset["market_data"] = market_data
                logger.info(f"Fetched market data")
            except Exception as e:
                logger.error(f"Error fetching market data: {e}")
        
        # Save dataset to disk
        self._save_dataset(dataset, dataset_name)
        
        return dataset

    async def _fetch_price_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        time_of_day: str,
        data_source: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch price data for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            time_of_day: Time of day filter
            data_source: Data source to use
            
        Returns:
            DataFrame with price data if successful, None otherwise
        """
        # Determine which data source to use
        if data_source == "auto":
            # Try Polygon first, then Alpha Vantage, then Yahoo Finance
            sources = []
            if self.use_polygon:
                sources.append("polygon")
            if self.use_alpha_vantage:
                sources.append("alpha_vantage")
            if self.use_yahoo_finance:
                sources.append("yahoo_finance")
        else:
            sources = [data_source]
        
        # Try each source until successful
        for source in sources:
            try:
                if source == "polygon" and self.use_polygon:
                    # Calculate days between end_date and start_date
                    days = (end_date - start_date).days + 1
                    df = await self.polygon.get_intraday_bars(symbol, minutes=1, days=days)
                    
                    if df is not None and not df.empty:
                        # Filter by date range
                        df = self._filter_by_date_range(df, start_date, end_date)
                        # Filter by time of day
                        df = self._filter_by_time_of_day(df, time_of_day)
                        return df
                
                elif source == "alpha_vantage" and self.use_alpha_vantage:
                    df = await self.alpha_vantage.get_intraday_prices(symbol, interval="1min")
                    
                    if df is not None and not df.empty:
                        # Filter by date range
                        df = self._filter_by_date_range(df, start_date, end_date)
                        # Filter by time of day
                        df = self._filter_by_time_of_day(df, time_of_day)
                        return df
                
                elif source == "yahoo_finance" and self.use_yahoo_finance:
                    df = await self.yahoo_finance.get_intraday_prices(symbol)
                    
                    if df is not None and not df.empty:
                        # Filter by date range
                        df = self._filter_by_date_range(df, start_date, end_date)
                        # Filter by time of day
                        df = self._filter_by_time_of_day(df, time_of_day)
                        return df
            
            except Exception as e:
                logger.warning(f"Error fetching price data for {symbol} from {source}: {e}")
        
        # If all sources failed, return None
        logger.warning(f"All data sources failed for {symbol}")
        return None

    def _filter_by_date_range(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            df: DataFrame to filter
            start_date: Start date
            end_date: End date
            
        Returns:
            Filtered DataFrame
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, attempting to convert")
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                logger.error("Cannot filter by date range: no datetime index or timestamp column")
                return df
        
        # Convert dates to pandas Timestamp for consistent comparison
        start = pd.Timestamp(start_date.date())
        end = pd.Timestamp(end_date.date()) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Filter by date range
        return df[(df.index >= start) & (df.index <= end)]

    def _filter_by_time_of_day(self, df: pd.DataFrame, time_of_day: str) -> pd.DataFrame:
        """
        Filter DataFrame by time of day.
        
        Args:
            df: DataFrame to filter
            time_of_day: Time of day filter
            
        Returns:
            Filtered DataFrame
        """
        if time_of_day == "all":
            return df
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex, cannot filter by time of day")
            return df
        
        # Define time ranges for different parts of the trading day
        # Market open: 9:30 AM - 10:30 AM ET
        # Intraday: 10:30 AM - 3:00 PM ET
        # Power hour: 3:00 PM - 4:00 PM ET
        
        if time_of_day == "market_open":
            start_time = time(9, 30)
            end_time = time(10, 30)
        elif time_of_day == "intraday":
            start_time = time(10, 30)
            end_time = time(15, 0)
        elif time_of_day == "power_hour":
            start_time = time(15, 0)
            end_time = time(16, 0)
        else:
            logger.warning(f"Unknown time_of_day: {time_of_day}, returning all data")
            return df
        
        # Filter by time of day
        mask = (df.index.time >= start_time) & (df.index.time < end_time)
        return df[mask]

    async def _fetch_news_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        data_source: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch news data for symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            data_source: Data source to use
            
        Returns:
            List of news items
        """
        news_items = []
        
        # Determine which data source to use
        if data_source == "auto":
            # Try Alpha Vantage first, then Polygon
            sources = []
            if self.use_alpha_vantage:
                sources.append("alpha_vantage")
            if self.use_polygon:
                sources.append("polygon")
        else:
            sources = [data_source]
        
        # Try each source until successful
        for source in sources:
            try:
                if source == "alpha_vantage" and self.use_alpha_vantage:
                    # Alpha Vantage news endpoint
                    for symbol in symbols:
                        symbol_news = await self.alpha_vantage.get_symbol_news(symbol, limit=10)
                        if symbol_news:
                            # Filter by date range
                            filtered_news = []
                            for item in symbol_news:
                                if "time_published" in item:
                                    try:
                                        pub_date = datetime.fromisoformat(item["time_published"].replace('Z', '+00:00'))
                                        if start_date <= pub_date <= end_date:
                                            filtered_news.append(item)
                                    except (ValueError, TypeError):
                                        # If date parsing fails, include the item anyway
                                        filtered_news.append(item)
                                else:
                                    filtered_news.append(item)
                            
                            news_items.extend(filtered_news)
                    
                    if news_items:
                        return news_items
                
                elif source == "polygon" and self.use_polygon:
                    # Polygon news endpoint
                    for symbol in symbols:
                        symbol_news = await self.polygon.get_news(symbol, limit=10)
                        if symbol_news:
                            # Filter by date range
                            filtered_news = []
                            for item in symbol_news:
                                if "published_utc" in item:
                                    try:
                                        pub_date = datetime.fromisoformat(item["published_utc"].replace('Z', '+00:00'))
                                        if start_date <= pub_date <= end_date:
                                            filtered_news.append(item)
                                    except (ValueError, TypeError):
                                        # If date parsing fails, include the item anyway
                                        filtered_news.append(item)
                                else:
                                    filtered_news.append(item)
                            
                            news_items.extend(filtered_news)
                    
                    if news_items:
                        return news_items
            
            except Exception as e:
                logger.warning(f"Error fetching news data from {source}: {e}")
        
        # If all sources failed or no news found, generate synthetic news
        if not news_items:
            logger.warning("No real news data found, generating synthetic news")
            news_items = self._generate_synthetic_news(symbols, start_date, end_date)
        
        return news_items

    def _generate_synthetic_news(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic news data for testing.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            List of synthetic news items
        """
        news_items = []
        
        # Templates for titles and summaries
        bullish_titles = [
            "{symbol} reports strong quarterly earnings",
            "{symbol} exceeds analyst expectations",
            "{symbol} announces new product launch",
            "Analysts upgrade {symbol} to 'buy'",
            "{symbol} expands into new markets",
            "{symbol} stock surges on positive news",
            "{symbol} reports record revenue growth",
            "CEO of {symbol} optimistic about future growth",
        ]
        
        bearish_titles = [
            "{symbol} misses earnings expectations",
            "Analysts downgrade {symbol} to 'sell'",
            "{symbol} facing regulatory challenges",
            "{symbol} reports disappointing sales figures",
            "Competition intensifies for {symbol}",
            "{symbol} warns of slowdown in growth",
            "{symbol} cuts guidance for next quarter",
            "Investors concerned about {symbol}'s debt levels",
        ]
        
        neutral_titles = [
            "{symbol} announces management changes",
            "{symbol} to present at upcoming conference",
            "{symbol} maintains market position",
            "Industry outlook remains stable for {symbol}",
            "{symbol} completes reorganization",
            "{symbol} holds annual shareholder meeting",
            "New partnerships announced for {symbol}",
            "{symbol} maintains dividend",
        ]
        
        # Corresponding summaries
        bullish_summaries = [
            "{symbol} reported quarterly earnings that exceeded analyst expectations, with revenue growth of {growth}% year-over-year.",
            "Analysts have upgraded {symbol} to a 'buy' rating, citing strong growth prospects and competitive positioning.",
            "{symbol} announced the launch of new products that are expected to significantly contribute to revenue growth in the coming quarters.",
            "The CEO of {symbol} expressed optimism about future growth, highlighting expansion plans and strong customer demand.",
            "{symbol} is expanding into new markets, which is expected to drive revenue growth and increase market share.",
        ]
        
        bearish_summaries = [
            "{symbol} reported quarterly earnings that fell short of analyst expectations, with revenue declining by {decline}% year-over-year.",
            "Analysts have downgraded {symbol} to a 'sell' rating, citing concerns about market saturation and increasing competition.",
            "{symbol} is facing regulatory challenges that could impact its business operations and financial performance.",
            "The CEO of {symbol} warned of a potential slowdown in growth due to macroeconomic headwinds and industry challenges.",
            "{symbol} has cut its guidance for the next quarter, indicating challenges in meeting previously set targets.",
        ]
        
        neutral_summaries = [
            "{symbol} announced management changes as part of its ongoing strategic restructuring efforts.",
            "{symbol} will be presenting at an upcoming industry conference to showcase its latest innovations and strategy.",
            "Industry analysts expect {symbol} to maintain its current market position despite competitive pressures.",
            "{symbol} completed a reorganization aimed at improving operational efficiency and reducing costs.",
            "{symbol} held its annual shareholder meeting where management discussed the company's performance and future outlook.",
        ]
        
        # Generate news over the date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for symbol in symbols:
            # Generate 1-3 news items per symbol
            num_news = np.random.randint(1, 4)
            
            for _ in range(num_news):
                # Select random date
                news_date = np.random.choice(date_range)
                
                # Randomly select news sentiment
                sentiment = np.random.choice(["bullish", "bearish", "neutral"], p=[0.4, 0.3, 0.3])
                
                if sentiment == "bullish":
                    title_template = np.random.choice(bullish_titles)
                    summary_template = np.random.choice(bullish_summaries)
                    sentiment_score = np.random.uniform(0.2, 0.8)
                elif sentiment == "bearish":
                    title_template = np.random.choice(bearish_titles)
                    summary_template = np.random.choice(bearish_summaries)
                    sentiment_score = np.random.uniform(-0.8, -0.2)
                else:
                    title_template = np.random.choice(neutral_titles)
                    summary_template = np.random.choice(neutral_summaries)
                    sentiment_score = np.random.uniform(-0.2, 0.2)
                
                # Fill templates
                growth = np.random.randint(5, 30)
                decline = np.random.randint(5, 30)
                
                title = title_template.format(symbol=symbol)
                summary = summary_template.format(symbol=symbol, growth=growth, decline=decline)
                
                # Create news item
                news_item = {
                    "title": title,
                    "summary": summary,
                    "source": np.random.choice(["Market News", "Financial Times", "Wall Street Journal", "Bloomberg", "CNBC"]),
                    "url": f"https://example.com/news/{symbol.lower()}/{news_date.strftime('%Y%m%d')}",
                    "published_at": news_date.isoformat(),
                    "symbols": [symbol],
                    "relevance_score": np.random.uniform(0.7, 1.0),
                    "sentiment_score": sentiment_score
                }
                
                news_items.append(news_item)
        
        # Add some market news
        market_titles = [
            "Market reaches new high as economic data improves",
            "Stocks fall amid recession fears",
            "Fed announces interest rate decision",
            "Inflation data comes in below expectations",
            "Market volatility increases as geopolitical tensions rise",
            "Economic outlook remains positive despite challenges",
            "Investors react to latest employment data",
            "Market trends suggest continued growth",
        ]
        
        market_summaries = [
            "The stock market reached new highs today as economic data showed improvements in key sectors.",
            "Stocks fell across the board as investors grew concerned about potential recession signals.",
            "The Federal Reserve announced its latest interest rate decision, impacting market expectations.",
            "The latest inflation data came in below analysts' expectations, potentially giving the Fed more flexibility.",
            "Market volatility increased as geopolitical tensions rose, creating uncertainty for investors.",
        ]
        
        # Add 5-10 market news items
        num_market_news = np.random.randint(5, 11)
        
        for _ in range(num_market_news):
            # Select random date
            news_date = np.random.choice(date_range)
            
            # Randomly select news sentiment
            sentiment = np.random.choice(["bullish", "bearish", "neutral"], p=[0.4, 0.3, 0.3])
            
            title = np.random.choice(market_titles)
            summary = np.random.choice(market_summaries)
            
            if sentiment == "bullish":
                sentiment_score = np.random.uniform(0.2, 0.8)
            elif sentiment == "bearish":
                sentiment_score = np.random.uniform(-0.8, -0.2)
            else:
                sentiment_score = np.random.uniform(-0.2, 0.2)
            
            # Create news item
            news_item = {
                "title": title,
                "summary": summary,
                "source": np.random.choice(["Market News", "Financial Times", "Wall Street Journal", "Bloomberg", "CNBC"]),
                "url": f"https://example.com/market-news/{news_date.strftime('%Y%m%d')}",
                "published_at": news_date.isoformat(),
                "symbols": [],  # No specific symbols
                "relevance_score": np.random.uniform(0.7, 1.0),
                "sentiment_score": sentiment_score
            }
            
            news_items.append(news_item)
        
        # Sort by date (newest first)
        news_items.sort(key=lambda x: x["published_at"], reverse=True)
        
        logger.info(f"Generated {len(news_items)} synthetic news items")
        
        return news_items

    async def _fetch_market_data(
        self,
        start_date: datetime,
        end_date: datetime,
        data_source: str
    ) -> Dict[str, Any]:
        """
        Fetch market data.
        
        Args:
            start_date: Start date
            end_date: End date
            data_source: Data source to use
            
        Returns:
            Dictionary with market data
        """
        market_data = {
            "market_status": {},
            "sector_performance": {},
            "economic_indicators": {}
        }
        
        # Determine which data source to use
        if data_source == "auto":
            # Try Polygon first, then Alpha Vantage
            sources = []
            if self.use_polygon:
                sources.append("polygon")
            if self.use_alpha_vantage:
                sources.append("alpha_vantage")
        else:
            sources = [data_source]
        
        # Try each source until successful
        for source in sources:
            try:
                if source == "polygon" and self.use_polygon:
                    # Get market status
                    market_status = await self.polygon.get_market_status()
                    if market_status:
                        market_data["market_status"] = market_status
                
                if source == "alpha_vantage" and self.use_alpha_vantage:
                    # Get sector performance
                    sector_performance = await self.alpha_vantage.get_sector_performance()
                    if sector_performance:
                        market_data["sector_performance"] = sector_performance
                    
                    # Get economic indicators
                    treasury_yield = await self.alpha_vantage.get_treasury_yield(interval="daily", maturity="10year")
                    if treasury_yield is not None and not treasury_yield.empty:
                        # Filter by date range
                        treasury_yield = treasury_yield[(treasury_yield.index >= start_date) & (treasury_yield.index <= end_date)]
                        # Convert to dictionary for JSON serialization
                        market_data["economic_indicators"]["treasury_yield"] = treasury_yield.reset_index().to_dict(orient="records")
            
            except Exception as e:
                logger.warning(f"Error fetching market data from {source}: {e}")
        
        # If market data is empty, generate synthetic data
        if not market_data["market_status"] and not market_data["sector_performance"] and not market_data["economic_indicators"]:
            logger.warning("No real market data found, generating synthetic market data")
            market_data = self._generate_synthetic_market_data(start_date, end_date)
        
        return market_data

    def _generate_synthetic_market_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate synthetic market data for testing.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with synthetic market data
        """
        # Generate synthetic market status
        market_status = {
            "market": np.random.choice(["open", "closed"]),
            "server_time": datetime.now().isoformat(),
            "exchanges": {
                "nasdaq": {
                    "name": "NASDAQ",
                    "type": "equity",
                    "market": "equities",
                    "status": np.random.choice(["open", "closed"]),
                    "session_start": "09:30:00",
                    "session_end": "16:00:00"
                },
                "nyse": {
                    "name": "NYSE",
                    "type": "equity",
                    "market": "equities",
                    "status": np.random.choice(["open", "closed"]),
                    "session_start": "09:30:00",
                    "session_end": "16:00:00"
                }
            }
        }
        
        # Generate synthetic sector performance
        sectors = [
            "Information Technology",
            "Health Care",
            "Financials",
            "Consumer Discretionary",
            "Communication Services",
            "Industrials",
            "Consumer Staples",
            "Energy",
            "Utilities",
            "Real Estate",
            "Materials"
        ]
        
        sector_performance = {
            "Rank A: Real-Time Performance": {
                sector: f"{np.random.uniform(-3.0, 3.0):.2f}%" for sector in sectors
            },
            "Rank B: 1 Day Performance": {
                sector: f"{np.random.uniform(-3.0, 3.0):.2f}%" for sector in sectors
            },
            "Rank C: 5 Day Performance": {
                sector: f"{np.random.uniform(-5.0, 5.0):.2f}%" for sector in sectors
            },
            "Rank D: 1 Month Performance": {
                sector: f"{np.random.uniform(-10.0, 10.0):.2f}%" for sector in sectors
            },
            "Rank E: 3 Month Performance": {
                sector: f"{np.random.uniform(-15.0, 15.0):.2f}%" for sector in sectors
            },
            "Rank F: Year-to-Date (YTD) Performance": {
                sector: f"{np.random.uniform(-20.0, 20.0):.2f}%" for sector in sectors
            },
            "Rank G: 1 Year Performance": {
                sector: f"{np.random.uniform(-25.0, 25.0):.2f}%" for sector in sectors
            },
            "Rank H: 3 Year Performance": {
                sector: f"{np.random.uniform(-40.0, 40.0):.2f}%" for sector in sectors
            },
            "Rank I: 5 Year Performance": {
                sector: f"{np.random.uniform(-50.0, 50.0):.2f}%" for sector in sectors
            },
            "Rank J: 10 Year Performance": {
                sector: f"{np.random.uniform(-100.0, 100.0):.2f}%" for sector in sectors
            }
        }
        
        # Generate synthetic economic indicators
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        treasury_yield = []
        
        for date in date_range:
            treasury_yield.append({
                "date": date.isoformat(),
                "value": np.random.uniform(1.5, 5.0)
            })
        
        return {
            "market_status": market_status,
            "sector_performance": sector_performance,
            "economic_indicators": {
                "treasury_yield": treasury_yield
            }
        }

    def _save_dataset(self, dataset: Dict[str, Any], dataset_name: str) -> None:
        """
        Save dataset to disk.
        
        Args:
            dataset: Dataset to save
            dataset_name: Name of the dataset
        """
        # Create file path
        file_path = os.path.join(self.data_dir, f"{dataset_name}.json")
        
        # Save as JSON
        with open(file_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved dataset to {file_path}")

    def load_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load dataset from disk.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset if found, None otherwise
        """
        # Create file path
        file_path = os.path.join(self.data_dir, f"{dataset_name}.json")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"Dataset {dataset_name} not found at {file_path}")
            return None
        
        # Load from JSON
        try:
            with open(file_path, 'r') as f:
                dataset = json.load(f)
            
            logger.info(f"Loaded dataset {dataset_name} from {file_path}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            return None

    def list_datasets(self) -> List[str]:
        """
        List available datasets.
        
        Returns:
            List of dataset names
        """
        # Get all JSON files in data directory
        dataset_files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
        
        # Extract dataset names (remove .json extension)
        dataset_names = [os.path.splitext(f)[0] for f in dataset_files]
        
        return dataset_names

    def add_labels(self, dataset_name: str, labels: Dict[str, Any]) -> bool:
        """
        Add labels to an existing dataset.
        
        Args:
            dataset_name: Name of the dataset
            labels: Labels to add
            
        Returns:
            True if successful, False otherwise
        """
        # Load dataset
        dataset = self.load_dataset(dataset_name)
        
        if dataset is None:
            return False
        
        # Update labels
        dataset["labels"].update(labels)
        
        # Save dataset
        self._save_dataset(dataset, dataset_name)
        
        logger.info(f"Added labels to dataset {dataset_name}")
        
        return True

    def convert_to_dataframe(self, dataset: Dict[str, Any], symbol: str) -> Optional[pd.DataFrame]:
        """
        Convert dataset price data to DataFrame.
        
        Args:
            dataset: Dataset to convert
            symbol: Symbol to extract
            
        Returns:
            DataFrame if successful, None otherwise
        """
        if "price_data" not in dataset or symbol not in dataset["price_data"]:
            logger.warning(f"No price data found for {symbol} in dataset")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset["price_data"][symbol])
        
        # Convert timestamp to datetime and set as index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        
        return df
