"""
Configuration settings for the trading system.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

class APISettings(BaseSettings):
    """API connection settings."""
    polygon_api_key: str = Field(..., env="POLYGON_API_KEY")
    alpha_vantage_api_key: str = Field(..., env="ALPHA_VANTAGE_API_KEY")
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    
    # Alpaca settings
    alpaca_api_key: str = Field(..., env="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(..., env="ALPACA_API_SECRET")
    alpaca_base_url: str = Field(..., env="ALPACA_BASE_URL")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }

class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    timescaledb_url: str = Field(..., env="TIMESCALEDB_URL")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }

class LLMSettings(BaseSettings):
    """LLM configuration settings."""
    model: str = Field("anthropic/claude-3-haiku-20240307", env="LLM_MODEL")
    max_tokens: int = Field(4096, env="LLM_MAX_TOKENS")
    temperature: float = Field(0.1, env="LLM_TEMPERATURE")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }

class TradingSettings(BaseSettings):
    """Trading configuration settings."""
    trading_mode: str = Field("paper", env="TRADING_MODE")
    max_positions: int = Field(3, env="MAX_POSITIONS")
    max_position_size: float = Field(2000.0, env="MAX_POSITION_SIZE")
    max_daily_risk: float = Field(150.0, env="MAX_DAILY_RISK")
    initial_universe_size: int = Field(300, env="INITIAL_UNIVERSE_SIZE")
    watchlist_size: int = Field(20, env="WATCHLIST_SIZE")
    candidate_size: int = Field(5, env="CANDIDATE_SIZE")
    
    @validator("trading_mode")
    def trading_mode_must_be_valid(cls, v):
        if v not in ["paper", "live"]:
            raise ValueError("trading_mode must be either 'paper' or 'live'")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }

class TimingSettings(BaseSettings):
    """Market timing settings."""
    market_open: str = Field("09:30", env="MARKET_OPEN")
    market_close: str = Field("16:00", env="MARKET_CLOSE")
    pre_market_scan: str = Field("08:30", env="PRE_MARKET_SCAN")
    post_market_analysis: str = Field("16:30", env="POST_MARKET_ANALYSIS")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }

class ModelSettings(BaseSettings):
    """ML model settings."""
    pattern_model_path: str = Field("models/pattern_recognition.pt", env="PATTERN_MODEL_PATH")
    ranking_model_path: str = Field("models/ranking_model.pkl", env="RANKING_MODEL_PATH")
    sentiment_model_path: str = Field("models/sentiment_model.pt", env="SENTIMENT_MODEL_PATH")
    exit_model_path: str = Field("models/exit_optimization.pt", env="EXIT_MODEL_PATH")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }

class LoggingSettings(BaseSettings):
    """Logging configuration settings."""
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_dir: str = os.path.join(ROOT_DIR, "logs")
    
    @validator("log_level")
    def log_level_must_be_valid(cls, v):
        if v not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError("log_level must be a valid Python logging level")
        return v
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "allow",
    }

class Settings(BaseSettings):
    """Main settings class that combines all configurations."""
    api: APISettings = APISettings()
    database: DatabaseSettings = DatabaseSettings()
    llm: LLMSettings = LLMSettings()
    trading: TradingSettings = TradingSettings()
    timing: TimingSettings = TimingSettings()
    model: ModelSettings = ModelSettings()
    logging: LoggingSettings = LoggingSettings()
    
    # Additional system-wide settings
    app_name: str = "Day Trading System"
    version: str = "1.0.0"
    data_dir: Path = ROOT_DIR / "data"
    models_dir: Path = ROOT_DIR / "models"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        Path(self.logging.log_dir).mkdir(exist_ok=True)

# Create global settings instance
settings = Settings()
