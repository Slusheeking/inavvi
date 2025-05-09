"""
Logging utilities for the trading system.
"""
import logging
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config.settings import settings

# Custom formatter with colors for console output
class ColorFormatter(logging.Formatter):
    """Logging formatter that adds colors to the log level."""
    
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',   # Red
        'CRITICAL': '\033[41m', # Red background
        'RESET': '\033[0m',    # Reset
    }
    
    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            log_message = f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

def setup_logger(name, log_level=None):
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: The name of the logger
        log_level: Optional log level (defaults to settings.logging.log_level)
        
    Returns:
        A configured logger instance
    """
    if log_level is None:
        log_level = settings.logging.log_level
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create log directory if it doesn't exist
    log_dir = Path(settings.logging.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create a file handler for rotating logs
    log_file = log_dir / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    
    # Create a handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = ColorFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name, log_level=None):
    """
    Get or create a logger with the specified name.
    This is a wrapper around setup_logger for consistency.
    
    Args:
        name: The name of the logger
        log_level: Optional log level (defaults to settings.logging.log_level)
        
    Returns:
        A configured logger instance
    """
    return setup_logger(name, log_level)

def log_execution_time(logger):
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger: The logger to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.6f} seconds")
            return result
        return wrapper
    return decorator

# Create main application logger
app_logger = setup_logger("trading_system")

# Export all required functions
__all__ = [
    "setup_logger",
    "get_logger",
    "log_execution_time",
    "ColorFormatter",
    "app_logger"
]