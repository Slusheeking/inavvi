"""
Configuration module for the trading system.

This module contains settings and configuration loading functionality.
"""

import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config.settings import settings  # noqa: E402 - Import needs to be after sys.path modification

__all__ = ["settings"]
