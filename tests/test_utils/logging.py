"""
Test utilities for logging configuration.

This module provides utilities to configure logging consistently across all tests.
"""

import logging
from typing import Optional, Union

# Import from rompy.logging
from rompy.logging import get_logger, config, LogLevel, LogFormat

# Global logger instance
logger = None


def _ensure_logger_initialized():
    """Ensure the logger is initialized."""
    global logger
    if logger is None:
        configure_test_logging()
        logger = get_logger(__name__)


def configure_test_logging(
    level: Optional[Union[int, str, LogLevel]] = None,
    format_str: Optional[Union[str, LogFormat]] = None,
) -> None:
    """Configure logging for tests.

    Args:
        level: Logging level as an int, string, or LogLevel enum. If None, uses INFO.
        format_str: Log format as a string or LogFormat enum. If None, uses a default format.
    """
    # Convert level to LogLevel if it's an int or string
    if level is None:
        level = LogLevel.INFO
    elif isinstance(level, int):
        level_name = logging.getLevelName(level)
        level = (
            LogLevel[level_name]
            if level_name in LogLevel.__members__
            else LogLevel.INFO
        )
    elif isinstance(level, str):
        level = (
            LogLevel[level.upper()]
            if level.upper() in LogLevel.__members__
            else LogLevel.INFO
        )

    # Convert format_str to LogFormat if it's a string
    if format_str is None:
        format_str = LogFormat.VERBOSE
    elif isinstance(format_str, str):
        format_str = (
            LogFormat[format_str.upper()]
            if format_str.upper() in LogFormat.__members__
            else LogFormat.VERBOSE
        )

    # Configure basic logging
    logging.basicConfig(
        level=level.value, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure ROMPY logging
    config.update(level=level, format=format_str)

    # Ensure the logger is properly initialized
    logging.getLogger(__name__).debug("Test logging configured at level %s", level)


def get_test_logger(name: str) -> logging.Logger:
    """Get a logger for tests.

    Args:
        name: Logger name (usually __name__).

    Returns:
        Configured logger instance.
    """
    _ensure_logger_initialized()
    return logging.getLogger(name)


# Initialize the logger when the module is imported
_ensure_logger_initialized()
