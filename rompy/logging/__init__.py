"""
Centralized logging and formatting utilities for ROMPY.

This module provides a unified approach to logging and output formatting across the ROMPY codebase.
It supports both console and file logging with configurable formatting and verbosity levels.

Basic usage:

    from rompy.logging import get_logger, config, BoxStyle

    # Configure logging
    config.update(level="DEBUG", log_dir="./logs")

    # Get a logger
    logger = get_logger(__name__)
    # Log messages
    logger.info("This is an info message")
    logger.success("Operation completed successfully")
    logger.error("Something went wrong")

    # Create a box
    logger.box("This is some content inside a box", title="My Box")

    # Create a status box
    logger.status_box("Processing complete!", BoxStyle.SUCCESS)

    # Create a bulleted list
    logger.bullet_list(["Item 1", "Item 2", "Item 3"])
"""

from .config import LogFormat, LoggingConfig, LogLevel, config
from .formatter import (AsciiGlyphs, BoxFormatter, BoxStyle, UnicodeGlyphs,
                        box, bullet_list, formatter, status_box)
from .logger import RompyLogger, get_logger

# Re-export commonly used items
__all__ = [
    # Config
    "LoggingConfig",
    "LogLevel",
    "LogFormat",
    "config",
    # Formatter
    "BoxStyle",
    "UnicodeGlyphs",
    "AsciiGlyphs",
    "BoxFormatter",
    "formatter",
    "box",
    "status_box",
    "bullet_list",
    # Logger
    "get_logger",
    "RompyLogger",
]
