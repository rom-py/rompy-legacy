"""
Logging configuration for ROMPY.

This module provides a centralized configuration system for logging across the ROMPY codebase.
It uses Pydantic for validation and environment variable support.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, ClassVar
from enum import Enum

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Available log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Available log formats."""

    SIMPLE = "simple"  # Just the message
    STANDARD = "standard"  # Level and message
    VERBOSE = "verbose"  # Timestamp, level, module, and message


class LoggingConfig(BaseSettings):
    """Centralized logging configuration for ROMPY.

    This class provides a singleton instance that can be configured once and
    accessed throughout the application. It supports both programmatic configuration
    and environment variables.

    Environment variables:
        ROMPY_LOG_LEVEL: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        ROMPY_LOG_FORMAT: Log format (simple, standard, verbose)
        ROMPY_LOG_DIR: Directory to save log files
        ROMPY_USE_ASCII: Use ASCII-only output (true/false)
        ROMPY_LOG_FILE: Name of the log file (default: 'rompy.log')
    """

    # Core settings
    level: LogLevel = Field(default=LogLevel.INFO, description="Default logging level")

    format: LogFormat = Field(
        default=LogFormat.VERBOSE, description="Log message format"
    )

    # File output
    log_dir: Optional[Path] = Field(
        default=None, description="Directory to save log files"
    )

    log_file: str = Field(default="rompy.log", description="Name of the log file")

    # Formatting
    use_ascii: bool = Field(
        default=False, description="Use ASCII-only characters for console output"
    )

    # Singleton instance
    _instance: ClassVar[Optional["LoggingConfig"]] = None

    class Config:
        env_prefix = "ROMPY_"
        case_sensitive = False
        extra = "ignore"
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def log_file_path(self) -> Optional[Path]:
        """Get the full path to the log file."""
        if self.log_dir is None:
            return None
        return self.log_dir / self.log_file

    def configure_logging(self) -> None:
        """Configure Python logging based on current settings."""
        # Remove all existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set log level
        root_logger.setLevel(self.level.value)

        # Create formatter based on format setting
        formatter = self._create_formatter()

        # Add console handler
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(self.level.value)
        root_logger.addHandler(console)

        # Add file handler if log_dir is set
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.log_file_path)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(self.level.value)
            root_logger.addHandler(file_handler)

    def _create_formatter(self) -> logging.Formatter:
        """Create a formatter based on the current configuration."""
        if self.format == LogFormat.SIMPLE:
            return logging.Formatter("%(message)s")
        elif self.format == LogFormat.STANDARD:
            return logging.Formatter("%(levelname)s: %(message)s")
        else:  # VERBOSE
            return logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)-20s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

    def update(self, **kwargs) -> None:
        """Update configuration and reconfigure logging if needed."""
        needs_reconfigure = False

        for key, value in kwargs.items():
            if key in self.model_fields and getattr(self, key) != value:
                setattr(self, key, value)
                needs_reconfigure = True

        if needs_reconfigure:
            self.configure_logging()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None


# Initialize default configuration
config = LoggingConfig()

# Configure logging with default settings when module is imported
config.configure_logging()
