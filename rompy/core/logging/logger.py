"""
Enhanced logger for ROMPY.

This module provides an enhanced logger that integrates with the ROMPY
logging configuration and formatting system.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TypeVar, cast

import ipdb

from .formatter import BoxFormatter, BoxStyle, formatter

T = TypeVar("T", bound="RompyLogger")


class RompyLogger(logging.Logger):
    """Enhanced logger with ROMPY-specific functionality."""

    def __init__(self, name: str, level: int = logging.NOTSET):
        """Initialize the logger.

        Args:
            name: The name of the logger
            level: The log level (defaults to logging.NOTSET)
        """
        # Initialize the base logger
        super().__init__(name, level)

        # Initialize the box formatter
        self._box_formatter = formatter

        # Ensure the formatter is properly set up
        if not hasattr(self, "_box_formatter") or self._box_formatter is None:
            from .formatter import formatter as default_formatter

            self._box_formatter = default_formatter

    @property
    def box_formatter(self) -> BoxFormatter:
        """Get the box formatter for this logger.

        Returns:
            BoxFormatter: The box formatter instance
        """
        if not hasattr(self, "_box_formatter") or self._box_formatter is None:
            from .formatter import formatter as default_formatter

            self._box_formatter = default_formatter
        return self._box_formatter

    def box(
        self,
        content: str,
        title: Optional[str] = None,
        style: BoxStyle = BoxStyle.SIMPLE,
    ) -> None:
        """Log a box with the given content and title.

        Args:
            content: The content to put in the box
            title: Optional title for the box
            style: Box style to use
        """
        box_content = self.box_formatter.box(content, title, style)
        for line in box_content.splitlines():
            self.info(line)

    def status_box(self, message: str, status: BoxStyle) -> None:
        """Log a status box with the given message and status.

        Args:
            message: The status message
            status: Status type (error, warning, success, etc.)
        """
        box_content = self.box_formatter.status_box(message, status)
        for line in box_content.splitlines():
            self.info(line)

    def bullet_list(self, items: list[str], indent: int = 2) -> None:
        """Log a bulleted list.

        Args:
            items: List items to log
            indent: Number of spaces to indent
        """
        bullet_content = self.box_formatter.bullet_list(items, indent)
        for line in bullet_content.splitlines():
            self.info(line)

    def success(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a success message."""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, self.box_formatter.success(message), args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        if self.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, self.box_formatter.error(message), args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        if self.isEnabledFor(logging.WARNING):
            self._log(
                logging.WARNING, self.box_formatter.warning(message), args, **kwargs
            )

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, self.box_formatter.info(message), args, **kwargs)


def get_logger(name: Optional[str] = None) -> RompyLogger:
    """Get a ROMPY logger instance.

    Args:
        name: Logger name. If None, returns the root logger.

    Returns:
        Configured RompyLogger instance
    """
    if name is None:
        name = "root"

    # Replace the default logger class with our custom one
    logging.setLoggerClass(RompyLogger)

    # Get or create the logger
    logger = logging.getLogger(name)

    # Ensure the logger is of our custom type
    if not isinstance(logger, RompyLogger):
        logger.__class__ = RompyLogger

    return cast(RompyLogger, logger)


# Set up the default logger class
logging.setLoggerClass(RompyLogger)

# Configure the root logger when the module is imported
root_logger = get_logger()
