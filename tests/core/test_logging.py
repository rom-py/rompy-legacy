"""
Tests for the rompy.core.logging module.

This module tests the centralized logging and formatting utilities.
"""

import os
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from rompy.core.logging import (
    LoggingConfig,
    LogLevel,
    LogFormat,
    BoxStyle,
    get_logger,
    RompyLogger,
    formatter,
)


class TestLoggingConfig:
    """Tests for the LoggingConfig class."""

    def setup_method(self):
        """Reset the config before each test."""
        LoggingConfig.reset()
        self.config = LoggingConfig()

    def test_default_values(self):
        """Test default configuration values."""
        assert self.config.level == LogLevel.INFO
        assert self.config.format == LogFormat.VERBOSE
        assert self.config.log_dir is None
        assert self.config.log_file == "rompy.log"
        assert self.config.use_ascii is False

    def test_update_configuration(self):
        """Test updating the configuration."""
        # Test updating a single value
        self.config.update(level=LogLevel.DEBUG)
        assert self.config.level == LogLevel.DEBUG

        # Test updating multiple values
        self.config.update(level=LogLevel.INFO, format=LogFormat.SIMPLE)
        assert self.config.level == LogLevel.INFO
        assert self.config.format == LogFormat.SIMPLE

    def test_log_file_path(self, tmp_path):
        """Test the log_file_path property."""
        # Test with log_dir set
        self.config.log_dir = tmp_path
        assert self.config.log_file_path == tmp_path / "rompy.log"

        # Test with log_dir None
        self.config.log_dir = None
        assert self.config.log_file_path is None

    def test_configure_logging(self, tmp_path):
        """Test that logging is properly configured."""
        # Configure logging with a temporary directory
        self.config.update(
            level=LogLevel.DEBUG,
            format=LogFormat.SIMPLE,
            log_dir=tmp_path,
            log_file="test.log",
        )

        # Get the root logger and verify its configuration
        root_logger = logging.getLogger()

        # Should have 2 handlers: one for console, one for file
        assert len(root_logger.handlers) == 2

        # Check log file was created
        log_file = tmp_path / "test.log"
        assert log_file.exists()


class TestRompyLogger:
    """Tests for the RompyLogger class."""

    def setup_method(self):
        """Set up test environment."""
        # Reset the config
        LoggingConfig.reset()
        self.config = LoggingConfig()

        # Set up a test logger
        self.logger = get_logger("test_logger")

        # Capture log output
        self.log_capture = []

        # Add a custom handler to capture log output
        class CaptureHandler(logging.Handler):
            def __init__(self, capture_list):
                super().__init__()
                self.capture_list = capture_list

            def emit(self, record):
                self.capture_list.append(self.format(record))

        self.handler = CaptureHandler(self.log_capture)
        self.handler.setFormatter(logging.Formatter("%(message)s"))

        # Remove existing handlers and add our capture handler
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.addHandler(self.handler)

    def test_log_messages(self):
        """Test basic log messages."""
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")
        self.logger.critical("Critical message")

        # Check that all messages were captured
        assert len(self.log_capture) == 5
        assert "Debug message" in self.log_capture[0]
        assert "Info message" in self.log_capture[1]
        assert "Warning message" in self.log_capture[2]
        assert "Error message" in self.log_capture[3]
        assert "Critical message" in self.log_capture[4]

    def test_box(self):
        """Test the box logging method."""
        self.logger.box("Test content", title="Test Box")

        # Check that the box was logged
        assert len(self.log_capture) > 0
        assert "Test Box" in "\n".join(self.log_capture)
        assert "Test content" in "\n".join(self.log_capture)

    def test_status_box(self):
        """Test the status_box logging method."""
        self.logger.status_box("Operation completed", BoxStyle.SUCCESS)

        # Check that the status box was logged
        assert len(self.log_capture) > 0
        assert "Operation completed" in "\n".join(self.log_capture)

    def test_bullet_list(self):
        """Test the bullet_list logging method."""
        items = ["Item 1", "Item 2", "Item 3"]
        self.logger.bullet_list(items)

        # Check that all items were logged
        output = "\n".join(self.log_capture)
        for item in items:
            assert item in output


class TestBoxFormatter:
    """Tests for the BoxFormatter class."""

    def setup_method(self):
        """Set up test environment."""
        LoggingConfig.reset()
        self.config = LoggingConfig()
        self.formatter = formatter

    def test_box_creation(self):
        """Test creating a box with different styles."""
        # Test with default style
        box_content = self.formatter.box("Test content", "Test Title")
        assert "Test Title" in box_content
        assert "Test content" in box_content

        # Test with different styles
        for style in BoxStyle:
            box_content = self.formatter.box("Test content", "Test Title", style)
            assert "Test content" in box_content

    def test_status_box_creation(self):
        """Test creating a status box."""
        for style in [
            BoxStyle.SUCCESS,
            BoxStyle.ERROR,
            BoxStyle.WARNING,
            BoxStyle.INFO,
        ]:
            box_content = self.formatter.status_box("Test message", style)
            assert "Test message" in box_content

    def test_bullet_list_creation(self):
        """Test creating a bulleted list."""
        items = ["Item 1", "Item 2", "Item 3"]
        list_content = self.formatter.bullet_list(items)

        for item in items:
            assert item in list_content

    def test_ascii_mode(self):
        """Test ASCII mode for formatter."""
        # Enable ASCII mode and update the formatter's config
        self.config.update(use_ascii=True)
        self.formatter.config = self.config  # Update the formatter's config

        # Create a box and check it uses ASCII characters
        box_content = self.formatter.box("Test content", "Test Title")

        # Check for ASCII characters
        assert "+" in box_content  # ASCII corner character
        assert "-" in box_content  # ASCII horizontal line
        assert "|" in box_content  # ASCII vertical line

        # Check that we're not using Unicode box-drawing characters
        assert "┌" not in box_content
        assert "─" not in box_content
        assert "│" not in box_content
