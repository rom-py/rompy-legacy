#!/usr/bin/env python
"""
Tests for the SwanDataGrid integration with the new logging system.

This module tests how the SwanDataGrid class integrates with the new logging
system and handles ASCII mode settings.
"""

import os
import sys
import importlib
import unittest
from unittest.mock import MagicMock, patch

import pytest

# Add the parent directory to the path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rompy.core.logging import get_logger, LoggingConfig, LogLevel, LogFormat, BoxStyle, formatter
from rompy.core.logging.formatter import BoxFormatter

# Initialize logger
logger = get_logger(__name__)

class MockSwanDataGrid:
    """A mock version of SwanDataGrid for testing logging integration."""
    
    def __init__(self):
        """Initialize with default values."""
        self.var = MagicMock()
        self.var.value = "TEST"
        self.logger = get_logger(__name__)
    
    def get_formatted_output(self, config: LoggingConfig):
        """Simulate a method that uses the logging formatter."""
        # Create a new formatter instance with the test config
        test_formatter = BoxFormatter(config=config)
        
        # Create a box with the test formatter
        box_content = test_formatter.box(
            "Variable: " + str(self.var.value),
            title="SWAN DATA GRID"
        )
        return box_content
    
    def log_with_arrow(self, config: LoggingConfig):
        """Simulate a method that logs with an arrow character based on ASCII mode."""
        # Create a new formatter instance with the test config
        test_formatter = BoxFormatter(config=config)
        
        # Get arrow based on current config
        return test_formatter.arrow("Source") + " Destination"


class TestSwanDataFormatting:
    """Test SwanDataGrid integration with logging settings."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment."""
        # Reset the singleton to ensure clean state
        LoggingConfig.reset()
        
        # Create test configs
        self.ascii_config = LoggingConfig()
        self.ascii_config.update(use_ascii=True)
        
        # Need to reset again to create a separate config
        LoggingConfig.reset()
        self.unicode_config = LoggingConfig()
        self.unicode_config.update(use_ascii=False)
        
        yield  # This is where the test runs
        
        # Teardown code (if any) goes here

    def test_ascii_mode_in_swan_data(self):
        """Test that the SwanDataGrid correctly uses ASCII mode."""
        # Create a SwanDataGrid instance
        grid = MockSwanDataGrid()
        
        # Test with ASCII mode on
        output = grid.get_formatted_output(self.ascii_config)
        lines = output.split("\n")
        
        # Check for ASCII box characters
        assert "+" in lines[0]  # ASCII corner
        assert "-" in lines[0]  # ASCII horizontal line
        assert "SWAN DATA GRID" in lines[1]  # Title is present
        
        # Check arrow format
        arrow_text = grid.log_with_arrow(self.ascii_config)
        assert "->" in arrow_text  # ASCII arrow
        
        # Test with Unicode mode
        output = grid.get_formatted_output(self.unicode_config)
        lines = output.split("\n")
        
        # Check for Unicode box characters
        assert "┌" in lines[0]  # Unicode corner
        assert "─" in lines[0]  # Unicode horizontal line
        assert "SWAN DATA GRID" in lines[1]  # Title is present
        
        # Check arrow format
        arrow_text = grid.log_with_arrow(self.unicode_config)
        assert "→" in arrow_text  # Unicode arrow

    def test_class_level_variable_initialization(self):
        """Test that the logging configuration is properly initialized from environment variables."""
        # Test with ASCII mode on
        with patch.dict(os.environ, {"ROMPY_USE_ASCII": "true"}, clear=True):
            # Reset the config to pick up the environment variable
            LoggingConfig.reset()
            config = LoggingConfig()
            
            # Check that the config is using ASCII
            assert config.use_ascii
            
            # Create a new formatter with this config
            formatter = BoxFormatter(config=config)
            
            # The formatter should be using ASCII glyphs
            assert formatter.glyphs.TOP_LEFT == '+'
            
        # Test with ASCII mode off (default)
        with patch.dict(os.environ, {}, clear=True):
            # Reset the config to pick up the environment variable
            LoggingConfig.reset()
            config = LoggingConfig()
            
            # Check that the config is not using ASCII by default
            assert not config.use_ascii
            
            # Create a new formatter with this config
            formatter = BoxFormatter(config=config)
            
            # The formatter should be using Unicode glyphs
            assert formatter.glyphs.TOP_LEFT == '┌'

    def test_environment_variable_integration(self):
        """Test that environment variables correctly affect the logging configuration."""
        # Test with ASCII mode on
        with patch.dict(os.environ, {"ROMPY_USE_ASCII": "true"}, clear=True):
            # Reset the config to pick up the environment variable
            LoggingConfig.reset()
            config = LoggingConfig()
            
            # Verify the config is using ASCII
            assert config.use_ascii
            
            # Create a formatter with this config
            formatter = BoxFormatter(config=config)
            
            # The formatter should be using ASCII glyphs
            assert formatter.glyphs.TOP_LEFT == '+'
            assert formatter.glyphs.ARROW == '->'
            
            # Create a SwanDataGrid instance with this config
            grid = MockSwanDataGrid()
            
            # The output should use ASCII characters
            output = grid.get_formatted_output(config)
            assert '+' in output  # ASCII corner
            assert '->' in grid.log_with_arrow(config)  # ASCII arrow
        
        # Test with ASCII mode off (default)
        with patch.dict(os.environ, {}, clear=True):
            # Reset the config to pick up the environment variable change
            LoggingConfig.reset()
            config = LoggingConfig()
            
            # Verify the config is not using ASCII by default
            assert not config.use_ascii
            
            # Create a formatter with this config
            formatter = BoxFormatter(config=config)
            
            # The formatter should be using Unicode glyphs
            assert formatter.glyphs.TOP_LEFT == '┌'
            assert formatter.glyphs.ARROW == '→'
            
            # Create a SwanDataGrid instance with this config
            grid = MockSwanDataGrid()
            
            # The output should use Unicode characters
            output = grid.get_formatted_output(config)
            assert '┌' in output  # Unicode corner
            assert '→' in grid.log_with_arrow(config)  # Unicode arrow format
            
            # Test with default config (should be Unicode)
            # Use the config we created earlier
            arrow_text = grid.log_with_arrow(config)
            assert "→" in arrow_text  # Unicode arrow
