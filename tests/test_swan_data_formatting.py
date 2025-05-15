#!/usr/bin/env python
"""
Tests for the SwanDataGrid integration with formatting.

This module tests how the SwanDataGrid class integrates with the formatting module
and handles ASCII mode settings.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
import importlib

# Import the formatting module
import rompy.formatting
from rompy.formatting import get_formatted_box


class MockSwanDataGrid:
    """A mock version of SwanDataGrid for testing formatting integration."""
    
    # Class-level variable that will be updated based on the formatting module
    _use_ascii_only = None
    
    def __init__(self):
        """Initialize with default values."""
        self.var = MagicMock()
        self.var.value = "TEST"
    
    def get_formatted_output(self):
        """Simulate a method that uses the formatting module."""
        return get_formatted_box(
            title="SWAN DATA GRID",
            content=["Variable: " + str(self.var.value)],
            use_ascii=self._use_ascii_only,
            width=40
        )
    
    def log_with_arrow(self):
        """Simulate a method that logs with an arrow character based on ASCII mode."""
        if self._use_ascii_only:
            return "Source -> Destination"
        else:
            return "Source → Destination"


class TestSwanDataFormatting:
    """Test SwanDataGrid integration with formatting settings."""

    def test_ascii_mode_in_swan_data(self):
        """Test that the SwanDataGrid correctly uses ASCII mode."""
        # Test with ASCII mode on
        with patch.dict(os.environ, {"ROMPY_ASCII_ONLY": "true"}, clear=True):
            # Reload the formatting module to pick up the environment change
            importlib.reload(rompy.formatting)
            
            # Set the class variable to match the formatting module
            MockSwanDataGrid._use_ascii_only = rompy.formatting.USE_ASCII_ONLY
            
            # Create a SwanDataGrid instance
            grid = MockSwanDataGrid()
            
            # Get formatted output
            output = grid.get_formatted_output()
            lines = output.split("\n")
            
            # Check for ASCII box characters
            assert "+" in lines[0]  # ASCII header
            assert "SWAN DATA GRID" in lines[1]  # Title is present
            
            # Check arrow format
            arrow_text = grid.log_with_arrow()
            assert "->" in arrow_text  # ASCII arrow
        
        # Test with ASCII mode off
        with patch.dict(os.environ, {"ROMPY_ASCII_ONLY": "false"}, clear=True):
            # Reload the formatting module to pick up the environment change
            importlib.reload(rompy.formatting)
            
            # Set the class variable to match the formatting module
            MockSwanDataGrid._use_ascii_only = rompy.formatting.USE_ASCII_ONLY
            
            # Create a SwanDataGrid instance
            grid = MockSwanDataGrid()
            
            # Get formatted output
            output = grid.get_formatted_output()
            lines = output.split("\n")
            
            # Check for Unicode box characters (using flexible assertions)
            assert any(c in lines[0] for c in ["┏", "┓", "╱", "╣"])  # Unicode header
            assert "SWAN DATA GRID" in lines[1]  # Title is present
            
            # Check arrow format
            arrow_text = grid.log_with_arrow()
            assert "→" in arrow_text  # Unicode arrow

    def test_class_level_variable_initialization(self):
        """Test that the class-level variables are properly initialized from the module."""
        # Test with ASCII mode on
        with patch.dict(os.environ, {"ROMPY_ASCII_ONLY": "true"}, clear=True):
            # Reload the formatting module to pick up the environment change
            importlib.reload(rompy.formatting)
            
            # Set the class variable to match the formatting module
            MockSwanDataGrid._use_ascii_only = rompy.formatting.USE_ASCII_ONLY
            
            # Verify the class variable was set correctly
            assert MockSwanDataGrid._use_ascii_only is True
        
        # Test with ASCII mode off
        with patch.dict(os.environ, {"ROMPY_ASCII_ONLY": "false"}, clear=True):
            # Reload the formatting module to pick up the environment change
            importlib.reload(rompy.formatting)
            
            # Set the class variable to match the formatting module
            MockSwanDataGrid._use_ascii_only = rompy.formatting.USE_ASCII_ONLY
            
            # Verify the class variable was set correctly
            assert MockSwanDataGrid._use_ascii_only is False

    def test_environment_variable_integration(self):
        """Test that environment variables correctly affect the SwanDataGrid class."""
        # Test with ASCII mode on via environment variable
        with patch.dict(os.environ, {"ROMPY_ASCII_ONLY": "true"}, clear=True):
            # Reload the formatting module to pick up the environment change
            importlib.reload(rompy.formatting)
            
            # Verify the global setting is correct
            assert rompy.formatting.USE_ASCII_ONLY is True
            
            # Set the class variable to match the formatting module
            MockSwanDataGrid._use_ascii_only = rompy.formatting.USE_ASCII_ONLY
            
            # Create a SwanDataGrid instance
            grid = MockSwanDataGrid()
            
            # Get formatted output and verify it uses ASCII
            output = grid.get_formatted_output()
            assert "+" in output  # ASCII character
            
            # Check arrow format
            arrow_text = grid.log_with_arrow()
            assert "->" in arrow_text  # ASCII arrow
