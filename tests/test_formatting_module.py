#!/usr/bin/env python
"""
Tests for the formatting module.

This module tests the centralized formatting functions and environment variable
handling in the rompy.formatting module.
"""

import os
import pytest
from unittest.mock import patch

from rompy.formatting import (
    USE_ASCII_ONLY,
    USE_SIMPLE_LOGS,
    get_ascii_mode,
    get_simple_logs,
    get_formatted_box,
    get_formatted_header_footer,
    log_box
)


class TestEnvironmentVariables:
    """Test environment variable handling for formatting settings."""

    def test_ascii_mode_default(self):
        """Test that ASCII mode is off by default."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to reset the global variables
            import importlib
            import rompy.formatting
            importlib.reload(rompy.formatting)
            
            assert not rompy.formatting.USE_ASCII_ONLY
            assert not rompy.formatting.get_ascii_mode()

    def test_ascii_mode_enabled(self):
        """Test that ASCII mode can be enabled via environment variable."""
        with patch.dict(os.environ, {"ROMPY_ASCII_ONLY": "true"}, clear=True):
            # Re-import to reset the global variables
            import importlib
            import rompy.formatting
            importlib.reload(rompy.formatting)
            
            assert rompy.formatting.USE_ASCII_ONLY
            assert rompy.formatting.get_ascii_mode()

    def test_simple_logs_default(self):
        """Test that simple logs mode is off by default."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to reset the global variables
            import importlib
            import rompy.formatting
            importlib.reload(rompy.formatting)
            
            assert not rompy.formatting.USE_SIMPLE_LOGS
            assert not rompy.formatting.get_simple_logs()

    def test_simple_logs_enabled(self):
        """Test that simple logs mode can be enabled via environment variable."""
        with patch.dict(os.environ, {"ROMPY_SIMPLE_LOGS": "true"}, clear=True):
            # Re-import to reset the global variables
            import importlib
            import rompy.formatting
            importlib.reload(rompy.formatting)
            
            assert rompy.formatting.USE_SIMPLE_LOGS
            assert rompy.formatting.get_simple_logs()

    def test_environment_variable_case_insensitivity(self):
        """Test that environment variables are case insensitive."""
        test_values = ["true", "TRUE", "True", "1", "yes", "YES"]
        
        for value in test_values:
            with patch.dict(os.environ, {"ROMPY_ASCII_ONLY": value}, clear=True):
                # Re-import to reset the global variables
                import importlib
                import rompy.formatting
                importlib.reload(rompy.formatting)
                
                assert rompy.formatting.USE_ASCII_ONLY
                assert rompy.formatting.get_ascii_mode()


class TestFormattedBox:
    """Test the get_formatted_box function."""

    def test_ascii_box(self):
        """Test creating a box with ASCII characters."""
        box = get_formatted_box(
            title="TEST BOX",
            content=["Line 1", "Line 2"],
            use_ascii=True,
            width=40
        )
        
        lines = box.split("\n")
        
        # Check box structure
        assert lines[0] == "+--------------------------------------+"  # Header
        assert "TEST BOX" in lines[1]  # Title
        assert "| Line 1" in lines[3]  # Content line 1
        assert "| Line 2" in lines[4]  # Content line 2
        assert lines[5] == "+--------------------------------------+"  # Footer

    def test_unicode_box(self):
        """Test creating a box with Unicode characters."""
        box = get_formatted_box(
            title="TEST BOX",
            content=["Line 1", "Line 2"],
            use_ascii=False,
            width=40
        )
        
        lines = box.split("\n")
        
        # Check box structure - using more flexible assertions for Unicode characters
        assert any(c in lines[0] for c in ["┏", "╱", "╣"])  # Unicode header contains special chars
        assert "TEST BOX" in lines[1]  # Title
        assert "Line 1" in lines[3]  # Content line 1
        assert "Line 2" in lines[4]  # Content line 2
        assert any(c in lines[5] for c in ["┛", "┗", "╯"])  # Unicode footer

    def test_box_without_title(self):
        """Test creating a box without a title."""
        box = get_formatted_box(
            content=["Line 1", "Line 2"],
            use_ascii=True,
            width=40
        )
        
        lines = box.split("\n")
        
        # Check box structure - should be just header, content, footer
        assert lines[0] == "+--------------------------------------+"  # Header
        assert "| Line 1" in lines[1]  # Content line 1
        assert "| Line 2" in lines[2]  # Content line 2
        assert lines[3] == "+--------------------------------------+"  # Footer

    def test_box_without_content(self):
        """Test creating a box without content."""
        box = get_formatted_box(
            title="TEST BOX",
            use_ascii=True,
            width=40
        )
        
        lines = box.split("\n")
        
        # Check box structure - should be just header, title, footer
        assert lines[0] == "+--------------------------------------+"  # Header
        assert "TEST BOX" in lines[1]  # Title
        assert lines[2] == "+--------------------------------------+"  # Footer

    def test_box_with_global_ascii_setting(self):
        """Test that the box uses the global ASCII setting if not specified."""
        with patch("rompy.formatting.USE_ASCII_ONLY", True):
            box = get_formatted_box(
                title="TEST BOX",
                width=40
            )
            
            lines = box.split("\n")
            assert lines[0] == "+--------------------------------------+"  # ASCII header

        with patch("rompy.formatting.USE_ASCII_ONLY", False):
            box = get_formatted_box(
                title="TEST BOX",
                width=40
            )
            
            lines = box.split("\n")
            assert any(c in lines[0] for c in ["┏", "╱", "╣"])  # Unicode header


class TestLogBox:
    """Test the log_box function."""

    def test_log_box_basic(self):
        """Test the basic functionality of log_box."""
        import logging
        from io import StringIO
        import sys
        
        # Setup a logger with a string buffer
        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False  # Don't propagate to root logger
        
        # Call log_box with the test logger
        log_box("TEST BOX", logger=logger)
        
        # Get the logged output
        output = log_buffer.getvalue()
        
        # Verify the box was logged correctly
        assert "TEST BOX" in output
        # Box should have multiple lines
        assert output.count('\n') >= 3
        
        # Clean up
        logger.removeHandler(handler)
    
    def test_log_box_no_empty_line(self):
        """Test log_box with add_empty_line=False."""
        import logging
        from io import StringIO
        
        # Setup a logger with a string buffer
        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        logger = logging.getLogger("test_logger2")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        
        # Call log_box with add_empty_line=False
        log_box("TEST BOX", logger=logger, add_empty_line=False)
        
        # Get the logged output
        output = log_buffer.getvalue()
        
        # The output should not end with two newlines
        assert not output.endswith('\n\n')
        
        # Clean up
        logger.removeHandler(handler)
    
    def test_log_box_respects_ascii_mode(self):
        """Test that log_box respects the ASCII mode setting."""
        import logging
        from io import StringIO
        
        # Setup a logger with a string buffer
        log_buffer = StringIO()
        handler = logging.StreamHandler(log_buffer)
        logger = logging.getLogger("test_logger3")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        
        # Call log_box with explicit ASCII mode
        log_box("ASCII BOX", logger=logger, use_ascii=True)
        
        # Get the logged output
        output = log_buffer.getvalue()
        
        # Check for ASCII characters
        assert "+" in output
        assert "-" in output
        
        # Clean up
        log_buffer.truncate(0)
        log_buffer.seek(0)
        
        # Call log_box with explicit Unicode mode
        log_box("UNICODE BOX", logger=logger, use_ascii=False)
        
        # Get the logged output
        output = log_buffer.getvalue()
        
        # Check for Unicode characters (at least one should be present)
        assert any(c in output for c in ["┏", "┃", "┗", "━", "┓", "┛"])
        
        # Clean up
        logger.removeHandler(handler)


class TestHeaderFooter:
    """Test the get_formatted_header_footer function."""

    def test_ascii_header_footer(self):
        """Test creating header and footer with ASCII characters."""
        header, footer, bullet = get_formatted_header_footer(
            title="TEST HEADER",
            use_ascii=True,
            width=40
        )
        
        # Check header and footer structure
        assert header.startswith("+")
        assert "TEST HEADER" in header
        assert header.endswith("+")
        assert footer == "+--------------------------------------+"
        assert bullet == "*"

    def test_unicode_header_footer(self):
        """Test creating header and footer with Unicode characters."""
        header, footer, bullet = get_formatted_header_footer(
            title="TEST HEADER",
            use_ascii=False,
            width=40
        )
        
        # Check header and footer structure - using more flexible assertions
        assert any(c in header for c in ["┏", "╱", "╣"])  # Unicode header start
        assert "TEST HEADER" in header
        assert any(c in header for c in ["┓", "╳", "╣"])  # Unicode header end
        assert any(c in footer for c in ["┛", "┗", "╯"])  # Unicode footer
        assert bullet == "•"

    def test_header_footer_without_title(self):
        """Test creating header and footer without a title."""
        header, footer, bullet = get_formatted_header_footer(
            use_ascii=True,
            width=40
        )
        
        # Check header and footer structure
        assert header == "+--------------------------------------+"
        assert footer == "+--------------------------------------+"
        assert bullet == "*"

    def test_header_footer_with_global_ascii_setting(self):
        """Test that header/footer uses the global ASCII setting if not specified."""
        with patch("rompy.formatting.USE_ASCII_ONLY", True):
            header, footer, bullet = get_formatted_header_footer(
                title="TEST HEADER",
                width=40
            )
            
            assert header.startswith("+")  # ASCII header
            assert bullet == "*"  # ASCII bullet

        with patch("rompy.formatting.USE_ASCII_ONLY", False):
            header, footer, bullet = get_formatted_header_footer(
                title="TEST HEADER",
                width=40
            )
            
            assert any(c in header for c in ["┏", "╱", "╣"])  # Unicode header
            assert bullet == "•"  # Unicode bullet
