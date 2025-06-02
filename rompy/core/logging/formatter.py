"""
Formatted output utilities for ROMPY.

This module provides utilities for creating consistent, visually appealing output
in both ASCII and Unicode modes. It handles boxes, headers, footers, and other
formatting elements.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Type, TypeVar

from pydantic import Field, field_validator, model_validator

from .config import LoggingConfig, LogLevel


class BoxStyle(str, Enum):
    """Predefined box styles for consistent output."""

    # Basic styles
    SIMPLE = "simple"
    ROUNDED = "rounded"
    DOUBLE = "double"
    # Status styles
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    PROCESSING = "processing"


@dataclass(frozen=True)
class Glyphs:
    """Unicode and ASCII glyphs for consistent output."""

    # Box corners
    TOP_LEFT: str
    TOP_RIGHT: str
    BOTTOM_LEFT: str
    BOTTOM_RIGHT: str
    # Lines
    HORIZONTAL: str
    VERTICAL: str
    # Connectors
    LEFT_T: str
    RIGHT_T: str
    TOP_T: str
    BOTTOM_T: str
    CROSS: str
    # Other
    ARROW: str
    BULLET: str
    CHECK: str
    CROSS_MARK: str
    ELLIPSIS: str


class UnicodeGlyphs(Glyphs):
    """Unicode glyphs for rich terminal output."""

    def __init__(self):
        super().__init__(
            TOP_LEFT="┌",
            TOP_RIGHT="┐",
            BOTTOM_LEFT="└",
            BOTTOM_RIGHT="┘",
            HORIZONTAL="─",
            VERTICAL="│",
            LEFT_T="┤",
            RIGHT_T="├",
            TOP_T="┴",
            BOTTOM_T="┬",
            CROSS="┼",
            ARROW="→",
            BULLET="•",
            CHECK="✓",
            CROSS_MARK="✗",
            ELLIPSIS="…",
        )


class AsciiGlyphs(Glyphs):
    """ASCII-only glyphs for compatibility."""

    def __init__(self):
        super().__init__(
            TOP_LEFT="+",
            TOP_RIGHT="+",
            BOTTOM_LEFT="+",
            BOTTOM_RIGHT="+",
            HORIZONTAL="-",
            VERTICAL="|",
            LEFT_T="+",
            RIGHT_T="+",
            TOP_T="+",
            BOTTOM_T="+",
            CROSS="+",
            ARROW="->",
            BULLET="*",
            CHECK="[OK]",
            CROSS_MARK="[X]",
            ELLIPSIS="...",
        )


class BoxFormatter:
    """Formatter for creating boxes and other visual elements."""

    def __init__(self, config: Optional[LoggingConfig] = None):
        """Initialize with optional logging config.

        Args:
            config: Logging configuration. If None, uses the global config.
        """
        self._config = config or LoggingConfig()
        print(f"BoxFormatter initialized with use_ascii={self._config.use_ascii}")
        self._glyphs: Optional[Glyphs] = None
        self._glyphs_is_ascii: bool = self._config.use_ascii

    @property
    def config(self) -> LoggingConfig:
        """Get the current config."""
        return self._config

    @config.setter
    def config(self, value: LoggingConfig) -> None:
        """Update the config and reset glyphs cache."""
        self._config = value
        self._glyphs = None
        self._glyphs_is_ascii = value.use_ascii

    @property
    def glyphs(self) -> Glyphs:
        """Get the appropriate glyphs based on current config."""
        # Always check the current config value to handle runtime changes
        current_ascii = self.config.use_ascii
        print(
            f"Accessing glyphs: current_ascii={current_ascii}, _glyphs_is_ascii={self._glyphs_is_ascii}, _glyphs={self._glyphs}"
        )

        if self._glyphs is None or self._glyphs_is_ascii != current_ascii:
            print(f"Creating new glyphs: use_ascii={current_ascii}")
            self._glyphs = AsciiGlyphs() if current_ascii else UnicodeGlyphs()
            self._glyphs_is_ascii = current_ascii
            print(
                f"Created {type(self._glyphs).__name__} with TOP_LEFT='{self._glyphs.TOP_LEFT}'"
            )
        return self._glyphs

    def box(
        self,
        content: str,
        title: Optional[str] = None,
        style: BoxStyle = BoxStyle.SIMPLE,
    ) -> str:
        """Create a box with optional title.

        Args:
            content: The content to put in the box
            title: Optional title for the box
            style: Box style to use

        Returns:
            Formatted box as a string
        """
        lines = content.splitlines()
        max_width = max(len(line) for line in lines + ([title] if title else []))

        # Create top border
        top_border = (
            self.glyphs.TOP_LEFT
            + self.glyphs.HORIZONTAL * (max_width + 2)
            + self.glyphs.TOP_RIGHT
        )

        # Create bottom border
        bottom_border = (
            self.glyphs.BOTTOM_LEFT
            + self.glyphs.HORIZONTAL * (max_width + 2)
            + self.glyphs.BOTTOM_RIGHT
        )

        # Create content lines
        content_lines = [
            f"{self.glyphs.VERTICAL} {line.ljust(max_width)} {self.glyphs.VERTICAL}"
            for line in lines
        ]

        # Add title if provided
        if title:
            title_line = f"{self.glyphs.VERTICAL} {title.center(max_width)} {self.glyphs.VERTICAL}"
            separator = (
                self.glyphs.RIGHT_T
                + self.glyphs.HORIZONTAL * (max_width + 2)
                + self.glyphs.LEFT_T
            )
            box_lines = (
                [top_border, title_line, separator] + content_lines + [bottom_border]
            )
        else:
            box_lines = [top_border] + content_lines + [bottom_border]

        return "\n".join(box_lines)

    def status_box(self, message: str, status: BoxStyle) -> str:
        """Create a status box with appropriate styling.

        Args:
            message: The status message
            status: Status type (error, warning, success, etc.)

        Returns:
            Formatted status box
        """
        # Map status to icons
        status_icons = {
            BoxStyle.SUCCESS: self.glyphs.CHECK,
            BoxStyle.ERROR: self.glyphs.CROSS_MARK,
            BoxStyle.WARNING: "!",
            BoxStyle.INFO: "i",
            BoxStyle.PROCESSING: "⌛",
        }

        icon = status_icons.get(status, "•")
        return self.box(message, title=f" {icon} {status.upper()} ", style=status)

    def bullet_list(self, items: List[str], indent: int = 2) -> str:
        """Create a bulleted list.

        Args:
            items: List items
            indent: Number of spaces to indent

        Returns:
            Formatted bullet list
        """
        indent_str = " " * indent
        return "\n".join(f"{indent_str}{self.glyphs.BULLET} {item}" for item in items)

    def arrow(self, text: str) -> str:
        """Format text with an arrow."""
        return f"{self.glyphs.ARROW} {text}"

    def success(self, text: str) -> str:
        """Format success message."""
        return f"{self.glyphs.CHECK} {text}"

    def error(self, text: str) -> str:
        """Format error message."""
        return f"{self.glyphs.CROSS_MARK} {text}"

    def warning(self, text: str) -> str:
        """Format warning message."""
        return f"! {text}"

    def info(self, text: str) -> str:
        """Format info message."""
        return f"{text}"


# Default formatter instance
formatter = BoxFormatter()


# Convenience functions that use the default formatter
def box(
    content: str, title: Optional[str] = None, style: BoxStyle = BoxStyle.SIMPLE
) -> str:
    """Create a box with optional title (using default formatter)."""
    return formatter.box(content, title, style)


def status_box(message: str, status: BoxStyle) -> str:
    """Create a status box (using default formatter)."""
    return formatter.status_box(message, status)


def bullet_list(items: List[str], indent: int = 2) -> str:
    """Create a bulleted list (using default formatter)."""
    return formatter.bullet_list(items, indent)


# Common glyph accessors
ARROW = UnicodeGlyphs().ARROW
BULLET = UnicodeGlyphs().BULLET
CHECK = UnicodeGlyphs().CHECK
CROSS_MARK = UnicodeGlyphs().CROSS_MARK
ELLIPSIS = UnicodeGlyphs().ELLIPSIS
