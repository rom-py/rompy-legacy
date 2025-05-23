import os
import logging
from typing import Tuple, Optional, Any, Dict, List

# Define constants for ASCII mode and simple logs
# Check if we should use ASCII-only formatting
USE_ASCII_ONLY = os.environ.get('ROMPY_ASCII_ONLY', '').lower() in ('1', 'true', 'yes')
# Check if simple logs are requested (no timestamps or module names)
USE_SIMPLE_LOGS = os.environ.get('ROMPY_SIMPLE_LOGS', '').lower() in ('1', 'true', 'yes')

# Define commonly used formatting elements based on ASCII mode
# Arrows for indicating actions or steps
ARROW = "->" if USE_ASCII_ONLY else "→"
# Bullet points for lists
BULLET = "*" if USE_ASCII_ONLY else "•"
# Default widths
DEFAULT_WIDTH = 72 if USE_ASCII_ONLY else 70

# Common table formatting elements
TABLE_FORMATS = {
    "ascii": {
        "top_line": "+{}-+{}-+",
        "header_line": "| {} | {} |",
        "separator": "+{}-+{}-+",
        "data_line": "| {} | {} |",
        "bottom_line": "+{}-+{}-+",
        "h_line": "-" * DEFAULT_WIDTH,
    },
    "unicode": {
        "top_line": "┏{}━┳{}━┓",
        "header_line": "┃ {} ┃ {} ┃",
        "separator": "┠{}━╋{}━┨",
        "data_line": "┃ {} ┃ {} ┃",
        "bottom_line": "┗{}━┻{}━┛",
        "h_line": "━" * DEFAULT_WIDTH,
    }
}

# Status box headers and footers
STATUS_BOX_TEMPLATES = {
    "processing": {
        "title": "PROCESSING",
        "width": DEFAULT_WIDTH
    },
    "completed": {
        "title": "COMPLETED",
        "width": DEFAULT_WIDTH
    },
    "error": {
        "title": "ERROR",
        "width": DEFAULT_WIDTH
    },
    "warning": {
        "title": "WARNING",
        "width": DEFAULT_WIDTH
    },
    "info": {
        "title": "INFORMATION",
        "width": DEFAULT_WIDTH
    }
}

# Helper functions to avoid circular imports
def get_ascii_mode() -> bool:
    """Return the current ASCII mode setting.

    This helper function makes it easier to access the ASCII mode setting
    from any module without potential circular import issues.

    Returns:
        bool: True if ASCII-only mode is enabled, False otherwise
    """
    return os.environ.get('ROMPY_ASCII_ONLY', '').lower() in ('1', 'true', 'yes')

def get_simple_logs() -> bool:
    """Return the current simple logs setting.

    This helper function makes it easier to access the simple logs setting
    from any module without potential circular import issues.

    Returns:
        bool: True if simple logs mode is enabled, False otherwise
    """
    return os.environ.get('ROMPY_SIMPLE_LOGS', '').lower() in ('1', 'true', 'yes')


def configure_logging(verbosity: int = 0, log_dir: Optional[str] = None) -> None:
    """Configure logging for ROMPY based on verbosity level and environment settings.
    
    Args:
        verbosity: Level of verbosity (0=INFO, 1=VERBOSE, 2=DEBUG)
        log_dir: Optional directory to save log files
    """
    # Set log level based on verbosity
    log_level = logging.INFO
    if verbosity >= 1:
        log_level = logging.INFO  # We use INFO for verbose output too
    if verbosity >= 2:
        log_level = logging.DEBUG

    # Check if simple logs are requested (no timestamps or module names)
    simple_logs = get_simple_logs()

    # Create a custom formatter with or without timestamp and level
    if simple_logs:
        # Simple format with just the message
        formatter = logging.Formatter('%(message)s')
    else:
        # Detailed format with timestamp, level, and module name
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)-20s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Configure console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(log_level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the console handler
    root_logger.addHandler(console)
    
    # Configure file handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'rompy.log'))
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)


def get_formatted_header_footer(
    title: str = None, 
    use_ascii: Optional[bool] = None, 
    width: Optional[int] = None
) -> Tuple[str, str, str]:
    """Create formatted header and footer for output blocks.
    
    Args:
        title: The title text to display in the header (optional)
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
        width: The width of the header/footer in characters (defaults to ASCII-appropriate width)
        
    Returns:
        A tuple containing (header, footer, bullet_char)
    """
    # If ASCII mode isn't specified, use the global setting
    if use_ascii is None:
        # For test compatibility, check the USE_ASCII_ONLY directly
        # This allows patching of the variable in tests to work
        use_ascii = USE_ASCII_ONLY
    
    # If width isn't specified, use a sensible default based on ASCII mode
    if width is None:
        width = DEFAULT_WIDTH
    
    if use_ascii:
        # Create ASCII-only header/footer
        bullet = "*"  # For test compatibility, use hardcoded value
        
        if title:
            header = f"+{'-' * (width - 2)}+"
            title_line = f"| {title.center(width - 4)} |"
            separator = f"+{'-' * (width - 2)}+"
            # Combine header with title
            header = f"{header}\n{title_line}\n{separator}"
        else:
            header = f"+{'-' * (width - 2)}+"
            
        footer = f"+{'-' * (width - 2)}+"
    else:
        # Create Unicode header/footer
        bullet = "•"  # For test compatibility, use hardcoded value
        
        if title:
            header = f"┏{'━' * (width - 2)}┓"
            title_line = f"┃ {title.center(width - 4)} ┃"
            separator = f"┠{'━' * (width - 2)}┨"
            # Combine header with title
            header = f"{header}\n{title_line}\n{separator}"
        else:
            header = f"┏{'━' * (width - 2)}┓"
            
        footer = f"┗{'━' * (width - 2)}┛"
    
    return header, footer, bullet


def get_formatted_box(
    title: str = None, 
    content: List[str] = None,
    use_ascii: Optional[bool] = None, 
    width: Optional[int] = None
) -> str:
    """Create a formatted box with a title and optional content.
    
    Args:
        title: The title text to display in the box
        content: Optional list of content lines to display in the box
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
        width: The width of the box in characters (defaults to ASCII-appropriate width)
        
    Returns:
        A string containing the formatted box
    """
    # If ASCII mode isn't specified, use the global setting
    if use_ascii is None:
        # For test compatibility, check the USE_ASCII_ONLY directly
        # This allows patching of the variable in tests to work
        use_ascii = USE_ASCII_ONLY
    
    # If width isn't specified, use a sensible default based on ASCII mode
    if width is None:
        width = DEFAULT_WIDTH
    
    if use_ascii:
        # Create ASCII-only box
        top = f"+{'-' * (width - 2)}+"
        bottom = f"+{'-' * (width - 2)}+"
        
        lines = []
        lines.append(top)
        
        if title:
            lines.append(f"| {title.center(width - 4)} |")
            if content:
                lines.append(f"+{'-' * (width - 2)}+")
        
        if content:
            for line in content:
                lines.append(f"| {line.ljust(width - 4)} |")
        
        lines.append(bottom)
        return "\n".join(lines)
    else:
        # Create Unicode box
        top = f"┏{'━' * (width - 2)}┓"
        bottom = f"┗{'━' * (width - 2)}┛"
        
        lines = []
        lines.append(top)
        
        if title:
            lines.append(f"┃ {title.center(width - 4)} ┃")
            if content:
                lines.append(f"┠{'━' * (width - 2)}┨")
        
        if content:
            for line in content:
                lines.append(f"┃ {line.ljust(width - 4)} ┃")
        
        lines.append(bottom)
        return "\n".join(lines)


def log_box(title: str, 
            logger=None, 
            use_ascii: Optional[bool] = None, 
            width: Optional[int] = None, 
            add_empty_line: bool = True) -> None:
    """Create a formatted box and log each line.
    
    This utility function creates a formatted box and logs each line to the specified
    logger, handling the common pattern of creating a box and then splitting it
    for logging.
    
    Args:
        title: The title text to display in the box
        logger: The logger to use (if None, imports and uses the root logger)
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
        width: The width of the box in characters (defaults to ASCII-appropriate width)
        add_empty_line: Whether to add an empty line after the box
    """
    # If ASCII mode isn't specified, use the global setting
    if use_ascii is None:
        # For test compatibility, check the USE_ASCII_ONLY directly
        # This allows patching of the variable in tests to work
        use_ascii = USE_ASCII_ONLY
    
    # If width isn't specified, use a sensible default based on ASCII mode
    if width is None:
        width = DEFAULT_WIDTH
    
    # Create the formatted box
    box = get_formatted_box(title=title, use_ascii=use_ascii, width=width)
    
    # Use the provided logger or get the root logger
    if logger is None:
        import logging
        logger = logging.getLogger()
    
    # Log each line of the box
    for line in box.split("\n"):
        logger.info(line)
    
    # Add an empty line if requested
    if add_empty_line:
        logger.info("")


def format_value(obj: Any) -> Optional[str]:
    """Format specific types of values for display.
    
    This utility function provides special formatting for specific types
    used throughout ROMPY, such as paths, timestamps, and configuration objects.
    
    Args:
        obj: The object to format
        
    Returns:
        A formatted string or None to use default formatting
    """
    from pathlib import Path
    from datetime import datetime, timedelta
    
    # Format Path objects
    if isinstance(obj, Path):
        return str(obj)
    
    # Format datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat(' ')
    
    # Format timedelta objects
    if isinstance(obj, timedelta):
        # Simple formatting for timedelta
        days = obj.days
        hours, remainder = divmod(obj.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days} days, {hours} hours, {minutes} minutes"
        elif hours > 0:
            return f"{hours} hours, {minutes} minutes"
        elif minutes > 0:
            return f"{minutes} minutes, {seconds} seconds"
        else:
            return f"{seconds} seconds"
    
    # Use default formatting for other types
    return None


def get_table_format(use_ascii: Optional[bool] = None) -> Dict:
    """Get the appropriate table formatting elements based on ASCII mode.
    
    Args:
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
        
    Returns:
        A dictionary of table formatting elements
    """
    if use_ascii is None:
        # For test compatibility, check the USE_ASCII_ONLY directly
        # This allows patching of the variable in tests to work
        use_ascii = USE_ASCII_ONLY
        
    return TABLE_FORMATS["ascii"] if use_ascii else TABLE_FORMATS["unicode"]


def format_table_row(key: str, value: str, use_ascii: Optional[bool] = None) -> str:
    """Format a key-value pair as a table row.
    
    Args:
        key: The key/label for the row
        value: The value to display
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
        
    Returns:
        A formatted table row string
    """
    if use_ascii is None:
        # For test compatibility, check the USE_ASCII_ONLY directly
        # This allows patching of the variable in tests to work
        use_ascii = USE_ASCII_ONLY
        
    format_dict = TABLE_FORMATS["ascii"] if use_ascii else TABLE_FORMATS["unicode"]
    return format_dict["data_line"].format(key, value)


def log_horizontal_line(logger=None, use_ascii: Optional[bool] = None) -> None:
    """Log a horizontal line for visual separation.
    
    Args:
        logger: The logger to use (if None, imports and uses the root logger)
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
    """
    if use_ascii is None:
        # For test compatibility, check the USE_ASCII_ONLY directly
        # This allows patching of the variable in tests to work
        use_ascii = USE_ASCII_ONLY
        
    # Use the provided logger or get the root logger
    if logger is None:
        import logging
        logger = logging.getLogger()
    
    line = TABLE_FORMATS["ascii"]["h_line"] if use_ascii else TABLE_FORMATS["unicode"]["h_line"]
    logger.info(line)


def get_status_box(status_type: str, custom_title: Optional[str] = None, use_ascii: Optional[bool] = None) -> str:
    """Get a pre-configured status box.
    
    Args:
        status_type: Type of status box ('processing', 'completed', 'error', 'warning', 'info')
        custom_title: Optional custom title to override the default
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
        
    Returns:
        A formatted box string
    """
    if status_type not in STATUS_BOX_TEMPLATES:
        status_type = "info"  # Default fallback
        
    template = STATUS_BOX_TEMPLATES[status_type]
    title = custom_title if custom_title else template["title"]
    
    # If ASCII mode isn't specified, use the global setting directly for test compatibility
    if use_ascii is None:
        use_ascii = USE_ASCII_ONLY
    
    return get_formatted_box(
        title=title,
        use_ascii=use_ascii,
        width=template["width"]
    )


def log_status(status_type: str, custom_title: Optional[str] = None, logger=None, add_empty_line: bool = True) -> None:
    """Log a pre-configured status box.
    
    Args:
        status_type: Type of status box ('processing', 'completed', 'error', 'warning', 'info')
        custom_title: Optional custom title to override the default
        logger: The logger to use (if None, imports and uses the root logger)
        add_empty_line: Whether to add an empty line after the box
    """
    # Use the provided logger or get the root logger
    if logger is None:
        import logging
        logger = logging.getLogger()
        
    box = get_status_box(status_type, custom_title)
    
    # Log each line of the box
    for line in box.split("\n"):
        logger.info(line)
    
    # Add an empty line if requested
    if add_empty_line:
        logger.info("")


def str_helper(lines: list, name: str, obj: Any, level: int, format_value_func=None) -> None:
    """Helper method to build a hierarchical string representation.
    
    Args:
        lines: List to append formatted string lines
        name: Name of the current object/field
        obj: The object to format
        level: Current indentation level
        format_value_func: Optional custom formatter function
    """
    indent = "  " * level
    
    # Handle None values
    if obj is None:
        lines.append(f"{indent}{name}: None")
        return
    
    # Check if there's a custom formatter
    if format_value_func is not None:
        custom_format = format_value_func(obj)
        if custom_format is not None:
            if "\n" in custom_format:
                # For multi-line string representations
                lines.append(f"{indent}{name}:")
                for line in custom_format.split("\n"):
                    lines.append(f"{indent}  {line}")
            else:
                lines.append(f"{indent}{name}: {custom_format}")
            return
    
    # Use the object's custom __str__ if it has one
    str_method = getattr(obj.__class__, "__str__", None)
    object_str_method = getattr(object, "__str__", None)
    
    if (str_method is not None and str_method is not object_str_method):
        # Use the object's custom __str__
        str_val = str(obj)
        if "\n" in str_val:
            # For multi-line string representations
            lines.append(f"{indent}{name}:")
            for line in str_val.split("\n"):
                lines.append(f"{indent}  {line}")
        else:
            lines.append(f"{indent}{name}: {str_val}")
    elif hasattr(obj, 'items') and callable(getattr(obj, 'items')):
        # Handle dictionary-like objects
        if not obj:
            lines.append(f"{indent}{name}: {{}}")
        else:
            lines.append(f"{indent}{name}:")
            for key, value in obj.items():
                str_helper(lines, str(key), value, level + 1, format_value_func)
    elif hasattr(obj, '__iter__') and not isinstance(obj, str):
        # Handle list-like objects
        if not obj:
            lines.append(f"{indent}{name}: []")
        else:
            lines.append(f"{indent}{name}:")
            for i, item in enumerate(obj):
                str_helper(lines, f"[{i}]", item, level + 1, format_value_func)
    else:
        # Default case for simple values
        lines.append(f"{indent}{name}: {obj}")