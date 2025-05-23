import os
import logging
from typing import Tuple, Optional, Any

# Define constants for ASCII mode and simple logs
# Check if we should use ASCII-only formatting
USE_ASCII_ONLY = os.environ.get('ROMPY_ASCII_ONLY', '').lower() in ('1', 'true', 'yes')
# Check if simple logs are requested (no timestamps or module names)
USE_SIMPLE_LOGS = os.environ.get('ROMPY_SIMPLE_LOGS', '').lower() in ('1', 'true', 'yes')

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
    title: str, 
    use_ascii: Optional[bool] = None, 
    width: int = 70
) -> Tuple[str, str, str]:
    """Create formatted header and footer for output blocks.
    
    Args:
        title: The title text to display in the header
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
        width: The width of the header/footer in characters
        
    Returns:
        A tuple containing (header, footer, bullet_char)
    """
    # If ASCII mode isn't specified, use the global setting
    if use_ascii is None:
        use_ascii = get_ascii_mode()
    
    if use_ascii:
        # Create ASCII-only header/footer
        header = f"+{'-' * (width - 2)}+"
        title_line = f"| {title.center(width - 4)} |"
        separator = f"+{'-' * (width - 2)}+"
        footer = f"+{'-' * (width - 2)}+"
        bullet = "*"
    else:
        # Create Unicode header/footer
        header = f"┏{'━' * (width - 2)}┓"
        title_line = f"┃ {title.center(width - 4)} ┃"
        separator = f"┠{'━' * (width - 2)}┨"
        footer = f"┗{'━' * (width - 2)}┛"
        bullet = "•"
    
    # Combine header with title
    header = f"{header}\n{title_line}\n{separator}"
    
    return header, footer, bullet


def get_formatted_box(
    title: str, 
    use_ascii: Optional[bool] = None, 
    width: int = 70
) -> str:
    """Create a formatted box with a title.
    
    Args:
        title: The title text to display in the box
        use_ascii: Whether to use ASCII-only characters (defaults to global setting)
        width: The width of the box in characters
        
    Returns:
        A string containing the formatted box
    """
    # If ASCII mode isn't specified, use the global setting
    if use_ascii is None:
        use_ascii = get_ascii_mode()
    
    if use_ascii:
        # Create ASCII-only box
        top = f"+{'-' * (width - 2)}+"
        title_line = f"| {title.center(width - 4)} |"
        bottom = f"+{'-' * (width - 2)}+"
        return f"{top}\n{title_line}\n{bottom}"
    else:
        # Create Unicode box
        top = f"┏{'━' * (width - 2)}┓"
        title_line = f"┃ {title.center(width - 4)} ┃"
        bottom = f"┗{'━' * (width - 2)}┛"
        return f"{top}\n{title_line}\n{bottom}"


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