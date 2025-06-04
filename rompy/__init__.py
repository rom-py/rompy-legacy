# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import warnings
from pathlib import Path

# Import the new logging system
from .core.logging import get_logger, LogLevel, LogFormat, LoggingConfig

# Initialize the logger with default configuration
logger = get_logger(__name__)

# Configure default logging if not already configured
LoggingConfig().configure_logging()

# Configure warnings to be less intrusive
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="intake.readers.readers")
warnings.filterwarnings(
    "ignore", message="A custom validator is returning a value other than `self`"
)

# __version__ = _version.get_versions()["version"]
__version__ = "0.3.1"

# Import and re-export formatting utilities
from .formatting import get_formatted_box, get_formatted_header_footer

# Import logging configuration
from .core.logging import LoggingConfig

# Root directory and templates directory paths
ROOT_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = ROOT_DIR / "templates"
