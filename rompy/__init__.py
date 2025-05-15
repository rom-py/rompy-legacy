# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import logging
import warnings
from pathlib import Path

# from . import _version

logger = logging.getLogger(__name__)

# Configure warnings to be less intrusive
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="intake.readers.readers")
warnings.filterwarnings("ignore", message="A custom validator is returning a value other than `self`")

# __version__ = _version.get_versions()["version"]
__version__ = "0.2.9"

# Import and re-export formatting utilities for backward compatibility
from .formatting import (
    USE_ASCII_ONLY,
    USE_SIMPLE_LOGS,
    get_ascii_mode as ROMPY_ASCII_MODE,
    get_simple_logs as ROMPY_SIMPLE_LOGS,
    get_formatted_box,
    get_formatted_header_footer
)

# Root directory and templates directory paths
ROOT_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = ROOT_DIR / "templates"
