# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import logging
import warnings
import os
from pathlib import Path

# from . import _version

logger = logging.getLogger(__name__)

# Configure warnings to be less intrusive
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="intake.readers.readers")
warnings.filterwarnings("ignore", message="A custom validator is returning a value other than `self`")

# __version__ = _version.get_versions()["version"]
__version__ = "0.2.9"

# Check if we should use ASCII-only formatting (globally accessible)
USE_ASCII_ONLY = os.environ.get('ROMPY_ASCII_ONLY', '').lower() in ('1', 'true', 'yes')

def ROMPY_ASCII_MODE():
    """Return the current ASCII mode setting.
    
    This helper function makes it easier to access the ASCII mode setting
    from any module without potential circular import issues.
    
    Returns:
        bool: True if ASCII-only mode is enabled, False otherwise
    """
    return os.environ.get('ROMPY_ASCII_ONLY', '').lower() in ('1', 'true', 'yes')

ROOT_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = ROOT_DIR / "templates"
