"""
Testing utilities package.

This package provides common utilities used by tests across different modules.
"""

from pathlib import Path
import sys

# Import utilities from modules
from .compare import compare_files, compare_nmls, compare_nmls_values, untar_file
from .schism import download_hycom

# Export the utility functions
__all__ = [
    # General comparison utilities
    "compare_files",
    "compare_nmls",
    "compare_nmls_values",
    "untar_file",
    # SCHISM-specific utilities
    "download_hycom",
]
