"""
SWAN Module for ROMPY

This module provides interfaces and utilities for working with the SWAN
(Simulating WAves Nearshore) model within the ROMPY framework.
"""

import os
from pathlib import Path

from rompy.logging import LoggingConfig, get_logger

logger = get_logger(__name__)

# Import SWAN components
from .boundary import Boundnest1
from .config import SwanConfig
from .data import SwanDataGrid
from .grid import SwanGrid

# Configure logging for the SWAN module
logging_config = LoggingConfig()
logging_config.configure_logging()

# Log module initialization
logger.debug("SWAN module initialized")
