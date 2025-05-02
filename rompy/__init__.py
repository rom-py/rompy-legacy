# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import logging
from pathlib import Path

# from . import _version

logger = logging.getLogger(__name__)

# __version__ = _version.get_versions()["version"]
__version__ = "0.2.7"

ROOT_DIR = Path(__file__).parent.resolve()
TEMPLATES_DIR = ROOT_DIR / "templates"
