# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

from ._version import get_versions
import intake
import logging
import os
import warnings

from .model import ModelRun

logger = logging.getLogger("rompy")


here = os.path.abspath(os.path.dirname(__file__))
cat = intake.open_catalog(os.path.join(here, "catalogs", "master.yaml"))


__version__ = get_versions()["version"]
del get_versions

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
