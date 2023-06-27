from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

import yaml
from pydantic import BaseModel, PrivateAttr, validator

from rompy import TEMPLATES_DIR
from rompy.types import RompyBaseModel

# from rompy.data import DataBlob, DataGrid

logger = logging.getLogger(__name__)


class BaseConfig(RompyBaseModel):
    """A base class for all templates

    Parameters
    ----------
    template : str
        The template directory
    checkout : str
        The git checkout branch, tag or commit. Only used if the template is a git repository
    _datefmt : str
        The date format to be rendered in the template
    """

    template: str = os.path.join(TEMPLATES_DIR, "base")
    checkout: str = "main"

    _datefmt: str = "%Y%m%d.%H%M%S"
    _generated_at: str = PrivateAttr()
    _generated_on: str = PrivateAttr()
    _generated_by: str = PrivateAttr()
