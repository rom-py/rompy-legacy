from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, root_validator, validator

from rompy import TEMPLATES_DIR
from rompy.configuration.base import BaseConfig
from rompy.core import RegularGrid
from rompy.data import DataGrid
from rompy.types import Coordinate






