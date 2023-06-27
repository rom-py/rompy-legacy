#!/usr/bin/env python

import numpy as np
import pandas as pd
import wavespectra as ws

import rompy
from rompy.filters import crop_filter, rename_filter, sort_filter
from rompy.swan import SwanModel

new_simulation = SwanModel(
    run_id="test_swan", template="../rompy/templates/swan", output_dir="simulations"
)

new_simulation.settings
