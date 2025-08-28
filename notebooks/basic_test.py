#!/usr/bin/env python


from rompy.swan import SwanModel

new_simulation = SwanModel(
    run_id="test_swan", template="../rompy/templates/swan", output_dir="simulations"
)

new_simulation.settings
