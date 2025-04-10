.. -*- mode: rst -*-

==========
Quickstart
==========

Installation
------------

Install `rompy` and its core dependencies from PyPI:

.. code-block:: bash

    pip install rompy

To include support for specific models like SWAN, SCHISM, XBeach, or extra data handling features, install with optional extras:

.. code-block:: bash

    pip install rompy[swan]
    pip install rompy[schism]
    pip install rompy[xbeach]
    pip install rompy[extra]
    pip install rompy[swan,xbeach,extra] # Install multiple extras

Or install the latest development version from source:

.. code-block:: bash

    git clone https://github.com/rom-py/rompy.git
    cd rompy
    pip install -e .[all] # Install with all extras in editable mode

Basic Usage: Declarative Setup (YAML)
--------------------------------------

The most common way to use `rompy` is via a YAML configuration file. This file defines both the runtime parameters (like the simulation period) and the detailed model setup.

1.  **Create a Configuration YAML (`config.yaml`):**
    Define your simulation runtime parameters and the model-specific configuration. This example uses a SWAN setup with a grid and a simple bathymetry input.

    .. code-block:: yaml

        # config.yaml
        run_id: swan_quickstart
        output_dir: ./swan_run
        period:
          start: 2023-01-01T00:00:00
          duration: 12h
          interval: 1h
        config:
          model_type: swanconfig # Selects the SWAN component-based config from rompy-swan plugin
          # --- Define SWAN Components ---
          cgrid: # Computational Grid
            model_type: regular
            grid:
              model_type: gridregular
              x0: 115.0
              y0: -33.0
              rot: 0.0
              nx: 50
              ny: 50
              dx: 0.01
              dy: 0.01
            spectrum:
              model_type: spectrum
              mdc: 36
              flow: 0.04
              fhigh: 1.0
          inpgrid: # Input Grids / Forcing Data
             model_type: data_interface
             bottom: # Bathymetry Input
               model_type: swandata # Specifies SwanDataGrid (from rompy-swan)
               var: BOTTOM # SWAN variable name
               source:
                 model_type: file # Specifies core SourceFile
                 uri: ./bathy_data.nc # Path to your bathymetry file
               z1: depth # Variable name for depth in bathy_data.nc
               coords: # Coordinate names in bathy_data.nc
                 x: longitude
                 y: latitude
             # --- Add other inputs like WIND here using appropriate SwanDataGrid ---
          lockup: # Simulation execution control
            model_type: lockup
            compute:
              model_type: stat # Stationary computation

2.  **Run `rompy` from the command line:**
    Pass the model type (`swanconfig`) and the configuration file path to the `rompy` CLI.

    .. code-block:: bash

        rompy swanconfig config.yaml

    Alternatively, use environment variables:

    .. code-block:: bash

        export ROMPY_MODEL=swanconfig
        export ROMPY_CONFIG=config.yaml
        rompy

    Or pass the YAML content directly:

    .. code-block:: bash

        export ROMPY_MODEL=swanconfig
        export ROMPY_CONFIG="$(cat config.yaml)"
        rompy

3.  **Run `rompy` from Python:**

    .. code-block:: python

        from rompy.model import ModelRun
        import yaml

        # Load configuration from YAML
        with open("config.yaml", "r") as f:
            config_dict = yaml.safe_load(f)

        # Create and generate the model run
        model = ModelRun(**config_dict)
        model.generate() # Or simply call model()

        print(f"Model run generated in: {model.staging_dir}")
        # Now you can execute the SWAN model using the files in ./swan_run/swan_quickstart

Basic Usage: Procedural Setup (Python)
----------------------------------------
You can also define the configuration directly in Python using the Pydantic models.

.. code-block:: python

    from rompy.model import ModelRun
    # Import the specific config class from the relevant model plugin
    # e.g., from rompy.swan.config import SwanConfigComponents
    # or from rompy_xbeach.config import Config as XBeachConfig
    from rompy.swan.config import SwanConfigComponents
    from rompy.core.source import SourceFile # Example core source
    from datetime import datetime

    # Define the SWAN configuration object
    swan_config = SwanConfigComponents(
        cgrid=dict(
            model_type="regular",
            grid=dict(
                model_type="gridregular",
                x0=115.0, y0=-33.0, rot=0.0,
                nx=50, ny=50, dx=0.01, dy=0.01
            ),
            spectrum=dict(
                model_type="spectrum",
                mdc=36, flow=0.04, fhigh=1.0
            )
        ),
        inpgrid=dict(
            model_type="data_interface",
            bottom=dict(
                model_type="swandata", # Model-specific data object from rompy-swan
                var="BOTTOM",
                source=SourceFile(uri="./bathy_data.nc"), # Core source object
                z1="depth",
                coords=dict(x="longitude", y="latitude"),
            )
        ),
        lockup=dict(
            model_type="lockup",
            compute=dict(model_type="stat")
        )
        # Add other components like physics, boundary, output as needed
    )

    # Define the ModelRun
    model = ModelRun(
        run_id="swan_procedural",
        output_dir="./swan_run_py",
        period=dict(start=datetime(2023, 1, 1), duration="12h", interval="1h"),
        config=swan_config # Pass the model-specific config object
    )

    # Generate the input files
    model.generate()

    print(f"Model run generated in: {model.staging_dir}")

Next Steps
----------
Explore the :doc:`core_concepts` to understand the building blocks (including Grids and Data Handling), dive into specific :doc:`models` for details on SWAN and SCHISM setups, and check out the :doc:`demo` notebooks for practical examples. See the `rompy-xbeach <https://github.com/rom-py/rompy-xbeach>`_ repository for XBeach integration examples.