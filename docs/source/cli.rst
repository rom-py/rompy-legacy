.. _cli:

Command Line Interface
======================

The ROMPY Command Line Interface (CLI) provides a convenient way to run ocean, wave, and hydrodynamic models with configuration files.

Basic Usage
------------

The basic syntax for running a model is:

.. code-block:: bash

    rompy <model> <config-file> [OPTIONS]

Where:
- ``<model>``: The type of model to run (e.g., swan, schism)
- ``<config-file>``: Path to a YAML or JSON configuration file

Available Models
-----------------

To list all available models, run:

.. code-block:: bash

    rompy --version

This will display the ROMPY version and a list of available models.

Options
-------

.. program:: rompy
.. option:: --zip, --no-zip

    Create a zip archive of the model files (default: False)

.. option:: -v, --verbose

    Increase verbosity (can be used multiple times for more detail)

.. option:: --log-dir PATH

    Directory to save log files

.. option:: --show-warnings

    Show Python warnings (default: False)

.. option:: --ascii-only

    Use ASCII-only characters in output (default: False)

.. option:: --simple-logs

    Use simple log format without timestamps and module names (default: False)

.. option:: --version

    Show version information and exit

Examples
--------

Run a SWAN model with a configuration file:

.. code-block:: bash

    rompy swan config.yml

Run with increased verbosity and save logs to a directory:

.. code-block:: bash

    rompy swan config.yml -v --log-dir ./logs

Run with ASCII-only output and simple logging format:

.. code-block:: bash

    rompy swan config.yml --ascii-only --simple-logs

Environment Variables
----------------------

You can set the following environment variables as an alternative to command-line options:

- ``ROMPY_MODEL``: Default model to use
- ``ROMPY_CONFIG``: Default configuration file
- ``ROMPY_ZIP``: Set to "1" to enable zip output
- ``ROMPY_LOG_DIR``: Directory for log files
- ``ROMPY_ASCII_ONLY``: Set to "1" for ASCII-only output
- ``ROMPY_SIMPLE_LOGS``: Set to "1" for simple log format

Configuration File Format
--------------------------

The configuration file can be in either YAML or JSON format. The structure depends on the specific model being used. Refer to the model's documentation for details.

Example YAML configuration:

.. code-block:: yaml

    model_type: "swan"
    start_time: "2023-01-01T00:00:00"
    end_time: "2023-01-02T00:00:00"
    time_step: 3600
    grid:
        nx: 100
        ny: 100
        dx: 1000
        dy: 1000
    # Additional model-specific parameters...

Exit Codes
----------

The CLI uses the following exit codes:

- ``0``: Success
- ``1``: Error running the model
- ``2``: Invalid arguments or configuration
