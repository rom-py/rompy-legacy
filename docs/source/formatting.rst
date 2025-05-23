============================
Formatting and Logging Framework
============================

Overview
--------

ROMPY includes a comprehensive formatting and logging framework that provides consistent visual presentation of information throughout the application. This framework includes:

1. **Customizable Logging**: Configure log levels, formats, and outputs
2. **Formatted Output Boxes**: Create consistent visual elements for status reports and information display
3. **String Representation**: A standardized approach to presenting model objects as strings
4. **Environment-based Configuration**: Control formatting behavior via environment variables

This framework is designed to be flexible yet consistent, allowing both programmatic and command-line usage.

Core Components
--------------

The formatting framework spans several files:

- ``rompy/formatting.py``: Core formatting utilities
- ``rompy/utils.py``: Helper functions for formatting
- ``rompy/core/types.py``: Base model with string representation capabilities
- ``rompy/cli.py``: Command-line interface with formatting options

Environment Variables
--------------------

The formatting behavior can be controlled with these environment variables:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``ROMPY_ASCII_ONLY``
     - ``False``
     - When set to ``True``, use only ASCII characters for boxes and formatting
   * - ``ROMPY_SIMPLE_LOGS``
     - ``False``
     - When set to ``True``, simplify log format (no timestamps or module names)
   * - ``ROMPY_LOG_DIR``
     - None
     - Directory to save log files (optional)

These can be set in your environment or passed via the CLI arguments.

Logging Configuration
--------------------

The ``configure_logging()`` function provides a unified way to set up logging:

.. code-block:: python

    from rompy.formatting import configure_logging
    
    # Configure with increasing verbosity (0=INFO, 1=VERBOSE, 2=DEBUG)
    configure_logging(verbosity=1, log_dir='/path/to/logs')

From the command line, logging can be configured with:

.. code-block:: bash

    rompy swan config.yml -v  # Verbose mode
    rompy swan config.yml -vv  # Debug mode
    rompy swan config.yml --log-dir /path/to/logs  # Save logs to directory
    rompy swan config.yml --simple-logs  # Use simple log format

Formatted Output Boxes
---------------------

The framework provides utilities to create consistent visual elements:

.. code-block:: python

    from rompy.formatting import get_formatted_box, get_formatted_header_footer
    
    # Create a simple box with a title
    box = get_formatted_box("Status Report", use_ascii=False, width=60)
    print(box)
    
    # Create header, footer, and bullet character for more complex output
    header, footer, bullet = get_formatted_header_footer(
        "Processing Results", 
        use_ascii=False, 
        width=70
    )
    
    # Use in output
    print(header)
    print(f"{bullet} Processed 100 items")
    print(f"{bullet} Found 5 anomalies")
    print(footer)

These utilities respect the ``ROMPY_ASCII_ONLY`` setting, determining whether to use Unicode or ASCII characters:

**Unicode Mode (default):**

.. code-block:: text

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃                        Status Report                         ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

**ASCII Mode:**

.. code-block:: text

    +------------------------------------------------------------+
    |                        Status Report                        |
    +------------------------------------------------------------+

String Representation
--------------------

The ``RompyBaseModel`` class in ``rompy/core/types.py`` provides a standardized string representation for all model objects:

.. code-block:: python

    from rompy.core.types import RompyBaseModel
    
    class MyModel(RompyBaseModel):
        # Model definition...
        pass
    
    model = MyModel(param1="value1", param2=42)
    print(model)  # Uses the hierarchical string representation

Output will be formatted hierarchically:

.. code-block:: text

    MyModel:
      param1: value1
      param2: 42
      nested_object:
        attr1: something
        attr2: [
          [0]: first item
          [1]: second item
        ]

Customizing String Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize the string representation by overriding the ``_format_value`` method:

.. code-block:: python

    class MyCustomModel(RompyBaseModel):
        def _format_value(self, obj):
            # Custom formatting for specific types
            if isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M")
            # Return None to use default formatting
            return None

Value Formatting
---------------

The ``format_value()`` function in ``formatting.py`` provides special formatting for common types:

- Path objects
- Datetime objects
- Timedelta objects

Usage from Command Line
----------------------

The formatting framework is accessible through CLI options:

.. code-block:: bash

    # Use ASCII-only characters
    rompy swan config.yml --ascii-only
    
    # Use simple log format (no timestamps or module names)
    rompy swan config.yml --simple-logs
    
    # Control verbosity
    rompy swan config.yml -v         # Verbose
    rompy swan config.yml -vv        # Debug
    
    # Specify log directory
    rompy swan config.yml --log-dir /path/to/logs

Integration Examples
------------------

Combining Multiple Features
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rompy.formatting import configure_logging, get_formatted_box
    import logging
    
    # Configure logging
    configure_logging(verbosity=1)
    logger = logging.getLogger(__name__)
    
    # Create a formatted status box
    status_box = get_formatted_box("Processing Started", width=50)
    
    # Log the status
    logger.info("\n" + status_box)
    
    # Continue with processing...
    logger.info("Processing item 1...")

CLI Application with Custom Formatting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import click
    from rompy.formatting import get_formatted_header_footer
    import os
    
    @click.command()
    @click.option("--ascii-only", is_flag=True, help="Use ASCII-only characters")
    def process(ascii_only):
        # Set environment variable
        os.environ['ROMPY_ASCII_ONLY'] = 'true' if ascii_only else 'false'
        
        # Create header and footer
        header, footer, bullet = get_formatted_header_footer(
            "Processing Results", 
            use_ascii=ascii_only, 
            width=60
        )
        
        # Display formatted output
        print(header)
        print(f"{bullet} Processed 100 items")
        print(f"{bullet} Found 5 anomalies")
        print(footer)

Best Practices
-------------

1. **Consistent Usage**: Always use the formatting utilities for output to maintain a consistent appearance
2. **Environment Awareness**: Check and respect the environment variables
3. **Customization**: Override the `_format_value` method for custom types rather than the entire `__str__` method
4. **CLI Options**: Provide formatting options in CLI applications
5. **Documentation**: Document the formatting options available to users

For more examples and detailed API documentation, see the API Reference section.