============================
Formatting and Logging Framework
============================

Overview
--------

ROMPY includes a comprehensive formatting and logging framework that provides consistent visual presentation of information throughout the application. This framework includes:

1. **Customizable Logging**: Configure log levels, formats, and outputs
2. **Formatted Output Boxes**: Create consistent visual elements for status reports and information display
3. **Common Formatting Elements**: Centralized symbols and elements for consistent appearance
4. **String Representation**: A standardized approach to presenting model objects as strings
5. **Environment-based Configuration**: Control formatting behavior via environment variables

This framework is designed to be flexible yet consistent, allowing both programmatic and command-line usage.

Core Components
--------------

The formatting framework spans several files:

- ``rompy/formatting.py``: Core formatting utilities and centralized formatting elements
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
     - When set to ``True``, use only ASCII characters for boxes and formatting (also adjusts default box width)
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

    from rompy.formatting import get_formatted_box, get_formatted_header_footer, log_box
    import logging
    
    # Create a simple box with a title (uses defaults from environment)
    box = get_formatted_box("Status Report")
    print(box)
    
    # Explicitly specify formatting options when needed
    box = get_formatted_box("Status Report", use_ascii=False)
    print(box)
    
    # Create header, footer, and bullet character for more complex output
    header, footer, bullet = get_formatted_header_footer("Processing Results")
    
    # Use in output
    print(header)
    print(f"{bullet} Processed 100 items")
    print(f"{bullet} Found 5 anomalies")
    print(footer)
    
    # For logging a box directly (most common use case)
    logger = logging.getLogger(__name__)
    log_box("PROCESSING STARTED", logger=logger)

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

The ``log_box()`` utility simplifies the common pattern of creating a box and logging each line:

.. code-block:: python

    from rompy.formatting import log_box
    import logging
    
    logger = logging.getLogger(__name__)
    
    # This single call:
    log_box("MODEL RUN CONFIGURATION", logger=logger)
    
    # Replaces this common pattern:
    # box = get_formatted_box(title="MODEL RUN CONFIGURATION")
    # for line in box.split("\n"):
    #     logger.info(line)
    # logger.info("")

Common Formatting Elements
-------------------------

The framework provides centralized formatting elements that can be imported and used directly:

.. code-block:: python

    from rompy.formatting import ARROW, BULLET, DEFAULT_WIDTH
    
    # Use in string formatting
    print(f"{ARROW} Processing step 1")
    print(f"{BULLET} First item in list")
    
    # The elements automatically adapt to ASCII mode
    # In ASCII mode: ARROW = "->" and BULLET = "*"
    # In Unicode mode: ARROW = "→" and BULLET = "•"

Pre-defined status boxes are also available:

.. code-block:: python

    from rompy.formatting import log_status
    
    # Log different types of status boxes
    log_status("processing", logger=logger)  # "PROCESSING"
    log_status("completed", logger=logger)   # "COMPLETED"
    log_status("error", logger=logger)       # "ERROR"
    log_status("warning", logger=logger)     # "WARNING"
    log_status("info", logger=logger)        # "INFORMATION"
    
    # With custom title
    log_status("processing", "CUSTOM TITLE", logger=logger)

Additional utility functions:

.. code-block:: python

    from rompy.formatting import log_horizontal_line, format_table_row
    
    # Log a horizontal line for visual separation
    log_horizontal_line(logger=logger)
    
    # Format key-value pairs as table rows
    logger.info(format_table_row("Parameter", "Value"))
    logger.info(format_table_row("Width", "70"))

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

Hierarchical String Representation Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``RompyBaseModel`` implements a recursive string representation system that:

1. **Traverses the Object Graph**: Recursively explores nested objects, including dictionaries, lists, and other RompyBaseModel instances
2. **Maintains Indentation**: Properly indents nested structures for readability
3. **Special Handling**: Applies special formatting for common types like datetimes, paths, etc.
4. **Smart Truncation**: Avoids excessive output for large collections

The string representation is implemented through two key methods:

* ``__str__``: Entry point that generates the full string representation
* ``_str_helper``: Recursive helper that builds the hierarchical structure

This system ensures consistent, readable output across all ROMPY models.

Customizing String Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can customize the string representation in two ways:

1. **Override the ``_format_value`` method** (recommended):

.. code-block:: python

    class MyCustomModel(RompyBaseModel):
        def _format_value(self, obj):
            # Custom formatting for specific types
            if isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M")
            elif isinstance(obj, Path):
                return f"PATH: {obj}"
            # Return None to use default formatting
            return None

This method is called for each value during string representation. Returning ``None`` will use the default formatting.

2. **Override the ``__str__`` method** (for complete customization):

.. code-block:: python

    class CompletelyCustomModel(RompyBaseModel):
        def __str__(self):
            # Completely custom string representation
            return f"Custom representation of {self.__class__.__name__}"

The ``_format_value`` approach is preferred as it maintains the hierarchical structure while allowing customization for specific types.

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

    from rompy.formatting import configure_logging, log_box, ARROW, BULLET
    import logging
    
    # Configure logging
    configure_logging(verbosity=1)
    logger = logging.getLogger(__name__)
    
    # Log a formatted status box (uses sensible defaults)
    log_box("Processing Started", logger=logger)
    
    # Continue with processing using centralized formatting elements
    logger.info(f"{ARROW} Processing item 1...")
    logger.info(f"{BULLET} Result: Success")

CLI Application with Custom Formatting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import click
    from rompy.formatting import log_status, ARROW, BULLET
    import os
    import logging
    
    @click.command()
    @click.option("--ascii-only", is_flag=True, help="Use ASCII-only characters")
    def process(ascii_only):
        # Set environment variable
        os.environ['ROMPY_ASCII_ONLY'] = 'true' if ascii_only else 'false'
        
        # Configure logging
        logger = logging.getLogger()
        
        # Use predefined status box
        log_status("processing", "PROCESSING RESULTS", logger=logger)
        
        # Use centralized formatting elements
        logger.info(f"{BULLET} Processed 100 items")
        logger.info(f"{BULLET} Found 5 anomalies")
        logger.info(f"{ARROW} Operation completed successfully")

Best Practices
-------------

1. **Consistent Usage**: Always use the formatting utilities for output to maintain a consistent appearance
2. **Environment Awareness**: Respect the environment variables rather than hardcoding formatting preferences
3. **Customization**: Override the `_format_value` method for custom types rather than the entire `__str__` method
4. **CLI Options**: Provide formatting options in CLI applications
5. **Smart Defaults**: Use the smart defaults provided by the formatting module instead of passing explicit parameters
6. **Documentation**: Document the formatting options available to users

When working with formatted boxes:

.. code-block:: python

    # DO: Use the log_box or log_status utility
    from rompy.formatting import log_box, log_status
    log_box("OPERATION COMPLETE", logger=logger)
    log_status("completed", logger=logger)
    
    # DON'T: Manually handle box formatting and logging
    from rompy.formatting import get_formatted_box
    box = get_formatted_box("OPERATION COMPLETE")
    for line in box.split("\n"):
        logger.info(line)
    logger.info("")
    
    # DON'T: Override default formatting parameters unless necessary
    # This is redundant as the function will detect ASCII mode automatically
    log_box("OPERATION COMPLETE", logger=logger, use_ascii=True)

When implementing formatting in new components:

.. code-block:: python

    # DO: Use the centralized formatting elements
    from rompy.formatting import ARROW, BULLET
    logger.info(f"{ARROW} Processing step")
    logger.info(f"{BULLET} List item")
    
    # DON'T: Create your own formatting elements
    from rompy.formatting import get_ascii_mode
    arrow = "->" if get_ascii_mode() else "→"
    logger.info(f"{arrow} Processing step")

For more examples and detailed API documentation, see the API Reference section.