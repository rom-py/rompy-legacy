============================
Formatting and Logging
============================

Overview
--------

ROMPY provides a comprehensive framework for consistent formatting and logging across the codebase. This framework ensures that:

1. Log messages are consistent and configurable
2. String representations of objects are clear and hierarchical
3. Output formatting is visually appealing and consistent
4. Configuration is flexible and environment-aware

Core Components
--------------

The framework consists of several key components:

1. **Centralized Logging System**
   - Consistent log formatting and handling
   - Environment variable configuration
   - Multiple log levels and output formats

2. **Hierarchical String Representation**
   - Clean, readable output of complex objects
   - Recursive handling of nested structures
   - Type-specific formatting

3. **Formatted Output**
   - Boxes and visual elements
   - Consistent headers and footers
   - Progress indicators

Logging System
-------------

ROMPY's logging system is built on Python's standard `logging` module but provides additional features and consistency.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from rompy.core.logging import get_logger
    
    # Get a logger for your module
    logger = get_logger(__name__)
    
    # Log messages at different levels
    logger.debug("Detailed debug information")
    logger.info("Informational message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical error")

Configuration
~~~~~~~~~~~~~

Logging can be configured via environment variables:

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Variable
     - Default
     - Description
   * - ``ROMPY_LOG_LEVEL``
     - ``INFO``
     - Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   * - ``ROMPY_LOG_FORMAT``
     - ``detailed``
     - Log format style (``simple`` or ``detailed``)
   * - ``ROMPY_LOG_FILE``
     - None
     - Optional file path for log output

Programmatic configuration is also available:

.. code-block:: python

    from rompy.core.logging import configure_logging
    
    configure_logging(
        level="DEBUG",
        format="detailed",
        log_file="rompy.log"
    )

Hierarchical String Representation
--------------------------------

All ROMPY models include a hierarchical string representation for better readability of complex objects.

Basic Usage
~~~~~~~~~~

.. code-block:: python

    class MyModel(RompyBaseModel):
        name: str
        value: float
        nested: dict
    
    obj = MyModel(name="test", value=42.0, nested={"a": 1, "b": 2})
    print(obj)

Output:

.. code-block:: text

    MyModel:
      name: test
      value: 42.0
      nested:
        a: 1
        b: 2

Custom Formatting
~~~~~~~~~~~~~~~~

Customize formatting by overriding the `_format_value` method:

.. code-block:: python

    class CustomModel(RompyBaseModel):
        timestamp: datetime
        
        def _format_value(self, obj: Any) -> Optional[str]:
            if isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M")
            return None

Formatted Output
---------------

ROMPY provides utilities for creating consistent, visually appealing output.

Boxes and Sections
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rompy.core.formatting import box, section
    
    # Create a simple box
    print(box("Important Message"))
    
    # Create a section with content
    print(section("Processing Results", ["Item 1", "Item 2", "Item 3"]))

Progress Indicators
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rompy.core.formatting import ProgressBar
    import time
    
    with ProgressBar("Processing", total=100) as pbar:
        for i in range(100):
            time.sleep(0.1)
            pbar.update(1)

Best Practices
-------------

1. **Logging**
   - Use appropriate log levels (DEBUG for detailed info, INFO for normal operations, etc.)
   - Include relevant context in log messages
   - Use structured logging for machine-readable output

2. **String Representation**
   - Keep string representations concise but informative
   - Include all relevant attributes
   - Handle nested objects appropriately

3. **Formatting**
   - Be consistent with formatting across the codebase
   - Use the provided utilities for common formatting needs
   - Consider readability in different output contexts (CLI, logs, etc.)

Example Integration
------------------

Here's how these components work together in a typical ROMPY module:

.. code-block:: python

    from rompy.core.logging import get_logger
    from rompy.core.formatting import section
    from rompy.core.types import RompyBaseModel
    
    logger = get_logger(__name__)
    
    class DataProcessor(RompyBaseModel):
        """Process data with logging and formatted output."""
        
        def process(self, data):
            logger.info("Starting data processing")
            
            with section("Processing Data"):
                # Process data here
                logger.debug(f"Processing {len(data)} items")
                
                # Log progress
                for i, item in enumerate(data, 1):
                    self._process_item(item)
                    logger.debug(f"Processed item {i}/{len(data)}")
            
            logger.info("Processing complete")
        
        def _process_item(self, item):
            # Process individual items
            pass
