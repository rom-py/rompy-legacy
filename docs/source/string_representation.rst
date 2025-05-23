===============================
Hierarchical String Representation
===============================

Overview
--------

ROMPY includes a sophisticated hierarchical string representation system for its models, making complex nested data structures easy to read and inspect. This system is built into the ``RompyBaseModel`` base class, which all ROMPY models inherit from.

The string representation system provides:

1. **Consistent Formatting**: All models display their data in a uniform, hierarchical format
2. **Recursive Exploration**: Nested objects, lists, and dictionaries are properly traversed and formatted
3. **Type-Specific Formatting**: Special handling for common types like dates, paths, and durations
4. **Customizable Output**: Ability to override formatting for specific types while maintaining structure

Core Implementation
------------------

The hierarchical string representation system is implemented in the ``RompyBaseModel`` class in ``rompy/core/types.py``. The system consists of three key methods:

.. code-block:: python

    def __str__(self) -> str:
        """Return a hierarchical string representation of the model."""
        lines = []
        self._str_helper(lines, name=self.__class__.__name__, obj=self, level=0)
        return "\n".join(lines)
    
    def _format_value(self, obj: Any) -> Optional[str]:
        """Format a value for string representation.
        
        Can be overridden by subclasses to customize formatting.
        Return None to use default formatting.
        """
        return None
    
    def _str_helper(self, lines: list, name: str, obj: Any, level: int) -> None:
        """Helper method to build a hierarchical string representation."""
        # Implementation recursively formats all nested objects

The ``__str__`` method is the entry point, which initializes the output and calls ``_str_helper`` to recursively build the string representation. The ``_format_value`` method provides a hook for customizing how specific types are formatted.

Output Format
------------

The hierarchical string representation produces output like:

.. code-block:: text

    ModelName:
      attribute1: value1
      nested_object:
        nested_attr1: value2
        nested_attr2: value3
      list_attribute:
        [0]: first_item
        [1]: second_item
      dictionary_attribute:
        key1: value4
        key2: value5

This format makes it easy to understand the structure of complex model objects, with consistent indentation and naming.

Customizing String Representation
--------------------------------

There are two main approaches to customizing the string representation:

1. Override the ``_format_value`` Method (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override the ``_format_value`` method to customize how specific types are formatted while maintaining the hierarchical structure:

.. code-block:: python

    class MyCustomModel(RompyBaseModel):
        def _format_value(self, obj: Any) -> Optional[str]:
            # Format datetime objects in a custom way
            if isinstance(obj, datetime):
                return obj.strftime("%Y-%m-%d %H:%M")
                
            # Format Path objects with a prefix
            if isinstance(obj, Path):
                return f"Path: {obj}"
                
            # Special formatting for your custom types
            if isinstance(obj, MySpecialType):
                return obj.get_custom_representation()
                
            # Return None for all other types to use default formatting
            return None

This approach allows you to intercept and format specific types without affecting the overall structure of the output. It's particularly useful for customizing how dates, paths, timedeltas, and other complex types are displayed.

2. Override the ``__str__`` Method (Complete Customization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complete control over the string representation, override the ``__str__`` method:

.. code-block:: python

    class CompletelyCustomModel(RompyBaseModel):
        def __str__(self) -> str:
            # Completely custom string representation
            return f"Custom representation of {self.__class__.__name__}"

However, this approach is discouraged for most cases as it breaks the consistent hierarchical formatting that makes ROMPY models easy to read and understand.

Examples from ROMPY
------------------

Several ROMPY classes implement custom string formatting. Here are a few examples from the codebase:

**BaseConfig** in ``rompy/core/config.py``:

.. code-block:: python

    def _format_value(self, obj):
        """Custom formatter for BaseConfig values."""
        # Format BaseConfig objects with header and structure
        if hasattr(obj, 'model_type') and isinstance(obj, BaseConfig):
            is_ascii = get_ascii_mode()
            if is_ascii:
                header = "+------------ MODEL CONFIGURATION ------------+"
                separator = "+-------------------------------------------+"
                footer = "+-------------------------------------------+"
            else:
                header = "┏━━━━━━━━━━ MODEL CONFIGURATION ━━━━━━━━━━━━┓"
                separator = "┠━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┨"
                footer = "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
            
            bullet = "*" if is_ascii else "•"
        
            lines = [header]
            lines.append(f"  {bullet} Model type: {obj.model_type}")
        
            # Add template info if available
            if hasattr(obj, 'template') and obj.template:
                template_path = obj.template
                if len(template_path) > 50:  # Truncate long paths
                    template_path = "..." + template_path[-47:]
                lines.append(f"  {bullet} Template: {template_path}")
        
            lines.append(footer)
            return "\n".join(lines)

**ModelRun** in ``rompy/model.py``:

.. code-block:: python

    def _format_value(self, obj):
        """Custom formatter for ModelRun values"""
        # Format TimeRange object with more detail
        if isinstance(obj, TimeRange):
            duration = obj.end - obj.start
            formatted_duration = obj.format_duration(duration)
            return (
                f"{obj.start.isoformat(' ')} to {obj.end.isoformat(' ')}\n"
                f"  Duration: {formatted_duration}\n"
                f"  Interval: {str(obj.interval)}\n"
                f"  Include End: {obj.include_end}"
            )

        # Format DateTime objects in readable format
        if isinstance(obj, datetime):
            return obj.isoformat(" ")
        
        # Format Path objects as strings
        if isinstance(obj, Path):
            return str(obj)

**SwanDataGrid** in ``rompy/swan/data.py``:

.. code-block:: python

    def _format_value(self, obj):
        """Format specific types for string representation.
        
        This formats custom types used in SwanDataGrid for better readability.
        """
        # Use formatting utilities imported at the top of the file

        # Get header, footer, and bullet character
        header, footer, bullet = get_formatted_header_footer(
            title="SWAN DATA GRID"
        )

        # Build content lines
        if isinstance(obj, SwanDataGrid):
            lines = [header]
            lines.append(f"  {bullet} Variable: {obj.var.value}")
            
            if obj.z1:
                lines.append(f"  {bullet} Primary variable: {obj.z1}")
                if obj.z2:
                    lines.append(f"  {bullet} Secondary variable: {obj.z2}")
            
            lines.append(f"  {bullet} Output: {obj.var.value}.grd")
            lines.append(footer)
            return "\n".join(lines)
        
        # Format other types using default mechanism
        return None

**TemplateRenderer** in ``rompy/core/render.py``:

.. code-block:: python

    def _format_value(self, obj) -> Optional[str]:
        """Format specific types of values for display.

        This method formats template rendering information with rich details.

        Args:
            obj: The object to format

        Returns:
            A formatted string or None to use default formatting
        """
        # Format TemplateRenderer object
        if isinstance(obj, TemplateRenderer):
            header, footer, bullet = get_formatted_header_footer(
                title="TEMPLATE RENDERING DETAILS"
            )

            lines = [header]
            lines.append(f"  {bullet} Template source: {obj.template}")
            lines.append(f"  {bullet} Output directory: {obj.output_dir}")

            if obj.checkout:
                lines.append(f"  {bullet} Template version: {obj.checkout}")

            # Add context summary if available
            context_items = len(obj.context)
            if context_items > 0:
                lines.append(f"  {bullet} Context items: {context_items}")

            lines.append(footer)
            return "\n".join(lines)

        # Use default formatting for other types
        return None

Integration with Logging and Formatting
-------------------------------------

The hierarchical string representation integrates with ROMPY's logging and formatting system:

.. code-block:: python

    import logging
    from rompy.core.types import RompyBaseModel
    
    logger = logging.getLogger(__name__)
    
    # Create a model
    model = MyComplexModel()
    
    # Log the model - will use the hierarchical string representation
    logger.info("Model state:")
    for line in str(model).split("\n"):
        logger.info(line)

This produces clean, well-formatted log output that clearly shows the structure of your model.

In the ROMPY codebase, this is used extensively in several places:

**ModelRun** in ``rompy/model.py``:

.. code-block:: python

    # First try to use the _format_value method directly if available
    if hasattr(self.config, "_format_value"):
        try:
            # Try using _format_value method directly for structured formatting
            formatted_config = self.config._format_value(self.config)
            if formatted_config:
                for line in formatted_config.split("\n"):
                    logger.info(line)
                # If we successfully used _format_value, we're done
                formatted = True
            else:
                # _format_value returned None, fall back to str()
                formatted = False
        except Exception as e:
            logger.debug(f"Error in _format_value: {str(e)}")
            formatted = False

**CLI** in ``rompy/cli.py``:

.. code-block:: python

    # Display model configuration using string representation
    try:
        config_str = str(args)
        for line in config_str.split('\n'):
            if verbose >= 2:  # Only show in debug mode
                logger.debug(line)
    except Exception as e:
        logger.debug(f"Error formatting configuration: {str(e)}")

**SwanConfig** in ``rompy/swan/config.py``:

.. code-block:: python

    # Log the configuration using string representation
    if verbose:
        logger.info("Configuration details:")
        config_lines = str(self).split("\n")
        for line in config_lines:
            logger.info(f"  {line}")

Best Practices
-------------

1. **Use Hierarchical Formatting**: Let the base class handle the structure when possible
2. **Override _format_value**: Customize specific type formatting without breaking the overall structure
3. **Keep It Simple**: Format for readability, not completeness
4. **Consider Log Output**: Remember that your string representation may be used in logs
5. **Format Sensitive Data Safely**: Redact or summarize sensitive fields
6. **Handle Large Collections**: Summarize or truncate large lists and dictionaries
7. **Add Contextual Information**: Include relevant metadata that helps understand the object
8. **Support ASCII Mode**: Consider using the `get_ascii_mode()` function for compatibility with both ASCII and Unicode output
9. **Fallback Gracefully**: Always have a fallback option if custom formatting fails
10. **Use Formatted Box Utilities**: Leverage the formatting module's box utilities for consistent appearance
11. **Skip Private Fields**: Fields starting with underscore (`_`) are typically skipped in string representation

For examples of effective string representation customization, see the implementation in various ROMPY models like ``BaseConfig``, ``ModelRun``, ``SwanDataGrid``, ``TemplateRenderer``, and other classes throughout the codebase.

Related Documentation
-------------------

- For information about ROMPY's general formatting system, see :doc:`formatting`
- For details on the command line interface, see :doc:`cli`
- For core concepts and model structure, see :doc:`core_concepts`

Implementation Details
--------------------

The string representation mechanism is implemented in ``RompyBaseModel`` in ``rompy/core/types.py``. The key methods are:

1. ``__str__``: The entry point that generates the full representation
2. ``_format_value``: Hook for customizing type formatting (meant to be overridden)
3. ``_str_helper``: Recursive helper that builds the hierarchical structure

The implementation carefully handles:

- Nested ``RompyBaseModel`` instances
- Dictionaries and collections
- Objects with custom ``__str__`` methods
- Multi-line string representations
- Proper indentation based on nesting level
- Skipping private fields (those starting with ``_``)

This approach provides a consistent, readable output format across the entire codebase while allowing for customization where needed.