"""
ROMPY Command Line Interface

This module provides the command-line interface for ROMPY.
"""

import json
import sys
import os
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from typing import Optional
from datetime import datetime

import click
import yaml

from rompy.model import ModelRun
from rompy.core.logging import get_logger, LoggingConfig, LogLevel, LogFormat

# Initialize the logger
logger = get_logger(__name__)

# Get installed entry points
installed = entry_points(group="rompy.config").names


def configure_logging(
    verbosity: int = 0,
    log_dir: Optional[str] = None,
    simple_logs: bool = False,
    ascii_only: bool = False,
    show_warnings: bool = False,
) -> None:
    """Configure logging based on verbosity level and other options.

    This function configures the logging system using the LoggingConfig class.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG
        log_dir: Directory to save log files
        simple_logs: Use simple log format without timestamps and module names
        ascii_only: Use ASCII-only characters in output
        show_warnings: Whether to show Python warnings
    """
    # Get the singleton instance of LoggingConfig
    logging_config = LoggingConfig()

    # Map verbosity to log level
    log_level = LogLevel.WARNING
    if verbosity >= 1:
        log_level = LogLevel.INFO
    if verbosity >= 2:
        log_level = LogLevel.DEBUG

    # Determine log format
    log_format = LogFormat.SIMPLE if simple_logs else LogFormat.VERBOSE

    # Prepare update parameters
    update_params = {"level": log_level, "format": log_format, "use_ascii": ascii_only}

    # Set log directory and file if provided
    if log_dir:
        from pathlib import Path

        log_file = f"rompy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        update_params["log_dir"] = Path(log_dir)
        update_params["log_file"] = log_file

    # Apply the configuration (update() will call configure_logging() if needed)
    logging_config.update(**update_params)

    # Configure warnings
    if not show_warnings:
        warnings.filterwarnings("ignore")
    else:
        # Show deprecation warnings
        warnings.filterwarnings("default", category=DeprecationWarning)

    # Log configuration
    logger.debug("Logging configured with level: %s", log_level.value)
    if log_dir:
        logger.info("Log directory: %s", log_dir)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "model", type=click.Choice(installed), envvar="ROMPY_MODEL", required=False
)
@click.argument("config", envvar="ROMPY_CONFIG", required=False)
@click.option("zip", "--zip/--no-zip", default=False, envvar="ROMPY_ZIP")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be used multiple times)",
)
@click.option("--log-dir", envvar="ROMPY_LOG_DIR", help="Directory to save log files")
@click.option(
    "--show-warnings/--hide-warnings", default=False, help="Show Python warnings"
)
@click.option(
    "--ascii-only/--unicode",
    default=False,
    help="Use ASCII-only characters in output",
    envvar="ROMPY_ASCII_ONLY",
)
@click.option(
    "--simple-logs/--detailed-logs",
    default=False,
    help="Use simple log format without timestamps and module names",
    envvar="ROMPY_SIMPLE_LOGS",
)
@click.option("--version", is_flag=True, help="Show version information and exit")
def main(
    model,
    config,
    zip,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
    version,
):
    """Run ROMPY model with the specified configuration.

    ROMPY (Regional Ocean Modeling PYthon) is a tool for generating and running
    ocean, wave, and hydrodynamic model configurations.

    Usage: rompy <model> config.yml

    Args:
        model: Model type to use (one of: {models})
        config: YAML or JSON configuration file

    Options:
        --zip/--no-zip          Create a zip archive of the model files
        -v, --verbose           Increase verbosity (can be used multiple times)
        --log-dir PATH          Directory to save log files
        --show-warnings         Show Python warnings
        --ascii-only            Use ASCII-only characters in output
        --simple-logs           Use simple log format without timestamps and module names
        --version               Show version information and exit

    Examples:
        rompy swan config.yml
        rompy schism my_config.json --ascii-only
        rompy swan config.yml --simple-logs -v
    """
    # Format the docstring with available models
    main.__doc__ = main.__doc__.format(models=", ".join(installed))
    # Configure warnings handling
    if not show_warnings:
        # Capture warnings to prevent them from being displayed
        warnings.filterwarnings("ignore")

    # Configure logging with all parameters
    configure_logging(
        verbosity=verbose,
        log_dir=log_dir,
        simple_logs=simple_logs,
        ascii_only=ascii_only,
        show_warnings=show_warnings,
    )

    # Get the logging config for reference
    logging_config = LoggingConfig()

    # Log the settings
    logger.debug(f"ASCII mode set to: {logging_config.use_ascii}")
    logger.debug(
        f"Simple logs mode set to: {logging_config.format == LogFormat.SIMPLE} (no timestamps or module names)"
    )

    # Import here to avoid circular imports
    import rompy

    # If --version flag is specified, show version and exit
    if version:
        logger.info(f"ROMPY Version: {rompy.__version__}")
        return

    # If no model or config is provided, show help and available models
    if not model or not config:
        logger.info(f"ROMPY Version: {rompy.__version__}")
        logger.info(f"Available models: {', '.join(installed)}")
        logger.info("Run 'rompy --help' for usage information")
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    try:
        with open(config, "r") as f:
            content = f.read()
    except (FileNotFoundError, IsADirectoryError, OSError):
        # Not a file, treat as raw string
        content = config

    logger.info("Loading config...")

    args = None
    # Try JSON first
    try:
        args = json.loads(content)
        logger.info("Parsed config as JSON")
    except json.JSONDecodeError:
        pass

    # If JSON failed, try YAML
    if args is None:
        try:
            args = yaml.safe_load(content)
            logger.info("Parsed config as YAML")
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config as JSON or YAML: {e}")
            raise click.UsageError("Config file is not valid JSON or YAML")

    # Log version and execution information
    logger.info(f"ROMPY Version: {rompy.__version__}")
    logger.info(f"Running model: {model}")
    logger.info(f"Configuration: {config}")

    # Create and run the model
    try:
        start_time = datetime.now()
        logger.info("Running model...")
        model = ModelRun(**args)
        model()

        if zip:
            logger.info("Zipping model outputs...")
            zip_file = model.zip()
            logger.info(f"Model archive created: {zip_file}")

        # Log completion time
        elapsed = datetime.now() - start_time
        logger.info(f"Model run completed in {elapsed.total_seconds():.2f} seconds")

        if log_dir:
            logger.info(f"Log directory: {log_dir}")
    except TypeError as e:
        if "unsupported format string" in str(e) and "timedelta" in str(e):
            logger.error(f"Error with time format: {str(e)}")
            logger.error(
                "This is likely due to formatting issues with time duration values"
            )
            if verbose > 0:
                logger.error("", exc_info=True)
        else:
            logger.error(f"Type error in model: {str(e)}", exc_info=verbose > 0)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running model: {str(e)}", exc_info=verbose > 0)
        sys.exit(1)
