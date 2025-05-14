import json
import logging
import sys
import os
import warnings
from importlib.metadata import entry_points
from datetime import datetime

import click
import yaml

from rompy.model import ModelRun

installed = entry_points(group="rompy.config").names
logger = logging.getLogger(__name__)

def configure_logging(verbosity=0):
    """Configure logging based on verbosity level.

    Args:
        verbosity (int): 0=WARNING, 1=INFO, 2=DEBUG
    """
    log_level = logging.WARNING
    if verbosity >= 1:
        log_level = logging.INFO
    if verbosity >= 2:
        log_level = logging.DEBUG

    # Create a custom formatter with timestamp and level
    # Format the logger name to a fixed width to align messages
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)-20s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    # Configure file handler if output directory exists
    log_file = None
    output_dir = os.environ.get('ROMPY_LOG_DIR', None)
    if output_dir and os.path.isdir(output_dir):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(output_dir, f'rompy_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

    # Configure root logger
    logging.root.setLevel(log_level)
    logging.root.handlers = []  # Remove existing handlers
    logging.root.addHandler(console)

    # Suppress specific warnings
    if log_level != logging.DEBUG:
        logging.getLogger('pydantic').setLevel(logging.ERROR)
        logging.getLogger('intake').setLevel(logging.WARNING)

        # Filter out common warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
        warnings.filterwarnings("ignore", category=UserWarning, module="intake.readers.readers")
        warnings.filterwarnings("ignore", message="A custom validator is returning a value other than `self`")

    return log_file


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument("model", type=click.Choice(installed), envvar="ROMPY_MODEL", required=False)
@click.argument("config", envvar="ROMPY_CONFIG", required=False)
@click.option("zip", "--zip/--no-zip", default=False, envvar="ROMPY_ZIP")
@click.option("-v", "--verbose", count=True, help="Increase verbosity (can be used multiple times)")
@click.option("--log-dir", envvar="ROMPY_LOG_DIR", help="Directory to save log files")
@click.option("--show-warnings/--hide-warnings", default=False, help="Show Python warnings")
@click.option("--ascii-only/--unicode", default=False, help="Use ASCII-only characters in output", envvar="ROMPY_ASCII_ONLY")
@click.option("--version", is_flag=True, help="Show version information and exit")
def main(model, config, zip, verbose, log_dir, show_warnings, ascii_only, version):
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
        --version               Show version information and exit

    Examples:
        rompy swan config.yml
        rompy schism my_config.json --ascii-only
    """
    # Format the docstring with available models
    main.__doc__ = main.__doc__.format(models=", ".join(installed))
    # Configure warnings handling
    if not show_warnings:
        # Capture warnings to prevent them from being displayed
        warnings.filterwarnings("ignore")

    # Set ASCII-only environment variable globally
    ascii_value = 'true' if ascii_only else 'false'
    os.environ['ROMPY_ASCII_ONLY'] = ascii_value

    # Force reloading of the ASCII_MODE value
    import rompy
    reload_value = rompy.ROMPY_ASCII_MODE()

    # Log the ascii mode setting
    logger.debug(f"ASCII mode set to: {ascii_only}")

    # Configure logging
    if log_dir:
        os.environ['ROMPY_LOG_DIR'] = log_dir
    log_file = configure_logging(verbose)

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
    logger.info(f"ASCII Mode: {'Enabled' if ascii_only else 'Disabled'}")
    # Ensure we're actually using the correct mode by checking the helper function
    from rompy import ROMPY_ASCII_MODE
    actual_mode = ROMPY_ASCII_MODE()
    if actual_mode != ascii_only:
        logger.warning(f"ASCII mode setting inconsistency detected - requested: {ascii_only}, actual: {actual_mode}")
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

        if log_file:
            logger.info(f"Log file saved to: {log_file}")
    except TypeError as e:
        if "unsupported format string" in str(e) and "timedelta" in str(e):
            logger.error(f"Error with time format: {str(e)}")
            logger.error("This is likely due to formatting issues with time duration values")
            if verbose > 0:
                logger.error("", exc_info=True)
        else:
            logger.error(f"Type error in model: {str(e)}", exc_info=verbose > 0)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running model: {str(e)}", exc_info=verbose > 0)
        sys.exit(1)
