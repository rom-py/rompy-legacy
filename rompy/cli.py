import json
import logging
from importlib.metadata import entry_points

import click
import yaml

from rompy.model import ModelRun

installed = entry_points(group="rompy.config").names
logger = logging.getLogger(__name__)


@click.command()
# @click.argument("model", type=click.Choice(installed), envvar="ROMPY_MODEL")
@click.argument("config", envvar="ROMPY_CONFIG")
@click.option("zip", "--zip/--no-zip", default=False, envvar="ROMPY_ZIP")
def main(config, zip):
    """Run model
    Usage: rompy <model> config.yml
    Args:
        model(str): model type
        config(str): yaml or json config file
    """
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

    model = ModelRun(**args)
    logger.info("Running model...")
    model()
    if zip:
        model.zip()
