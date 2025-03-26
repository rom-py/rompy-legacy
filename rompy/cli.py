# -*- coding: utf-8 -*-

import json
import logging
from importlib.metadata import entry_points

import click
import yaml

from .model import ModelRun

logging.basicConfig(level=logging.INFO)

installed = entry_points(group="rompy.config").names


@click.command()
@click.argument("model", type=click.Choice(installed), envvar="ROMPY_MODEL")
@click.argument("config", envvar="ROMPY_CONFIG")
@click.option("zip", "--zip/--no-zip", default=False, envvar="ROMPY_ZIP")
def main(model, config, zip):
    """Run model
    Usage: rompy <model> config.yml
    Args:
        model(str): model type
        config(str): yaml or json config file
    """
    try:
        # First try to open it as a file
        with open(config, "r") as f:
            content = f.read()
    except (FileNotFoundError, IsADirectoryError, OSError):
        # If not a file, treat it as raw content
        content = config

    try:
        # Try to parse as yaml
        args = yaml.load(content, Loader=yaml.Loader)
        model = ModelRun(**args)
    except TypeError:
        model = ModelRun.model_validate_json(json.loads(content))
    model()
    if zip:
        model.zip()


if __name__ == "__main__":
    main()
