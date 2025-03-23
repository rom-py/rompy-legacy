# -*- coding: utf-8 -*-

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
@click.option(
    "--kwargs",
    "-k",
    multiple=True,
    help="additional key value pairs in the format key:value",
)
def main(model, config, zip, kwargs):
    """Run model
    Usage: rompy <model> config.yml
    Args:
        model(str): model type
        config(str): yaml config file
    """
    try:
        # First try to open it as a file
        with open(config, "r") as f:
            args = yaml.load(f, Loader=yaml.Loader)
    except (FileNotFoundError, IsADirectoryError, OSError):
        # If not a file, treat it as raw YAML content
        args = yaml.load(config, Loader=yaml.Loader)

    kw = {}
    for item in kwargs:
        split = item.split(":")
        kw.update({split[0]: split[1]})
        current = getattr(instance, split[0])
        setattr(instance, split[0], type(current)(split[1]))
    model = ModelRun(**args, **kw)
    model()
    if zip:
        model.zip()


if __name__ == "__main__":
    main()
