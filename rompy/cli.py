# -*- coding: utf-8 -*-

import logging

import click
import yaml

from .model import ModelRun

logging.basicConfig(level=logging.INFO)

installed = []
try:
    from rompy.core import BaseConfig

    installed.append("base")
except ImportError:
    pass
try:
    from rompy.swan import SwanConfig

    installed.append("swan")
except ImportError:
    pass


@click.command()
@click.argument("model", type=click.Choice(installed))
@click.argument("config", type=click.File("r"))
@click.option(
    "--kwargs",
    "-k",
    multiple=True,
    help="additional key value pairs in the format key:value",
)
def main(model, config, kwargs):
    """Run model
    Usage: rompy <model> config.yml
    Args:
        model(str): model type
        config(str): yaml config file
    """
    args = yaml.load(config, Loader=yaml.Loader)

    kw = {}
    for item in kwargs:
        split = item.split(":")
        kw.update({split[0]: split[1]})
        current = getattr(instance, split[0])
        setattr(instance, split[0], type(current)(split[1]))
    model = ModelRun(**args, **kw)
    model()


if __name__ == "__main__":
    main()
