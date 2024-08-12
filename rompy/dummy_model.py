"""This is where the ModelRun class will be defined."""
from typing import Union
from pydantic import create_model, Field

from rompy.registry import TypeRegistry


def create_model_run():
    """Dynamically generate the ModelRun class."""
    types = TypeRegistry.get_types()
    config_type = Union[tuple(types.values())]
    
    return create_model(
        "ModelRun",
        config=(config_type, Field(..., discriminator='model_type'))
    )
