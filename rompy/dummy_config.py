from typing import List, Literal

from pydantic import BaseModel

from rompy.registry import Registry


class ConfigModel(BaseModel):
    config_type: str


@Registry.register_config("swan")
class SwanConfig(ConfigModel):
    config_type: Literal["swan"] = "swan"


@Registry.register_config("schism")
class SchismConfig(ConfigModel):
    config_type: Literal["schism"] = "schism"


@Registry.register_config("xbeach")
class XbeachConfig(ConfigModel):
    config_type: Literal["xbeach"] = "xbeach"
