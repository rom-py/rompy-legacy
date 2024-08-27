from typing import List, Literal

from pydantic import BaseModel

from rompy.registry import BaseConfigModel, Registry


@Registry.register_config("swan")
class SwanConfig(BaseConfigModel):
    config_type: Literal["swan"] = "swan"


@Registry.register_config("schism")
class SchismConfig(BaseConfigModel):
    config_type: Literal["schism"] = "schism"


@Registry.register_config("xbeach")
class XbeachConfig(BaseConfigModel):
    config_type: Literal["xbeach"] = "xbeach"
