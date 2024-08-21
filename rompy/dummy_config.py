from typing import Literal

from pydantic import BaseModel

from rompy.dummy_model import ModelRun
from rompy.registry import TypeRegistry


@TypeRegistry.register("swan")
class SwanConfig(BaseModel):
    model_type: Literal["swan"] = "swan"


@TypeRegistry.register("schism")
class SchismConfig(BaseModel):
    model_type: Literal["schism"] = "schism"


@TypeRegistry.register("xbeach")
class XbeachConfig(BaseModel):
    model_type: Literal["xbeach"] = "xbeach"


if __name__ == "__main__":
    kwargs = {"config": {"model_type": "swan"}}
    print(ModelRun(**kwargs))

    kwargs = {"config": {"model_type": "schism"}}
    print(ModelRun(**kwargs))

    kwargs = {"config": {"model_type": "xbeach"}}
    print(ModelRun(**kwargs))

    from pydantic import ValidationError

    try:
        kwargs = {"config": {"model_type": "foo"}}
        print(ModelRun(**kwargs))
    except ValidationError as e:
        print(e)

    kwargs = {"config": {"model_type": "xbeach"}}
    mr = ModelRun(**kwargs)
    print(mr.dummy_property)