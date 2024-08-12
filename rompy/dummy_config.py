"""A dummy config module using defining and registering some dummy Config classes."""
from typing import Literal
from pydantic import BaseModel

from rompy.registry import register_type
from rompy.dummy_model import create_model_run


@register_type("swan")
class SwanConfig(BaseModel):
    model_type: Literal["swan"]


@register_type("schism")
class SchismConfig(BaseModel):
    model_type: Literal["schism"]


@register_type("xbeach")
class XbeachConfig(BaseModel):
    model_type: Literal["xbeach"]


ModelRun = create_model_run()


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