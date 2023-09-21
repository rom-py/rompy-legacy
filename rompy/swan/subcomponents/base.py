"""Base class for SWAN sub-components."""
from typing import Literal
from abc import ABC
from pydantic import ConfigDict, Field

from rompy.core import RompyBaseModel


class BaseSubComponent(RompyBaseModel, ABC):
    """Base class for SWAN sub-components.

    This class is not intended to be used directly, but to be subclassed by other
    SWAN sub-components to implement the following common behaviour:

    * Define a `render()` method to render a CMD string from the subcomponent
    * Forbid extra arguments so only implemented fields must be specified

    """

    model_type: Literal["subcomponent"] = Field(description="Model type discriminator")
    model_config = ConfigDict(extra="forbid")

    def cmd(self) -> str:
        return self.model_type.upper()

    def render(self) -> str:
        """Render the sub-component to a string."""
        return self.cmd()
