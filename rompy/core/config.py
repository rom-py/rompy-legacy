import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import ConfigDict, Field

from .types import RompyBaseModel

logger = logging.getLogger(__name__)


DEFAULT_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "base")

class BaseOutputsResponse(RompyBaseModel):
    """Response model for model outputs.
    
    Includes paths to wave spectra and wave parameters output files.
    """
    wave_spectra: Path = Field(
        description="Paths to wave spectra output files",
        default_factory=dict,
    )
    wave_parameters: Path = Field(
        description="Paths to wave parameters output files",
        default_factory=dict,
    )
    wave_hotfiles: Path = Field(
        description="Paths to wave hotfiles output files",
        default_factory=dict,
    )


class BaseResponse(RompyBaseModel):
    """Base response model for config outputs.
    
    Contains common fields for all model responses: outputs and hotfiles paths.
    Specific model implementations will extend this class with model-specific outputs.
    """
    outputs: BaseOutputsResponse = Field(
        description="The outputs of the model run",
        default_factory=dict,
    )




class BaseConfig(RompyBaseModel):
    """Base class for model templates.

    The template class provides the object that is used to set up the model configuration.
    When implemented for a given model, can move along a scale of complexity
    to suit the application.

    In its most basic form, as implemented in this base object, it consists of path to a cookiecutter template
    with the class providing the context for the {{config}} values in that template. Note that any
    {{runtime}} values are filled from the ModelRun object.

    If the template is a git repo, the checkout parameter can be used to specify a branch or tag and it
    will be cloned and used.

    If the object is callable, it will be colled prior to rendering the template. This mechanism can be
    used to perform tasks such as fetching exteral data, or providing additional context to the template
    beyond the arguments provided by the user..
    """

    model_type: Literal["base"] = "base"
    template: Optional[str] = Field(
        description="The path to the model template",
        default=DEFAULT_TEMPLATE,
    )
    checkout: Optional[str] = Field(
        description="The git branch to use if the template is a git repo",
        default="main",
    )
    model_config = ConfigDict(extra="allow")

    def __call__(self, *args, **kwargs):
        return self






class SchismResponse(BaseResponse):
    """Response model for SCHISM model outputs.
    
    Includes paths to spectra, parameters, and oceanum_parameters output files.
    """
    spectra: dict[str, str] = Field(
        description="Paths to spectra output files",
        default_factory=dict,
    )
    parameters: dict[str, str] = Field(
        description="Paths to parameters output files",
        default_factory=dict,
    )
    oceanum_parameters: dict[str, str] = Field(
        description="Paths to oceanum parameters output files",
        default_factory=dict,
    )

