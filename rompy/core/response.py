import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Type, Union

from pydantic import Field

from .types import RompyBaseModel
from .config import BaseConfigResponse

logger = logging.getLogger(__name__)


class ModelRunResponse(RompyBaseModel):
    """Response model for ModelRun.

    Includes all config response data along with execution timing information.
    """
    model_type: str = Field(
        description="The model type for the run",
    )
    output_dir: Path = Field(
        description="The output directory for the run",
    )
    config_response: BaseConfigResponse = Field(
        description="The config response containing output paths",
    )
    execution_start_time: datetime = Field(
        description="The time when model execution started",
    )
    execution_end_time: datetime = Field(
        description="The time when model execution completed",
    )
    duration: timedelta = Field(
        description="The duration of the model execution",
    )


def get_response_class(model_type: str) -> Type[BaseConfigResponse]:
    """Get the response class for a specific model type using the entry points system.
    
    This function uses the rompy.response entry points group to dynamically load
    the appropriate response class for a given model type.
    
    Parameters
    ----------
    model_type : str
        The model type identifier (e.g., 'swan', 'schism')
        
    Returns
    -------
    Type[BaseConfigResponse]
        The registered response class for the model type
        
    Raises
    ------
    KeyError
        If no response class is registered for the model type
    """
    from rompy.utils import load_entry_points
    
    # Load all response classes from entry points
    response_classes = load_entry_points("rompy.response")
    
    # Create a dictionary mapping model types to their response classes
    model_responses = {}
    for response_class in response_classes:
        if hasattr(response_class, "_model_type"):
            model_responses[response_class._model_type] = response_class
    
    # Return the response class for the given model type
    if model_type in model_responses:
        return model_responses[model_type]
    
    # Fallback to base response class
    logger.warning(f"No response class found for model type '{model_type}', using BaseConfigResponse")
    return BaseConfigResponse