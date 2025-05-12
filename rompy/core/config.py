import logging
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import ConfigDict, Field

from .types import RompyBaseModel
from rompy import ROMPY_ASCII_MODE

logger = logging.getLogger(__name__)



DEFAULT_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "base")


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
        
    def __str__(self):
        """Return a formatted string representation of the config.
        
        This method provides a human-readable representation of the configuration 
        that can be used in logs and other output.
        """
        # Get all fields excluding internal ones
        fields = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and k != 'model_config'
        }
        
        # Format with tabular headers
        lines = []
        
        # Use helper function to avoid circular imports
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()
        
        if USE_ASCII_ONLY:
            # ASCII version
            lines.append("+------------------------------------------------------------------------+")
            lines.append("|                       BASE CONFIGURATION                               |")
            lines.append("+------------------------------------------------------------------------+")
            lines.append("+-----------------------------+-------------------------------------+")
            lines.append(f"| MODEL TYPE                  | {self.model_type:<35} |")
            lines.append("+-----------------------------+-------------------------------------+")
            
            # Add configuration parameters
            if len(fields) > 1:  # If we have more fields than just model_type
                lines.append("+------------------------------------------------------------------------+")
                lines.append("| CONFIGURATION PARAMETERS                                                |")
                lines.append("+------------------------------------------------------------------------+")
                
                for key, value in fields.items():
                    if key == 'model_type':
                        continue
                    
                    if hasattr(value, '__str__') and not isinstance(value, (str, int, float, bool)):
                        # For complex objects that have their own __str__ method
                        lines.append(f"   * {key}")
                        str_val = str(value).replace('\n', '\n     ')
                        lines.append(f"     {str_val}")
                    else:
                        # For simple values
                        lines.append(f"   * {key:<20}: {value}")
        else:
            # Unicode version
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append("┃                       BASE CONFIGURATION                           ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
            lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            lines.append(f"┃ MODEL TYPE                  ┃ {self.model_type:<35} ┃")
            lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
            
            # Add configuration parameters
            if len(fields) > 1:  # If we have more fields than just model_type
                lines.append("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
                lines.append("┃ CONFIGURATION PARAMETERS                                          ┃")
                lines.append("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
                
                for key, value in fields.items():
                    if key == 'model_type':
                        continue
                    
                    if hasattr(value, '__str__') and not isinstance(value, (str, int, float, bool)):
                        # For complex objects that have their own __str__ method
                        lines.append(f"   • {key}")
                        str_val = str(value).replace('\n', '\n     ')
                        lines.append(f"     {str_val}")
                    else:
                        # For simple values
                        lines.append(f"   • {key:<20}: {value}")
                
        return "\n".join(lines)
