import glob
import logging
import os
import platform
import shutil
import zipfile as zf
import time as time_module
from datetime import datetime
from pathlib import Path
from typing import Union
import textwrap
import sys

import ipdb
from pydantic import Field

from rompy.utils import load_entry_points
from rompy.formatting import (
    get_formatted_box,
    get_formatted_header_footer
)

# Import these at module level to avoid scoping issues
from rompy.formatting import USE_ASCII_ONLY, USE_SIMPLE_LOGS

from .core.config import BaseConfig
from .core.render import render
from .core.time import TimeRange
from .core.types import RompyBaseModel

logger = logging.getLogger(__name__)


# Accepted config types are defined in the entry points of the rompy.config group
CONFIG_TYPES = load_entry_points("rompy.config")

def _load_backends():
    """Load backends from entry points with fallback handling."""
    run_backends = {}
    postprocessors = {}
    pipeline_backends = {}
    
    # Load run backends
    try:
        for backend in load_entry_points("rompy.run"):
            name = backend.__name__.lower().replace('runbackend', '')
            run_backends[name] = backend
    except Exception as e:
        logger.warning(f"Failed to load run backends: {e}")
    
    # Load postprocessors
    try:
        for proc in load_entry_points("rompy.postprocess"):
            name = proc.__name__.lower().replace('postprocessor', '')
            postprocessors[name] = proc
    except Exception as e:
        logger.warning(f"Failed to load postprocessors: {e}")
    
    # Load pipeline backends
    try:
        for backend in load_entry_points("rompy.pipeline"):
            name = backend.__name__.lower().replace('pipelinebackend', '')
            pipeline_backends[name] = backend
    except Exception as e:
        logger.warning(f"Failed to load pipeline backends: {e}")
    
    return run_backends, postprocessors, pipeline_backends

# Load backends from entry points
RUN_BACKENDS, POSTPROCESSORS, PIPELINE_BACKENDS = _load_backends()


class ModelRun(RompyBaseModel):
    """A model run.

    It is intented to be model agnostic.
    It deals primarily with how the model is to be run, i.e. the period of the run
    and where the output is going. The actual configuration of the run is
    provided by the config object.

    Further explanation is given in the rompy.core.Baseconfig docstring.
    """
    # Make formatting variables accessible to all class methods
    _use_ascii_only = USE_ASCII_ONLY
    _use_simple_logs = USE_SIMPLE_LOGS

    run_id: str = Field("run_id", description="The run id")
    period: TimeRange = Field(
        TimeRange(
            start=datetime(2020, 2, 21, 4),
            end=datetime(2020, 2, 24, 4),
            interval="15M",
        ),
        description="The time period to run the model",
    )
    output_dir: Path = Field("./simulations", description="The output directory")
    config: Union[CONFIG_TYPES] = Field(
        default_factory=BaseConfig,
        description="The configuration object",
        discriminator="model_type",
    )
    delete_existing: bool = Field(False, description="Delete existing output directory")
    model_type: Literal["modelrun"] = Field(
        "modelrun", description="The model type for SCHISM."
    )
    _datefmt: str = "%Y%m%d.%H%M%S"
    _staging_dir: Path = None

    @property
    def staging_dir(self):
        """The directory where the model is staged for execution

        returns
        -------
        staging_dir : str
        """

        if self._staging_dir is None:
            self._staging_dir = self._create_staging_dir()
        return self._staging_dir

    def _create_staging_dir(self):
        odir = Path(self.output_dir) / self.run_id
        if self.delete_existing and odir.exists():
            shutil.rmtree(odir)
        odir.mkdir(parents=True, exist_ok=True)
        return odir

    @property
    def _generation_medatadata(self):
        return dict(
            _generated_at=str(datetime.utcnow()),
            _generated_by=os.environ.get("USER"),
            _generated_on=platform.node(),
        )

    def generate(self) -> str:
        """Generate the model input files

        returns
        -------
        staging_dir : str

        """
        logger.info("")
        # Use the formatted box utility with class variables
        box = get_formatted_box(
            title="MODEL RUN CONFIGURATION",
            use_ascii=self._use_ascii_only,
            width=72 if self._use_ascii_only else 70
        )
        for line in box.split("\n"):
            logger.info(line)
        logger.info("")

        # Format model settings in a more structured way
        config_type = type(self.config).__name__
        duration = self.period.end - self.period.start
        formatted_duration = self.period.format_duration(duration)

        # Use a formatted two-column layout with class variables
        if self._use_ascii_only:
            logger.info("+-----------------------------+-------------------------------------+")
            logger.info(f"| Run ID                      | {self.run_id:<35} |")
            logger.info(f"| Model Type                  | {config_type:<35} |")
            logger.info(f"| Start Time                  | {self.period.start.isoformat():<35} |")
            logger.info(f"| End Time                    | {self.period.end.isoformat():<35} |")
            logger.info(f"| Duration                    | {formatted_duration:<35} |")
            logger.info(f"| Time Interval               | {str(self.period.interval):<35} |")
            logger.info(f"| Output Directory            | {str(self.output_dir):<35} |")
        else:
            logger.info( "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            logger.info(f"┃ Run ID                      ┃ {self.run_id:<35} ┃")
            logger.info(f"┃ Model Type                  ┃ {config_type:<35} ┃")
            logger.info(f"┃ Start Time                  ┃ {self.period.start.isoformat():<35} ┃")
            logger.info(f"┃ End Time                    ┃ {self.period.end.isoformat():<35} ┃")
            logger.info(f"┃ Duration                    ┃ {formatted_duration:<35} ┃")
            logger.info(f"┃ Time Interval               ┃ {str(self.period.interval):<35} ┃")
            logger.info(f"┃ Output Directory            ┃ {str(self.output_dir):<35} ┃")

        if hasattr(self.config, 'description') and self.config.description:
            if self._use_ascii_only:
                logger.info(f"| Description                | {self.config.description:<35} |")
            else:
                logger.info(f"┃ Description                ┃ {self.config.description:<35} ┃")

        if self._use_ascii_only:
            logger.info("+-----------------------------+-------------------------------------+")
        else:
            logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        # Display detailed configuration info using the hierarchical __str__ method
        logger.info("")
        logger.info(f"MODEL CONFIGURATION ({type(self.config).__name__}):")
        
        # First try to use the _format_value method directly if available
        if hasattr(self.config, '_format_value'):
            try:
                # Try using _format_value method directly for structured formatting
                formatted_config = self.config._format_value(self.config)
                if formatted_config:
                    for line in formatted_config.split('\n'):
                        logger.info(line)
                    # If we successfully used _format_value, we're done
                    formatted = True
                else:
                    # _format_value returned None, fall back to str()
                    formatted = False
            except Exception as e:
                logger.debug(f"Error in _format_value: {str(e)}")
                formatted = False
        else:
            formatted = False
            
        # If _format_value didn't work or isn't available, fall back to str()
        if not formatted:
            try:
                # Use hierarchical string representation from RompyBaseModel
                config_lines = str(self.config).split('\n')
                for line in config_lines:
                    logger.info(line)
            except Exception as e:
                # If anything goes wrong with config formatting, log the error and minimal info
                logger.info(f"Using {type(self.config).__name__} configuration")
                logger.debug(f"Configuration string formatting error: {str(e)}")

        logger.info("")
        # Use helper functions to avoid circular imports
        from rompy import ROMPY_ASCII_MODE, ROMPY_SIMPLE_LOGS
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()
        
        # Use the formatted box utility
        box = get_formatted_box(
            title="STARTING MODEL GENERATION",
            use_ascii=USE_ASCII_ONLY,
            width=72 if USE_ASCII_ONLY else 70
        )
        for line in box.split("\n"):
            logger.info(line)
        logger.info(f"Preparing input files in {self.output_dir}")

        # Collect context data
        cc_full = {}
        cc_full["runtime"] = self.model_dump()
        cc_full["runtime"].update(self._generation_medatadata)
        cc_full["runtime"].update({"_datefmt": self._datefmt})

        # Process configuration
        logger.info("Processing model configuration...")
        if callable(self.config):
            # Run the __call__() method of the config object if it is callable passing
            # the runtime instance, and fill in the context with what is returned
            logger.info("Running configuration callable...")
            cc_full["config"] = self.config(self)
        else:
            # Otherwise just fill in the context with the config instance itself
            logger.info("Using static configuration...")
            cc_full["config"] = self.config

        # Render templates
        logger.info(f"Rendering model templates to {self.output_dir}/{self.run_id}...")
        staging_dir = render(
            cc_full, self.config.template, self.output_dir, self.config.checkout
        )

        logger.info("")
        # Use helper functions to avoid circular imports
        from rompy import ROMPY_ASCII_MODE, ROMPY_SIMPLE_LOGS
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()
        
        # Use the formatted box utility
        box = get_formatted_box(
            title="MODEL GENERATION COMPLETE",
            use_ascii=USE_ASCII_ONLY,
            width=72 if USE_ASCII_ONLY else 70
        )
        for line in box.split("\n"):
            logger.info(line)
        logger.info(f"Model files generated at: {staging_dir}")
        return staging_dir

    def zip(self) -> str:
        """Zip the input files for the model run

        This function zips the input files for the model run and returns the
        name of the zip file. It also cleans up the staging directory leaving
        only the settings.json file that can be used to reproduce the run.

        returns
        -------
        zip_fn : str
        """
        logger.info("")
        # Use the formatted box utility with class variables
        box = get_formatted_box(
            title="ARCHIVING MODEL FILES",
            use_ascii=self._use_ascii_only,
            width=72 if self._use_ascii_only else 70
        )
        for line in box.split("\n"):
            logger.info(line)

        # Always remove previous zips
        zip_fn = Path(str(self.staging_dir) + ".zip")
        if zip_fn.exists():
            logger.info(f"Removing existing archive at {zip_fn}")
            zip_fn.unlink()

        # Count files to be archived
        file_count = sum([len(fn) for _, _, fn in os.walk(self.staging_dir)])
        logger.info(f"Archiving {file_count} files from {self.staging_dir}")

        # Create zip archive
        with zf.ZipFile(zip_fn, mode="w", compression=zf.ZIP_DEFLATED) as z:
            for dp, dn, fn in os.walk(self.staging_dir):
                for filename in fn:
                    source_path = os.path.join(dp, filename)
                    rel_path = os.path.relpath(source_path, self.staging_dir)
                    z.write(source_path, rel_path)

        # Clean up staging directory
        logger.info(f"Cleaning up staging directory {self.staging_dir}")
        shutil.rmtree(self.staging_dir)

        logger.info(f"Archive created successfully: {zip_fn}")
        # Draw a horizontal line based on ASCII mode using class variables
        if self._use_ascii_only:
            logger.info("------------------------------------------------------------------------")
        else:
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return zip_fn

    def __call__(self):
        return self.generate()

    def _format_value(self, obj):
        """Custom formatter for ModelRun values"""
        from .core.time import TimeRange
        from datetime import datetime, timedelta
        from pathlib import Path
        
        # Format TimeRange object with more detail
        if isinstance(obj, TimeRange):
            duration = obj.end - obj.start
            formatted_duration = obj.format_duration(duration)
            return (
                f"{obj.start.isoformat(' ')} to {obj.end.isoformat(' ')}\n"
                f"  Duration: {formatted_duration}\n"
                f"  Interval: {str(obj.interval)}\n"
                f"  Include End: {obj.include_end}"
            )
        
        # Format DateTime objects in readable format 
        if isinstance(obj, datetime):
            return obj.isoformat(' ')
            
        # Format timedelta objects using TimeRange's format_duration
        if isinstance(obj, timedelta):
            from .core.time import TimeRange
            # We need to create a TimeRange instance to use the format_duration method
            tr = TimeRange(start=datetime.now(), duration=obj)
            return tr.format_duration(obj)
        
        # Format Path objects as strings
        if isinstance(obj, Path):
            return str(obj)
            
        # Format config with just its type name to avoid recursive dump
        if hasattr(obj, 'model_type') and isinstance(obj, RompyBaseModel):
            config_type = type(obj).__name__
            description = getattr(obj, 'description', '')
            result = f"{config_type}"
            if description:
                result += f" - {description}"
            return result
            
        # Use default formatting for everything else
        return None
