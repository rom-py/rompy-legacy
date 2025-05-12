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

from pydantic import Field

from rompy.utils import load_entry_points

from .core.config import BaseConfig
from .core.render import render
from .core.time import TimeRange
from .core.types import RompyBaseModel

logger = logging.getLogger(__name__)


# Accepted config types are defined in the entry points of the rompy.config group
CONFIG_TYPES = load_entry_points("rompy.config")


class ModelRun(RompyBaseModel):
    """A model run.

    It is intented to be model agnostic.
    It deals primarily with how the model is to be run, i.e. the period of the run
    and where the output is going. The actual configuration of the run is
    provided by the config object.

    Further explanation is given in the rompy.core.Baseconfig docstring.
    """

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
        # Use helper function to avoid circular imports
        from rompy import ROMPY_ASCII_MODE
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()

        logger.info("")
        if USE_ASCII_ONLY:
            logger.info("+------------------------------------------------------------------------+")
            logger.info("|                       MODEL RUN CONFIGURATION                           |")
            logger.info("+------------------------------------------------------------------------+")
        else:
            logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            logger.info("┃                       MODEL RUN CONFIGURATION                     ┃")
            logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        logger.info("")

        # Format model settings in a more structured way
        config_type = type(self.config).__name__
        duration = self.period.end - self.period.start
        formatted_duration = self.period.format_duration(duration)

        if USE_ASCII_ONLY:
            logger.info("+-----------------------------+-------------------------------------+")
            logger.info(f"| Run ID                     | {self.run_id:<35} |")
            logger.info(f"| Model Type                 | {config_type:<35} |")
            logger.info(f"| Start Time                 | {self.period.start.isoformat():<35} |")
            logger.info(f"| End Time                   | {self.period.end.isoformat():<35} |")
            logger.info(f"| Duration                   | {formatted_duration:<35} |")
            logger.info(f"| Time Interval              | {str(self.period.interval):<35} |")
            logger.info(f"| Output Directory           | {str(self.output_dir):<35} |")
        else:
            logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            logger.info(f"┃ Run ID                     ┃ {self.run_id:<35} ┃")
            logger.info(f"┃ Model Type                 ┃ {config_type:<35} ┃")
            logger.info(f"┃ Start Time                 ┃ {self.period.start:<35} ┃")
            logger.info(f"┃ End Time                   ┃ {self.period.end:<35} ┃")
            logger.info(f"┃ Duration                   ┃ {formatted_duration:<35} ┃")
            logger.info(f"┃ Time Interval              ┃ {str(self.period.interval):<35} ┃")
            logger.info(f"┃ Output Directory           ┃ {str(self.output_dir):<35} ┃")

        if hasattr(self.config, 'description') and self.config.description:
            if USE_ASCII_ONLY:
                logger.info(f"| Description                | {self.config.description:<35} |")
            else:
                logger.info(f"┃ Description                ┃ {self.config.description:<35} ┃")

        if USE_ASCII_ONLY:
            logger.info("+-----------------------------+-------------------------------------+")
        else:
            logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

        # Display minimal configuration info - detailed config will be logged by the
        # config implementation after the generate() call to avoid duplication
        if hasattr(self.config, '__str__'):
            logger.info("")
            logger.info(f"Configuration type: {type(self.config).__name__}")
            # Only log the first section to provide minimal context
            try:
                config_lines = str(self.config).split('\n')
                first_section = []
                for i, line in enumerate(config_lines):
                    if i < 5:  # Header lines
                        first_section.append(line)
                    else:
                        break
                # Add indication that there's more
                first_section.append("...")
                for line in first_section:
                    logger.info(line)
            except Exception:
                # If anything goes wrong with config formatting, just log minimal info
                logger.info(f"Using {type(self.config).__name__} configuration")

        logger.info("")
        # Use helper function to avoid circular imports
        from rompy import ROMPY_ASCII_MODE
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()

        if USE_ASCII_ONLY:
            logger.info("+------------------------------------------------------------------------+")
            logger.info("|                      STARTING MODEL GENERATION                          |")
            logger.info("+------------------------------------------------------------------------+")
        else:
            logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            logger.info("┃                      STARTING MODEL GENERATION                    ┃")
            logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
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
        # Use helper function to avoid circular imports
        from rompy import ROMPY_ASCII_MODE
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()

        if USE_ASCII_ONLY:
            logger.info("+------------------------------------------------------------------------+")
            logger.info("|                      MODEL GENERATION COMPLETE                          |")
            logger.info("+------------------------------------------------------------------------+")
        else:
            logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            logger.info("┃                      MODEL GENERATION COMPLETE                    ┃")
            logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
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
        # Use helper function to avoid circular imports
        from rompy import ROMPY_ASCII_MODE
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()

        logger.info("")
        if USE_ASCII_ONLY:
            logger.info("+------------------------------------------------------------------------+")
            logger.info("|                        ARCHIVING MODEL FILES                            |")
            logger.info("+------------------------------------------------------------------------+")
        else:
            logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
            logger.info("┃                        ARCHIVING MODEL FILES                      ┃")
            logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")

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
        # Use helper function to avoid circular imports
        from rompy import ROMPY_ASCII_MODE
        USE_ASCII_ONLY = ROMPY_ASCII_MODE()

        if USE_ASCII_ONLY:
            logger.info("------------------------------------------------------------------------")
        else:
            logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return zip_fn

    def __call__(self):
        return self.generate()

    def __str__(self):
        """Create a formatted string representation of the model run"""
        # Format the model run information in a more structured way
        config_type = type(self.config).__name__

        repr = "\n"
        repr += f"Run ID:     {self.run_id}\n"
        repr += f"Period:     {self.period.start} to {self.period.end}\n"
        repr += f"            Duration: {self.period.format_duration(self.period.end - self.period.start)}\n"
        repr += f"            Interval: {str(self.period.interval)}\n"
        repr += f"            Include End: {self.period.include_end}\n"
        repr += f"\n"
        repr += f"Output Dir: {self.output_dir}\n"
        repr += f"Config:     {config_type}\n"

        # Add additional config info if available
        if hasattr(self.config, 'description') and self.config.description:
            repr += f"Description: {self.config.description}\n"

        return repr
