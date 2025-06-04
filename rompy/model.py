"""
Model run implementation for ROMPY.

This module provides the ModelRun class which is the main entry point for
running models with ROMPY.
"""

import glob
import os
import platform
import shutil
import sys
import textwrap
import time as time_module
import zipfile as zf
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import Field

from rompy.core.config import BaseConfig
from rompy.core.logging import LogFormat, LoggingConfig, LogLevel, get_logger
from rompy.core.render import render
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel
from rompy.utils import load_entry_points

# Initialize the logger
logger = get_logger(__name__)


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

    # Initialize formatting variables in __init__

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
        # Import formatting utilities
        from rompy.formatting import format_table_row, get_formatted_box, log_box

        # Format model settings in a structured way
        config_type = type(self.config).__name__
        duration = self.period.end - self.period.start
        formatted_duration = self.period.format_duration(duration)

        # Create table rows for the model run info
        rows = [
            format_table_row("Run ID", str(self.run_id)),
            format_table_row("Model Type", config_type),
            format_table_row("Start Time", self.period.start.isoformat()),
            format_table_row("End Time", self.period.end.isoformat()),
            format_table_row("Duration", formatted_duration),
            format_table_row("Time Interval", str(self.period.interval)),
            format_table_row("Output Directory", str(self.output_dir)),
        ]

        # Add description if available
        if hasattr(self.config, "description") and self.config.description:
            rows.append(format_table_row("Description", self.config.description))

        # Create a formatted table with proper alignment
        formatted_rows = []
        key_lengths = []

        # First pass: collect all valid rows and calculate max key length
        for row in rows:
            try:
                # Split the row by the box-drawing vertical line character
                parts = [p.strip() for p in row.split("┃") if p.strip()]
                if len(parts) >= 2:  # We expect at least key and value parts
                    key = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else ""
                    key_lengths.append(len(key))
                    formatted_rows.append((key, value))
            except Exception as e:
                logger.warning(f"Error processing row '{row}': {e}")

        if not formatted_rows:
            logger.warning("No valid rows found for model run configuration table")
            return self._staging_dir

        max_key_len = max(key_lengths) if key_lengths else 0

        # Format the rows with proper alignment
        aligned_rows = []
        for key, value in formatted_rows:
            aligned_row = f"{key:>{max_key_len}} : {value}"
            aligned_rows.append(aligned_row)

        # Log the box with the model run info
        log_box(title="MODEL RUN CONFIGURATION", logger=logger, add_empty_line=False)

        # Log each row of the content with proper indentation
        for row in aligned_rows:
            logger.info(f"  {row}")

        # Log the bottom of the box
        log_box(
            title=None, logger=logger, add_empty_line=True  # Just the bottom border
        )

        # Display detailed configuration info using the new formatting framework
        from rompy.formatting import log_box

        # Create a box with the configuration type as title
        log_box(f"MODEL CONFIGURATION ({config_type})")

        # Use the model's string representation which now uses the new formatting
        try:
            # The __str__ method of RompyBaseModel already handles the formatting
            config_str = str(self.config)
            for line in config_str.split("\n"):
                logger.info(line)
        except Exception as e:
            # If anything goes wrong with config formatting, log the error and minimal info
            logger.info(f"Using {type(self.config).__name__} configuration")
            logger.debug(f"Configuration string formatting error: {str(e)}")

        logger.info("")
        # Use helper functions to avoid circular imports
        # No need to import or set USE_ASCII_ONLY, we use get_ascii_mode() directly

        # Use the log_box utility function
        from rompy.formatting import log_box

        log_box(
            title="STARTING MODEL GENERATION",
            logger=logger,
            add_empty_line=False,
        )
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
        # No need to import or set USE_ASCII_ONLY, we use get_ascii_mode() directly

        # Use the log_box utility function
        from rompy.formatting import log_box

        log_box(
            title="MODEL GENERATION COMPLETE",
            logger=logger,
            add_empty_line=False,
        )
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
        # Use the log_box utility function
        from rompy.formatting import log_box

        log_box(
            title="ARCHIVING MODEL FILES",
            logger=logger,
        )

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

        from rompy.formatting import log_box

        log_box(
            f"✓ Archive created successfully: {zip_fn}",
            logger=logger,
            add_empty_line=False,
        )
        return zip_fn

    def __call__(self):
        return self.generate()

    # Formatting is now handled by the formatting module
