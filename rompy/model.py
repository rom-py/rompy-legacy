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
from typing import Any, Dict, Literal, Optional, Union

from pydantic import Field, model_validator

from rompy.backends import BackendConfig
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


def _load_backends():
    """Load backends from entry points with fallback handling."""
    run_backends = {}
    postprocessors = {}
    pipeline_backends = {}

    # Load run backends
    try:
        for backend in load_entry_points("rompy.run"):
            name = backend.__name__.lower().replace("runbackend", "")
            run_backends[name] = backend
    except Exception as e:
        logger.warning(f"Failed to load run backends: {e}")

    # Load postprocessors
    try:
        for proc in load_entry_points("rompy.postprocess"):
            name = proc.__name__.lower().replace("postprocessor", "")
            postprocessors[name] = proc
    except Exception as e:
        logger.warning(f"Failed to load postprocessors: {e}")

    # Load pipeline backends
    try:
        for backend in load_entry_points("rompy.pipeline"):
            name = backend.__name__.lower().replace("pipelinebackend", "")
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

    # Initialize formatting variables in __init__

    model_type: Literal["modelrun"] = Field("modelrun", description="The model type.")
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
    run_id_subdir: bool = Field(
        True, description="Use run_id subdirectory in the output directory"
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
        if self.run_id_subdir:
            odir = Path(self.output_dir) / self.run_id
        else:
            odir = Path(self.output_dir)
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
        cc_full["runtime"]["staging_dir"] = self.staging_dir
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

        staging_dir = render(
            cc_full, self.config.template, self.output_dir, self.config.checkout
        )

        logger.info("")
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

    def __str__(self):
        repr = f"\nrun_id: {self.run_id}"
        repr += f"\nperiod: {self.period}"
        repr += f"\noutput_dir: {self.output_dir}"
        repr += f"\nconfig: {type(self.config)}\n"
        return repr

    def run(self, backend: BackendConfig, workspace_dir: Optional[str] = None) -> bool:
        """
        Run the model using the specified backend configuration.

        This method uses Pydantic configuration objects that provide type safety
        and validation for all backend parameters.

        Args:
            backend: Pydantic configuration object (LocalConfig, DockerConfig, etc.)
            workspace_dir: Path to generated workspace directory (optional)

        Returns:
            True if execution was successful, False otherwise

        Raises:
            TypeError: If backend is not a BackendConfig instance

        Examples:
            from rompy.backends import LocalConfig, DockerConfig

            # Local execution
            model.run(LocalConfig(timeout=3600, command="python run.py"))

            # Docker execution
            model.run(DockerConfig(image="swan:latest", cpu=4, memory="2g"))
        """
        if not isinstance(backend, BackendConfig):
            raise TypeError(
                f"Backend must be a BackendConfig instance, "
                f"got {type(backend).__name__}"
            )

        logger.debug(f"Using backend config: {type(backend).__name__}")

        # Get the backend class directly from the configuration
        backend_class = backend.get_backend_class()
        backend_instance = backend_class()

        # Pass the config object and workspace_dir to the backend
        return backend_instance.run(self, config=backend, workspace_dir=workspace_dir)

    def postprocess(self, processor: str = "noop", **kwargs) -> Dict[str, Any]:
        """
        Postprocess the model outputs using the specified processor.

        This method uses entry points to load and execute the appropriate postprocessor.
        Available processors are automatically discovered from the rompy.postprocess entry point group.

        Built-in processors:
        - "noop": A placeholder processor that does nothing but returns success

        Args:
            processor: Name of the postprocessor to use (default: "noop")
            **kwargs: Additional processor-specific parameters

        Returns:
            Dictionary with results from the postprocessing

        Raises:
            ValueError: If the specified processor is not available
        """
        # Get the requested postprocessor class from entry points
        if processor not in POSTPROCESSORS:
            available = list(POSTPROCESSORS.keys())
            raise ValueError(
                f"Unknown postprocessor: {processor}. "
                f"Available processors: {', '.join(available)}"
            )

        # Create an instance and process the outputs
        processor_class = POSTPROCESSORS[processor]
        processor_instance = processor_class()
        return processor_instance.process(self, **kwargs)

    def pipeline(self, pipeline_backend: str = "local", **kwargs) -> Dict[str, Any]:
        """
        Run the complete model pipeline (generate, run, postprocess) using the specified pipeline backend.

        This method executes the entire model workflow from input generation through running
        the model to postprocessing outputs. It uses entry points to load and execute the
        appropriate pipeline backend from the rompy.pipeline entry point group.

        Built-in pipeline backends:
        - "local": Execute the complete pipeline locally using the existing ModelRun methods

        Args:
            pipeline_backend: Name of the pipeline backend to use (default: "local")
            **kwargs: Additional backend-specific parameters. Common parameters include:
                - run_backend: Backend to use for the run stage (for local pipeline)
                - processor: Processor to use for postprocessing (for local pipeline)
                - run_kwargs: Additional parameters for the run stage
                - process_kwargs: Additional parameters for postprocessing

        Returns:
            Dictionary with results from the pipeline execution

        Raises:
            ValueError: If the specified pipeline backend is not available
        """
        # Get the requested pipeline backend class from entry points
        if pipeline_backend not in PIPELINE_BACKENDS:
            available = list(PIPELINE_BACKENDS.keys())
            raise ValueError(
                f"Unknown pipeline backend: {pipeline_backend}. "
                f"Available backends: {', '.join(available)}"
            )

        # Create an instance and execute the pipeline
        backend_class = PIPELINE_BACKENDS[pipeline_backend]
        backend_instance = backend_class()
        return backend_instance.execute(self, **kwargs)
