import glob
import logging
import os
import platform
import shutil
import zipfile as zf
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Union

from pydantic import Field

from rompy.utils import load_entry_points

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
        logger.info("-----------------------------------------------------")
        logger.info("Model settings:")
        logger.info(self)
        logger.info("-----------------------------------------------------")
        logger.info(f"Generating model input files in {self.output_dir}")

        cc_full = {}
        cc_full["runtime"] = self.model_dump()
        cc_full["runtime"].update(self._generation_medatadata)
        cc_full["runtime"].update({"_datefmt": self._datefmt})

        if callable(self.config):
            # Run the __call__() method of the config object if it is callable passing
            # the runtime instance, and fill in the context with what is returned
            cc_full["config"] = self.config(self)
        else:
            # Otherwise just fill in the context with the config instance itself
            cc_full["config"] = self.config

        staging_dir = render(
            cc_full, self.config.template, self.output_dir, self.config.checkout
        )

        logger.info("")
        logger.info(f"Successfully generated project in {staging_dir}")
        logger.info("-----------------------------------------------------")
        return staging_dir

    def zip(self) -> str:
        """Zip the input files for the model run

        This function zips the input files for the model run and returns the
        name of the zip file. It also cleans up the staging directory leaving
        only the settings.json file that can be used to repoducte the run.

        returns
        -------
        zip_fn : str
        """

        # Always remove previous zips
        zip_fn = Path(str(self.staging_dir) + ".zip")
        if zip_fn.exists():
            zip_fn.unlink()

        with zf.ZipFile(zip_fn, mode="w", compression=zf.ZIP_DEFLATED) as z:
            for dp, dn, fn in os.walk(self.staging_dir):
                for filename in fn:
                    z.write(
                        os.path.join(dp, filename),
                        os.path.relpath(os.path.join(dp, filename), self.staging_dir),
                    )
        shutil.rmtree(self.staging_dir)
        logger.info(f"Successfully zipped project to {zip_fn}")
        return zip_fn

    def __call__(self):
        return self.generate()

    def __str__(self):
        repr = f"\nrun_id: {self.run_id}"
        repr += f"\nperiod: {self.period}"
        repr += f"\noutput_dir: {self.output_dir}"
        repr += f"\nconfig: {type(self.config)}\n"
        return repr

    def run(self, backend: str = "local", **kwargs) -> bool:
        """
        Run the model using the specified backend.

        This method uses entry points to load and execute the appropriate run backend.
        Available backends are automatically discovered from the rompy.run entry point group.
        
        Built-in backends:
        - "local": Execute the model locally using the system's Python interpreter
        - "docker": Execute the model inside a Docker container (if available)

        Args:
            backend: Name of the run backend to use (default: "local")
            **kwargs: Additional backend-specific parameters

        Returns:
            True if execution was successful, False otherwise

        Raises:
            ValueError: If the specified backend is not available
        """
        # Get the requested backend class from entry points
        if backend not in RUN_BACKENDS:
            available = list(RUN_BACKENDS.keys())
            raise ValueError(
                f"Unknown run backend: {backend}. "
                f"Available backends: {', '.join(available)}"
            )

        # Create an instance and run the model
        backend_class = RUN_BACKENDS[backend]
        backend_instance = backend_class()
        return backend_instance.run(self, **kwargs)

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
