import glob
import logging
import os
import platform
import zipfile as zf
from datetime import datetime
from typing import Optional

from pydantic import Field

from rompy.swan import SwanConfig

from .core import BaseConfig, RompyBaseModel, TimeRange
from .core.render import render

logger = logging.getLogger(__name__)


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
    output_dir: str = Field("./simulations", description="The output directory")
    config: BaseConfig | SwanConfig = Field(
        BaseConfig(), description="The configuration object"
    )
    _datefmt: str = "%Y%m%d.%H%M%S"

    @property
    def staging_dir(self):
        """The directory where the model is staged for execution

        returns
        -------
        staging_dir : str
        """

        odir = os.path.join(self.output_dir, self.run_id)
        os.makedirs(odir, exist_ok=True)
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
        cc_full["runtime"] = self.dict()
        cc_full["runtime"].update(self._generation_medatadata)
        cc_full["runtime"].update({"_datefmt": self._datefmt})
        # TODO calculate from period
        cc_full["runtime"]["frequency"] = "0.25 HR"

        if callable(self.config):
            cc_full["config"] = self.config(self)
        else:
            cc_full["config"] = self.config

        staging_dir = render(
            cc_full, self.config.template, self.output_dir, self.config.checkout
        )

        logger.info("")
        logger.info(f"Successfully generated project in {self.output_dir}")
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
        zip_fn = self.staging_dir + "/simulation.zip"
        if os.path.exists(zip_fn):
            os.remove(zip_fn)

        # Zip the input files
        run_files = glob.glob(self.staging_dir + "/**/*", recursive=True)
        with zf.ZipFile(zip_fn, mode="w", compression=zf.ZIP_DEFLATED) as z:
            for f in run_files:
                # strip off the path prefix
                z.write(f, f.replace(self.staging_dir, ""))

        # Clean up run files leaving the settings.json
        for f in run_files:
            if not os.path.basename(f) == "settings.json":
                if os.path.isfile(f):
                    os.remove(f)

        # Clean up any directories
        for f in run_files:
            if os.path.isdir(f):
                os.rmdir(f)

        return zip_fn

    def __call__(self):
        return self.generate()

    def __str__(self):
        ret = f"\nperiod: \n{self.period}\n\n"
        ret += f"output_dir: \n{self.output_dir}\n\n"
        ret += f"config: \n{self.config}"
        return ret
