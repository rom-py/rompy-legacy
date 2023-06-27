"""Base model wrapper.

Principles
----------
pydantic
    - BaseModel with useful capability
    - Type checking of input

"""
from __future__ import annotations
import logging
import shutil
from pathlib import Path
from pydantic import BaseModel


logger = logging.getLogger(__file__)


class Model(BaseModel):
    """Base model class.

    Parameters
    ----------
    id: str
        Model id.
    kind: str
        Model kind, this should be set by inheritted classes.
    workdir: str
        Workspace directory path.
    children: list[Model]
        Model child nests.
    loglevel: str | int
        Logger level, see https://docs.python.org/3/library/logging.html#levels.

    """
    id: str
    kind: str
    workdir: Path
    children: list[Model] = []
    loglevel: str | int = "INFO"

    def _set_workspace(self):
        """Create workspace environment."""
        logger.info(f"Creating workspace directory: {self.workdir}")
        self.workdir.mkdir(exist_ok=True, parents=True)

    def _del_workspace(self):
        """Delete workspace environment."""
        logger.warning(f"Deleting workspace directory: {self.workdir}")
        shutil.rmtree(self.workdir)

    def _fetch_data(self):
        """Fetch all forcing data."""
        pass

    def _write_cmd(self):
        """Write model command file."""
        pass

    def preprocess(self):
        """Create workspace and all files to run the model."""
        self._set_workspace()
        self._fetch_data()
        self._write_cmd()

    def run(self):
        """Execute model binary."""
        pass

    def postprocess(self):
        """Postprocess and upload model results."""

