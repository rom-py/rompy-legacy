from pathlib import Path
from shutil import rmtree

import pytest
import yaml
from utils import compare_nmls


# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

pytest.importorskip("rompy.schism")

from rompy.model import ModelRun
from rompy.schism import SCHISMGrid
from rompy.schism.config import SchismCSIROConfig, SchismCSIROMigrationConfig

here = Path(__file__).parent


@pytest.fixture
def config():
    """Loads and returns configuration from YAML file."""
    here = Path(__file__).parent
    config_path = here / "configs" / "csiro.yaml"
    with config_path.open() as file:
        return yaml.safe_load(file)


@pytest.mark.skip(
    "Not working because it looks like the csiro template uses non default values"
)
def test_schism_migration(config):
    """Test the swantemplate function."""
    rmtree("simulations", ignore_errors=True)
    model_run = ModelRun(**config)
    model_run.generate()
    config.update({"run_id": "migration"})
    config["config"].update({"model_type": "schismcsiromigration"})
    migration_run = ModelRun(**config)
    migration_run.generate()
    compare_nmls("simulations/migration/param.nml", "simulations/test_schism/param.nml")


# if __name__ == "__main__":
#     here = Path(__file__).parent
#     config_path = here / "configs" / "csiro.yaml"
#     with config_path.open() as file:
#         config = yaml.safe_load(file)
#     test_schism_migration(config)
