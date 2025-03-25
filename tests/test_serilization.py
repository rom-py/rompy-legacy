import os
from pathlib import Path

import pytest
from envyaml import EnvYAML

from rompy.schism.config import SCHISMConfig, SCHISMGrid
from rompy.schism.namelists import NML, Param, Wwminput
from rompy.swan.config import SwanConfigComponents
from rompy.core import DataBlob

here = Path(__file__).parent
os.environ["ROMPY_PATH"] = str(here.parent.parent)



@pytest.fixture(scope="module")
def schism_config():
    yield SCHISMConfig(
        grid=SCHISMGrid(
            hgrid=DataBlob(id="hgrid", source=here / "test_data" / "hgrid.gr3"),
            drag=1,
        ),
        nml=NML(
            param=Param(**{"core": {"ipre": 1}}),
            wwminput=Wwminput(**{"proc": {"dimmode": 2}}),
        ),
    )


@pytest.fixture(scope="module")
def swan_config():
    yield EnvYAML(here / "swan" / "swan_model.yml")


def test_swan(swan_config):
    """Test that the model serializes correctly."""

    swan = SwanConfigComponents(**swan_config)
    serialized = swan.model_dump()

    swan2 = SwanConfigComponents(**serialized)
    assert swan == swan2

def test_schism(schism_config):
    """Test that the model serializes correctly."""
    
    # Serialize the model to JSON
    serialized = schism_config.model_dump()
    
    # Create a new instance from the serialized data
    schism2 = SCHISMConfig(**serialized)
    
    # Test individual components to ensure proper serialization/deserialization
    assert schism_config.model_type == schism2.model_type
    assert schism_config.grid.grid_type == schism2.grid.grid_type
    assert schism_config.grid.hgrid == schism2.grid.hgrid
    
    # Check GR3Generator fields
    # The original config has a GR3Generator with a value, and the deserialized one should too
    if hasattr(schism_config.grid.drag, 'value'):
        if isinstance(schism2.grid.drag, (int, float)):
            assert schism_config.grid.drag.value == schism2.grid.drag
        else:
            assert schism_config.grid.drag.value == schism2.grid.drag.value
    
    # Namelist should be equal
    assert schism_config.nml == schism2.nml
