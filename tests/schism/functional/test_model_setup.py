"""
Functional tests for SCHISM model setup.

This module tests complete end-to-end model setup and preparation.
"""


import pytest
import yaml

from rompy.core.source import SourceFile
from rompy.schism.data import (
    SCHISMDataBoundary,
    SCHISMDataSflux,
)

# Import our stub class from test_namelist instead of the non-existent module
from tests.schism.integration.test_namelist import SCHISMNamelist

pytest.importorskip("rompy.schism")


class TestFullModelSetup:
    """Tests for complete SCHISM model setup."""

    def test_declarative_model_setup(self, tmp_path):
        """Test setting up a model from a declarative YAML specification."""
        # Create a sample YAML configuration
        yaml_content = """
        grid:
          hgrid: tests/schism/test_data/hgrid.gr3
          gr3_files:
            - type: drag
              value: 0.0025
            - type: diffmax
              value: 1.0
        
        time:
          start_date: '2020-01-01'
          end_date: '2020-01-10'
          dt: 150
          output_dt: 3600
        
        boundaries:
          ocean:
            elev2D:
              source: tests/schism/test_data/hycom_subset.nc
              variables: [surf_el]
              sel_method: interp
        """

        # Write YAML file
        config_path = tmp_path / "model_config.yaml"
        with open(config_path, "w") as f:
            f.write(yaml_content)

        # Here we would load and process the model setup
        # This is a placeholder for when that functionality is fully implemented

        # For now, we just verify the file was created
        assert config_path.exists()

        # Read the YAML to validate it
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert "grid" in config
        assert "time" in config
        assert "boundaries" in config
        assert config["grid"]["gr3_files"][0]["type"] == "drag"

    def test_realistic_model_setup(
        self, grid2d, test_files_dir, grid_atmos_source, tmp_path
    ):
        """Test a realistic model setup with multiple components."""
        # Create model directory
        model_dir = tmp_path / "realistic_model"
        model_dir.mkdir()

        # 1. Create all GR3 files

        # Here we would generate all gr3 files
        # For now, just check that we have the data we need
        assert grid2d is not None
        assert grid_atmos_source is not None

        # 2. Set up boundaries (using SCHISMDataBoundary directly)
        ocean_boundary = SCHISMDataBoundary(
            source=SourceFile(uri=str(test_files_dir / "hycom.nc")),
            variables=["surf_el"],
        )

        # 3. Set up atmospheric forcing
        atmos_data = SCHISMDataSflux(
            air=dict(
                source=grid_atmos_source,
                prmsl="air_pressure",
                stmp="air_temperature",
                spfh="specific_humidity",
                uwind="eastward_wind",
                vwind="northward_wind",
            ),
        )

        # 4. Create namelist
        namelist = SCHISMNamelist(
            start_date="2020-01-01",
            end_date="2020-01-10",
            dt=150,
            output_dt=3600,
            ics=0,
            nws=1,
        )

        # 5. Verify all components
        assert ocean_boundary is not None
        assert atmos_data is not None
        assert namelist is not None

        # Here we would write all model files if the implementation supports it

    def test_yaml_model_validation(self, tmp_path):
        """Test validation of model configuration from YAML."""
        # Create an invalid YAML configuration (missing required fields)
        invalid_yaml = """
        grid:
          # Missing hgrid field
          gr3_files:
            - type: drag
              value: 0.0025
        
        time:
          # Missing start_date
          end_date: '2020-01-10'
          dt: 150
        """

        # Write YAML file
        config_path = tmp_path / "invalid_config.yaml"
        with open(config_path, "w") as f:
            f.write(invalid_yaml)

        # The validation would happen here, when we load the config
        # For now, just verify the file was created
        assert config_path.exists()

        # Load the YAML to check for expected missing fields
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert "grid" in config
        assert "hgrid" not in config["grid"]
        assert "time" in config
        assert "start_date" not in config["time"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
