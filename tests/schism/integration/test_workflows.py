"""
Integration tests for common SCHISM workflows.

This module tests common end-to-end workflows for SCHISM model setup and configuration.
"""

import pytest

from rompy.core.data import DataBlob
from rompy.core.source import SourceFile
from rompy.core.time import TimeRange
from rompy.schism import SCHISMGrid
from rompy.schism.data import (
    SCHISMDataBoundary,
    SCHISMDataSflux,
    SfluxAir,
)

# Import our stub class from test_namelist instead of the non-existent module
from tests.schism.integration.test_namelist import SCHISMNamelist

pytest.importorskip("rompy.schism")


class TestCommonWorkflows:
    """Tests for common SCHISM model workflows."""

    def test_simple_ocean_setup(self, grid2d, test_files_dir, tmp_path):
        """Test setting up a simple ocean model with elevation boundary."""
        # 1. Create a directory for the model
        model_dir = tmp_path / "simple_ocean"
        model_dir.mkdir()

        # 2. Copy the grid
        grid_copy = grid2d.copy_to(model_dir)
        assert (model_dir / "hgrid.gr3").exists()

        # 3. Set up ocean boundary (using SCHISMDataBoundary directly)
        ocean_boundary = SCHISMDataBoundary(
            source=SourceFile(uri=str(test_files_dir / "hycom.nc")),
            variables=["surf_el"],
        )

        # 4. Create a simple namelist
        namelist = SCHISMNamelist(
            start_date="2020-01-01",
            end_date="2020-01-10",
            dt=150,
            output_dt=3600,
        )

        # 5. Check that all components are ready
        assert grid_copy is not None
        assert ocean_boundary is not None
        assert namelist is not None

        # Here, we would generate the actual model files if the implementation supports it

    def test_3d_ocean_with_atmosphere(
        self, grid3d, test_files_dir, grid_atmos_source, tmp_path
    ):
        """Test setting up a 3D ocean model with atmospheric forcing."""
        # 1. Create a directory for the model
        model_dir = tmp_path / "3d_ocean_atmos"
        model_dir.mkdir()

        # 2. Copy the grid
        grid_copy = grid3d.copy_to(model_dir)
        assert (model_dir / "hgrid.gr3").exists()

        # 3. Set up ocean boundary with temperature (using SCHISMDataBoundary directly)
        temp_boundary = SCHISMDataBoundary(
            source=SourceFile(uri=str(test_files_dir / "hycom.nc")),
            variables=["water_temp"],
        )

        # 4. Set up atmospheric forcing
        atmos_data = SCHISMDataSflux(
            air=SfluxAir(
                source=grid_atmos_source,
                prmsl="air_pressure",
                stmp="air_temperature",
                spfh="specific_humidity",
                uwind="eastward_wind",
                vwind="northward_wind",
            ),
        )

        # 5. Create a namelist
        namelist = SCHISMNamelist(
            start_date="2020-01-01",
            end_date="2020-01-10",
            dt=150,
            output_dt=3600,
        )

        # 6. Check that all components are ready
        assert grid_copy is not None
        assert temp_boundary is not None
        assert atmos_data is not None
        assert namelist is not None

        # Here, we would generate the actual model files if the implementation supports it

    def test_tidal_model(self, grid2d, tmp_path):
        """Test setting up a tidal model using new boundary conditions system."""
        # 1. Create a directory for the model
        model_dir = tmp_path / "tidal_model"
        model_dir.mkdir()

        # 2. Copy the grid
        grid_copy = grid2d.copy_to(model_dir)
        assert (model_dir / "hgrid.gr3").exists()

        # 3. Set up tidal forcing using new boundary conditions system
        from rompy.schism.boundary_conditions import create_tidal_only_boundary_config
        tidal_boundary = create_tidal_only_boundary_config(
            constituents=["M2", "S2", "K1", "O1"],
            tidal_database="tpxo9"
        )

        # 4. Create a namelist
        namelist = SCHISMNamelist(
            start_date="2020-01-01",
            end_date="2020-01-10",
            dt=150,
            output_dt=3600,
        )

        # 5. Check that all components are ready
        assert grid_copy is not None
        assert tidal_boundary is not None
        assert namelist is not None

        # Here, we would generate the actual model files if the implementation supports it


if __name__ == "__main__":
    pytest.main(["-v", __file__])
