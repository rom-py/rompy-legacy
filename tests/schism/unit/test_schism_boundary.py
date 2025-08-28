"""
Unit tests for SCHISM boundary functionality.

This module tests the boundary components and data handling for SCHISM.
"""


import pytest

from rompy.core.data import DataBlob
from rompy.schism.data import SCHISMDataBoundary

pytest.importorskip("rompy.schism")


class TestBoundaryHandling:
    """Tests for boundary handling functionality."""

    def test_create_boundary(self, grid2d):
        """Test creation of boundaries from a grid."""
        # Get boundary nodes
        boundary_nodes = (
            grid2d.get_boundary_nodes()
            if hasattr(grid2d, "get_boundary_nodes")
            else grid2d.ocean_boundary()[0]
        )

        # Verify we have boundary nodes
        assert len(boundary_nodes) > 0

    def test_data_boundary_creation(self, grid2d, hycom_bnd2d, tmp_path):
        """Test creating a data boundary object."""
        # Create a data boundary
        data_boundary = SCHISMDataBoundary(
            id="test_boundary",
            source=hycom_bnd2d.source,  # Use SourceFile directly from hycom_bnd2d
            variables=["surf_el"],
            sel_method="interp",
            time_buffer=[0, 1],
        )

        # Test that data boundary was created successfully
        assert data_boundary is not None
        assert data_boundary.variables == ["surf_el"]

        # Ensure boundary format is correct for the grid

    def test_ocean_boundary(self, grid2d, hycom_bnd2d):
        """Test ocean boundary data handling."""
        # Create ocean boundary data directly
        ocean_boundary = SCHISMDataBoundary(
            id="elev2D",
            source=hycom_bnd2d.source,  # Use SourceFile directly
            variables=["surf_el"],
        )

        assert ocean_boundary is not None
        assert ocean_boundary.id == "elev2D"

    def test_3d_boundary(self, grid3d, hycom_bnd_temp_3d, tmp_path):
        """Test 3D boundary data handling."""
        # Create 3D temperature boundary
        temp_boundary = SCHISMDataBoundary(
            id="TEM_3D",
            source=hycom_bnd_temp_3d.source,  # Use SourceFile directly
            variables=["water_temp"],
            sel_method="interp",
        )

        # Test that 3D boundary was created successfully
        assert temp_boundary is not None
        assert temp_boundary.variables == ["water_temp"]

        # Ensure boundary format is correct for the grid

    def test_boundary_validation(self):
        """Test validation in boundary objects."""
        # Test with missing required fields
        with pytest.raises(Exception):
            # No source provided
            SCHISMDataBoundary(
                id="validation_test",
                variables=["surf_el"],
            )

        with pytest.raises(Exception):
            # Empty variables list
            SCHISMDataBoundary(
                id="validation_test2",
                source=DataBlob(source="dummy"),
                variables=[],
            )

    def test_boundary_to_file(self, grid2d, hycom_bnd2d, tmp_path):
        """Test writing boundary data to files."""
        # Create a data boundary
        data_boundary = SCHISMDataBoundary(
            id="test_boundary_file",
            source=hycom_bnd2d.source,  # Use SourceFile directly
            variables=["surf_el"],
            sel_method="interp",
        )

        # Here we would test writing to a file, if that functionality is implemented
        # This is a placeholder for when that functionality is available


if __name__ == "__main__":
    pytest.main(["-v", __file__])
