"""
Unit tests for SCHISM grid functionality.

This module tests the core grid components of the SCHISM implementation.
"""


import pytest

from rompy.schism.grid import GR3Generator

pytest.importorskip("rompy.schism")


# Include basic grid tests
class TestSCHISMGrid:
    """Tests for the SCHISM grid components."""

    def test_vgrid_model(self):
        """Test the SchismVGrid model class."""
        from rompy.schism.vgrid import VGrid

        # Create vertical grid
        vgrid = VGrid(ivcor=2, nvrt=10, h_c=20.0, theta_b=0.5, theta_f=5.0)  # SZ type

        # Validate properties
        assert hasattr(vgrid, "ivcor")
        assert hasattr(vgrid, "nvrt")
        assert hasattr(vgrid, "h_c")
        assert hasattr(vgrid, "theta_b")
        assert hasattr(vgrid, "theta_f")

        # Validate serialization
        vgrid_dict = vgrid.model_dump()
        assert "ivcor" in vgrid_dict
        assert "nvrt" in vgrid_dict

    def test_grid_model_structure(self):
        """Test the model class structure (not the instantiation)."""
        from rompy.schism.grid import SCHISMGrid

        # Just test the class structure and model fields, not instantiation
        assert hasattr(SCHISMGrid, "model_fields")
        fields = SCHISMGrid.model_fields

        # Check field names (only check for hgrid and vgrid which are definitely present)
        assert "hgrid" in fields
        assert "vgrid" in fields

        # Check model methods
        assert hasattr(SCHISMGrid, "model_dump")
        assert hasattr(SCHISMGrid, "model_json_schema")


class TestGR3Generator:
    """Tests for GR3 file generation."""

    @pytest.mark.parametrize(
        "gr3_type,value",
        [
            ("drag", 0.0025),
            ("diffmin", 1.0e-6),
            ("diffmax", 1.0),
            ("watertype", 1),
            ("albedo", 0.15),
            ("windrot_geo2proj", 0.0),
        ],
    )
    def test_gr3_generation(self, gr3_type, value, hgrid_path, tmp_path):
        """Test that GR3 files can be generated properly."""
        # Skip if no test grid available
        if hgrid_path is None:
            pytest.skip("No hgrid.gr3 file found for testing")

        # Create generator
        generator = GR3Generator(hgrid=hgrid_path, gr3_type=gr3_type, value=value)

        # Generate GR3 file
        output_file = generator.generate(tmp_path)

        # Check that file was created and has expected format
        assert output_file.exists()

        # Validate basic content
        with open(output_file, "r") as f:
            # Check header
            header = f.readline().strip()
            assert gr3_type in header

            # Check second line has two integers
            counts_line = f.readline().strip()
            parts = counts_line.split()
            assert len(parts) == 2
            assert all(p.isdigit() for p in parts)

            # Check at least one node line
            node_line = f.readline().strip()
            parts = node_line.split()
            assert len(parts) >= 4  # id, x, y, value

            # Check value is as expected
            node_value = float(parts[-1])
            assert abs(node_value - value) < 1e-6

    def test_generate_all_gr3_files(self, hgrid_path, tmp_path):
        """Test generating all supported GR3 file types."""
        if hgrid_path is None:
            pytest.skip("No hgrid.gr3 file found for testing")

        # Define supported GR3 types with standard values
        gr3_types = {
            "drag": 0.0025,
            "diffmin": 1.0e-6,
            "diffmax": 1.0,
            "watertype": 1,
            "albedo": 0.15,
            "windrot_geo2proj": 0.0,
        }

        for gr3_type, value in gr3_types.items():
            generator = GR3Generator(hgrid=hgrid_path, gr3_type=gr3_type, value=value)
            output_file = generator.generate(tmp_path)
            assert output_file.exists(), f"Failed to generate {gr3_type}.gr3 file"

    def test_gr3_error_handling(self, hgrid_path, tmp_path):
        """Test error handling in GR3 generation."""
        # Test with invalid type
        with pytest.raises(ValueError):
            GR3Generator(hgrid=hgrid_path, gr3_type="invalid_type", value=1.0)

        # Test with missing value
        with pytest.raises(Exception):
            generator = GR3Generator(hgrid=hgrid_path, gr3_type="drag", value=None)
            generator.generate(tmp_path)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
