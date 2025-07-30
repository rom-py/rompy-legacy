"""
Test for GR3 file generation using PyLibs implementation.
"""

import os
import pytest
import tempfile
from pathlib import Path
import shutil

from rompy.schism.grid import GR3Generator


# Skip this test if required files don't exist
def get_test_hgrid():
    """Find a suitable hgrid.gr3 file for testing"""
    # Try to find a test grid in the test data directory
    test_data_dir = Path(__file__).parent.parent / "data"

    # Look for hgrid.gr3 files in the test data directory
    potential_files = list(test_data_dir.glob("**/hgrid.gr3"))

    if potential_files:
        return potential_files[0]

    # If no file found, skip the test
    pytest.skip("No hgrid.gr3 file found for testing")
    return None


@pytest.mark.parametrize(
    "gr3_type,value",
    [
        ("drag", 0.0025),
        ("diffmin", 1.0),
        ("diffmax", 1.0),
        ("watertype", 1),
        ("albedo", 0.1),
        ("windrot_geo2proj", 0.0),
    ],
)
def test_gr3_generation(gr3_type, value):
    """Test that GR3 file can be generated properly with PyLibs implementation"""
    hgrid_path = get_test_hgrid()

    if hgrid_path is None:
        pytest.skip("No hgrid.gr3 file found for testing")

    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create generator
        generator = GR3Generator(hgrid=hgrid_path, gr3_type=gr3_type, value=value)

        # Generate GR3 file
        output_file = generator.generate(tmpdir)

        # Check that file was created
        assert output_file.exists(), f"Failed to generate {gr3_type}.gr3 file"


def test_generate_all_gr3_files():
    """
    Generate all GR3 files at once to verify the implementation works correctly.
    """
    hgrid_path = get_test_hgrid()
    if hgrid_path is None:
        pytest.skip("No hgrid.gr3 file found for testing")

    # Output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Generate all GR3 file types
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
            output_file = generator.generate(output_dir)
            assert output_file.exists(), f"Failed to generate {gr3_type}.gr3 file"


if __name__ == "__main__":
    # When run directly, generate test GR3 files
    hgrid_path = get_test_hgrid()
    if hgrid_path:
        output_dir = Path(__file__).parent / "gr3_files"
        output_dir.mkdir(exist_ok=True, parents=True)

        # Generate all GR3 file types
        gr3_types = {
            "drag": 0.0025,
            "diffmin": 1.0e-6,
            "diffmax": 1.0,
            "watertype": 1,
            "albedo": 0.15,
            "windrot_geo2proj": 0.0,
        }

        for gr3_type, value in gr3_types.items():
            print(f"Generating {gr3_type}.gr3...")
            generator = GR3Generator(hgrid=hgrid_path, gr3_type=gr3_type, value=value)
            output_file = generator.generate(output_dir)
            print(f"Generated {output_file}")
