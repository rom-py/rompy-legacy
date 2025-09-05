import re
import tempfile
from datetime import datetime
from pathlib import Path

# Import necessary modules from rompy
from rompy.schism.boundary_core import TidalBoundary  # Backward compatibility alias
from rompy.schism.boundary_core import (
    BoundaryConfig,
    ElevationType,
    TidalDataset,
    VelocityType,
)


def validate_bctides_file(file_path):
    """Check if the bctides.in file exists and has basic content.

    Parameters
    ----------
    file_path : str or Path
        Path to bctides.in file

    Returns
    -------
    bool
        True if file exists and has expected content structure
    """
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {path}")
        return False

    with open(path, "r") as f:
        content = f.read()

    # Check for minimum expected content
    if not re.search(r"\d+\s+\d+\.\d+", content):  # ntip tip_dp line
        print("Missing ntip/tip_dp line")
        return False

    if not re.search(r"\d+\s+!nbfr", content):  # nbfr line
        print("Missing nbfr line")
        return False

    if not re.search(r"\d+\s+!nope", content):  # nope line
        print("Missing nope line")
        return False

    # Basic checks pass
    return True


def test_simple_bctides_format():
    """Simple test to validate bctides.in format."""
    # Create a temporary directory for our test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Find grid file path
        grid_path = Path(__file__).parent / "hgrid_20kmto60km_schism_testing.gr3"
        if not grid_path.exists():
            grid_path = Path(__file__).parent / "test_data" / "hgrid.gr3"

        if not grid_path.exists():
            print("No suitable grid file found for testing")
            return False

        # Create a simple tidal boundary
        boundary = TidalBoundary(
            grid_path=str(grid_path),
            tidal_data=TidalDataset(model="OCEANUM-atlas", constituents=["M2", "S2"]),
        )

        # Configure a simple tidal boundary
        config = BoundaryConfig(
            elev_type=ElevationType.HARMONIC, vel_type=VelocityType.HARMONIC
        )
        boundary.set_boundary_config(0, config)

        # Set run parameters
        boundary.set_run_parameters(datetime(2023, 1, 1), 2.0)  # 2 days

        # Write the bctides.in file
        output_file = Path(temp_dir) / "bctides.in"
        try:
            boundary.write_boundary_file(output_file)
            print(f"Successfully wrote bctides.in to {output_file}")

            # Validate the file
            is_valid = validate_bctides_file(output_file)
            print(f"Validation result: {'PASS' if is_valid else 'FAIL'}")

            # Print the first few lines for inspection
            with open(output_file, "r") as f:
                head = "".join(f.readlines()[:20])
            print("\nFirst 20 lines of bctides.in:")
            print("-" * 40)
            print(head)
            print("-" * 40)

            return is_valid

        except Exception as e:
            print(f"Error generating bctides.in: {e}")
            return False


if __name__ == "__main__":
    # Run the test directly
    result = test_simple_bctides_format()
    print(f"\nTest {'passed' if result else 'failed'}")
