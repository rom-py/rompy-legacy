import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

# Import Bctides class directly
from rompy.schism.bctides import Bctides


def test_files_dir():
    """Get the directory containing test files."""
    return Path(os.path.dirname(os.path.abspath(__file__)))


def validate_bctides_format(file_path):
    """Validate the format of a bctides.in file."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.split("!")[0].strip() for line in lines]
    lines = [line for line in lines if line]

    line_index = 0

    # Parse ntip and tip_dp (earth tidal potential)
    parts = lines[line_index].split()
    if len(parts) < 2:
        return False, "Missing ntip and tip_dp values"

    try:
        ntip = int(parts[0])
        float(parts[1])
    except ValueError:
        return False, "Invalid ntip or tip_dp values"

    line_index += 1

    # Parse tidal potential constituents if any
    if ntip > 0:
        for i in range(ntip):
            # Constituent name
            if line_index >= len(lines):
                return False, f"Missing constituent name for potential {i+1}"
            constituent = lines[line_index].strip()
            line_index += 1

            # Species, amplitude, frequency, nodal factor, earth equilibrium argument
            if line_index >= len(lines):
                return False, f"Missing tidal potential parameters for {constituent}"

            parts = lines[line_index].split()
            if len(parts) != 5:
                return False, f"Invalid tidal potential format for {constituent}"

            try:
                int(parts[0])
                float(parts[1])
                float(parts[2])
                float(parts[3])
                float(parts[4])
            except ValueError:
                return False, f"Invalid tidal potential values for {constituent}"

            line_index += 1

    # Parse nbfr (tidal boundary forcing frequencies)
    if line_index >= len(lines):
        return False, "Missing nbfr value"

    try:
        nbfr = int(lines[line_index])
    except ValueError:
        return False, "Invalid nbfr value"

    line_index += 1

    # Parse frequency info for each constituent
    for i in range(nbfr):
        # Constituent name
        if line_index >= len(lines):
            return False, f"Missing constituent name for frequency {i+1}"

        constituent = lines[line_index].strip()
        line_index += 1

        # Frequency, nodal factor, earth equilibrium argument
        if line_index >= len(lines):
            return False, f"Missing frequency parameters for {constituent}"

        parts = lines[line_index].split()
        if len(parts) != 3:
            return False, f"Invalid frequency format for {constituent}"

        try:
            float(parts[0])
            float(parts[1])
            float(parts[2])
        except ValueError:
            return False, f"Invalid frequency values for {constituent}"

        line_index += 1

    # Parse nope (number of open boundary segments)
    if line_index >= len(lines):
        return False, "Missing nope value"

    try:
        int(lines[line_index])
    except ValueError:
        return False, "Invalid nope value"

    return True, "File format is valid"


class MockGrid:
    """Mock grid class for testing."""

    def __init__(self):
        # Basic grid properties
        self.ne = 100  # Number of elements
        self.np = 60  # Number of nodes
        self.nob = 1  # Number of open boundaries
        self.nobn = np.array([10], dtype=np.int32)  # Number of nodes per boundary
        self.iobn = [
            np.array(range(10), dtype=np.int32)
        ]  # Node indices for each boundary
        self.x = np.array([float(i) for i in range(60)])  # Longitudes
        self.y = np.array([float(i) for i in range(60)])  # Latitudes


def test_basic_bctides_format(tidal_data_files):
    """Test that a basic bctides.in file can be created and has correct format."""
    # Create a mock grid
    grid = MockGrid()

    # Create dummy flags for one boundary segment
    flags = [[3, 3, 0, 0]]  # Tidal elevation, tidal velocity, no temp/salt BC

    # Create a Bctides instance
    bctides = Bctides(
        hgrid=grid,
        flags=flags,
        constituents=["M2", "S2"],
        tidal_database=tidal_data_files,
        tidal_model="OCEANUM-atlas",
    )

    # Set start time and duration
    bctides._start_time = datetime(2023, 1, 1)
    bctides._rnday = 5.0

    # Override interpolation method with a mock that returns constant values
    def mock_interpolate(self, lons, lats, tnames, data_type):
        num_nodes = len(lons)
        num_constituents = len(tnames)
        if data_type == "h":
            arr = np.zeros((num_nodes, num_constituents, 2))
            arr[..., 0] = 0.5  # amplitude
            arr[..., 1] = 45.0  # phase
            return arr
        elif data_type == "uv":
            arr = np.zeros((num_nodes, num_constituents, 4))
            arr[..., 0] = 0.1  # u_amp
            arr[..., 1] = 30.0  # u_phase
            arr[..., 2] = 0.1  # v_amp
            arr[..., 3] = 60.0  # v_phase
            return arr
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    # Assign our mock method to the instance
    bctides._interpolate_tidal_data = mock_interpolate.__get__(
        bctides, bctides.__class__
    )

    # Set empty constants to avoid file writing issues
    bctides.ethconst = {}
    bctides.vthconst = {}

    # Test both original and patched versions
    test_versions = [
        ("Original", bctides),
    ]

    for version_name, bctides_version in test_versions:
        print(f"\nTesting {version_name} version:")

        # Write the bctides.in file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            bctides_version.write_bctides(tmp_path)

            # Print the file contents for analysis
            print_bctides_file(tmp_path)

            # Validate the file format
            is_valid, message = validate_bctides_format(tmp_path)
            assert is_valid, message

            # Additional checks - read the file and examine specific sections
            with open(tmp_path, "r") as f:
                content = f.read()

            # Check constituent names (case-insensitive)
            content_lower = content.lower()
            assert "m2" in content_lower, "M2 constituent not found in output"
            assert "s2" in content_lower, "S2 constituent not found in output"

            # Check nbfr section
            with open(tmp_path, "r") as f:
                lines = f.readlines()

            # Find line with nbfr
            nbfr_line = None
            for i, line in enumerate(lines):
                if "nbfr" in line:
                    nbfr_line = i
                    break

                # If no explicit marker, look for a line that just has the number of constituents
                if line.strip().isdigit() and int(line.strip()) == len(bctides.tnames):
                    nbfr_line = i
                    break

            assert nbfr_line is not None, "nbfr line not found"

            # The nbfr value should be the number of constituents
            nbfr_value = int(lines[nbfr_line].split("!")[0].strip())
            assert nbfr_value == len(
                bctides.tnames
            ), f"nbfr ({nbfr_value}) doesn't match number of constituents ({len(bctides.tnames)})"

            # Check for constituent presence (case-insensitive)
            # Since SCHISM is case-insensitive, we just verify constituents are present
            content_lower = content.lower()
            assert "m2" in content_lower, "M2 constituent not found in any case"
            assert "s2" in content_lower, "S2 constituent not found in any case"

            # Log case information for debugging
            counts = {"M2": content.count("M2"), "m2": content.count("m2")}
            print(f"Case counts in {version_name} version: {counts}")

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def print_bctides_file(file_path):
    """Print the contents of a bctides.in file for analysis."""
    print("\n==== BCTIDES FILE CONTENTS ====")
    with open(file_path, "r") as f:
        print(f.read())
    print("==== END OF FILE CONTENTS ====\n")


if __name__ == "__main__":
    test_basic_bctides_format()
    print("All tests passed!")
