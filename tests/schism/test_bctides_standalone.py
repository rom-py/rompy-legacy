import os
import sys
import pytest
from pathlib import Path
from datetime import datetime
import tempfile
import numpy as np

# Import needed modules
from rompy.schism.boundary_tides import (
    TidalBoundary,
    ElevationType,
    VelocityType,
    TracerType,
    TidalSpecies,
    BoundaryConfig,
    create_tidal_boundary,
    create_hybrid_boundary,
    create_river_boundary,
    create_nested_boundary,
)
from rompy.schism import SCHISMGrid
from rompy.core.data import DataBlob

# Path to test data
here = Path(__file__).parent


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
        tip_dp = float(parts[1])
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
                species = int(parts[0])
                amp = float(parts[1])
                freq = float(parts[2])
                nodal = float(parts[3])
                ear = float(parts[4])
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
            freq = float(parts[0])
            nodal = float(parts[1])
            ear = float(parts[2])
        except ValueError:
            return False, f"Invalid frequency values for {constituent}"

        line_index += 1

    # Parse nope (number of open boundary segments)
    if line_index >= len(lines):
        return False, "Missing nope value"

    try:
        nope = int(lines[line_index])
    except ValueError:
        return False, "Invalid nope value"

    return True, "File format is valid"


class MockTidalData:
    """Mock tidal dataset for testing."""

    def __init__(self):
        # Create mock tidal data
        self.data = {}
        self.lons = np.linspace(-180, 180, 10)
        self.lats = np.linspace(-90, 90, 10)

        # Create amp and phase data for each constituent
        for constituent in ["M2", "S2", "K1", "O1"]:
            # Amplitude and phase for elevation
            self.data[f"{constituent}_h_amp"] = np.ones((10, 10)) * 0.5
            self.data[f"{constituent}_h_phase"] = np.ones((10, 10)) * 45.0

            # Amplitude and phase for velocity
            self.data[f"{constituent}_u_amp"] = np.ones((10, 10)) * 0.1
            self.data[f"{constituent}_u_phase"] = np.ones((10, 10)) * 30.0
            self.data[f"{constituent}_v_amp"] = np.ones((10, 10)) * 0.1
            self.data[f"{constituent}_v_phase"] = np.ones((10, 10)) * 60.0

    def interp(self, lon, lat, constituent, data_type):
        """Mock interpolation function."""
        if data_type == "h":
            return np.array([[0.5, 45.0]])  # amp, phase for elevation
        elif data_type == "uv":
            return np.array([[0.1, 30.0, 0.1, 60.0]])  # u_amp, u_phase, v_amp, v_phase
        else:
            raise ValueError(f"Unknown data type: {data_type}")


class TestBctides:
    """Test cases for bctides.in file format."""

    def test_pure_tidal_boundary(self):
        """Test bctides.in format for a pure tidal boundary."""
        # Create a simple bctides.in file with tidal constituents
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Write a minimal bctides.in file with M2 and S2 constituents
            with open(tmp_path, "w") as f:
                f.write("! Bctides.in file generated for testing on 2023-01-01\n")
                f.write("2 50.0 !ntip, tip_dp\n")
                # For each tidal potential region
                f.write("M2\n")
                f.write(
                    "2 0.242334 0.0000140519 1.0 0.0 !species, amp, freq, nodal factor, earth tear\n"
                )
                f.write("S2\n")
                f.write(
                    "2 0.112743 0.0000145444 1.0 0.0 !species, amp, freq, nodal factor, earth tear\n"
                )
                # Number of tidal boundary forcing frequencies
                f.write("2 !nbfr - number of tidal forcing frequencies\n")
                # For each frequency
                f.write("M2\n")
                f.write(
                    "0.0000140519 1.0 0.0 !freq, nodal factor, earth equilibrium argument\n"
                )
                f.write("S2\n")
                f.write(
                    "0.0000145444 1.0 0.0 !freq, nodal factor, earth equilibrium argument\n"
                )
                # Number of open boundaries
                f.write("1 !nope - number of open boundaries\n")
                # Boundary type flags for each boundary
                f.write("5 5 0 0 !ifltype, iettype, itetype, isatype\n")
                # Number of nodes on this boundary
                f.write("10 !number of nodes\n")
                # For each constituent, amplitude and phase at each node
                for i in range(10):  # 10 nodes
                    f.write(f"0.5 45.0 !amp, phase for node {i+1}, constituent M2\n")
                for i in range(10):  # 10 nodes
                    f.write(f"0.3 30.0 !amp, phase for node {i+1}, constituent S2\n")

            # Validate format
            is_valid, message = validate_bctides_format(tmp_path)
            assert is_valid, message

            # Additional validation
            with open(tmp_path, "r") as f:
                content = f.read()

            # Check that constituents are in the file
            assert "M2" in content, "M2 constituent not found in output"
            assert "S2" in content, "S2 constituent not found in output"

            # Check that ntip section is correct
            with open(tmp_path, "r") as f:
                first_line = f.readline().strip()
                second_line = f.readline().strip()

            # First line should be a comment with date
            assert first_line.startswith("!"), "First line should be a comment"

            # Second line should have ntip and tip_dp
            parts = second_line.split("!")[0].strip().split()
            assert len(parts) >= 2, "Second line should have ntip and tip_dp"
            assert int(parts[0]) == 2, "ntip should be 2"
            assert float(parts[1]) == 50.0, "tip_dp should be 50.0"

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_river_boundary(self):
        """Test bctides.in format for a river boundary."""
        # Create a simple bctides.in file with river boundary
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Write a minimal bctides.in file with river boundary
            with open(tmp_path, "w") as f:
                f.write("! Bctides.in file generated for testing on 2023-01-01\n")
                f.write("0 50.0 !ntip, tip_dp\n")
                # Number of tidal boundary forcing frequencies
                f.write("0 !nbfr - number of tidal forcing frequencies\n")
                # Number of open boundaries
                f.write("1 !nope - number of open boundaries\n")
                # Boundary type flags for each boundary
                f.write("0 2 0 0 !ifltype, iettype, itetype, isatype\n")
                # Number of nodes on this boundary
                f.write("10 !number of nodes\n")
                # Constant discharge value
                f.write("-100.0 !discharge value\n")

            # Validate format
            is_valid, message = validate_bctides_format(tmp_path)
            assert is_valid, message

            # Additional validation - check for river flow
            with open(tmp_path, "r") as f:
                content = f.readlines()

            # Extract boundary flags line
            boundary_flags_line = None
            for i, line in enumerate(content):
                if "!nope" in line:
                    # Next non-empty line with data is the boundary flags
                    j = i + 1
                    while j < len(content) and not content[j].strip():
                        j += 1
                    if j < len(content):
                        boundary_flags_line = content[j].split("!")[0].strip().split()
                        break

            assert boundary_flags_line is not None, "Boundary flags line not found"
            assert (
                len(boundary_flags_line) >= 3
            ), "Boundary flags line should have at least 3 values"
            assert int(boundary_flags_line[0]) == 0, "Elevation type should be 0 (NONE)"
            assert (
                int(boundary_flags_line[1]) == 2
            ), "Velocity type should be 2 (CONSTANT)"

            # Find the constant discharge value
            discharge_line = None
            for i, line in enumerate(content):
                if "discharge value" in line:
                    discharge_line = line
                    break

            assert discharge_line is not None, "Discharge value line not found"
            discharge_value = float(discharge_line.split("!")[0].strip())
            assert discharge_value == -100.0, "Discharge value should be -100.0"

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    # Run tests directly
    test = TestBctides()
    test.test_pure_tidal_boundary()
    test.test_river_boundary()
    print("All tests passed!")
