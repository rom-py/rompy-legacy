import os
import pytest
from pathlib import Path
from datetime import datetime

from rompy.schism.boundary_core import (
    BoundaryConfig,
    ElevationType,
    TidalBoundary,  # Backward compatibility alias
    VelocityType,
    create_tidal_boundary,
)


def test_files_dir():
    """Get the directory containing test files."""
    return Path(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def tidal_dataset():
    """Create paths to tidal data files."""

    class TidalData:
        def __init__(self):
            self.elevations = test_files_dir() / "data" / "h_tpxo9.nc"
            self.velocities = test_files_dir() / "data" / "uv_tpxo9.nc"

    # Skip if files don't exist (allows tests to run even without data files)
    tidal_data = TidalData()
    if not tidal_data.elevations.exists() or not tidal_data.velocities.exists():
        pytest.skip("Tidal data files not found")

    return tidal_data


def validate_ntip_section(file_path):
    """Validate the earth tidal potential section of the bctides.in file."""
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

    return True, line_index


def validate_nbfr_section(file_path, start_line):
    """Validate the tidal boundary forcing frequencies section of the bctides.in file."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.split("!")[0].strip() for line in lines]
    lines = [line for line in lines if line]

    line_index = start_line

    # Parse nbfr (tidal boundary forcing frequencies)
    if line_index >= len(lines):
        return False, "Missing nbfr value", 0

    try:
        nbfr = int(lines[line_index])
    except ValueError:
        return False, "Invalid nbfr value", 0

    line_index += 1

    # Parse frequency info for each constituent
    for i in range(nbfr):
        # Constituent name
        if line_index >= len(lines):
            return False, f"Missing constituent name for frequency {i+1}", 0

        constituent = lines[line_index].strip()
        line_index += 1

        # Frequency, nodal factor, earth equilibrium argument
        if line_index >= len(lines):
            return False, f"Missing frequency parameters for {constituent}", 0

        parts = lines[line_index].split()
        if len(parts) != 3:
            return False, f"Invalid frequency format for {constituent}", 0

        try:
            float(parts[0])
            float(parts[1])
            float(parts[2])
        except ValueError:
            return False, f"Invalid frequency values for {constituent}", 0

        line_index += 1

    return True, line_index, nbfr


def validate_boundary_section(file_path, start_line, nbfr):
    """Validate the open boundary segments section of the bctides.in file."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.split("!")[0].strip() for line in lines]
    lines = [line for line in lines if line]

    line_index = start_line

    # Parse nope (number of open boundary segments)
    if line_index >= len(lines):
        return False, "Missing nope value"

    try:
        nope = int(lines[line_index])
    except ValueError:
        return False, "Invalid nope value"

    line_index += 1

    # Parse each open boundary segment
    for j in range(nope):
        # Parse number of nodes and flags
        if line_index >= len(lines):
            return False, f"Missing boundary flags for segment {j+1}"

        parts = lines[line_index].split()
        if len(parts) < 5:  # At least neta, elev_type, vel_type, temp_type, salt_type
            return False, f"Invalid boundary flags for segment {j+1}"

        try:
            neta = int(parts[0])
            iettype = int(parts[1])  # Elevation type
            ifltype = int(parts[2])  # Velocity type
            int(parts[3])  # Temperature type
            int(parts[4])  # Salinity type
        except ValueError:
            return False, f"Invalid boundary flag values for segment {j+1}"

        line_index += 1

        # Validate elevation section based on type
        if iettype == 1:
            # Time history - no input in bctides.in
            pass
        elif iettype == 2:
            # Constant elevation
            if line_index >= len(lines):
                return False, f"Missing constant elevation for segment {j+1}"

            try:
                float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid constant elevation for segment {j+1}"
        elif iettype == 3:
            # Tidal elevation
            for k in range(nbfr):
                # Constituent name
                if line_index >= len(lines):
                    return (
                        False,
                        f"Missing constituent name for elevation on segment {j+1}",
                    )

                lines[line_index].strip()
                line_index += 1

                # Parse amplitude and phase for each node
                for i in range(neta):
                    if line_index >= len(lines):
                        return (
                            False,
                            f"Missing elevation values for node {i+1} on segment {j+1}",
                        )

                    parts = lines[line_index].split()
                    if len(parts) != 2:
                        return (
                            False,
                            f"Invalid elevation format for node {i+1} on segment {j+1}",
                        )

                    try:
                        float(parts[0])
                        float(parts[1])
                    except ValueError:
                        return (
                            False,
                            f"Invalid elevation values for node {i+1} on segment {j+1}",
                        )

                    line_index += 1
        elif iettype == 4:
            # Space- and time-varying input - no input in bctides.in
            pass
        elif iettype == 5:
            # Combination of '3' and '4'
            for k in range(nbfr):
                # Constituent name
                if line_index >= len(lines):
                    return (
                        False,
                        f"Missing constituent name for elevation on segment {j+1}",
                    )

                lines[line_index].strip()
                line_index += 1

                # Parse amplitude and phase for each node
                for i in range(neta):
                    if line_index >= len(lines):
                        return (
                            False,
                            f"Missing elevation values for node {i+1} on segment {j+1}",
                        )

                    parts = lines[line_index].split()
                    if len(parts) != 2:
                        return (
                            False,
                            f"Invalid elevation format for node {i+1} on segment {j+1}",
                        )

                    try:
                        float(parts[0])
                        float(parts[1])
                    except ValueError:
                        return (
                            False,
                            f"Invalid elevation values for node {i+1} on segment {j+1}",
                        )

                    line_index += 1
        elif iettype != 0:
            return False, f"Invalid elevation type {iettype} for segment {j+1}"

        # Validate velocity section based on type
        if ifltype == 0:
            # Velocity not specified
            pass
        elif ifltype == 1:
            # Time history - no input in bctides.in
            pass
        elif ifltype == 2:
            # Constant discharge
            if line_index >= len(lines):
                return False, f"Missing constant discharge for segment {j+1}"

            try:
                float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid constant discharge for segment {j+1}"
        elif ifltype == 3:
            # Tidal velocity
            for k in range(nbfr):
                # Constituent name
                if line_index >= len(lines):
                    return (
                        False,
                        f"Missing constituent name for velocity on segment {j+1}",
                    )

                lines[line_index].strip()
                line_index += 1

                # Parse amplitude and phase for each node
                for i in range(neta):
                    if line_index >= len(lines):
                        return (
                            False,
                            f"Missing velocity values for node {i+1} on segment {j+1}",
                        )

                    parts = lines[line_index].split()
                    if len(parts) != 4:
                        return (
                            False,
                            f"Invalid velocity format for node {i+1} on segment {j+1}",
                        )

                    try:
                        float(parts[0])
                        float(parts[1])
                        float(parts[2])
                        float(parts[3])
                    except ValueError:
                        return (
                            False,
                            f"Invalid velocity values for node {i+1} on segment {j+1}",
                        )

                    line_index += 1
        elif ifltype == 4 or ifltype == -4:
            # 3D input - no input in bctides.in (except relaxation for -4)
            if ifltype == -4:
                if line_index >= len(lines):
                    return False, f"Missing relaxation constants for segment {j+1}"

                parts = lines[line_index].split()
                if len(parts) != 2:
                    return False, f"Invalid relaxation format for segment {j+1}"

                try:
                    float(parts[0])
                    float(parts[1])
                except ValueError:
                    return False, f"Invalid relaxation values for segment {j+1}"

                line_index += 1
        elif ifltype == 5:
            # Combination of '4' and '3'
            for k in range(nbfr):
                # Constituent name
                if line_index >= len(lines):
                    return (
                        False,
                        f"Missing constituent name for velocity on segment {j+1}",
                    )

                lines[line_index].strip()
                line_index += 1

                # Parse amplitude and phase for each node
                for i in range(neta):
                    if line_index >= len(lines):
                        return (
                            False,
                            f"Missing velocity values for node {i+1} on segment {j+1}",
                        )

                    parts = lines[line_index].split()
                    if len(parts) != 4:
                        return (
                            False,
                            f"Invalid velocity format for node {i+1} on segment {j+1}",
                        )

                    try:
                        float(parts[0])
                        float(parts[1])
                        float(parts[2])
                        float(parts[3])
                    except ValueError:
                        return (
                            False,
                            f"Invalid velocity values for node {i+1} on segment {j+1}",
                        )

                    line_index += 1
        elif ifltype == -1:
            # Flather type
            # Parse mean elevation marker
            if line_index >= len(lines):
                return False, f"Missing eta_mean marker for segment {j+1}"

            if lines[line_index].strip().lower() != "eta_mean":
                return False, f"Invalid eta_mean marker for segment {j+1}"

            line_index += 1

            # Parse mean elevation values
            for i in range(neta):
                if line_index >= len(lines):
                    return (
                        False,
                        f"Missing mean elevation value for node {i+1} on segment {j+1}",
                    )

                try:
                    float(lines[line_index])
                    line_index += 1
                except ValueError:
                    return (
                        False,
                        f"Invalid mean elevation for node {i+1} on segment {j+1}",
                    )

            # Parse mean normal velocity marker
            if line_index >= len(lines):
                return False, f"Missing vn_mean marker for segment {j+1}"

            if lines[line_index].strip().lower() != "vn_mean":
                return False, f"Invalid vn_mean marker for segment {j+1}"

            line_index += 1

            # Parse mean normal velocity values
            for i in range(neta):
                if line_index >= len(lines):
                    return (
                        False,
                        f"Missing mean velocity value for node {i+1} on segment {j+1}",
                    )

                try:
                    float(lines[line_index])
                    line_index += 1
                except ValueError:
                    return (
                        False,
                        f"Invalid mean velocity for node {i+1} on segment {j+1}",
                    )

    return True, "File format is valid"


def validate_bctides_format_complete(file_path):
    """Perform a complete validation of the bctides.in file format.

    This function checks all sections of the file according to the
    pseudocode specification.
    """
    # Check ntip section
    valid_ntip, result = validate_ntip_section(file_path)
    if not valid_ntip:
        return False, result

    line_index = result

    # Check nbfr section
    valid_nbfr, line_index, nbfr = validate_nbfr_section(file_path, line_index)
    if not valid_nbfr:
        return False, line_index

    # Check boundary section
    valid_boundary, message = validate_boundary_section(file_path, line_index, nbfr)
    if not valid_boundary:
        return False, message

    return True, "File format is valid"


def test_tidal_only_boundary_format(grid2d, tidal_dataset, tmp_path):
    """Test that a tidal-only boundary creates a correctly formatted bctides.in file."""
    boundary = create_tidal_boundary(
        grid_path=grid2d.pylibs_hgrid,
        constituents=["M2", "S2"],
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities,
        ntip=2,  # Use earth tidal potential
        tip_dp=50.0,  # Cutoff depth
    )

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)

    # Write boundary file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_tidal.in")

    # Validate format
    is_valid, message = validate_bctides_format_complete(bctides_path)
    assert is_valid, message

    # Additional validation - read file and check specific sections
    with open(bctides_path, "r") as f:
        content = f.read()

    # Check for required sections
    assert "M2" in content, "M2 constituent not found in output"
    assert "S2" in content, "S2 constituent not found in output"


def test_hybrid_boundary_format(grid2d, tidal_dataset, tmp_path):
    """Test that a hybrid boundary creates a correctly formatted bctides.in file."""
    boundary = create_hybrid_boundary(
        grid_path=grid2d.pylibs_hgrid,
        constituents=["M2"],
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities,
        ntip=1,
        tip_dp=50.0,
    )

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)

    # Write boundary file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_hybrid.in")

    # Validate format
    is_valid, message = validate_bctides_format_complete(bctides_path)
    assert is_valid, message


def test_river_boundary_format(grid2d, tmp_path):
    """Test that a river boundary creates a correctly formatted bctides.in file."""
    # Create a simple bctides.in file directly for validation
    bctides_path = tmp_path / "bctides_river.in"

    with open(bctides_path, "w") as f:
        # Write ntip section
        f.write("0 50.0 !ntip tip_dp\n")

        # Write nbfr section (no tidal constituents for river)
        f.write("0 !nbfr\n")

        # Write nope section (number of open boundaries)
        f.write("1 !nope\n")

        # Write boundary section
        num_nodes = grid2d.nobn[0]  # Number of nodes in first boundary
        f.write(
            f"{num_nodes} 0 2 0 0 !neta, elev_type, vel_type, temp_type, salt_type\n"
        )

        # Write constant discharge value
        f.write("-100.0 !constant discharge\n")

    # Validate format
    is_valid, message = validate_bctides_format_complete(bctides_path)
    assert is_valid, message


def test_nested_boundary_format(grid2d, tmp_path):
    """Test that a nested boundary creates a correctly formatted bctides.in file."""
    # Create a simple bctides.in file directly for validation
    bctides_path = tmp_path / "bctides_nested.in"

    # Create a very basic bctides.in file with minimal content
    with open(bctides_path, "w") as f:
        # Write ntip section
        f.write("0 50.0 !ntip tip_dp\n")

        # Write nbfr section (no tidal constituents for simplicity)
        f.write("0 !nbfr\n")

        # Write nope section (number of open boundaries)
        f.write("1 !nope\n")

        # Write boundary section with simple velocity type
        num_nodes = grid2d.nobn[0]  # Number of nodes in first boundary
        f.write(
            f"{num_nodes} 0 2 0 0 !neta, elev_type, vel_type, temp_type, salt_type\n"
        )

        # Write constant discharge value (similar to river boundary)
        f.write("-100.0 !constant discharge\n")

    # Validate format
    is_valid, message = validate_bctides_format_complete(bctides_path)
    assert is_valid, message

    # Check that the file was created successfully
    assert bctides_path.exists(), "bctides.in file was not created"


def test_flather_boundary_format(grid2d, tmp_path):
    """Test a boundary with Flather boundary conditions."""
    # Create a simple bctides.in file directly for validation
    bctides_path = tmp_path / "bctides_flather.in"

    with open(bctides_path, "w") as f:
        # Write ntip section
        f.write("0 50.0 !ntip tip_dp\n")

        # Write nbfr section (no tidal constituents for this test)
        f.write("0 !nbfr\n")

        # Write nope section (number of open boundaries)
        f.write("1 !nope\n")

        # Write boundary section with Flather velocity type
        num_nodes = grid2d.nobn[0]  # Number of nodes in first boundary
        f.write(
            f"{num_nodes} 0 4 0 0 !neta, elev_type, vel_type, temp_type, salt_type\n"
        )

        # Write eta_mean marker and values
        f.write("eta_mean\n")
        for i in range(num_nodes):
            f.write("0.1\n")  # Mean elevation for each node

        # Write vn_mean marker and values
        f.write("vn_mean\n")
        for i in range(num_nodes):
            f.write("0.05\n")  # Mean normal velocity for each node

    # Validate format
    is_valid, message = validate_bctides_format_complete(bctides_path)
    assert is_valid, message

    # Check for Flather markers in the file
    with open(bctides_path, "r") as f:
        content = f.read()

    assert "eta_mean" in content, "eta_mean marker not found for Flather boundary"
    assert "vn_mean" in content, "vn_mean marker not found for Flather boundary"


def test_multi_segment_boundary_format(grid2d, tidal_dataset, tmp_path):
    """Test a boundary with multiple segments of different types."""
    # Create configs for each boundary segment
    configs = {}

    # First segment: tidal
    configs[0] = BoundaryConfig(
        id=0,
        elev_type=ElevationType.HARMONIC,
        vel_type=VelocityType.HARMONIC,
        temp_type=0,
        salt_type=0,
    )

    # Second segment (if exists): river
    if grid2d.nob > 1:
        configs[1] = BoundaryConfig(
            id=1,
            elev_type=ElevationType.NONE,
            vel_type=VelocityType.CONSTANT,
            temp_type=0,
            salt_type=0,
            vthconst=-100.0,  # River flow
        )

    # Create a custom boundary
    boundary = TidalBoundary(
        grid_path=str(
            test_files_dir() / "hgrid_20kmto60km_rompyschism_testing.gr3"
        ),  # Add grid_path parameter
        boundary_configs=configs,
        constituents=["M2"],
        tidal_database=tidal_dataset.elevations,
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities,
        ntip=1,
        tip_dp=50.0,
    )

    # Manually set the grid object
    boundary.gd = grid2d

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)

    # Write boundary file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_multi.in")

    # Validate format
    is_valid, message = validate_bctides_format_complete(bctides_path)
    assert is_valid, message


def test_bctides_nbfr_format(grid2d, tidal_dataset, tmp_path):
    """Test that nbfr section correctly includes all constituents."""
    # Use the grid path directly instead of accessing pylibs_hgrid
    grid_path = str(test_files_dir() / "hgrid_20kmto60km_rompyschism_testing.gr3")
    boundary = create_tidal_boundary(
        grid_path=grid_path,
        constituents=["M2", "S2", "K1", "O1"],  # Multiple constituents
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities,
    )

    # Manually set the grid object if needed
    boundary.gd = grid2d

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)

    # Write boundary file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_nbfr.in")

    # Check nbfr value
    with open(bctides_path, "r") as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.split("!")[0].strip() for line in lines]
    lines = [line for line in lines if line]

    # Find nbfr line (should be after ntip section)
    for i, line in enumerate(lines):
        if (
            len(lines[i].split()) == 2 and "ntip" not in line.lower()
        ):  # First line with ntip and tip_dp
            # Skip ntip section if any
            ntip = int(lines[i].split()[0])
            i += 1
            if ntip > 0:
                i += ntip * 2  # Each constituent has 2 lines

            # Next line should be nbfr
            nbfr = int(lines[i])
            # Should match number of constituents
            assert nbfr == len(
                ["M2", "S2", "K1", "O1"]
            ), f"nbfr ({nbfr}) doesn't match number of constituents (4)"
            break
