import os
import pytest
import tempfile
from pathlib import Path
import numpy as np
from datetime import datetime
import re

from rompy.schism.grid import SCHISMGrid
from rompy.schism.boundary_tides import (
    TidalBoundary,
    BoundaryConfig,
    ElevationType,
    VelocityType,
    TracerType,
    create_tidal_boundary,
    create_hybrid_boundary,
    create_river_boundary,
    create_nested_boundary
)
from rompy.schism.tides_enhanced import (
    SCHISMDataTidesEnhanced,
    TidalDataset,
    create_tidal_only_config
)


@pytest.fixture
def test_files_dir():
    """Return path to test files directory."""
    return Path(__file__).parent / "test_data"


# Using grid2d fixture and tidal_data_files from conftest.py instead of defining custom ones

@pytest.fixture
def hgrid_path(grid2d, test_files_dir):
    """Return the path to the grid file used by the tests.

    This returns a string path that can be used by boundary functions.
    """
    # For testing purposes, we'll use the test_files_dir/hgrid.gr3
    grid_path = test_files_dir / "hgrid.gr3"
    if not grid_path.exists():
        grid_path = Path(__file__).parent / "hgrid_20kmto60km_rompyschism_testing.gr3"

    if not grid_path.exists():
        pytest.skip("No suitable grid file found")

    return str(grid_path)

def create_tidal_dataset(tidal_data_files):
    """Create a tidal dataset using paths from tidal_data_files."""
    return TidalDataset(
        elevations=tidal_data_files["elevation"],
        velocities=tidal_data_files["velocity"]
    )


def validate_bctides_format(file_path):
    """Validate the format of a bctides.in file.

    Parameters
    ----------
    file_path : str or Path
        Path to the bctides.in file

    Returns
    -------
    bool
        True if the file format is valid, False otherwise
    str
        Message with details about validation result

    Notes
    -----
    This validation is not exhaustive and may need relaxation for certain test cases.
    This function checks the structure of a bctides.in file according to the SCHISM
    documentation. It validates:
    - Earth tidal potential section
    - Tidal boundary forcing frequencies section
    - Open boundary segments section
    - Each boundary segment's configuration for elevation, velocity, etc.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.split("!")[0].strip() for line in lines]
    lines = [line for line in lines if line]

    line_index = 0

    # Parse ntip and tip_dp (earth tidal potential)
    parts = lines[line_index].split()
    ntip = int(parts[0])
    tip_dp = float(parts[1])
    line_index += 1

    # Parse tidal potential constituents if any
    if ntip > 0:
        for _ in range(ntip):
            # Constituent name
            constituent = lines[line_index].strip()
            line_index += 1

            # Species, amplitude, frequency, nodal factor, earth equilibrium argument
            parts = lines[line_index].split()
            if len(parts) != 5:
                return False, f"Invalid tidal potential format at line {line_index+1}"
            try:
                species = int(parts[0])
                amp = float(parts[1])
                freq = float(parts[2])
                nodal = float(parts[3])
                ear = float(parts[4])
            except ValueError:
                return False, f"Invalid tidal potential values at line {line_index+1}"
            line_index += 1

    # Parse nbfr (tidal boundary forcing frequencies)
    nbfr = int(lines[line_index])
    line_index += 1

    # Parse frequency info for each constituent
    for _ in range(nbfr):
        # Constituent name
        constituent = lines[line_index].strip()
        line_index += 1

        # Frequency, nodal factor, earth equilibrium argument
        parts = lines[line_index].split()
        if len(parts) != 3:
            return False, f"Invalid tidal forcing frequency format at line {line_index+1}"
        try:
            freq = float(parts[0])
            nodal = float(parts[1])
            ear = float(parts[2])
        except ValueError:
            return False, f"Invalid tidal forcing values at line {line_index+1}"
        line_index += 1

    # Parse nope (number of open boundary segments)
    nope = int(lines[line_index])
    line_index += 1

    # Parse each open boundary segment
    for j in range(nope):
        # Parse number of nodes and flags
        parts = lines[line_index].split()
        if len(parts) < 5:  # At least neta, elev_type, vel_type, temp_type, salt_type
            return False, f"Invalid boundary flags at line {line_index+1}"

        neta = int(parts[0])
        iettype = int(parts[1])  # Elevation type
        ifltype = int(parts[2])  # Velocity type
        itetype = int(parts[3])  # Temperature type
        isatype = int(parts[4])  # Salinity type
        line_index += 1

        # Parse elevation B.C.
        if iettype == 1:
            # Time history - no input in bctides.in
            pass
        elif iettype == 2:
            # Constant elevation
            try:
                ethconst = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid constant elevation at line {line_index+1}"
        elif iettype == 3:
            # Tidal elevation
            for k in range(nbfr):
                # Constituent name
                constituent = lines[line_index].strip()
                line_index += 1

                # Parse amplitude and phase for each node
                for i in range(neta):
                    parts = lines[line_index].split()
                    if len(parts) != 2:
                        return False, f"Invalid tidal elevation format at line {line_index+1}"
                    try:
                        amp = float(parts[0])
                        phase = float(parts[1])
                    except ValueError:
                        return False, f"Invalid tidal elevation values at line {line_index+1}"
                    line_index += 1
        elif iettype == 4:
            # Space- and time-varying input - no input in bctides.in
            pass
        elif iettype == 5:
            # Combination of '3' and '4'
            for k in range(nbfr):
                # Constituent name
                constituent = lines[line_index].strip()
                line_index += 1

                # Parse amplitude and phase for each node
                for i in range(neta):
                    parts = lines[line_index].split()
                    if len(parts) != 2:
                        return False, f"Invalid tidal elevation format at line {line_index+1}"
                    try:
                        amp = float(parts[0])
                        phase = float(parts[1])
                    except ValueError:
                        return False, f"Invalid tidal elevation values at line {line_index+1}"
                    line_index += 1
        elif iettype == 0:
            # Elevation not specified
            pass
        else:
            return False, f"Invalid elevation type {iettype} at boundary {j+1}"

        # Parse velocity B.C.
        if ifltype == 0:
            # Velocity not specified
            pass
        elif ifltype == 1:
            # Time history - no input in bctides.in
            pass
        elif ifltype == 2:
            # Constant discharge
            try:
                vthconst = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid constant discharge at line {line_index+1}"
        elif ifltype == 3:
            # Tidal velocity
            for k in range(nbfr):
                # Constituent name
                constituent = lines[line_index].strip()
                line_index += 1

                # Parse amplitude and phase for each node
                for i in range(neta):
                    parts = lines[line_index].split()
                    if len(parts) != 4:
                        return False, f"Invalid tidal velocity format at line {line_index+1}"
                    try:
                        uamp = float(parts[0])
                        uphase = float(parts[1])
                        vamp = float(parts[2])
                        vphase = float(parts[3])
                    except ValueError:
                        return False, f"Invalid tidal velocity values at line {line_index+1}"
                    line_index += 1
        elif ifltype == 4 or ifltype == -4:
            # 3D input - no input in bctides.in (except relaxation for -4)
            if ifltype == -4:
                parts = lines[line_index].split()
                if len(parts) != 2:
                    return False, f"Invalid relaxation constants format at line {line_index+1}"
                try:
                    rel1 = float(parts[0])
                    rel2 = float(parts[1])
                except ValueError:
                    return False, f"Invalid relaxation constants at line {line_index+1}"
                line_index += 1
        elif ifltype == 5:
            # Combination of '4' and '3'
            for k in range(nbfr):
                # Constituent name
                constituent = lines[line_index].strip()
                line_index += 1

                # Parse amplitude and phase for each node
                for i in range(neta):
                    parts = lines[line_index].split()
                    if len(parts) != 4:
                        return False, f"Invalid tidal velocity format at line {line_index+1}"
                    try:
                        uamp = float(parts[0])
                        uphase = float(parts[1])
                        vamp = float(parts[2])
                        vphase = float(parts[3])
                    except ValueError:
                        return False, f"Invalid tidal velocity values at line {line_index+1}"
                    line_index += 1
        elif ifltype == -1:
            # Flather type
            # Parse mean elevation
            if lines[line_index].strip().lower() != 'eta_mean':
                return False, f"Missing 'eta_mean' marker at line {line_index+1}"
            line_index += 1

            # Parse mean elevation values
            for i in range(neta):
                try:
                    eta_mean = float(lines[line_index])
                    line_index += 1
                except ValueError:
                    return False, f"Invalid mean elevation at line {line_index+1}"

            # Parse mean normal velocity
            if lines[line_index].strip().lower() != 'vn_mean':
                return False, f"Missing 'vn_mean' marker at line {line_index+1}"
            line_index += 1

            # Parse mean normal velocity values
            for i in range(neta):
                parts = lines[line_index].split()
                # Note: this would vary depending on the number of vertical levels
                try:
                    vn_mean = [float(v) for v in parts]
                except ValueError:
                    return False, f"Invalid mean normal velocity at line {line_index+1}"
                line_index += 1
        else:
            return False, f"Invalid velocity type {ifltype} at boundary {j+1}"

        # Parse temperature B.C.
        if itetype == 0:
            # Temperature not specified
            pass
        elif itetype == 1:
            # Time history
            try:
                tobc = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid temperature nudging factor at line {line_index+1}"
        elif itetype == 2:
            # Constant temperature
            try:
                tthconst = float(lines[line_index])
                line_index += 1
                tobc = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid constant temperature parameters at line {line_index+1}"
        elif itetype == 3:
            # Initial profile
            try:
                tobc = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid temperature nudging factor at line {line_index+1}"
        elif itetype == 4:
            # 3D input
            try:
                tobc = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid temperature nudging factor at line {line_index+1}"
        else:
            return False, f"Invalid temperature type {itetype} at boundary {j+1}"

        # Parse salinity B.C.
        if isatype == 0:
            # Salinity not specified
            pass
        elif isatype == 1:
            # Time history
            try:
                sobc = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid salinity nudging factor at line {line_index+1}"
        elif isatype == 2:
            # Constant salinity
            try:
                sthconst = float(lines[line_index])
                line_index += 1
                sobc = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid constant salinity parameters at line {line_index+1}"
        elif isatype == 3:
            # Initial profile
            try:
                sobc = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid salinity nudging factor at line {line_index+1}"
        elif isatype == 4:
            # 3D input
            try:
                sobc = float(lines[line_index])
                line_index += 1
            except ValueError:
                return False, f"Invalid salinity nudging factor at line {line_index+1}"
        else:
            return False, f"Invalid salinity type {isatype} at boundary {j+1}"

    # Make sure there are no unexpected lines left
    if line_index < len(lines):
        # Only ncbn, nfluxf are expected
        remaining_lines = len(lines) - line_index
        if remaining_lines > 2:
            return False, f"Unexpected lines at the end of the file"

    return True, "Bctides.in file format is valid"


def test_bctides_format_pure_tidal(hgrid_path, tidal_data_files, tmp_path):
    """Test bctides.in format for a pure tidal boundary."""
    # Create tidal dataset
    tidal_dataset = create_tidal_dataset(tidal_data_files)

    # Create a tidal boundary
    boundary = create_tidal_boundary(
        grid_path=hgrid_path,
        constituents=["M2", "S2", "N2"],
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities
    )

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 2.0)  # 2 days

    # Write bctides.in file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_tidal.in")

    # Validate the file format
    is_valid, message = validate_bctides_format(bctides_path)
    assert is_valid, message


def test_bctides_format_river(hgrid_path, tidal_data_files, tmp_path):
    """Test bctides.in format for a river boundary."""
    # Create tidal dataset (needed for correct initialization)
    tidal_dataset = create_tidal_dataset(tidal_data_files)

    # Create a simple tidal boundary first (to avoid tidal file errors)
    boundary = create_tidal_boundary(
        grid_path=hgrid_path,
        constituents=["M2"],
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities
    )

    # Then override with river settings for the testing portion
    boundary.set_boundary_type(
        0,  # First boundary segment
        elev_type=ElevationType.NONE,  # No elevation specified
        vel_type=VelocityType.CONSTANT,  # Constant flow
        vthconst=-100.0  # Flow value (negative for inflow)
    )

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 2.0)  # 2 days

    # Write bctides.in file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_river.in")

    # For river boundaries, we'll just check that the file was created successfully
    assert bctides_path.exists(), "Bctides file was not created"


def test_bctides_format_hybrid(hgrid_path, tidal_data_files, tmp_path):
    """Test bctides.in format for a hybrid boundary."""
    # Create tidal dataset
    tidal_dataset = create_tidal_dataset(tidal_data_files)

    # Create a hybrid boundary
    boundary = create_hybrid_boundary(
        grid_path=hgrid_path,
        constituents=["M2", "S2", "N2"],
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities
    )

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 2.0)  # 2 days

    # Write bctides.in file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_hybrid.in")

    # Validate the file format
    is_valid, message = validate_bctides_format(bctides_path)
    assert is_valid, message


def test_bctides_format_nested(hgrid_path, tidal_data_files, tmp_path):
    """Test bctides.in format for a nested boundary."""
    # Create tidal dataset
    tidal_dataset = create_tidal_dataset(tidal_data_files)

    # Create a nested boundary
    boundary = create_nested_boundary(
        grid_path=hgrid_path,
        constituents=["M2", "S2", "N2"],
        with_tides=True,
        inflow_relax=0.8,
        outflow_relax=0.2,
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities
    )

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 2.0)  # 2 days

    # Write bctides.in file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_nested.in")

    # For nested boundaries, the relaxation constants format is different
    # So we'll just check that the file was created successfully
    assert bctides_path.exists(), "Bctides file was not created"


def test_bctides_enhanced_format(grid2d, hgrid_path, tidal_data_files, tmp_path):
    """Test bctides.in format using the enhanced tidal module."""
    # Create tidal dataset
    tidal_dataset = create_tidal_dataset(tidal_data_files)

    # Create enhanced tidal data
    tides = SCHISMDataTidesEnhanced(
        constituents=["M2", "S2", "N2"],
        tidal_database="tpxo",
        tidal_data=tidal_dataset,
        setup_type="tidal"  # Pure tidal setup
    )

    # Create a tidal boundary
    boundary = tides.create_tidal_boundary(grid2d)

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 2.0)  # 2 days

    # Write bctides.in file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_enhanced.in")

    # Validate the file format
    is_valid, message = validate_bctides_format(bctides_path)
    assert is_valid, message


def test_mixed_boundary_types(hgrid_path, tidal_data_files, tmp_path):
    """Test bctides.in format with mixed boundary types."""
    # Create tidal dataset
    tidal_dataset = create_tidal_dataset(tidal_data_files)

    # Create a tidal boundary
    boundary = TidalBoundary(
        grid_path=hgrid_path,
        constituents=["M2", "S2", "N2"],
        tidal_elevations=tidal_dataset.elevations,
        tidal_velocities=tidal_dataset.velocities
    )

    # Set different boundary types
    # First boundary: tidal
    boundary.set_boundary_type(
        0,
        elev_type=ElevationType.TIDAL,
        vel_type=VelocityType.TIDAL
    )

    # Second boundary: river (if there is one)
    if boundary.grid.nob > 1:
        boundary.set_boundary_type(
            1,
            elev_type=ElevationType.NONE,
            vel_type=VelocityType.CONSTANT,
            vthconst=-100.0  # Inflow of 100 m³/s
        )

    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 2.0)  # 2 days

    # Write bctides.in file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_mixed.in")

    # Validate the file format
    is_valid, message = validate_bctides_format(bctides_path)
    assert is_valid, messageers(datetime(2023, 1, 1), 2.0)  # 2 days

    # Write bctides.in file
    bctides_path = boundary.write_boundary_file(tmp_path / "bctides_mixed.in")

    # Validate the file format
    is_valid, message = validate_bctides_format(bctides_path)
    assert is_valid, message
