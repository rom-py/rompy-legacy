import os
import pytest
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime

from rompy.schism.boundary_core import (
    BoundaryHandler,
    TidalBoundary,  # Backward compatibility alias
    BoundaryConfig,
    ElevationType,
    VelocityType,
    create_tidal_boundary,
)
from rompy.schism.bctides import Bctides
from rompy.schism.grid import SCHISMGrid


def validate_constituent_case_consistency(file_path):
    """Validate that the case of constituent names is consistent throughout the file."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Remove comments and empty lines
    lines = [line.split("!")[0].strip() for line in lines]
    lines = [line for line in lines if line]

    # Initialize dictionary to track case usage
    constituents = {}

    # Process the file to find ntip, nbfr, and constituents
    i = 0

    # Parse ntip
    parts = lines[i].split()
    ntip = int(parts[0])
    i += 1

    # Skip tidal potential constituents
    if ntip > 0:
        for _ in range(ntip):
            constituent = lines[i].strip()
            if constituent not in constituents:
                constituents[constituent] = []
            constituents[constituent].append(i)
            i += 1

            # Skip parameters line
            i += 1

    # Parse nbfr
    nbfr = int(lines[i])
    i += 1

    # Parse frequency constituents
    for _ in range(nbfr):
        constituent = lines[i].strip()
        if constituent not in constituents:
            constituents[constituent] = []
        constituents[constituent].append(i)
        i += 1

        # Skip frequency line
        i += 1

    # Parse nope (number of open boundary segments)
    nope = int(lines[i])
    i += 1

    # Parse each boundary segment
    for j in range(nope):
        # Skip boundary flags line
        i += 1

        # Look for constituent lines in this segment (for elevation and velocity)
        while i < len(lines):
            line = lines[i]

            # Check if this line is a constituent name
            if line.upper() in [const.upper() for const in constituents.keys()]:
                # Find the matching constituent (case-insensitive)
                for const in constituents.keys():
                    if line.upper() == const.upper():
                        constituents[const].append(i)
                        break

            i += 1

            # Check if we've reached the end of the boundary segments
            if i < len(lines) and (
                lines[i].endswith("!ncbn") or lines[i].endswith("!nfluxf")
            ):
                break

    # For each constituent, check if the case is consistent
    inconsistent = []
    for const, positions in constituents.items():
        # Extract the actual strings at these positions
        instances = [lines[pos] for pos in positions]

        # Check if all instances have the same case
        if len(set(instances)) > 1:
            inconsistent.append(f"{const}: {instances}")

    return inconsistent


def test_tidal_boundary_constituent_consistency(
    grid2d, tidal_dataset, mock_tidal_data, monkeypatch
):
    """Test that constituent names in the bctides.in file have consistent case using real grid."""
    # Use the mock_tidal_data function for interpolation
    monkeypatch.setattr(Bctides, "_interpolate_tidal_data", mock_tidal_data)

    # Create boundary configs for tidal boundary
    configs = {}
    configs[0] = BoundaryConfig(
        id=0,
        elev_type=ElevationType.HARMONIC,
        vel_type=VelocityType.HARMONIC,
        temp_type=0,
        salt_type=0,
    )

    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Get grid path from grid2d fixture
        grid_path = str(grid2d.hgrid.source)

        # Create a TidalBoundary instance with only the constituents in the test dataset (M2, S2, N2)
        boundary = TidalBoundary(
            grid_path=grid_path, boundary_configs=configs, tidal_data=tidal_dataset
        )

        # Set run parameters
        boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)

        # Write the boundary file
        bctides_path = boundary.write_boundary_file(tmp_path)

        # Check for case consistency
        inconsistencies = validate_constituent_case_consistency(bctides_path)

        # Debug output
        with open(bctides_path, "r") as f:
            content = f.read()
            print(f"\nBCTIDES CONTENT:\n{content}\n")

        # There should be no inconsistencies
        assert (
            len(inconsistencies) == 0
        ), f"Inconsistent constituent cases: {inconsistencies}"

        # Check each constituent for case consistency
        for constituent in ["M2", "S2"]:
            upper_count = content.count(constituent)
            lower_count = content.count(constituent.lower())

            # Either all uppercase or all lowercase is acceptable, but mixing is not
            assert upper_count == 0 or lower_count == 0, f"Mixed case for {constituent}"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_create_tidal_boundary_wrapper(
    grid2d, tidal_dataset, mock_tidal_data, monkeypatch
):
    """Test the create_tidal_boundary wrapper function with real grid and data."""
    # Use the mock_tidal_data function for interpolation
    monkeypatch.setattr(Bctides, "_interpolate_tidal_data", mock_tidal_data)

    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Get grid path from grid2d fixture
        grid_path = str(grid2d.hgrid.source)

        # Create the boundary with the wrapper function
        boundary = create_tidal_boundary(
            grid_path=grid_path,
            tidal_database=tidal_dataset.tidal_database,
            constituents=tidal_dataset.constituents,
            tidal_model=tidal_dataset.tidal_model,
        )

        # Set run parameters
        boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)

        # Write the boundary file
        bctides_path = boundary.write_boundary_file(tmp_path)

        # Check for case consistency
        inconsistencies = validate_constituent_case_consistency(bctides_path)
        assert (
            len(inconsistencies) == 0
        ), f"Inconsistent constituent cases: {inconsistencies}"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_tidal_boundary_with_different_grids(
    request, grid2d, grid3d, tidal_dataset, mock_tidal_data, monkeypatch
):
    """Test tidal boundary with different grid types."""
    # Use the mock_tidal_data function for interpolation
    monkeypatch.setattr(Bctides, "_interpolate_tidal_data", mock_tidal_data)

    # Test with each grid type
    for grid_fixture in [grid2d, grid3d]:
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Get grid path
            grid_path = str(grid_fixture.hgrid.source)

            # Create the boundary
            boundary = create_tidal_boundary(
                grid_path=grid_path,
                tidal_database=tidal_dataset.tidal_database,
                constituents=tidal_dataset.constituents,
                tidal_model=tidal_dataset.tidal_model,
            )

            # Set run parameters
            boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)

            # Write the boundary file
            bctides_path = boundary.write_boundary_file(tmp_path)

            # Check for case consistency
            inconsistencies = validate_constituent_case_consistency(bctides_path)
            assert (
                len(inconsistencies) == 0
            ), f"Inconsistent constituent cases: {inconsistencies}"

        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def test_case_consistency(grid2d, tidal_dataset, mock_tidal_data, monkeypatch):
    """Test that constituent names in the bctides.in file have consistent case."""
    # Use the mock_tidal_data function for interpolation
    monkeypatch.setattr(Bctides, "_interpolate_tidal_data", mock_tidal_data)

    # Create a temporary file for output
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Create a boundary
        grid_path = str(grid2d.hgrid.source)
        boundary = create_tidal_boundary(
            grid_path=grid_path,
            tidal_database=tidal_dataset.tidal_database,
            constituents=tidal_dataset.constituents,
            tidal_model=tidal_dataset.tidal_model,
        )

        # Write the boundary file
        boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)
        boundary.write_boundary_file(tmp_path)

        # Check for case consistency
        inconsistencies = validate_constituent_case_consistency(tmp_path)
        assert (
            len(inconsistencies) == 0
        ), f"Found case inconsistencies: {inconsistencies}"

        # Check specifically for each constituent
        with open(tmp_path, "r") as f:
            content = f.read()

        for constituent in ["M2", "S2"]:
            # All occurrences should be same case (either all uppercase or all lowercase)
            upper_count = content.count(constituent)
            lower_count = content.count(constituent.lower())
            assert (
                upper_count == 0 or lower_count == 0
            ), f"Mixed case found for {constituent}"

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
