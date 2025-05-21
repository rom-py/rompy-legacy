import os
import sys
import pytest
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from rompy.schism.boundary_tides import (
    TidalBoundary, BoundaryConfig, ElevationType, VelocityType,
    create_tidal_boundary
)
from rompy.schism.bctides import Bctides

# Directory for test data
def test_files_dir():
    """Get the directory containing test files."""
    return Path(os.path.dirname(os.path.abspath(__file__)))

class MockGrid:
    """Mock grid class for testing."""
    def __init__(self):
        # Basic grid properties
        self.ne = 100  # Number of elements
        self.np = 60   # Number of nodes
        self.nob = 1   # Number of open boundaries
        self.nobn = np.array([10], dtype=np.int32)  # Number of nodes per boundary
        self.iobn = [np.array(range(10), dtype=np.int32)]  # Node indices for each boundary
        self.x = np.array([float(i) for i in range(60)])  # Longitudes
        self.y = np.array([float(i) for i in range(60)])  # Latitudes
        
    @property
    def pylibs_hgrid(self):
        """Return self as mock pylibs hgrid."""
        return self

def validate_constituent_case_consistency(file_path):
    """Validate that the case of constituent names is consistent throughout the file."""
    with open(file_path, "r") as f:
        content = f.read()
    
    # Initialize dictionary to track case usage
    constituents = {}
    
    # Extract all constituent names
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    # Remove comments and empty lines
    lines = [line.split("!")[0].strip() for line in lines]
    lines = [line for line in lines if line]
    
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
            if i < len(lines) and (lines[i].endswith("!ncbn") or lines[i].endswith("!nfluxf")):
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

def test_tidal_boundary_constituent_consistency():
    """Test that constituent names in the bctides.in file have consistent case."""
    # Mock the interpolate method
    mock_interpolate = MagicMock(return_value=np.array([[0.5, 45.0] for _ in range(10)]))
    """Test that constituent names in the bctides.in file have consistent case."""
    # Configure mock
    mock_interpolate.return_value = np.array([[0.5, 45.0] for _ in range(10)])
    
    # Create a mock grid
    grid = MockGrid()
    
    # Create boundary configs for tidal boundary
    configs = {}
    configs[0] = BoundaryConfig(
        id=0,
        elev_type=ElevationType.TIDAL,
        vel_type=VelocityType.TIDAL,
        temp_type=0,
        salt_type=0
    )
    
    # Create a TidalBoundary instance
    boundary = TidalBoundary(
        grid_path=str(test_files_dir() / "dummy_path.gr3"),
        boundary_configs=configs,
        constituents=["M2", "S2", "K1", "O1"],
        tidal_database=None
    )
    
    # Replace the grid
    boundary.grid = grid
    
    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # Write the boundary file
        bctides_path = boundary.write_boundary_file(tmp_path)
        
        # Check for case consistency
        inconsistencies = validate_constituent_case_consistency(bctides_path)
        
        # Print the file contents for debugging
        print("\n==== BCTIDES FILE CONTENTS ====")
        with open(bctides_path, "r") as f:
            print(f.read())
        print("==== END OF FILE CONTENTS ====\n")
        
        # There should be no inconsistencies
        assert len(inconsistencies) == 0, f"Inconsistent constituent cases: {inconsistencies}"
        
        # Count occurrences of upper and lowercase constituent names
        with open(bctides_path, "r") as f:
            content = f.read()
        
        for constituent in ["M2", "S2", "K1", "O1"]:
            upper_count = content.count(constituent)
            lower_count = content.count(constituent.lower())
            print(f"Constituent {constituent}: {upper_count} uppercase, {lower_count} lowercase")
            
            # Either all uppercase or all lowercase is acceptable, but mixing is not
            assert upper_count == 0 or lower_count == 0, f"Mixed case for {constituent}"
    
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def test_create_tidal_boundary_wrapper():
    """Test the create_tidal_boundary wrapper function."""
    # Mock the interpolate method
    mock_interpolate = MagicMock(return_value=np.array([[0.5, 45.0] for _ in range(10)]))
    """Test the create_tidal_boundary wrapper function."""
    # Configure mock
    mock_interpolate.return_value = np.array([[0.5, 45.0] for _ in range(10)])
    
    # Create a mock grid
    grid = MockGrid()
    
    # Mock grid_path to use
    grid_path = test_files_dir() / "dummy_path.gr3"
    
    # Create a boundary with the wrapper function
    with patch('pylib.read_schism_hgrid', return_value=grid):
        boundary = create_tidal_boundary(
            grid_path=grid_path,
            constituents=["M2", "S2"],
            tidal_elevations=None,
            tidal_velocities=None
        )
    
    # Set run parameters
    boundary.set_run_parameters(datetime(2023, 1, 1), 5.0)
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # Write the boundary file
        bctides_path = boundary.write_boundary_file(tmp_path)
        
        # Check for case consistency
        inconsistencies = validate_constituent_case_consistency(bctides_path)
        assert len(inconsistencies) == 0, f"Inconsistent constituent cases: {inconsistencies}"
    
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def fix_bctides_case_issue():
    """
    Create and apply a patch to fix the case consistency issue in Bctides.
    
    This function modifies the Bctides.write_bctides method to maintain consistent
    case for constituent names throughout the bctides.in file.
    """
    # Original method
    original_write_bctides = Bctides.write_bctides
    
    def patched_write_bctides(self, output_file):
        """Patched version of write_bctides that maintains consistent case."""
        # Ensure we have start_time and rnday
        if not self._start_time or self._rnday is None:
            raise ValueError(
                "start_time and rnday must be set before calling write_bctides"
            )

        # Get tidal factors
        self._get_tidal_factors()

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Writing bctides.in to {output_file}")
        
        with open(output_file, "w") as f:
            # Write header with date information
            if isinstance(self._start_time, datetime):
                f.write(
                    f"!{self._start_time.month:02d}/{self._start_time.day:02d}/{self._start_time.year:4d} "
                    f"{self._start_time.hour:02d}:00:00 UTC\n"
                )
            else:
                # Assume it's a list [year, month, day, hour]
                year, month, day, hour = self._start_time
                f.write(f"!{month:02d}/{day:02d}/{year:4d} {hour:02d}:00:00 UTC\n")

            # Write tidal potential information
            if self.ntip > 0:
                f.write(
                    f" {len(self.tnames)} {self.cutoff_depth:.3f} !number of earth tidal potential, "
                    f"cut-off depth for applying tidal potential\n"
                )

                # Write each constituent's potential information
                for i, tname in enumerate(self.tnames):
                    # Use original case
                    f.write(f"{tname}\n")

                    # Determine species type based on constituent name
                    species_type = 2  # Default to semi-diurnal
                    if tname.upper() in ["O1", "K1", "P1", "Q1"]:
                        species_type = 1  # Diurnal
                    elif tname.upper() in ["MM", "MF"]:
                        species_type = 0  # Long period

                    f.write(
                        f"{species_type} {self.amp[i]:<.6f} {self.freq[i]:<.9e} "
                        f"{self.nodal[i]:7.5f} {self.tear[i]:.2f}\n"
                    )
            else:
                # No earth tidal potential
                f.write(
                    " 0 50.000 !number of earth tidal potential, cut-off depth for applying tidal potential\n"
                )

            # Write frequency info
            n_constituents = len(self.tnames) + (1 if len(self.ethconst) > 0 else 0)
            f.write(f"{n_constituents} !nbfr\n")

            # Write Z0 (mean sea level) if ethconst provided
            if len(self.ethconst) > 0:
                f.write("Z0\n  0.0 1.0 0.0\n")

            # Write frequency info for each constituent
            for i, tname in enumerate(self.tnames):
                # Use original case, not lowercase
                f.write(
                    f"{tname}\n  {self.freq[i]:<.9e} {self.nodal[i]:7.5f} {self.tear[i]:.2f}\n"
                )

            # Write open boundary information
            f.write(f"{self.gd.nob} !nope\n")

            # For each open boundary
            for ibnd in range(self.gd.nob):
                # Get boundary nodes
                nodes = self.gd.iobn[ibnd]
                num_nodes = self.gd.nobn[ibnd]

                # Write boundary flags (ensure we have enough flags defined)
                bnd_flags = (
                    self.flags[ibnd] if ibnd < len(self.flags) else self.flags[0]
                )
                flag_str = " ".join(map(str, bnd_flags))
                f.write(f"{num_nodes} {flag_str} !ocean\n")

                # Get boundary coordinates
                lons = self.gd.x[nodes]
                lats = self.gd.y[nodes]

                # Write elevation boundary conditions

                # First, handle constant elevation if provided
                if len(self.ethconst) > 0:
                    f.write("Z0\n")
                    eth_val = self.ethconst[ibnd] if ibnd < len(self.ethconst) else 0.0
                    for n in range(num_nodes):
                        f.write(f"{eth_val} 0.0\n")

                # Then write tidal constituents
                for i, tname in enumerate(self.tnames):
                    logger.info(f"Processing tide {tname} for boundary {ibnd+1}")

                    # Interpolate tidal data for this constituent
                    try:
                        tidal_data = self._interpolate_tidal_data(
                            lons, lats, tname, "h"
                        )

                        # Write header for constituent - use original case, not lowercase
                        f.write(f"{tname}\n")

                        # Write amplitude and phase for each node
                        for n in range(num_nodes):
                            f.write(f"{tidal_data[n,0]:8.6f} {tidal_data[n,1]:.6f}\n")
                    except Exception as e:
                        # Log error but continue with other constituents
                        logger.error(
                            f"Error processing tide {tname} for boundary {ibnd+1}: {e}"
                        )
                        raise

                # Write velocity boundary conditions

                # First, handle constant velocity if provided
                if len(self.vthconst) > 0:
                    f.write("Z0\n")
                    vth_val = self.vthconst[ibnd] if ibnd < len(self.vthconst) else 0.0
                    for n in range(num_nodes):
                        f.write("0.0 0.0 0.0 0.0\n")

                # Then write tidal constituents
                for i, tname in enumerate(self.tnames):
                    # Write header for constituent - use original case, not lowercase
                    f.write(f"{tname}\n")

                    # Try to interpolate velocity data
                    if self.tidal_velocities and os.path.exists(self.tidal_velocities):
                        vel_data = self._interpolate_tidal_data(lons, lats, tname, "uv")

                        # Write u/v amplitude and phase for each node
                        for n in range(num_nodes):
                            f.write(
                                f"{vel_data[n,0]:8.6f} {vel_data[n,1]:.6f} "
                                f"{vel_data[n,2]:8.6f} {vel_data[n,3]:.6f}\n"
                            )
                    else:
                        # If no velocity file, use zeros to ensure file structure is complete
                        logger.warning(
                            f"No velocity data available for {tname}, using zeros"
                        )
                        for n in range(num_nodes):
                            f.write("0.0 0.0 0.0 0.0\n")

            # Add remaining sections
            f.write("0 !ncbn: total # of flow bnd segments with discharge\n")
            f.write("0 !nfluxf: total # of flux boundary segments\n")

        logger.info(f"Successfully wrote bctides.in to {output_file}")
        return output_file
    
    # Return the patched method
    return patched_write_bctides

if __name__ == "__main__":
    # Apply the patch
    Bctides.write_bctides = fix_bctides_case_issue()
    
    # Run the tests directly
    test_tidal_boundary_constituent_consistency()
    test_create_tidal_boundary_wrapper()
    print("All tests passed!")