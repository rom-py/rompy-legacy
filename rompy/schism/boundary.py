"""
Boundary data handling for SCHISM.

This module provides classes for handling SCHISM boundary data,
particularly focusing on boundary data using PyLibs functionality directly.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import xarray as xr

# Ensure PyLibs is in path
sys.path.append("/home/tdurrant/source/pylibs")

logger = logging.getLogger(__name__)

# Import PyLibs functions directly
from pylib import *
from src.schism_file import (
    read_schism_hgrid,
    read_schism_bpfile,
    schism_grid,
    read_schism_vgrid,
    compute_zcor,
)


class BoundaryData:
    """Handler for SCHISM boundary data using PyLibs directly."""

    def __init__(
        self,
        grid_path: Union[str, Path],
        source_data: Union[xr.Dataset, pd.DataFrame, Dict[str, Any]] = None,
        variables: List[str] = None,
        boundary_indexes: Optional[List[int]] = None,
    ):
        """Initialize the boundary data handler.

        Parameters
        ----------
        grid_path : str or Path
            Path to the SCHISM grid file
        source_data : xr.Dataset, pd.DataFrame, or dict, optional
            Source data for boundary conditions
        variables : list of str, optional
            Variables to extract from source_data
        boundary_indexes : list of int, optional
            Indexes of boundaries to process, if None all open boundaries are used
        """
        self.grid_path = Path(grid_path) if grid_path is not None else None
        self.variables = variables or []
        self.boundary_indexes = boundary_indexes
        self.source_data = source_data

        # Load grid using PyLibs if path is provided
        if self.grid_path is not None and os.path.exists(self.grid_path):
            self.grid = read_schism_hgrid(str(self.grid_path))
        else:
            self.grid = None

    def set_source_data(self, source_data, variables=None):
        """Set the source data for boundary conditions.

        Parameters
        ----------
        source_data : xr.Dataset, pd.DataFrame, or dict
            Source data for boundary conditions
        variables : list of str, optional
            Variables to extract from source_data
        """
        self.source_data = source_data
        if variables:
            self.variables = variables

    def extract_boundary_points(self):
        """Extract boundary points from the SCHISM grid.

        Returns
        -------
        dict
            Dictionary with boundary indexes as keys and boundary point coordinates as values
        """
        if self.grid is None:
            logger.error("Grid is not loaded, cannot extract boundary points")
            return {}

        # Make sure boundaries are computed
        if hasattr(self.grid, "compute_bnd"):
            self.grid.compute_bnd()

        # Get open boundaries
        boundaries = {}

        if hasattr(self.grid, "nob") and self.grid.nob > 0:
            # If specific boundary indexes were requested, filter to those
            if self.boundary_indexes is not None:
                for i in self.boundary_indexes:
                    if i < self.grid.nob:
                        boundaries[i] = {
                            "nodes": self.grid.iobn[i],
                            "coordinates": self._get_coordinates_for_nodes(
                                self.grid.iobn[i]
                            ),
                        }
            else:
                # Otherwise, include all open boundaries
                for i in range(self.grid.nob):
                    boundaries[i] = {
                        "nodes": self.grid.iobn[i],
                        "coordinates": self._get_coordinates_for_nodes(
                            self.grid.iobn[i]
                        ),
                    }

        return boundaries

    def _get_coordinates_for_nodes(self, nodes):
        """Get coordinates for a set of nodes.

        Parameters
        ----------
        nodes : list or array
            Node indexes

        Returns
        -------
        dict
            Dictionary with 'x' and 'y' keys for coordinates
        """
        if self.grid is None:
            return {"x": [], "y": []}

        return {"x": self.grid.x[nodes], "y": self.grid.y[nodes]}

    def interpolate_to_boundary(self, data, boundary_coords, method="linear"):
        """Interpolate data to boundary points.

        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            Data to interpolate
        boundary_coords : dict
            Dictionary with 'x' and 'y' keys for target coordinates
        method : str
            Interpolation method, e.g., 'linear', 'nearest'

        Returns
        -------
        xr.Dataset or xr.DataArray
            Interpolated data at boundary points
        """
        # Simplified implementation using native PyLibs functions
        # This would be customized based on the specific needs and data formats

        # Example: interpolate using PyLibs' interpolate function
        result = None

        # Implementation would depend on the specific structure of input data
        # and PyLibs' interpolation capabilities

        return result

    def create_boundary_dataset(self, time_range=None):
        """Create a boundary dataset for SCHISM.

        Parameters
        ----------
        time_range : tuple or list, optional
            Time range (start, end) to filter the data

        Returns
        -------
        xr.Dataset
            Dataset formatted for SCHISM boundary input
        """
        # Extract boundary points
        boundaries = self.extract_boundary_points()
        if not boundaries:
            logger.error("No boundary points found, cannot create boundary dataset")
            return None

        # Create empty dataset
        ds = xr.Dataset()

        # Process source data and interpolate to boundary points
        # The actual implementation would depend on:
        # 1. Format of source_data (xr.Dataset, pd.DataFrame, etc.)
        # 2. Structure of the boundary data needed by the model
        # 3. Interpolation approach (nearest, linear, etc.)

        # If time_range is provided, filter the data

        # For each variable, interpolate to the boundary points

        # Combine into a single dataset with the appropriate structure

        return ds

    def write_boundary_file(self, output_path, time_range=None):
        """Write boundary data to a file for SCHISM.

        Parameters
        ----------
        output_path : str or Path
            Path to write the boundary file
        time_range : tuple or list, optional
            Time range (start, end) to filter the data

        Returns
        -------
        Path
            Path to the written file
        """
        # Create boundary dataset
        ds = self.create_boundary_dataset(time_range)
        if ds is None:
            logger.error("Failed to create boundary dataset, cannot write file")
            return None

        # Write to file using appropriate PyLibs functions or xarray
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        try:
            ds.to_netcdf(output_path)
            logger.info(f"Successfully wrote boundary data to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error writing boundary data to {output_path}: {e}")
            return None


class Boundary3D(BoundaryData):
    """Handler specifically for 3D boundary data using PyLibs directly."""

    def __init__(
        self,
        grid_path: Union[str, Path],
        source_data: Union[xr.Dataset, pd.DataFrame, Dict[str, Any]] = None,
        variables: List[str] = None,
        boundary_indexes: Optional[List[int]] = None,
        vertical_coords: Optional[Union[str, List[float]]] = None,
    ):
        """Initialize the 3D boundary data handler.

        Parameters
        ----------
        grid_path : str or Path
            Path to the SCHISM grid file
        source_data : xr.Dataset, pd.DataFrame, or dict, optional
            Source data for boundary conditions
        variables : list of str, optional
            Variables to extract from source_data
        boundary_indexes : list of int, optional
            Indexes of boundaries to process, if None all open boundaries are used
        vertical_coords : str or list of float, optional
            Vertical coordinate specification
        """
        super().__init__(grid_path, source_data, variables, boundary_indexes)

        # Additional properties for 3D
        self.vertical_coords = vertical_coords
        self.vgrid = None

        # Load vgrid if possible
        vgrid_path = Path(grid_path).parent / "vgrid.in"
        if vgrid_path.exists():
            try:
                self.vgrid = read_schism_vgrid(str(vgrid_path))
                logger.info(f"Loaded vgrid from {vgrid_path}")
            except Exception as e:
                logger.warning(f"Failed to load vgrid from {vgrid_path}: {e}")

    def create_boundary_dataset(self, time_range=None):
        """Create a 3D boundary dataset for SCHISM.

        Parameters
        ----------
        time_range : tuple or list, optional
            Time range (start, end) to filter the data

        Returns
        -------
        xr.Dataset
            Dataset formatted for SCHISM boundary input
        """
        # This method would be implemented to handle 3D data specifically
        # The implementation would be similar to the parent class, but with
        # additional handling for the vertical dimension

        # Start by calling parent method to get basic dataset
        ds = super().create_boundary_dataset(time_range)

        # Add vertical dimension processing

        return ds
