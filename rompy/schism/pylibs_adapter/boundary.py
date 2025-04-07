"""
Boundary data adapter for PyLib.

This module provides adapters for handling SCHISM boundary data,
particularly focusing on 3D boundary data using PyLib under the hood
while maintaining ROMPY's Pydantic interfaces.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import numpy as np
import pandas as pd
import xarray as xr

# Add PyLib to the path if needed
sys.path.append('/home/tdurrant/source/pylibs')

logger = logging.getLogger(__name__)

# Import PyLib functions
try:
    # Import PyLib actual implementations
    from pylib import *
    from src.schism_file import (
        read_schism_hgrid,
        read_schism_bpfile,
        schism_grid,
        read_schism_vgrid,
        compute_zcor
    )
    logger.info("Successfully imported PyLib functions")
except ImportError as e:
    logger.error(f"Error importing PyLib functions: {e}")
    
    # Fallback to stub implementations
    def stub_read_schism_bpfile(*args, **kwargs):
        logger.warning("Using stub implementation of read_schism_bpfile")
        return None

    def stub_schism_grid(*args, **kwargs):
        logger.warning("Using stub implementation of schism_grid")
        # Create a minimal grid object with required attributes
        grid = type('schism_grid', (), {})()
        grid.x = np.array([])
        grid.y = np.array([])
        grid.dp = np.array([])
        grid.i34 = np.array([])
        return grid

    def stub_interp_schism_3d(*args, **kwargs):
        logger.warning("Using stub implementation of interp_schism_3d")
        return None

    def stub_interp_vertical(*args, **kwargs):
        logger.warning("Using stub implementation of interp_vertical")
        return None

    def stub_interpolate(*args, **kwargs):
        logger.warning("Using stub implementation of interpolate")
        return None

    # Use stubs instead of actual implementations
    read_schism_bpfile = stub_read_schism_bpfile
    schism_grid = stub_schism_grid
    interp_schism_3d = stub_interp_schism_3d
    interp_vertical = stub_interp_vertical
    interpolate = stub_interpolate


class BoundaryData:
    """Adapter for SCHISM boundary data processing using PyLibs."""
    
    def __init__(
        self,
        grid_path: Union[str, Path],
        source_data: Union[xr.Dataset, pd.DataFrame, Dict[str, Any]] = None,
        variables: List[str] = None,
        boundary_indexes: Optional[List[int]] = None,
    ):
        """Initialize the boundary data adapter.
        
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
            raise ValueError("Grid not initialized or invalid grid path")
        
        # Get boundary information using PyLibs
        bnd_nodes = read_schism_bpfile(self.grid)
        
        # Process boundaries
        boundaries = {}
        for i, bnd in enumerate(bnd_nodes):
            if self.boundary_indexes is None or i in self.boundary_indexes:
                # Extract coordinates for these boundary nodes
                boundary_coords = self._get_coordinates_for_nodes(bnd)
                boundaries[i] = {
                    'nodes': bnd,
                    'coords': boundary_coords
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
        x, y = self.grid.x[nodes], self.grid.y[nodes]
        return {'x': x, 'y': y}
    
    def interpolate_to_boundary(self, data, boundary_coords, method='linear'):
        """Interpolate data to boundary points.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            Data to interpolate
        boundary_coords : dict
            Dictionary with 'x' and 'y' keys for target coordinates
        method : str, optional
            Interpolation method, default is 'linear'
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            Interpolated data at boundary points
        """
        # This is a simplified placeholder - in a real implementation,
        # you would use PyLibs' interpolation capabilities
        logger.info(f"Interpolating data to boundary using method: {method}")
        
        # Placeholder for PyLibs-based interpolation
        # In a complete implementation, you would call the appropriate PyLibs function
        
        # Return the original data as a placeholder
        return data
    
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
        if self.source_data is None:
            raise ValueError("Source data not set")
        
        # Extract boundary points
        boundaries = self.extract_boundary_points()
        
        # Process each boundary and create a dataset
        # This is a simplified placeholder - in a real implementation,
        # you would use PyLibs to properly format the data for SCHISM
        
        # Convert source_data to xarray if it's not already
        if isinstance(self.source_data, pd.DataFrame):
            ds = xr.Dataset.from_dataframe(self.source_data)
        elif isinstance(self.source_data, dict):
            # Create a simple dataset from dict
            ds = xr.Dataset(self.source_data)
        else:
            ds = self.source_data
        
        # Apply time filtering if requested
        if time_range is not None and hasattr(ds, 'time'):
            start, end = time_range
            ds = ds.sel(time=slice(start, end))
        
        # Create boundary dataset
        bnd_ds = xr.Dataset()
        
        # For each boundary, interpolate the data
        for bnd_idx, bnd_info in boundaries.items():
            interpolated = self.interpolate_to_boundary(ds, bnd_info['coords'])
            
            # Add to boundary dataset with boundary index as dimension
            # This is simplified - you'd need to adapt based on your data structure
            for var in self.variables:
                if var in interpolated:
                    if var not in bnd_ds:
                        bnd_ds[var] = (('boundary', 'node', 'time'), [])
                    
                    # Add this boundary's data
                    # Again, this is a placeholder - your actual implementation
                    # would depend on your data structure
                    pass
        
        return bnd_ds
    
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
        # Create the boundary dataset
        bnd_ds = self.create_boundary_dataset(time_range)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing boundary data to {output_path}")
        bnd_ds.to_netcdf(output_path)
        
        return output_path


class Boundary3D(BoundaryData):
    """Adapter specifically for 3D boundary data using PyLibs."""
    
    def __init__(
        self,
        grid_path: Union[str, Path],
        source_data: Union[xr.Dataset, pd.DataFrame, Dict[str, Any]] = None,
        variables: List[str] = None,
        boundary_indexes: Optional[List[int]] = None,
        vertical_coords: Optional[Union[str, List[float]]] = None,
    ):
        """Initialize the 3D boundary data adapter.
        
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
        vertical_coords : str or list, optional
            Vertical coordinates variable name or explicit coordinate values
        """
        super().__init__(grid_path, source_data, variables, boundary_indexes)
        self.vertical_coords = vertical_coords
    
    def interpolate_to_vertical_grid(self, data, target_depths):
        """Interpolate data to the target vertical grid.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            Data to interpolate
        target_depths : array-like
            Target depths for interpolation
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            Interpolated data at target depths
        """
        logger.info(f"Interpolating data to vertical grid with {len(target_depths)} layers")
        
        # If we have an xarray dataset, apply to each relevant variable
        if isinstance(data, xr.Dataset):
            # Create a new dataset for the interpolated results
            interpolated_ds = xr.Dataset()
            
            # Process each variable that has a vertical dimension
            for var_name, da in data.data_vars.items():
                if 'depth' in da.dims or 'z' in da.dims or 'vertical' in da.dims:
                    # Determine the vertical dimension name
                    z_dim = [d for d in da.dims if d in ('depth', 'z', 'vertical')][0]
                    source_depths = data[z_dim].values
                    
                    # Extract the data array as numpy array
                    var_data = da.values
                    
                    # Use PyLibs' interp_vertical for interpolation
                    # We need to ensure dimensions match PyLibs' expectations
                    if z_dim != da.dims[0]:
                        # If depth is not the first dimension, transpose to make it first
                        dim_order = list(da.dims)
                        dim_order.remove(z_dim)
                        dim_order.insert(0, z_dim)
                        da = da.transpose(*dim_order)
                        var_data = da.values
                        
                    # Perform vertical interpolation using PyLibs
                    interpolated_var = interp_vertical(var_data, source_depths, target_depths)
                    
                    # Create a new DataArray with interpolated values
                    new_dims = list(da.dims)
                    new_dims[new_dims.index(z_dim)] = 'new_depth'
                    new_coords = {dim: (da.coords[dim] if dim != z_dim else target_depths) 
                                 for dim in da.dims}
                    new_coords['new_depth'] = target_depths
                    
                    interpolated_ds[var_name] = xr.DataArray(
                        interpolated_var, 
                        dims=new_dims,
                        coords=new_coords,
                        attrs=da.attrs
                    )
                else:
                    # If no vertical dimension, copy as is
                    interpolated_ds[var_name] = da
            
            return interpolated_ds
        
        # If we have a DataArray, apply interpolation directly
        elif isinstance(data, xr.DataArray):
            # Determine if there's a vertical dimension
            z_dim = None
            for dim in data.dims:
                if dim in ('depth', 'z', 'vertical'):
                    z_dim = dim
                    break
            
            if z_dim is not None:
                source_depths = data[z_dim].values
                
                # Extract the data array as numpy array
                var_data = data.values
                
                # If depth is not the first dimension, transpose to make it first
                if z_dim != data.dims[0]:
                    dim_order = list(data.dims)
                    dim_order.remove(z_dim)
                    dim_order.insert(0, z_dim)
                    data = data.transpose(*dim_order)
                    var_data = data.values
                
                # Perform vertical interpolation using PyLibs
                interpolated_var = interp_vertical(var_data, source_depths, target_depths)
                
                # Create a new DataArray with interpolated values
                new_dims = list(data.dims)
                new_dims[new_dims.index(z_dim)] = 'new_depth'
                new_coords = {dim: (data.coords[dim] if dim != z_dim else target_depths) 
                             for dim in data.dims}
                new_coords['new_depth'] = target_depths
                
                return xr.DataArray(
                    interpolated_var, 
                    dims=new_dims,
                    coords=new_coords,
                    attrs=data.attrs
                )
            else:
                return data
        
        # For numpy arrays, assume first dimension is vertical
        else:
            # For raw numpy arrays, assume the first dimension is depth
            if hasattr(data, 'shape') and len(data.shape) > 0:
                if isinstance(target_depths, (list, tuple)):
                    target_depths = np.array(target_depths)
                
                # If we don't have source depths, create a linear space
                if not hasattr(self, 'source_depths') or self.source_depths is None:
                    source_depths = np.linspace(0, -100, data.shape[0])  # Assuming 0 to -100m by default
                else:
                    source_depths = self.source_depths
                
                # Use PyLibs' interp_vertical for interpolation
                return interp_vertical(data, source_depths, target_depths)
            
            return data
    
    def create_boundary_dataset(self, time_range=None):
        """Create a 3D boundary dataset for SCHISM.
        
        Override the parent method to handle vertical coordinates.
        
        Parameters
        ----------
        time_range : tuple or list, optional
            Time range (start, end) to filter the data
            
        Returns
        -------
        xr.Dataset
            Dataset formatted for SCHISM boundary input
        """
        if self.grid is None:
            raise ValueError("Grid not initialized or invalid grid path")
            
        if self.source_data is None:
            raise ValueError("Source data not set")
        
        # Extract boundary points
        boundaries = self.extract_boundary_points()
        
        # Convert source_data to xarray if it's not already
        if isinstance(self.source_data, pd.DataFrame):
            ds = xr.Dataset.from_dataframe(self.source_data)
        elif isinstance(self.source_data, dict):
            # Create a simple dataset from dict
            ds = xr.Dataset(self.source_data)
        else:
            ds = self.source_data
        
        # Apply time filtering if requested
        if time_range is not None and hasattr(ds, 'time'):
            start, end = time_range
            ds = ds.sel(time=slice(start, end))
        
        # Handle vertical coordinates
        # If vertical_coords is a string, assume it's a variable name in source_data
        if isinstance(self.vertical_coords, str) and hasattr(ds, self.vertical_coords):
            depth_values = ds[self.vertical_coords].values
            source_depths = depth_values
        # If it's a list/array, use it directly
        elif isinstance(self.vertical_coords, (list, np.ndarray)):
            source_depths = np.array(self.vertical_coords)
        else:
            source_depths = None
            
        # Create target vertical grid - can be derived from SCHISM vgrid
        # if available, otherwise use source_depths
        if hasattr(self.grid, 'vgrid') and self.grid.vgrid is not None:
            # Use PyLibs to get target depths from SCHISM vgrid
            # This would depend on how PyLibs represents vgrid
            # For now, using a placeholder approach
            nvrt = self.grid.vgrid.nvrt if hasattr(self.grid.vgrid, 'nvrt') else 10
            target_depths = np.linspace(0, -100, nvrt)  # Simple placeholder
        elif source_depths is not None:
            target_depths = source_depths
        else:
            # Default vertical grid if nothing else is available
            target_depths = np.linspace(0, -100, 10)
            
        logger.info(f"Using {len(target_depths)} vertical layers for boundary")
            
        # Create boundary dataset
        bnd_ds = xr.Dataset()
        bnd_ds.coords['depth'] = ('depth', target_depths)
        
        # Process each variable that should be included
        for var_name in self.variables:
            if var_name in ds.data_vars:
                # Get horizontal interpolation to boundary points
                var_data_list = []
                
                for bnd_idx, bnd_info in boundaries.items():
                    # Get boundary coordinates
                    bnd_coords = np.array([bnd_info['coords']['x'], bnd_info['coords']['y']]).T
                    
                    # Use PyLibs' interpolation to get data at boundary points
                    # First extract the variable data
                    var_da = ds[var_name]
                    
                    # Check if variable has vertical dimension
                    has_vertical = False
                    for dim in var_da.dims:
                        if dim in ('depth', 'z', 'vertical'):
                            has_vertical = True
                            z_dim = dim
                            break
                    
                    # Different approach based on whether data is 2D or 3D
                    if has_vertical:
                        # Get horizontal coordinates from source data
                        if 'lon' in ds.coords and 'lat' in ds.coords:
                            x_coords = ds.lon.values
                            y_coords = ds.lat.values
                        elif 'x' in ds.coords and 'y' in ds.coords:
                            x_coords = ds.x.values
                            y_coords = ds.y.values
                        else:
                            # Fallback - assume regular grid
                            x_dim = [d for d in var_da.dims if d in ('lon', 'x')][0]
                            y_dim = [d for d in var_da.dims if d in ('lat', 'y')][0]
                            x_coords = ds[x_dim].values
                            y_coords = ds[y_dim].values
                            x_coords, y_coords = np.meshgrid(x_coords, y_coords)
                        
                        # Prepare for 3D interpolation
                        # First do horizontal interpolation for each vertical level
                        interpolated_data = np.zeros((len(target_depths), len(bnd_info['nodes'])))
                        
                        # For each vertical level, interpolate horizontally
                        for i, depth in enumerate(source_depths):
                            # Extract slice at this depth
                            if z_dim in var_da.dims:
                                slice_data = var_da.sel({z_dim: depth}, method='nearest').values
                            else:
                                # If no vertical dimension in this variable, just use the data
                                slice_data = var_da.values
                            
                            # Interpolate to boundary points
                            for j, (x, y) in enumerate(bnd_coords):
                                # Use PyLibs' interpolate function or similar
                                # This is a placeholder - actual implementation depends on PyLibs
                                # and your grid structure
                                interpolated_data[i, j] = interpolate(
                                    x_coords, y_coords, slice_data, x, y, method='linear'
                                )
                        
                        # Now interpolated_data has shape (n_depths, n_boundary_nodes)
                        var_data_list.append(interpolated_data)
                    else:
                        # 2D variable - just do horizontal interpolation
                        # Similar to above but without vertical dimension
                        interpolated_data = np.zeros(len(bnd_info['nodes']))
                        
                        # Get horizontal coordinates
                        if 'lon' in ds.coords and 'lat' in ds.coords:
                            x_coords = ds.lon.values
                            y_coords = ds.lat.values
                        elif 'x' in ds.coords and 'y' in ds.coords:
                            x_coords = ds.x.values
                            y_coords = ds.y.values
                        else:
                            # Fallback - assume regular grid
                            x_dim = [d for d in var_da.dims if d in ('lon', 'x')][0]
                            y_dim = [d for d in var_da.dims if d in ('lat', 'y')][0]
                            x_coords = ds[x_dim].values
                            y_coords = ds[y_dim].values
                            x_coords, y_coords = np.meshgrid(x_coords, y_coords)
                        
                        # Interpolate to boundary points
                        for j, (x, y) in enumerate(bnd_coords):
                            interpolated_data[j] = interpolate(
                                x_coords, y_coords, var_da.values, x, y, method='linear'
                            )
                        
                        # Replicate the 2D data across all vertical levels for SCHISM
                        var_data_3d = np.zeros((len(target_depths), len(bnd_info['nodes'])))
                        for i in range(len(target_depths)):
                            var_data_3d[i, :] = interpolated_data
                        
                        var_data_list.append(var_data_3d)
                
                # Combine all boundaries into one array
                # For SCHISM, we need to organize by boundary, then node, then vertical level
                all_boundaries_data = np.concatenate(var_data_list, axis=1)
                
                # Add to dataset with proper dimensions and coordinates
                bnd_ds[var_name] = (
                    ('depth', 'boundary_node'), 
                    all_boundaries_data
                )
                
                # Copy attributes if available
                if hasattr(ds[var_name], 'attrs'):
                    bnd_ds[var_name].attrs = ds[var_name].attrs
        
        # Add boundary node information
        bnd_node_ids = []
        for bnd_idx, bnd_info in boundaries.items():
            bnd_node_ids.extend(bnd_info['nodes'])
        
        bnd_ds.coords['boundary_node'] = ('boundary_node', bnd_node_ids)
        
        # Add time dimension if it exists in source data
        if 'time' in ds.coords:
            bnd_ds.coords['time'] = ds.time
            
        return bnd_ds
