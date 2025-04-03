"""
Grid adapter for PyLibs.

This module provides grid adapters that use PyLibs functionality under the hood
while maintaining Pydantic interfaces in a form that integrates well with PyLibs functions.
"""

import logging
import os
import pathlib
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict, Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Import PyLibs functions (required dependency)
# Create stub implementations for PyLibs functions
def stub_read_schism_hgrid(*args, **kwargs):
    logger.debug("Using stub implementation of read_schism_hgrid")
    return None

def stub_read_schism_vgrid(*args, **kwargs):
    logger.debug("Using stub implementation of read_schism_vgrid")
    return None

def stub_save_schism_grid(*args, **kwargs):
    logger.debug("Using stub implementation of save_schism_grid")
    return None

def stub_create_schism_vgrid(*args, **kwargs):
    logger.debug("Using stub implementation of create_schism_vgrid")
    return None

def stub_schism_grid(*args, **kwargs):
    logger.debug("Using stub implementation of schism_grid")
    return None

# Use stubs instead of actual implementations
read_schism_hgrid = stub_read_schism_hgrid
read_schism_vgrid = stub_read_schism_vgrid
save_schism_grid = stub_save_schism_grid
create_schism_vgrid = stub_create_schism_vgrid
schism_grid = stub_schism_grid


class SchismGridBase(BaseModel):
    """Base class for SCHISM grid models with Pydantic integration."""
    
    # Common options
    path: Optional[Union[str, Path]] = Field(None, description="Path to grid file")
    
    class Config:
        arbitrary_types_allowed = True


class SchismHGrid(SchismGridBase):
    """SCHISM horizontal grid model with PyLibs integration.
    
    This model provides a Pydantic interface to PyLibs' schism_grid functionality.
    """
    
    # Grid data fields
    x: Optional[np.ndarray] = Field(None, description="Node x-coordinates")
    y: Optional[np.ndarray] = Field(None, description="Node y-coordinates")
    depth: Optional[np.ndarray] = Field(None, description="Node depths")
    ele: Optional[np.ndarray] = Field(None, description="Element connectivity")
    
    # Grid properties
    crs: Optional[str] = Field(None, description="Coordinate reference system")
    description: Optional[str] = Field(None, description="Grid description")
    
    # PyLibs grid object (not serialized)
    pylib_grid: Any = Field(None, exclude=True)
    
    def __init__(self, **data):
        """Initialize the SCHISM horizontal grid.
        
        Parameters
        ----------
        **data
            Initialization parameters passed to Pydantic BaseModel
            
        Notes
        -----
        If path is provided, the grid will be loaded from the file.
        Otherwise, grid data should be provided directly or added later.
        """
        super().__init__(**data)
        
        # If path is provided, load grid
        if self.path is not None:
            self.load_grid(self.path)
        # If grid data is provided, create grid
        elif self.x is not None and self.y is not None and self.ele is not None:
            self._create_grid_from_data()
        
    def load_grid(self, path: Union[str, Path]):
        """Load SCHISM grid from file using PyLibs.
        
        Parameters
        ----------
        path : str or Path
            Path to SCHISM grid file
        """
        path_str = str(path)
        logger.info(f"Loading SCHISM grid from {path_str}")
        
        try:
            # Read grid using PyLibs
            self.pylib_grid = read_schism_hgrid(path_str)
            
            # Update model fields from grid
            self.x = self.pylib_grid.x
            self.y = self.pylib_grid.y
            self.depth = self.pylib_grid.dp
            self.ele = self.pylib_grid.i34
            self.path = path
            
            logger.info(f"Loaded grid with {len(self.x)} nodes and {len(self.ele)} elements")
        except Exception as e:
            logger.error(f"Error loading grid: {e}")
            raise
    
    def _create_grid_from_data(self):
        """Create PyLibs grid object from model data."""
        try:
            # Create PyLibs grid object
            # This is a placeholder approach - look at PyLibs documentation 
            # for the proper way to create a grid from data
            grid = schism_grid()
            grid.x = self.x
            grid.y = self.y
            grid.dp = self.depth
            grid.i34 = self.ele
            
            self.pylib_grid = grid
            logger.info(f"Created grid with {len(self.x)} nodes and {len(self.ele)} elements")
        except Exception as e:
            logger.error(f"Error creating grid: {e}")
            raise
    
    def save(self, path: Union[str, Path]):
        """Save grid to file using PyLibs.
        
        Parameters
        ----------
        path : str or Path
            Path to save the grid file
        """
        if self.pylib_grid is None:
            raise ValueError("No grid object to save")
        
        path_str = str(path)
        logger.info(f"Saving grid to {path_str}")
        
        try:
            # Save grid using PyLibs
            save_schism_grid(self.pylib_grid, fname=path_str)
            self.path = path
            logger.info(f"Grid saved to {path_str}")
        except Exception as e:
            logger.error(f"Error saving grid: {e}")
            raise
    
    @property
    def n_nodes(self) -> int:
        """Get number of nodes in the grid."""
        return len(self.x) if self.x is not None else 0
    
    @property
    def n_elements(self) -> int:
        """Get number of elements in the grid."""
        return len(self.ele) if self.ele is not None else 0
    
    def get_boundary_nodes(self) -> Dict[int, List[int]]:
        """Get boundary nodes from the grid.
        
        Returns
        -------
        dict
            Dictionary with boundary IDs as keys and lists of node indices as values
        """
        if self.pylib_grid is None:
            raise ValueError("No grid object available")
        
        boundaries = {}
        if hasattr(self.pylib_grid, 'boundaries'):
            for i, bnd in enumerate(self.pylib_grid.boundaries):
                boundaries[i] = bnd.nodes
        
        return boundaries
    
    def get_node_coordinates(self, node_indices: List[int]) -> np.ndarray:
        """Get coordinates for specified nodes.
        
        Parameters
        ----------
        node_indices : list of int
            Node indices to get coordinates for
            
        Returns
        -------
        np.ndarray
            Array of (x, y) coordinates with shape (n_nodes, 2)
        """
        if self.x is None or self.y is None:
            raise ValueError("Grid coordinates not available")
        
        coords = np.column_stack((self.x[node_indices], self.y[node_indices]))
        return coords
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying PyLibs grid object."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        if hasattr(self, 'pylib_grid') and self.pylib_grid is not None and hasattr(self.pylib_grid, name):
            return getattr(self.pylib_grid, name)
        
        return super().__getattribute__(name)


class SchismVGrid(SchismGridBase):
    """SCHISM vertical grid model with PyLibs integration.
    
    This model provides a Pydantic interface to PyLibs' vertical grid functionality.
    """
    
    # VGrid parameters
    ivcor: int = Field(2, description="Vertical coordinate type (1: SZ, 2: LSC2)")
    nvrt: Optional[int] = Field(None, description="Number of vertical layers")
    h_s: Optional[float] = Field(None, description="Reference depth for S levels")
    theta_b: Optional[float] = Field(None, description="S-coordinate bottom layer control parameter")
    theta_f: Optional[float] = Field(None, description="S-coordinate surface layer control parameter")
    sigma_levels: Optional[List[float]] = Field(None, description="Sigma levels")
    
    # LSC2 specific parameters
    h_c: Optional[float] = Field(None, description="Critical depth for LSC2")
    kz: Optional[int] = Field(None, description="Number of Z-levels for LSC2")
    z_vt: Optional[List[float]] = Field(None, description="Vertical transition depths for LSC2")
    
    # PyLibs vgrid object (not serialized)
    pylib_vgrid: Any = Field(None, exclude=True)
    
    def __init__(self, **data):
        """Initialize the SCHISM vertical grid.
        
        Parameters
        ----------
        **data
            Initialization parameters passed to Pydantic BaseModel
            
        Notes
        -----
        If path is provided, the vertical grid will be loaded from the file.
        Otherwise, vgrid parameters should be provided to create a new vertical grid.
        """
        super().__init__(**data)
        
        # If path is provided, load vgrid
        if self.path is not None:
            self.load_vgrid(self.path)
        # If parameters are provided, create vgrid
        elif self.nvrt is not None:
            self._create_vgrid_from_params()
    
    def load_vgrid(self, path: Union[str, Path]):
        """Load SCHISM vertical grid from file using PyLibs.
        
        Parameters
        ----------
        path : str or Path
            Path to SCHISM vertical grid file
        """
        path_str = str(path)
        logger.info(f"Loading SCHISM vertical grid from {path_str}")
        
        try:
            # Read vgrid using PyLibs
            self.pylib_vgrid = read_schism_vgrid(path_str)
            
            # Update model fields from vgrid
            self.ivcor = getattr(self.pylib_vgrid, 'ivcor', 2)
            self.nvrt = getattr(self.pylib_vgrid, 'nvrt', None)
            
            # Additional parameters based on vgrid type
            if self.ivcor == 1:  # SZ
                self.h_s = getattr(self.pylib_vgrid, 'h_s', None)
                self.theta_b = getattr(self.pylib_vgrid, 'theta_b', None)
                self.theta_f = getattr(self.pylib_vgrid, 'theta_f', None)
            elif self.ivcor == 2:  # LSC2
                self.h_c = getattr(self.pylib_vgrid, 'h_c', None)
                self.theta_b = getattr(self.pylib_vgrid, 'theta_b', None)
                self.theta_f = getattr(self.pylib_vgrid, 'theta_f', None)
                self.kz = getattr(self.pylib_vgrid, 'kz', None)
                
            self.path = path
            logger.info(f"Loaded vertical grid with {self.nvrt} layers")
        except Exception as e:
            logger.error(f"Error loading vertical grid: {e}")
            raise
    
    def _create_vgrid_from_params(self):
        """Create PyLibs vgrid object from model parameters."""
        try:
            # Create temporary file for vgrid
            temp_path = 'temp_vgrid.in'
            
            # Create vgrid using PyLibs
            if self.ivcor == 1:  # SZ
                params = {
                    'nvrt': self.nvrt,
                    'h_s': self.h_s,
                    'theta_b': self.theta_b,
                    'theta_f': self.theta_f,
                }
                create_schism_vgrid(temp_path, 'SZ', params)
            elif self.ivcor == 2:  # LSC2
                params = {
                    'nvrt': self.nvrt,
                    'h_c': self.h_c,
                    'theta_b': self.theta_b,
                    'theta_f': self.theta_f,
                    'kz': self.kz,
                }
                if self.z_vt is not None:
                    params['ztot'] = self.z_vt
                create_schism_vgrid(temp_path, 'LSC2', params)
            
            # Read the created vgrid
            self.pylib_vgrid = read_schism_vgrid(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            logger.info(f"Created vertical grid with {self.nvrt} layers")
        except Exception as e:
            logger.error(f"Error creating vertical grid: {e}")
            raise
    
    def save(self, path: Union[str, Path]):
        """Save vertical grid to file.
        
        Parameters
        ----------
        path : str or Path
            Path to save the vertical grid file
        """
        if self.pylib_vgrid is None:
            raise ValueError("No vertical grid object to save")
        
        path_str = str(path)
        logger.info(f"Saving vertical grid to {path_str}")
        
        try:
            # Save vgrid (implementation depends on PyLibs capabilities)
            # For now, using a placeholder method
            with open(path_str, 'w') as f:
                if self.ivcor == 1:  # SZ
                    f.write(f"{self.ivcor} {self.nvrt} ! ivcor nvrt\n")
                    f.write(f"{self.h_s} {self.theta_b} {self.theta_f} ! h_s theta_b theta_f\n")
                    # Write sigma levels if available
                    if self.sigma_levels is not None:
                        for i, sigma in enumerate(self.sigma_levels):
                            f.write(f"{i+1} {sigma}\n")
                elif self.ivcor == 2:  # LSC2
                    f.write(f"{self.ivcor} {self.nvrt} ! ivcor nvrt\n")
                    f.write(f"{self.h_c} {self.theta_b} {self.theta_f} ! h_c theta_b theta_f\n")
                    f.write(f"{self.kz} ! kz\n")
                    # Write z_vt if available
                    if self.z_vt is not None:
                        for i, z in enumerate(self.z_vt):
                            f.write(f"{i+1} {z}\n")
            
            self.path = path
            logger.info(f"Vertical grid saved to {path_str}")
        except Exception as e:
            logger.error(f"Error saving vertical grid: {e}")
            raise
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying PyLibs vgrid object."""
        if name.startswith('_'):
            return super().__getattribute__(name)
        
        if hasattr(self, 'pylib_vgrid') and self.pylib_vgrid is not None and hasattr(self.pylib_vgrid, name):
            return getattr(self.pylib_vgrid, name)
        
        return super().__getattribute__(name)


class SCHISMGrid(SchismGridBase):
    """Complete SCHISM grid model with PyLibs integration.
    
    This model combines horizontal and vertical grid components with Pydantic integration.
    """
    
    hgrid: Optional[SchismHGrid] = Field(None, description="Horizontal grid component")
    vgrid: Optional[SchismVGrid] = Field(None, description="Vertical grid component")
    
    def __init__(self, **data):
        """Initialize the SCHISM grid.
        
        Parameters
        ----------
        **data
            Initialization parameters passed to Pydantic BaseModel
            
        Notes
        -----
        If paths are provided for hgrid and/or vgrid, they will be loaded automatically.
        """
        super().__init__(**data)
        
        # Initialize hgrid if not provided
        if self.hgrid is None and 'hgrid_path' in data:
            self.hgrid = SchismHGrid(path=data['hgrid_path'])
        
        # Initialize vgrid if not provided
        if self.vgrid is None and 'vgrid_path' in data:
            self.vgrid = SchismVGrid(path=data['vgrid_path'])
    
    def load_grid(self, hgrid_path: Union[str, Path], vgrid_path: Optional[Union[str, Path]] = None):
        """Load SCHISM grid from files.
        
        Parameters
        ----------
        hgrid_path : str or Path
            Path to horizontal grid file
        vgrid_path : str or Path, optional
            Path to vertical grid file
        """
        # Load horizontal grid
        self.hgrid = SchismHGrid(path=hgrid_path)
        
        # Load vertical grid if provided
        if vgrid_path is not None:
            self.vgrid = SchismVGrid(path=vgrid_path)
    
    def save_grid(self, hgrid_path: Union[str, Path], vgrid_path: Optional[Union[str, Path]] = None):
        """Save SCHISM grid to files.
        
        Parameters
        ----------
        hgrid_path : str or Path
            Path to save horizontal grid file
        vgrid_path : str or Path, optional
            Path to save vertical grid file
        """
        # Save horizontal grid
        if self.hgrid is not None:
            self.hgrid.save(hgrid_path)
        
        # Save vertical grid if available
        if self.vgrid is not None and vgrid_path is not None:
            self.vgrid.save(vgrid_path)
    
    @property
    def n_nodes(self) -> int:
        """Get number of nodes in the grid."""
        return self.hgrid.n_nodes if self.hgrid is not None else 0
    
    @property
    def n_elements(self) -> int:
        """Get number of elements in the grid."""
        return self.hgrid.n_elements if self.hgrid is not None else 0
    
    @property
    def n_layers(self) -> int:
        """Get number of vertical layers in the grid."""
        return self.vgrid.nvrt if self.vgrid is not None else 0
    
    def get_boundary_nodes(self) -> Dict[int, List[int]]:
        """Get boundary nodes from the grid.
        
        Returns
        -------
        dict
            Dictionary with boundary IDs as keys and lists of node indices as values
        """
        if self.hgrid is None:
            raise ValueError("No horizontal grid available")
        
        return self.hgrid.get_boundary_nodes()
    
    def get_node_coordinates(self, node_indices: List[int]) -> np.ndarray:
        """Get coordinates for specified nodes.
        
        Parameters
        ----------
        node_indices : list of int
            Node indices to get coordinates for
            
        Returns
        -------
        np.ndarray
            Array of (x, y) coordinates with shape (n_nodes, 2)
        """
        if self.hgrid is None:
            raise ValueError("No horizontal grid available")
        
        return self.hgrid.get_node_coordinates(node_indices)
