"""
Plotting utilities and common functions for SCHISM visualization.

This module provides utility functions for common plotting operations,
data processing, and visualization setup used across SCHISM plotting modules.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

logger = logging.getLogger(__name__)


def validate_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Validate that a file exists.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to file to validate
        
    Returns
    -------
    bool
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def setup_cartopy_axis():
    """
    Set up cartopy projection for geographic plotting.
    
    Returns
    -------
    projection : cartopy.crs projection or None
        Cartopy projection if available, None otherwise
    """
    try:
        import cartopy.crs as ccrs
        return ccrs.PlateCarree()
    except ImportError:
        logger.warning("Cartopy not available, using regular matplotlib projection")
        return None


def get_geographic_extent(lons: np.ndarray, lats: np.ndarray, buffer: float = 0.1) -> Tuple[float, float, float, float]:
    """
    Get geographic extent from longitude and latitude arrays.
    
    Parameters
    ----------
    lons : np.ndarray
        Longitude values
    lats : np.ndarray  
        Latitude values
    buffer : float, optional
        Buffer to add around extent as fraction of range. Default is 0.1.
        
    Returns
    -------
    extent : Tuple[float, float, float, float]
        Geographic extent as (lon_min, lon_max, lat_min, lat_max)
    """
    lon_min, lon_max = np.nanmin(lons), np.nanmax(lons)
    lat_min, lat_max = np.nanmin(lats), np.nanmax(lats)
    
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    
    lon_buffer = lon_range * buffer
    lat_buffer = lat_range * buffer
    
    return (
        lon_min - lon_buffer,
        lon_max + lon_buffer, 
        lat_min - lat_buffer,
        lat_max + lat_buffer
    )


def setup_colormap(
    data: np.ndarray,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    levels: Optional[Union[int, List[float]]] = None
) -> Tuple[str, Normalize, Optional[np.ndarray]]:
    """
    Set up colormap and normalization for data visualization.
    
    Parameters
    ----------
    data : np.ndarray
        Data array for colormap setup
    cmap : str, optional
        Colormap name. Default is "viridis".
    vmin : Optional[float]
        Minimum value for normalization
    vmax : Optional[float]
        Maximum value for normalization  
    levels : Optional[Union[int, List[float]]]
        Contour levels if needed
        
    Returns
    -------
    cmap_name : str
        Colormap name
    norm : Normalize
        Normalization object
    levels_array : Optional[np.ndarray]
        Contour levels array if specified
    """
    # Handle vmin/vmax
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
        
    # Create normalization
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Handle levels
    levels_array = None
    if levels is not None:
        if isinstance(levels, int):
            levels_array = np.linspace(vmin, vmax, levels)
        else:
            levels_array = np.array(levels)
            
    return cmap, norm, levels_array


def add_grid_overlay(
    ax: plt.Axes,
    grid: Any,
    alpha: float = 0.3,
    color: str = "gray",
    linewidth: float = 0.5
):
    """
    Add grid overlay to existing plot.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes object to add grid to
    grid : Any
        SCHISM grid object
    alpha : float, optional
        Transparency of grid lines. Default is 0.3.
    color : str, optional
        Color of grid lines. Default is "gray".
    linewidth : float, optional
        Width of grid lines. Default is 0.5.
    """
    try:
        # Use pylibs plotting if available
        if hasattr(grid, 'pylibs_hgrid'):
            hgrid = grid.pylibs_hgrid
            # Plot triangulation
            ax.triplot(hgrid.x, hgrid.y, hgrid.elnode, 
                      alpha=alpha, color=color, linewidth=linewidth)
        else:
            logger.warning("Grid overlay not supported for this grid type")
    except Exception as e:
        logger.warning(f"Could not add grid overlay: {e}")


def add_boundary_overlay(
    ax: plt.Axes,
    grid: Any,
    boundary_colors: Dict[str, str] = None,
    linewidth: float = 2.0
):
    """
    Add boundary overlay to existing plot.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes object to add boundaries to
    grid : Any
        SCHISM grid object
    boundary_colors : Dict[str, str], optional
        Colors for different boundary types
    linewidth : float, optional
        Width of boundary lines. Default is 2.0.
    """
    if boundary_colors is None:
        boundary_colors = {"ocean": "red", "land": "green", "tidal": "blue"}
    
    try:
        # Plot ocean boundaries
        if hasattr(grid, 'ocean_boundary'):
            x_ocean, y_ocean = grid.ocean_boundary()
            if len(x_ocean) > 0:
                ax.plot(x_ocean, y_ocean, 
                       color=boundary_colors.get("ocean", "red"),
                       linewidth=linewidth, label="Ocean Boundary")
        
        # Plot land boundaries  
        if hasattr(grid, 'land_boundary'):
            x_land, y_land = grid.land_boundary()
            if len(x_land) > 0:
                ax.plot(x_land, y_land,
                       color=boundary_colors.get("land", "green"), 
                       linewidth=linewidth, label="Land Boundary")
                       
    except Exception as e:
        logger.warning(f"Could not add boundary overlay: {e}")


def detect_file_type(file_path: Union[str, Path]) -> str:
    """
    Detect SCHISM file type from filename.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to SCHISM file
        
    Returns
    -------
    file_type : str
        Detected file type
    """
    file_path = Path(file_path)
    name = file_path.name.lower()
    
    if name.endswith('.gr3'):
        return 'gr3'
    elif name.endswith('.th.nc'):
        if 'sal_3d' in name:
            return 'salinity_3d'
        elif 'tem_3d' in name:
            return 'temperature_3d'
        elif 'uv3d' in name:
            return 'velocity_3d'
        elif 'elev2d' in name:
            return 'elevation_2d'
        else:
            return 'boundary_th'
    elif name.endswith('bctides.in'):
        return 'bctides'
    elif 'sflux' in name:
        return 'atmospheric'
    else:
        return 'unknown'


def load_schism_data(file_path: Union[str, Path]) -> xr.Dataset:
    """
    Load SCHISM data file as xarray Dataset.
    
    Parameters
    ----------
    file_path : Union[str, Path]
        Path to SCHISM data file
        
    Returns
    -------
    ds : xr.Dataset
        Loaded dataset
        
    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_type = detect_file_type(file_path)
    
    if file_type in ['salinity_3d', 'temperature_3d', 'velocity_3d', 'elevation_2d', 'boundary_th', 'atmospheric']:
        try:
            ds = xr.open_dataset(file_path)
            return ds
        except Exception as e:
            raise ValueError(f"Could not load NetCDF file {file_path}: {e}")
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def get_variable_info(ds: xr.Dataset, var_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset variable.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable
    var_name : str
        Variable name
        
    Returns
    -------
    info : Dict[str, Any]
        Variable information dictionary
    """
    if var_name not in ds.data_vars:
        raise ValueError(f"Variable {var_name} not found in dataset")
    
    var = ds[var_name]
    
    info = {
        'name': var_name,
        'dims': var.dims,
        'shape': var.shape,
        'dtype': var.dtype,
        'units': var.attrs.get('units', 'unknown'),
        'long_name': var.attrs.get('long_name', var_name),
        'description': var.attrs.get('description', ''),
        'min_value': float(np.nanmin(var.values)),
        'max_value': float(np.nanmax(var.values)),
        'has_time': 'time' in var.dims,
        'has_depth': any(dim in var.dims for dim in ['depth', 'level', 'sigma', 'z']),
        'spatial_dims': [dim for dim in var.dims if dim in ['lon', 'lat', 'x', 'y', 'node']]
    }
    
    return info


def create_time_subset(
    ds: xr.Dataset,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    time_idx: Optional[int] = None
) -> xr.Dataset:
    """
    Create time subset of dataset.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    start_time : Optional[str]
        Start time for subset (ISO format)
    end_time : Optional[str]
        End time for subset (ISO format)
    time_idx : Optional[int]
        Single time index to extract
        
    Returns
    -------
    ds_subset : xr.Dataset
        Time-subsetted dataset
    """
    if 'time' not in ds.dims:
        logger.warning("Dataset does not have time dimension")
        return ds
    
    if time_idx is not None:
        return ds.isel(time=time_idx)
    
    if start_time or end_time:
        return ds.sel(time=slice(start_time, end_time))
    
    return ds


def convert_quads_to_triangles(elnode: np.ndarray) -> np.ndarray:
    """
    Convert SCHISM quad elements to triangles for matplotlib compatibility.
    
    SCHISM may store triangular elements in quadrilateral format using special values
    (like -1 or -2) for missing nodes. These are filtered out properly.
    
    Parameters
    ----------
    elnode : np.ndarray
        Element connectivity array (n_elements, 4) for quads or (n_elements, 3) for triangles
        
    Returns
    -------
    triangles : np.ndarray
        Triangle connectivity array (n_triangles, 3)
    """
    if elnode.shape[1] == 3:
        # Already triangles - filter out invalid elements
        valid_mask = np.all(elnode >= 0, axis=1)
        return elnode[valid_mask]
    elif elnode.shape[1] == 4:
        # Handle mixed quads and triangles
        triangles_list = []
        
        for i, elem in enumerate(elnode):
            # Count valid nodes (non-negative indices)
            valid_nodes = elem[elem >= 0]
            
            if len(valid_nodes) == 3:
                # Triangular element - use first 3 valid nodes
                triangles_list.append(elem[:3])
            elif len(valid_nodes) == 4:
                # True quadrilateral - split into 2 triangles
                triangles_list.append(elem[[0, 1, 2]])
                triangles_list.append(elem[[0, 2, 3]])
            # Skip elements with < 3 valid nodes
        
        if not triangles_list:
            raise ValueError("No valid triangular elements found")
            
        return np.array(triangles_list, dtype=elnode.dtype)
    else:
        raise ValueError(f"Unsupported element type with {elnode.shape[1]} nodes per element")


def ensure_0_based_indexing(elnode: np.ndarray) -> np.ndarray:
    """
    Ensure element connectivity uses 0-based indexing for matplotlib.
    
    SCHISM grids may use special values like -1 or -2 to indicate invalid/missing nodes
    (e.g., triangular elements stored in quadrilateral format). These are preserved.
    
    Parameters
    ----------
    elnode : np.ndarray
        Element connectivity array
        
    Returns
    -------
    elnode_0based : np.ndarray
        Element connectivity with 0-based indexing
    """
    # Create a copy to avoid modifying the original
    elnode_copy = elnode.copy()
    
    # Mask for valid nodes (non-negative values)
    valid_mask = elnode_copy >= 0
    
    if valid_mask.sum() == 0:
        raise ValueError("No valid node indices found in element connectivity")
    
    # Get minimum valid node index
    min_valid = elnode_copy[valid_mask].min()
    
    if min_valid == 1:
        # Convert from 1-based to 0-based indexing (only for valid nodes)
        elnode_copy[valid_mask] = elnode_copy[valid_mask] - 1
        return elnode_copy
    elif min_valid == 0:
        # Already 0-based
        return elnode_copy
    else:
        raise ValueError(f"Unexpected minimum valid node index: {min_valid}")


def prepare_triangulation_data(hgrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare SCHISM grid data for matplotlib triangulation.
    
    Parameters
    ----------
    hgrid : schism_grid
        SCHISM grid object from pylibs
        
    Returns
    -------
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates  
    triangles : np.ndarray
        Triangle connectivity (n_triangles, 3) with 0-based indexing
    """
    x = hgrid.x
    y = hgrid.y
    elnode = hgrid.elnode
    
    # Ensure 0-based indexing
    elnode_0based = ensure_0_based_indexing(elnode)
    
    # Convert to triangles if needed
    triangles = convert_quads_to_triangles(elnode_0based)
    
    return x, y, triangles


def format_scientific_notation(value: float, precision: int = 2) -> str:
    """
    Format value in scientific notation for plot labels.
    
    Parameters
    ----------
    value : float
        Value to format
    precision : int, optional
        Number of decimal places. Default is 2.
        
    Returns
    -------
    formatted : str
        Formatted string
    """
    if abs(value) <= 1e-3 or abs(value) >= 1e4:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"


def create_diverging_colormap_levels(
    data: np.ndarray,
    num_levels: int = 21,
    center: float = 0.0
) -> np.ndarray:
    """
    Create levels for diverging colormap centered on a value.
    
    Parameters
    ----------
    data : np.ndarray
        Data array
    num_levels : int, optional
        Number of levels. Default is 21.
    center : float, optional
        Center value for diverging colormap. Default is 0.0.
        
    Returns
    -------
    levels : np.ndarray
        Colormap levels
    """
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    
    # Determine maximum absolute deviation from center
    max_dev = max(abs(vmin - center), abs(vmax - center))
    
    # Create symmetric levels around center
    return np.linspace(center - max_dev, center + max_dev, num_levels)


def add_scale_bar(
    ax: plt.Axes,
    length_km: float = 50,
    location: str = 'lower right',
    fontsize: int = 10
):
    """
    Add scale bar to geographic plot.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes object to add scale bar to
    length_km : float, optional
        Scale bar length in kilometers. Default is 50.
    location : str, optional
        Scale bar location. Default is 'lower right'.
    fontsize : int, optional
        Font size for scale bar text. Default is 10.
    """
    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        
        # Add scale bar (assuming PlateCarree projection)
        scalebar = ScaleBar(
            1, units='m', length_fraction=0.25,
            location=location, fontsize=fontsize
        )
        ax.add_artist(scalebar)
        
    except ImportError:
        logger.warning("matplotlib-scalebar not available, skipping scale bar")
    except Exception as e:
        logger.warning(f"Could not add scale bar: {e}")


def save_plot(
    fig: plt.Figure,
    filename: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight',
    **kwargs
):
    """
    Save plot with standard settings.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    filename : Union[str, Path]
        Output filename
    dpi : int, optional
        Resolution in dots per inch. Default is 300.
    bbox_inches : str, optional
        Bounding box setting. Default is 'tight'.
    **kwargs : dict
        Additional arguments for fig.savefig
    """
    try:
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        logger.info(f"Plot saved to {filename}")
    except Exception as e:
        logger.error(f"Could not save plot to {filename}: {e}")
        raise