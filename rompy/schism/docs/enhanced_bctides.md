# Enhanced SCHISM Tidal Boundary Handling

This documentation explains the enhanced tidal boundary handling implementation in rompy/schism, which provides a more flexible and comprehensive approach to setting up SCHISM boundary conditions.

## Overview

The enhanced tidal boundary implementation:

1. Supports all SCHISM boundary condition types 
2. Provides a clean, object-oriented interface
3. Offers factory methods for common configurations
4. Integrates with the existing `BoundaryData` infrastructure
5. Uses Pydantic for robust configuration validation

## Key Components

### 1. `TidalBoundary` Class

A specialized boundary handler that extends `BoundaryData`:

```python
from rompy.schism.boundary_tides import TidalBoundary, ElevationType, VelocityType, TracerType

# Create a tidal boundary handler
boundary = TidalBoundary(
    grid_path="path/to/hgrid.gr3",
    constituents=["M2", "S2", "K1", "O1"],
    tidal_database="tpxo",
    tidal_elevations="path/to/h_tpxo9.nc",
    tidal_velocities="path/to/uv_tpxo9.nc"
)

# Configure boundaries
boundary.set_boundary_type(
    0,  # Boundary index
    elev_type=ElevationType.TIDAL,
    vel_type=VelocityType.TIDAL
)

# Set simulation parameters
boundary.set_run_parameters(start_time, run_days)

# Write bctides.in file
boundary.write_boundary_file("path/to/bctides.in")
```

### 2. `SCHISMDataTidesEnhanced` Class

An enhanced version of `SCHISMDataTides` that uses the new boundary infrastructure:

```python
from rompy.schism.tides_enhanced import SCHISMDataTidesEnhanced
from rompy.core.time import TimeRange

# Create configuration
tides = SCHISMDataTidesEnhanced(
    constituents=["M2", "S2", "K1", "O1"],
    tidal_database="tpxo",
    tidal_elevations="path/to/h_tpxo9.nc",
    tidal_velocities="path/to/uv_tpxo9.nc",
    setup_type="tidal"  # Predefined configuration type
)

# Generate bctides.in file
tides.get(
    destdir="path/to/output",
    grid=grid_instance,
    time=TimeRange(start=start_time, end=end_time)
)
```

### 3. Factory Functions

Convenient functions for creating common configurations:

```python
from rompy.schism.boundary_tides import (
    create_tidal_boundary,
    create_hybrid_boundary,
    create_river_boundary,
    create_nested_boundary
)

# Create a pure tidal boundary
tidal = create_tidal_boundary(
    grid_path="path/to/hgrid.gr3",
    constituents=["M2", "S2", "K1", "O1"]
)

# Create a river boundary
river = create_river_boundary(
    grid_path="path/to/hgrid.gr3",
    river_flow=-500.0  # Negative for inflow
)
```

## Boundary Condition Types

The implementation supports all boundary types defined in the SCHISM documentation:

### Elevation Types (`ElevationType`)

- `NONE` (0): Not specified
- `TIMESERIES` (1): Time history from elev.th
- `CONSTANT` (2): Constant elevation
- `TIDAL` (3): Tidal elevation
- `SPACETIME` (4): Space and time-varying from elev2D.th.nc
- `TIDALSPACETIME` (5): Combination of tide and external file

### Velocity Types (`VelocityType`)

- `NONE` (0): Not specified
- `TIMESERIES` (1): Time history from flux.th
- `CONSTANT` (2): Constant discharge
- `TIDAL` (3): Tidal velocity
- `SPACETIME` (4): Space and time-varying from uv3D.th.nc
- `TIDALSPACETIME` (5): Combination of tide and external file
- `FLATHER` (-1): Flather type radiation boundary
- `RELAXED` (-4): 3D input with relaxation

### Tracer Types (`TracerType` - for temperature and salinity)

- `NONE` (0): Not specified
- `TIMESERIES` (1): Time history from temp/salt.th
- `CONSTANT` (2): Constant value
- `INITIAL` (3): Initial profile for inflow
- `SPACETIME` (4): 3D input

## Common Configurations

### 1. Tidal-Only Boundary

All boundaries use tidal elevations and velocities:

```python
from rompy.schism.tides_enhanced import create_tidal_only_config

config = create_tidal_only_config(
    constituents=["M2", "S2", "K1", "O1"],
    tidal_database="tpxo",
    tidal_elevations="path/to/h_tpxo9.nc",
    tidal_velocities="path/to/uv_tpxo9.nc"
)
```

### 2. Hybrid Boundary (Tides + External Data)

Boundaries use a combination of tidal constituents and external data:

```python
from rompy.schism.tides_enhanced import create_hybrid_config

config = create_hybrid_config(
    constituents=["M2", "S2", "K1", "O1"],
    tidal_database="tpxo",
    tidal_elevations="path/to/h_tpxo9.nc",
    tidal_velocities="path/to/uv_tpxo9.nc"
)
```

### 3. River Boundary

One boundary is a river with constant flow, others can be tidal:

```python
from rompy.schism.tides_enhanced import create_river_config

config = create_river_config(
    river_boundary_index=0,
    river_flow=-500.0,  # Negative for inflow
    other_boundaries="tidal",  # Other boundaries use tides
    constituents=["M2", "S2", "K1", "O1"]
)
```

### 4. Nested Model Boundary

Boundaries use external data with relaxation, optionally with tides:

```python
from rompy.schism.tides_enhanced import create_nested_config

config = create_nested_config(
    with_tides=True,
    inflow_relax=0.8,
    outflow_relax=0.8,
    constituents=["M2", "S2", "K1", "O1"]
)
```

## Customizing Boundary Configurations

For more control, you can directly specify configurations for each boundary:

```python
from rompy.schism.tides_enhanced import SCHISMDataTidesEnhanced, BoundarySetup
from rompy.schism.boundary_tides import ElevationType, VelocityType, TracerType

# Create configurations for each boundary
boundaries = {
    0: BoundarySetup(  # Ocean boundary with tides
        elev_type=ElevationType.TIDAL,
        vel_type=VelocityType.TIDAL
    ),
    1: BoundarySetup(  # River boundary
        elev_type=ElevationType.NONE,
        vel_type=VelocityType.CONSTANT,
        const_flow=-200.0  # Negative for inflow
    ),
    2: BoundarySetup(  # Nested boundary with relaxation
        elev_type=ElevationType.SPACETIME,
        vel_type=VelocityType.RELAXED,
        temp_type=TracerType.SPACETIME,
        salt_type=TracerType.SPACETIME,
        inflow_relax=0.9,
        outflow_relax=0.9
    )
}

# Create the configuration
config = SCHISMDataTidesEnhanced(
    constituents=["M2", "S2", "K1", "O1"],
    boundaries=boundaries
)
```

## Advanced Features

### Earth Tidal Potential

To enable earth tidal potential:

```python
config = create_tidal_only_config(
    constituents=["M2", "S2", "K1", "O1"],
    ntip=1  # Enable earth tidal potential
)
```

### Flather Radiation Boundary

For a Flather radiation boundary:

```python
boundary.set_boundary_type(
    boundary_index=1,
    elev_type=ElevationType.NONE,
    vel_type=VelocityType.FLATHER,
    eta_mean=[0.0, 0.0, 0.0],  # Mean elevation at each node
    vn_mean=[[0.1], [0.1], [0.1]]  # Mean normal velocity at each node
)
```

### Relaxation Parameters for Nested Boundaries

For type `-4` boundaries with relaxation:

```python
boundary.set_boundary_type(
    boundary_index=0,
    elev_type=ElevationType.SPACETIME,
    vel_type=VelocityType.RELAXED,
    inflow_relax=0.8,  # Strong nudging for inflow
    outflow_relax=0.2  # Weak nudging for outflow
)
```

## Integration with SCHISM Configuration

The enhanced tidal handling integrates with the SCHISM configuration system:

```python
from rompy.schism.config import SCHISMConfig
from rompy.schism.grid import SCHISMGrid
from rompy.schism.tides_enhanced import create_tidal_only_config
from rompy.schism.data import SCHISMData

# Create grid
grid = SCHISMGrid(hgrid="path/to/hgrid.gr3")

# Create tidal data
tides = create_tidal_only_config(constituents=["M2", "S2", "K1", "O1"])

# Create SCHISM data
data = SCHISMData(tides=tides)

# Create SCHISM configuration
config = SCHISMConfig(grid=grid, data=data)
```

## Migration from Legacy Code

To migrate from the legacy implementation:

1. Replace `SCHISMDataTides` with `SCHISMDataTidesEnhanced`
2. Use factory functions for common configurations
3. Use enum values for boundary types instead of raw integers
4. Take advantage of the boundary-specific configuration options

Legacy code:
```python
tides = SCHISMDataTides(
    constituents=["M2", "S2", "K1", "O1"],
    flags=[[3, 3, 0, 0]],
    ethconst=[0.0],
    vthconst=[0.0]
)
```

Enhanced code:
```python
tides = SCHISMDataTidesEnhanced(
    constituents=["M2", "S2", "K1", "O1"],
    setup_type="tidal"  # Equivalent to flags=[[3, 3, 0, 0]]
)
```

## Full Example

A complete example showing how to set up a mixed boundary configuration:

```python
from rompy.core.time import TimeRange
from rompy.schism.grid import SCHISMGrid
from rompy.schism.tides_enhanced import SCHISMDataTidesEnhanced, BoundarySetup
from rompy.schism.boundary_tides import ElevationType, VelocityType, TracerType

# Create grid
grid = SCHISMGrid(hgrid="path/to/hgrid.gr3")

# Create time range
time_range = TimeRange(
    start=datetime(2022, 1, 1),
    end=datetime(2022, 2, 1)
)

# Boundary configurations
boundaries = {
    0: BoundarySetup(  # Main ocean boundary with tides
        elev_type=ElevationType.TIDAL,
        vel_type=VelocityType.TIDAL
    ),
    1: BoundarySetup(  # River inflow
        elev_type=ElevationType.NONE,
        vel_type=VelocityType.CONSTANT,
        const_flow=-500.0,  # Negative for inflow
        temp_type=TracerType.CONSTANT,
        const_temp=15.0,
        salt_type=TracerType.CONSTANT,
        const_salt=0.1
    ),
    2: BoundarySetup(  # Nested boundary
        elev_type=ElevationType.SPACETIME,
        vel_type=VelocityType.RELAXED,
        temp_type=TracerType.SPACETIME,
        salt_type=TracerType.SPACETIME,
        inflow_relax=0.9,
        outflow_relax=0.9
    )
}

# Create tidal configuration
tides = SCHISMDataTidesEnhanced(
    constituents=["M2", "S2", "K1", "O1"],
    tidal_database="tpxo",
    tidal_elevations="path/to/h_tpxo9.nc",
    tidal_velocities="path/to/uv_tpxo9.nc",
    boundaries=boundaries,
    ntip=1  # Enable earth tidal potential
)

# Generate bctides.in
output_dir = Path("output_directory")
output_dir.mkdir(exist_ok=True)
tides.get(output_dir, grid, time_range)
```