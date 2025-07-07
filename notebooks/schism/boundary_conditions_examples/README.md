# SCHISM Boundary Conditions Configuration Examples

This directory contains comprehensive examples of SCHISM boundary condition configurations using the new unified boundary conditions system. These examples showcase different boundary condition setups documented in the ROMPY SCHISM boundary conditions documentation.

## Directory Structure

```
boundary_conditions_examples/
├── README.md                    # This file
├── 01_tidal_only/              # Pure tidal boundary configurations
├── 02_hybrid/                  # Combined tidal + external data
├── 03_river/                   # River boundary configurations
├── 04_nested/                  # Nested model configurations
└── 05_advanced/                # Advanced and custom configurations
```

## Quick Start Guide

### Understanding Boundary Types

The new unified boundary conditions system uses integer codes for different boundary types:

**Elevation Types (`elev_type`):**
- `0`: NONE - No elevation boundary condition
- `1`: TIMEHIST - Time history from elev.th
- `2`: CONSTANT - Constant elevation
- `3`: HARMONIC - Pure harmonic tidal elevation
- `4`: EXTERNAL - Time-varying elevation from external data
- `5`: HARMONICEXTERNAL - Combined harmonic and external elevation

**Velocity Types (`vel_type`):**
- `0`: NONE - No velocity boundary condition
- `1`: TIMEHIST - Time history from flux.th
- `2`: CONSTANT - Constant velocity/flow rate
- `3`: HARMONIC - Pure harmonic tidal velocity
- `4`: EXTERNAL - Time-varying velocity from external data
- `5`: HARMONICEXTERNAL - Combined harmonic and external velocity
- `6`: FLATHER - Flather radiation boundary
- `7`: RELAXED - Relaxation boundary (for nesting)

**Tracer Types (`temp_type`, `salt_type`):**
- `0`: NONE - No tracer boundary condition
- `1`: TIMEHIST - Time history from temp/salt.th
- `2`: CONSTANT - Constant tracer value
- `3`: INITIAL - Initial profile for inflow
- `4`: EXTERNAL - Time-varying tracer from external data

### Common Tidal Constituents

**Available in Test Data:**
- `[M2, S2, N2]` - Only these constituents are available in the provided test data files

**Example Sets for Real Applications:**
- **Basic Set**: `[M2, S2, N2, K1, O1]` - Major semi-diurnal and diurnal
- **Extended Set**: `[M2, S2, N2, K2, K1, O1, P1, Q1]` - For high-accuracy applications
- **Full Set**: `[M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, Ssa]` - Including long-period

**Note:** All examples in this directory use only `[M2, S2, N2]` to match the available test data. For real applications, you should use appropriate tidal data files that contain the full set of constituents needed for your region.

### Setup Types

The `setup_type` field provides pre-configured boundary setups:
- `"tidal"`: Pure tidal forcing
- `"hybrid"`: Tidal + external data
- `"river"`: River boundary configuration
- `"nested"`: Nested model configuration

## Available Examples

### 01_tidal_only/ - Pure Tidal Forcing

- **`basic_tidal.yaml`**: Pure tidal forcing with M2, S2, N2 constituents (elev_type=3, vel_type=3)
- **`extended_tidal.yaml`**: Tidal-only setup with refined timestep and additional namelist parameters  
- **`tidal_with_potential.yaml`**: Tidal forcing with earth tidal potential and self-attraction loading

### 02_hybrid/ - Combined Tidal + External Data

- **`hybrid_elevation.yaml`**: Combined tidal and external elevation data (elev_type=5)
- **`full_hybrid.yaml`**: Complete hybrid setup with tidal+external for elevation, velocity, temperature, and salinity

### 03_river/ - River Boundary Configurations

- **`simple_river.yaml`**: Single river inflow with constant flow/tracers plus tidal ocean boundary
- **`multi_river.yaml`**: Multiple river boundaries with different flow rates and tracer properties

### 04_nested/ - Nested Model Configurations

- **`nested_with_tides.yaml`**: Nested boundary conditions with relaxation and tidal forcing

### 05_advanced/ - Advanced Configurations

*Note: Advanced examples are currently disabled as they require specialized grid configurations with multiple open boundaries.*

## Usage Instructions

### Basic Workflow

1. **Choose your use case**: Select the directory that matches your modeling scenario
2. **Copy a template**: Use the closest example as a starting point
3. **Modify paths**: Update file paths to point to your data
4. **Adjust parameters**: Modify constituents, time periods, and boundary types as needed
5. **Validate**: Check that your configuration loads without errors

### File Path Conventions

All examples use relative paths from the project root:
```yaml
# Tidal data
elevations: tests/schism/test_data/tpxo9-neaus/h_m2s2n2.nc
velocities: tests/schism/test_data/tpxo9-neaus/u_m2s2n2.nc

# Grid files  
source: tests/schism/test_data/hgrid.gr3

# Ocean data
uri: tests/schism/test_data/hycom.nc
```

### Customizing for Your Domain

When adapting these examples:

1. **Update grid files**: Replace with your domain's hgrid.gr3 and vgrid.in
2. **Update file paths**: Change all `tests/schism/test_data/` paths to point to your actual data files
3. **Update tidal data**: Use tidal data covering your domain with full constituent sets
4. **Update tidal constituents**: Replace `[M2, S2, N2]` with appropriate constituents for your region (e.g., `[M2, S2, N2, K1, O1]` for most coastal applications)
5. **Update ocean data**: Use appropriate ocean model data (HYCOM, CMEMS, etc.)
6. **Adjust time period**: Set realistic start/end times for your simulation

## Common Patterns

### Data Source Configuration

```yaml
# Simple file source
source: 
  model_type: file
  uri: path/to/data.nc

# With coordinate mapping
coords:
  t: time
  x: xlon
  y: ylat
  z: depth

# With variable selection
variables:
  - surf_el
  - temperature
  - salinity
```

### Boundary Setup Pattern

```yaml
boundaries:
  0:  # Boundary index (0 applies to all open boundaries)
    elev_type: 5    # HARMONICEXTERNAL
    vel_type: 3     # HARMONIC
    temp_type: 4    # EXTERNAL
    salt_type: 4    # EXTERNAL
    # Data sources for external types
    elev_source:
      data_type: boundary
      source: 
        model_type: file
        uri: path/to/elevation.nc
      variables: [surf_el]
```

### Hotstart Integration

```yaml
data:
  data_type: schism
  # ... other configurations ...
  hotstart:
    enabled: true
    temp_var: temperature
    salt_var: salinity
    output_filename: hotstart.nc
    coords:
      t: time
      x: xlon
      y: ylat
      z: depth
```

## Best Practices

### Performance Considerations

1. **Start simple**: Begin with basic tidal-only configurations
2. **Run from project root**: The test runner script ensures you're running from the correct directory
3. **Validate data**: Ensure all data files cover your domain and time period
4. **Check units**: River flows are in m³/s (negative for inflow)
5. **Optimize constituents**: Use only necessary tidal constituents

## Common Pitfalls

1. **Limited test constituents**: Examples use only `[M2, S2, N2]` - update for real applications
2. **File paths**: Ensure all file paths are correct relative to your project root directory
3. **Working directory**: Always run the test script from within the ROMPY repository
4. **Mismatched coordinates**: Ensure coordinate names match your data files
5. **Wrong boundary indices**: Check that boundary indices match your grid
6. **Inconsistent time periods**: Ensure all data covers the simulation period
7. **Missing dependencies**: Include all required data sources
8. **Insufficient tidal constituents**: Use region-appropriate constituent sets for accurate results

### Debugging Tips

1. **Check validation errors**: Read error messages carefully
2. **Verify file paths**: Ensure all files exist and are readable
3. **Test with short runs**: Start with short time periods for testing
4. **Use diagnostic output**: Enable relevant output flags for debugging

## Related Documentation

- **SCHISM Boundary Conditions Documentation**: `rompy/docs/source/schism/boundary_conditions.rst`
- **Data Sources**: Core data handling documentation  
- **Grid Configuration**: SCHISM grid setup documentation

## Contributing New Examples

When adding new examples:

1. **Follow naming conventions**: Use descriptive, consistent names
2. **Add comprehensive comments**: Explain the purpose and key features
3. **Include use case description**: Document when to use this configuration  
4. **Test thoroughly**: Ensure examples work with test data
5. **Update this README**: Add your example to the appropriate section

## Support

For questions about these examples:
1. Check the boundary conditions documentation first
2. Review similar examples in the appropriate directory
3. Look at the test files for additional patterns
4. Consult the ROMPY development team for complex scenarios