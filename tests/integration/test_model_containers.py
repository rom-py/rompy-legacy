"""
Integration tests that run SCHISM and SWAN via their Docker containers,
reusing existing test data and model configurations.

These tests require Docker and will automatically build the required images:
  - SCHISM image tagged as 'schism' (built from docker/schism/Dockerfile)
  - SWAN image tagged as 'oceanum/swan:latest' (built from docker/swan/Dockerfile)

They are marked to skip if:
  - Docker is unavailable
  - Running in CI environment (auto-detected or explicitly set)
  - SKIP_DOCKER_BUILDS or ROMPY_SKIP_DOCKER_BUILDS environment variables are set

To force Docker builds in CI, set ROMPY_ENABLE_DOCKER_IN_CI=1
"""

import os
import subprocess
from pathlib import Path

import pytest

from rompy.backends.config import DockerConfig
from rompy.model import ModelRun
from rompy.run.docker import DockerRunBackend


@pytest.mark.slow
def test_schism_container_basic_config(
    tmp_path, docker_available, should_skip_docker_builds
):
    if not docker_available:
        pytest.skip("Docker not available")
    if should_skip_docker_builds:
        pytest.skip("Skipping Potential Docker build tests in CI environment")
    """Run SCHISM via container using components-based configuration.

    Uses inline component configuration equivalent to basic_tidal.yaml to avoid
    external YAML dependency while testing the same functionality.
    """
    import tarfile

    from rompy.core.data import DataBlob
    from rompy.schism.boundary_core import TidalDataset
    from rompy.schism.config import SCHISMConfig
    from rompy.schism.data import (BoundarySetupWithSource, SCHISMData,
                                   SCHISMDataBoundaryConditions)
    from rompy.schism.grid import SCHISMGrid
    from rompy.schism.namelists import NML, Param

    # Paths
    # Use paths relative to this test file (tests/integration/test_model_containers.py)
    test_dir = Path(__file__).parent
    tides_dir = test_dir.parent / "schism" / "test_data" / "tides"
    tides_archive = tides_dir / "oceanum-atlas.tar.gz"

    # Extract tidal atlas if not already extracted (matches example runner)
    if tides_archive.exists() and not (tides_dir / "OCEANUM-atlas").exists():
        with tarfile.open(tides_archive, "r:gz") as tar:
            tar.extractall(path=tides_dir)

    # Create SCHISM grid component
    from rompy.core.data import DataBlob
    from rompy.core.types import RompyBaseModel
    from rompy.schism.grid import SCHISMGrid
    from rompy.schism.namelists.param import Param, Schout

    hgrid_blob = DataBlob(
        id="hgrid",
        source=str(test_dir.parent / "schism" / "test_data" / "hgrid.gr3"),
    )
    grid_config = SCHISMGrid(hgrid=hgrid_blob, drag=2.5e-3, crs="epsg:4326")

    # Create boundary conditions with tidal setup
    boundary_conditions = SCHISMDataBoundaryConditions(
        data_type="boundary_conditions",
        setup_type="tidal",
        tidal_data=TidalDataset(
            tidal_database=tides_dir,
            tidal_model="OCEANUM-atlas",
            constituents=["M2", "S2", "N2"],
            nodal_corrections=False,
            tidal_potential=False,
            extrapolate_tides=True,
        ),
        boundaries={
            0: BoundarySetupWithSource(
                elev_type=3, vel_type=3, temp_type=0, salt_type=0
            )
        },
    )

    # Create SCHISM data component
    data_config = SCHISMData(
        data_type="schism", boundary_conditions=boundary_conditions
    )

    # Create namelist configuration
    nml_config = NML(
        param=Param(
            core={
                "dt": 150.0,
                "ibc": 1,  # Barotropic
                "ibtp": 0,  # Don't solve tracer transport - no tracers
                "nspool": 24,  # number of time steps to spool
                "ihfskip": 1152,  # number of time steps per output file
            },
            schout={
                "iof_hydro__1": 1,  # elevation
                "iof_hydro__26": 1,  # vel. vector
                "iout_sta": 0,  # Disable station output to avoid requiring station.in file
                "nspool_sta": 4,  # number of time steps to spool for sta
            },
        )
    )

    # Create SCHISM configuration
    schism_config = SCHISMConfig(
        model_type="schism", grid=grid_config, data=data_config, nml=nml_config
    )

    # Create ModelRun with components configuration
    model_run = ModelRun(
        output_dir=str(tmp_path),
        period={"start": "20230101T00", "end": "20230101T12", "interval": 3600},
        run_id="basic_tidal_example",
        delete_existing=True,
        config=schism_config,
    )

    # Minimal container run: use mpirun with 6 processes (4 scribes + 2 compute) and latest compiled SCHISM
    run_cmd = "schism_v5.13.0 4"

    # Get dockerfile paths for DockerRunBackend to handle building if needed
    repo_root = Path(__file__).resolve().parents[2]
    context_path = repo_root / "docker" / "schism"

    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),  # Relative to build context
        build_context=context_path,
        executable=run_cmd,
        mpiexec="mpirun",
        cpu=6,
        env_vars={
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        },
    )

    # Run the model (this will generate inputs automatically)
    result = model_run.run(backend=docker_config)

    assert result is True

    # Get the generated directory from the model_run (it will be set after generation)
    generated_dir = Path(model_run.staging_dir)

    # Verify outputs
    outputs_dir = generated_dir / "outputs"
    assert outputs_dir.exists(), f"SCHISM outputs directory not created: {outputs_dir}"

    # Check for the specific expected SCHISM output file
    out2d_file = outputs_dir / "out2d_1.nc"
    assert out2d_file.exists(), f"Expected SCHISM output file not found: {out2d_file}"
    assert (
        out2d_file.stat().st_size > 512
    ), "Output file too small; model may have failed early"

    # Verify output file structure using xarray
    import xarray as xr

    ds = xr.open_dataset(out2d_file)
    print(ds)
    # Check for required dimensions
    assert "time" in ds.dims, "Missing 'time' dimension in SCHISM output"
    assert (
        "nSCHISM_hgrid_node" in ds.dims
    ), "Missing 'nSCHISM_hgrid_node' dimension in SCHISM output"

    # Check that dimensions have reasonable sizes
    assert (
        ds.dims["time"] > 1
    ), f"Time dimension too small: {ds.dims['time']} (expected > 1)"
    assert (
        ds.dims["nSCHISM_hgrid_node"] > 1
    ), f"Node dimension too small: {ds.dims['nSCHISM_hgrid_node']} (expected > 1)"

    ds.close()


@pytest.mark.slow
def test_swan_container_basic_config(
    tmp_path, docker_available, should_skip_docker_builds
):
    if not docker_available:
        pytest.skip("Docker not available")
    if should_skip_docker_builds:
        pytest.skip("Skipping Potential Docker build tests in CI environment")
    """Test SWAN container with framework integration - validates template rendering and Docker execution."""
    from rompy.swan.components.boundary import BOUNDSPEC, INITIAL
    from rompy.swan.components.cgrid import REGULAR
    from rompy.swan.components.group import INPGRIDS, LOCKUP, OUTPUT
    from rompy.swan.components.inpgrid import CURVILINEAR, INPGRID, READINP
    from rompy.swan.components.inpgrid import REGULAR as INPGRID_REGULAR
    from rompy.swan.components.inpgrid import UNSTRUCTURED
    from rompy.swan.components.lockup import COMPUTE_NONSTAT
    from rompy.swan.components.output import BLOCK
    from rompy.swan.components.physics import (BREAKING_CONSTANT,
                                               FRICTION_MADSEN, GEN3)
    from rompy.swan.components.startup import COORDINATES, MODE, PROJECT, SET
    from rompy.swan.config import SwanConfigComponents
    from rompy.swan.subcomponents.boundary import CONSTANTPAR, DEFAULT, SIDE
    from rompy.swan.subcomponents.physics import ST6
    from rompy.swan.subcomponents.readgrid import GRIDREGULAR, READINP
    from rompy.swan.subcomponents.spectrum import PM, SHAPESPEC, SPECTRUM
    from rompy.swan.subcomponents.startup import SPHERICAL
    from rompy.swan.subcomponents.time import NONSTATIONARY

    # Create cgrid component using actual component classes
    cgrid_config = REGULAR(
        spectrum=SPECTRUM(mdc=36, flow=0.04, fhigh=0.4),
        grid=GRIDREGULAR(
            xp=115.68,  # Grid origin x
            yp=-32.76,  # Grid origin y
            alp=0.0,  # Grid rotation
            xlen=0.05,  # Grid length x (50 * 0.001)
            ylen=0.03,  # Grid length y (30 * 0.001)
            mx=50,  # Number of grid points x
            my=30,  # Number of grid points y
        ),
    )

    # Synthetic bathymetry and wind using inpgrids approach with actual component classes
    inpgrid_config = INPGRIDS(
        inpgrids=[
            INPGRID_REGULAR(
                grid_type="bottom",
                xpinp=115.68,
                ypinp=-32.76,
                alpinp=0.0,
                mxinp=50,
                myinp=30,
                dxinp=0.001,
                dyinp=0.001,
                excval=-999.0,
                readinp=READINP(grid_type="bottom", fname1="bottom.txt"),
            ),
            INPGRID_REGULAR(
                grid_type="wind",
                xpinp=115.68,
                ypinp=-32.76,
                alpinp=0.0,
                mxinp=50,
                myinp=30,
                dxinp=0.001,
                dyinp=0.001,
                excval=-999.0,
                readinp=READINP(grid_type="wind", fname1="wind.txt"),
                nonstationary=NONSTATIONARY(
                    tbeg="2023-01-01T00:00:00",
                    tend="2023-01-01T06:00:00",
                    delt="PT1H",
                    tfmt=1,
                    dfmt="hr",
                ),
            ),
        ]
    )

    # Simple boundary forcing configuration using actual component classes
    boundary_config = BOUNDSPEC(
        shapespec=SHAPESPEC(shape=PM()),  # PM spectrum
        location=SIDE(side="west", direction="ccw"),
        data=CONSTANTPAR(
            hs=1.0,  # Significant wave height 1m
            per=8.0,  # Wave period 8s
            dir=90.0,  # Wave direction from east (90 degrees)
            dd=15.0,  # Directional spread 15 degrees
        ),
    )

    # Basic startup configuration using actual component classes
    startup_config = {
        "project": PROJECT(name="Test Container", nr="0001"),  # Max 16 chars
        "set": SET(
            level=0.0,
            direction_convention="nautical",
            maxerr=3,  # Maximum allowed error tolerance
        ),
        "mode": MODE(kind="nonstationary", dim="twodimensional"),
        "coordinates": COORDINATES(kind=SPHERICAL()),
    }

    # Physics configuration using actual component classes
    physics_config = {
        "gen": GEN3(
            source_terms=ST6(
                a1sds=4.75e-7,  # Required ST6 parameter
                a2sds=7.0e-5,  # Required ST6 parameter
            )
        ),
        "friction": FRICTION_MADSEN(kn=0.015),
        "breaking": BREAKING_CONSTANT(alpha=1.0, gamma=0.73),
    }

    # Initial condition (cold start) using actual component classes
    initial_config = INITIAL(kind=DEFAULT())

    # Simple output configuration using actual component classes
    output_config = OUTPUT(
        block=BLOCK(
            sname="COMPGRID",
            fname="./swangrid.nc",
            output=["depth", "hsign", "tps", "dir"],
            times={"dfmt": "min"},
        )
    )

    # Lockup configuration with COMPUTE command using actual component classes
    lockup_config = LOCKUP(
        compute=COMPUTE_NONSTAT(
            times=NONSTATIONARY(
                tbeg="2023-01-01T00:00:00",
                tend="2023-01-01T06:00:00",
                delt="PT1H",
                tfmt=1,
                dfmt="min",
            )
        )
    )

    # Simple synthetic bathymetry and wind file generation
    def create_synthetic_files(staging_dir):
        """Create synthetic bathymetry and wind files for testing."""
        # Create bathymetry file with uniform 10m depth
        bottom_file = staging_dir / "bottom.txt"
        depths = []
        for j in range(30):  # my points
            for i in range(50):  # mx points
                depths.append("10.0")
        bottom_file.write_text("\n".join(depths) + "\n")

        # Create wind file with constant 10 m/s easterly wind (u=10, v=0)
        wind_file = staging_dir / "wind.txt"
        winds = []
        for j in range(30):  # my points
            for i in range(50):  # mx points
                winds.append("10.0 0.0")  # u=10 m/s (easterly), v=0 m/s
        wind_file.write_text("\n".join(winds) + "\n")

        return bottom_file, wind_file

    config = SwanConfigComponents(
        startup=startup_config,
        cgrid=cgrid_config,
        inpgrid=inpgrid_config,
        boundary=boundary_config,
        initial=initial_config,
        physics=physics_config,
        output=output_config,
        lockup=lockup_config,
    )

    model_run = ModelRun(
        run_id="test_swan_container",
        period=dict(start="20230101T00", duration="6h", interval="1h"),
        output_dir=str(tmp_path),
        config=config,
    )

    # Generate model files
    model_run.generate()

    # Create synthetic bathymetry and wind files after generation
    staging_dir = Path(model_run.staging_dir)
    bottom_file, wind_file = create_synthetic_files(staging_dir)
    print(f"Created synthetic files: {bottom_file.name}, {wind_file.name}")

    # Use single processor to avoid MPI root issues and segfaults
    run_cmd = 'bash -c "cd /app/run_id && swan.exe"'

    # Get dockerfile paths for DockerRunBackend to handle building if needed
    repo_root = Path(__file__).resolve().parents[2]
    context_path = repo_root / "docker" / "swan"

    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),  # Relative to build context
        build_context=context_path,
        executable=run_cmd,
        cpu=1,  # Single CPU since we're not using MPI
    )

    result = model_run.run(backend=docker_config)

    # Note: result may be False due to SWAN segfault (no wave forcing),
    # but this test validates framework integration and template fixes

    generated_dir = Path(model_run.output_dir) / model_run.run_id

    # Main success: INPUT file generated properly using SwanConfigComponents
    input_file = generated_dir / "INPUT"
    assert input_file.exists(), "INPUT file should exist"

    input_content = input_file.read_text()
    assert "COMPUTE NONST" in input_content, "COMPUTE command should be present"

    # Check if SWAN produced any output files (bonus if it works)
    output_files = list((generated_dir).glob("*.nc"))
    assert output_files, "No SWAN output .nc files found in generated directory"

    # Verify output file structure using xarray (similar to SCHISM test)
    import numpy as np
    import xarray as xr

    # Check the main output file
    main_output = output_files[0]  # Usually swangrid.nc
    ds = xr.open_dataset(main_output)
    print(ds)

    # Check for required dimensions
    assert "time" in ds.dims, "Missing 'time' dimension in SWAN output"
    assert "longitude" in ds.dims, "Missing 'longitude' dimension in SWAN output"
    assert "latitude" in ds.dims, "Missing 'latitude' dimension in SWAN output"

    # Check for key wave variables
    assert (
        "hs" in ds.data_vars
    ), "Missing significant wave height 'hs' variable in SWAN output"
    assert "depth" in ds.data_vars, "Missing 'depth' variable in SWAN output"

    # Check dimensions are reasonable
    assert ds.dims["time"] > 0, "Time dimension should be positive"
    assert ds.dims["longitude"] > 0, "Longitude dimension should be positive"
    assert ds.dims["latitude"] > 0, "Latitude dimension should be positive"

    # Check that we have some non-NaN wave height values
    hs_values = ds.hs.values
    assert not np.all(np.isnan(hs_values)), "All wave height values are NaN"

    ds.close()
