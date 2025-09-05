from pathlib import Path

import pytest

from rompy.backends.config import DockerConfig
from rompy.model import ModelRun
from rompy.run.docker import DockerRunBackend


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
    from rompy.swan.components.physics import BREAKING_CONSTANT, FRICTION_MADSEN, GEN3
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
