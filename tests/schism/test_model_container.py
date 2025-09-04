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
    from rompy.schism.data import (
        BoundarySetupWithSource,
        SCHISMData,
        SCHISMDataBoundaryConditions,
    )
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
