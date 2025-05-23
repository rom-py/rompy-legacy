import os
from pathlib import Path

import pytest
import matplotlib.pyplot as plt

pytest.importorskip("rompy.schism")
import numpy as np
import xarray as xr

from rompy.core.data import DataBlob
from rompy.core.time import TimeRange
from rompy.schism import SCHISMGrid
from rompy.schism.data import SCHISMDataTides, TidalDataset, SCHISMData
from rompy.schism.config import SCHISMConfig

HERE = Path(__file__).parent
import logging

logging.basicConfig(level=logging.INFO)


@pytest.fixture
def tidal_data():
    """Setup tidal data for testing"""
    if not (HERE / "test_data" / "tpxo9-neaus" / "h_m2s2n2.nc").exists():
        from tests.utils import untar_file

        untar_file(HERE / "test_data" / "tpxo9-neaus.tar.gz", HERE / "test_data/")

    tides = SCHISMDataTides(
        tidal_data=TidalDataset(
            elevations=HERE / "test_data" / "tpxo9-neaus" / "h_m2s2n2.nc",
            velocities=HERE / "test_data" / "tpxo9-neaus" / "u_m2s2n2.nc",
        ),
        constituents=["M2", "S2", "N2"],
    )
    return tides


def test_plot_boundaries(tmp_path, grid2d, tidal_data):
    """Test plotting boundaries method"""
    # Generate tidal data
    tidal_data.get(
        destdir=tmp_path,
        grid=grid2d,
        time=TimeRange(start="2023-01-01", end="2023-01-02", dt=3600),
    )

    # Create SCHISMData object with tidal data
    schism_data = SCHISMData(tides=tidal_data)

    # Create SCHISMConfig object
    config = SCHISMConfig(model_type="schism", grid=grid2d, data=schism_data)

    # Test plotting boundaries
    fig, ax = config.plot_tidal_boundaries(title="Test Boundary Plot")

    # Save the plot to the output directory
    output_file = tmp_path / "boundary_plot.png"
    fig.savefig(output_file)
    plt.close(fig)

    # Verify the output file was created
    assert output_file.exists(), f"Failed to create boundary plot at {output_file}"
    logging.info(f"Created boundary plot: {output_file}")


def test_plot_tidal_stations(tmp_path, grid2d, tidal_data):
    """Test plotting tidal amplitude/phase at boundary stations"""
    # Generate tidal data
    tidal_data.get(
        destdir=tmp_path,
        grid=grid2d,
        time=TimeRange(start="2023-01-01", end="2023-01-02", dt=3600),
    )

    # Create SCHISMData object with tidal data
    schism_data = SCHISMData(tides=tidal_data)

    # Create SCHISMConfig object
    config = SCHISMConfig(model_type="schism", grid=grid2d, data=schism_data)

    # Test plotting M2 amplitude
    fig, ax = config.plot_tidal_stations(constituent="M2", property_type="amp")

    # Save the plot to the output directory
    output_file = tmp_path / "tidal_amp_plot.png"
    fig.savefig(output_file)
    plt.close(fig)

    # Verify the output file was created
    assert (
        output_file.exists()
    ), f"Failed to create tidal amplitude plot at {output_file}"
    logging.info(f"Created tidal amplitude plot: {output_file}")

    # Test plotting M2 phase
    fig, ax = config.plot_tidal_stations(constituent="M2", property_type="phase")

    # Save the plot to the output directory
    output_file = tmp_path / "tidal_phase_plot.png"
    fig.savefig(output_file)
    plt.close(fig)

    # Verify the output file was created
    assert output_file.exists(), f"Failed to create tidal phase plot at {output_file}"
    logging.info(f"Created tidal phase plot: {output_file}")


def test_plot_tidal_rose(tmp_path, grid2d, tidal_data):
    """Test plotting tidal rose diagram"""
    # Generate tidal data
    tidal_data.get(
        destdir=tmp_path,
        grid=grid2d,
        time=TimeRange(start="2023-01-01", end="2023-01-02", dt=3600),
    )

    # Create SCHISMData object with tidal data
    schism_data = SCHISMData(tides=tidal_data)

    # Create SCHISMConfig object
    config = SCHISMConfig(model_type="schism", grid=grid2d, data=schism_data)

    # Get the boundary indices
    pylibs_hgrid = grid2d.pylibs_hgrid

    # Ensure boundaries are computed
    if not hasattr(pylibs_hgrid, "nob") or pylibs_hgrid.nob is None:
        pylibs_hgrid.compute_bnd()

    if pylibs_hgrid.nob > 0:
        # Test plotting tidal rose for the first station on the first boundary
        fig = config.plot_tidal_rose(station_idx=0, boundary_idx=0)

        # Save the plot to the output directory
        output_file = tmp_path / "tidal_rose_plot.png"
        fig.savefig(output_file)
        plt.close(fig)

        # Verify the output file was created
        assert (
            output_file.exists()
        ), f"Failed to create tidal rose plot at {output_file}"
        logging.info(f"Created tidal rose plot: {output_file}")
    else:
        logging.warning("No open boundaries found, skipping tidal rose plot test")


def test_plot_tidal_dataset(tmp_path, tidal_data, grid2d):
    """Test plotting tidal dataset coverage"""
    # No need to call get() for this test as it doesn't depend on that

    # Create SCHISMData object with tidal data
    schism_data = SCHISMData(tides=tidal_data)

    # Create SCHISMConfig object
    config = SCHISMConfig(model_type="schism", grid=grid2d, data=schism_data)

    # Test plotting dataset coverage
    fig = config.plot_tidal_dataset()

    # Save the plot to the output directory
    output_file = tmp_path / "tidal_dataset_plot.png"
    fig.savefig(output_file)
    plt.close(fig)

    # Verify the output file was created
    assert output_file.exists(), f"Failed to create tidal dataset plot at {output_file}"
    logging.info(f"Created tidal dataset coverage plot: {output_file}")


def test_full_workflow(tmp_path, grid2d, tidal_data):
    """Test a full workflow with all plotting methods"""
    # Generate tidal data
    tidal_data.get(
        destdir=tmp_path,
        grid=grid2d,
        time=TimeRange(start="2023-01-01", end="2023-01-02", dt=3600),
    )

    # Create SCHISMData object with tidal data
    schism_data = SCHISMData(tides=tidal_data)

    # Create SCHISMConfig object
    config = SCHISMConfig(model_type="schism", grid=grid2d, data=schism_data)

    # Create a figure with multiple subplots for different visualizations
    fig = plt.figure(figsize=(15, 10))

    # Plot boundaries
    ax1 = plt.subplot(2, 2, 1)
    config.plot_tidal_boundaries(ax=ax1, title="SCHISM Grid Boundaries")

    # Plot M2 amplitude
    ax2 = plt.subplot(2, 2, 2)
    config.plot_tidal_stations(constituent="M2", property_type="amp", ax=ax2)

    # Plot S2 amplitude
    ax3 = plt.subplot(2, 2, 3)
    config.plot_tidal_stations(constituent="S2", property_type="amp", ax=ax3)

    # Plot N2 amplitude
    ax4 = plt.subplot(2, 2, 4)
    config.plot_tidal_stations(constituent="N2", property_type="amp", ax=ax4)

    plt.tight_layout()

    # Save the plot to the output directory
    output_file = tmp_path / "tidal_visualization_dashboard.png"
    fig.savefig(output_file)
    plt.close(fig)

    # Verify the output file was created
    assert output_file.exists(), f"Failed to create dashboard plot at {output_file}"
    logging.info(f"Created tidal visualization dashboard: {output_file}")

    # Create a separate tidal rose plot
    pylibs_hgrid = grid2d.pylibs_hgrid
    if hasattr(pylibs_hgrid, "nob") and pylibs_hgrid.nob > 0:
        fig = config.plot_tidal_rose(station_idx=0, boundary_idx=0)
        output_file = tmp_path / "tidal_rose.png"
        fig.savefig(output_file)
        plt.close(fig)

        assert (
            output_file.exists()
        ), f"Failed to create tidal rose plot at {output_file}"
        logging.info(f"Created tidal rose plot: {output_file}")

    # Create dataset coverage plot
    fig = config.plot_tidal_dataset()
    output_file = tmp_path / "dataset_coverage.png"
    fig.savefig(output_file)
    plt.close(fig)

    assert (
        output_file.exists()
    ), f"Failed to create dataset coverage plot at {output_file}"
    logging.info(f"Created dataset coverage plot: {output_file}")

    return True
