#!/usr/bin/env python3
"""
SCHISM Real Data Model Run and Plotting Demonstration

This script demonstrates:
1. Initializing a SCHISM model run using a real configuration file
2. Generating model input files
3. Creating comprehensive plots of the generated data
4. Demonstrating all plotting capabilities with real SCHISM data
5. Plotting tidal input data (TPXO) and SCHISM boundary data (bctides.in)
6. Comprehensive tidal analysis with multiple visualization types

Features include:
- Basic grid structure and bathymetry plots
- Boundary condition visualization
- Atmospheric forcing plots
- Tidal input data visualization (elevation, velocity, amplitude/phase maps)
- SCHISM boundary data plots (actual data used by the model)
- Comprehensive tidal analysis overview
- Model validation and quality assessment

Usage:
    python schism_real_data_plotting_demo.py [--config path/to/config.yaml] [--save-plots] [--output-dir path]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_modify_config(config_path: Path, output_base: Path) -> dict:
    """Load the demo config and modify paths for local execution."""
    logger.info(f"Loading configuration from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Modify output directory to use our local output path
    config["output_dir"] = str(output_base / "model_run")
    config["delete_existing"] = True  # Always clean up for demo

    # Ensure we have a short period for demo purposes
    if "period" in config:
        # Change end time to just 6 hours after start for demo
        start_time = config["period"]["start"]
        if "T" in start_time:
            date_part, time_part = start_time.split("T")
            hour = int(time_part[:2])
            new_hour = (hour + 6) % 24
            config["period"]["end"] = f"{date_part}T{new_hour:02d}"
        else:
            config["period"]["end"] = config["period"]["start"]  # Fallback to same time

    # Fix paths in the config to work from current directory
    def fix_paths(obj):
        """Recursively fix paths in config."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str):
                    # Fix various path patterns
                    if value.startswith("../../"):
                        # Convert ../../tests/... to tests/...
                        obj[key] = value.replace("../../", "")
                        logger.info(f"Fixed path: {value} -> {obj[key]}")
                    elif "/home/tdurrant/source/rompy/rompy/" in value:
                        # Fix absolute paths to relative
                        obj[key] = value.replace(
                            "/home/tdurrant/source/rompy/rompy/", ""
                        )
                        logger.info(f"Fixed absolute path: {value} -> {obj[key]}")
                elif isinstance(value, (dict, list)):
                    fix_paths(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    if item.startswith("../../"):
                        obj[i] = item.replace("../../", "")
                        logger.info(f"Fixed path: {item} -> {obj[i]}")
                    elif "/home/tdurrant/source/rompy/rompy/" in item:
                        obj[i] = item.replace("/home/tdurrant/source/rompy/rompy/", "")
                        logger.info(f"Fixed absolute path: {item} -> {obj[i]}")
                elif isinstance(item, (dict, list)):
                    fix_paths(item)

    fix_paths(config)

    logger.info(f"Modified config for demo - output will go to: {config['output_dir']}")
    return config


def initialize_model_run(config_dict: dict):
    """Initialize and generate a SCHISM model run."""
    logger.info("=== Initializing SCHISM Model Run ===")

    try:
        from rompy.model import ModelRun

        # Create model run instance
        model_run = ModelRun(**config_dict)
        logger.info(f"Created model run: {model_run.run_id}")
        logger.info(f"Period: {model_run.period}")
        logger.info(f"Output directory: {model_run.staging_dir}")

        # Generate model input files
        logger.info("Generating model input files...")
        staging_dir = model_run.generate()
        logger.info(f"Model files generated in: {staging_dir}")

        return model_run, Path(staging_dir)

    except Exception as e:
        logger.error(f"Error initializing model run: {e}")
        raise


def find_generated_files(staging_dir: Path) -> dict:
    """Find and catalog the generated SCHISM input files."""
    logger.info("=== Cataloging Generated Files ===")

    files = {
        "grid": None,
        "vgrid": None,
        "bctides": None,
        "atmospheric": [],
        "boundary_data": [],
        "other": [],
    }

    # Look for common SCHISM files
    for file_path in staging_dir.rglob("*"):
        if file_path.is_file():
            name = file_path.name.lower()

            if name.endswith(".gr3") or name == "hgrid":
                files["grid"] = file_path
                logger.info(f"Found grid file: {file_path}")
            elif name == "vgrid.in" or name == "vgrid":
                files["vgrid"] = file_path
                logger.info(f"Found vgrid file: {file_path}")
            elif name == "bctides.in":
                files["bctides"] = file_path
                logger.info(f"Found bctides file: {file_path}")
            elif (
                "sflux" in name
                or name.endswith(".nc")
                and any(x in name for x in ["air", "wind", "prc"])
            ):
                files["atmospheric"].append(file_path)
                logger.info(f"Found atmospheric file: {file_path}")
            elif name.endswith(".th.nc") or name.endswith(".th"):
                files["boundary_data"].append(file_path)
                logger.info(f"Found boundary data file: {file_path}")
            else:
                files["other"].append(file_path)

    logger.info(f"File catalog complete:")
    logger.info(f"  Grid files: {1 if files['grid'] else 0}")
    logger.info(f"  Atmospheric files: {len(files['atmospheric'])}")
    logger.info(f"  Boundary data files: {len(files['boundary_data'])}")
    logger.info(f"  Other files: {len(files['other'])}")

    return files


def create_plots_from_real_data(
    files: dict, model_run, save_plots: bool = False, output_dir: Path = None
):
    """Create comprehensive plots using the real generated SCHISM data."""
    logger.info("=== Creating Plots from Real SCHISM Data ===")

    try:
        # Import plotting components
        from rompy.schism.plotting import SchismPlotter

        logger.info("Successfully imported SCHISM plotting components")

        # Initialize plotter with the generated files
        if files["grid"]:
            logger.info(f"Initializing plotter with grid file: {files['grid']}")
            plotter = SchismPlotter(config=model_run.config, grid_file=files["grid"])
        else:
            logger.warning("No grid file found, using config only")
            plotter = SchismPlotter(config=model_run.config)

        # Create plot output directory
        if save_plots and output_dir:
            plot_dir = output_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            logger.info(f"Plots will be saved to: {plot_dir}")
        else:
            plot_dir = None

        # Collect figures for interactive display
        interactive_figures = []

        # 1. Basic Grid Plots
        logger.info("Creating basic grid plots...")
        try:
            fig, ax = plotter.plot_grid(figsize=(12, 10))
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "01_grid_structure.png", dpi=150, bbox_inches="tight"
                )
                logger.info("Saved grid structure plot")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create grid plot: {e}")

        # 2. Bathymetry Plot
        logger.info("Creating bathymetry plot...")
        try:
            fig, ax = plotter.plot_bathymetry(figsize=(12, 10))
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "02_bathymetry.png", dpi=150, bbox_inches="tight"
                )
                logger.info("Saved bathymetry plot")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create bathymetry plot: {e}")

        # 3. Boundaries Plot
        logger.info("Creating boundaries plot...")
        try:
            fig, ax = plotter.plot_boundaries(figsize=(12, 10))
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "03_boundaries.png", dpi=150, bbox_inches="tight"
                )
                logger.info("Saved boundaries plot")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create boundaries plot: {e}")

        # 4. Atmospheric Forcing Plots (if available)
        if files["atmospheric"]:
            logger.info("Creating atmospheric forcing plots...")
            for i, atm_file in enumerate(files["atmospheric"]):
                try:
                    fig, ax = plotter.plot_atmospheric_data(
                        atmospheric_file=atm_file, figsize=(14, 10)
                    )
                    if save_plots and plot_dir:
                        fig.savefig(
                            plot_dir / f"04_atmospheric_forcing_{i+1}.png",
                            dpi=150,
                            bbox_inches="tight",
                        )
                        logger.info(f"Saved atmospheric forcing plot {i+1}")
                        plt.close(fig)
                    else:
                        interactive_figures.append(fig)
                except Exception as e:
                    logger.warning(
                        f"Could not create atmospheric forcing plot for {atm_file}: {e}"
                    )

        # 5. Boundary Data Plots (if available)
        if files["boundary_data"]:
            logger.info("Creating boundary data plots...")
            for i, boundary_file in enumerate(
                files["boundary_data"][:3]
            ):  # Limit to first 3
                try:
                    fig, ax = plotter.plot_boundary_data(boundary_file, figsize=(12, 8))
                    if save_plots and plot_dir:
                        fig.savefig(
                            plot_dir / f"05_boundary_data_{i+1}.png",
                            dpi=150,
                            bbox_inches="tight",
                        )
                        logger.info(f"Saved boundary data plot {i+1}")
                        plt.close(fig)
                    else:
                        interactive_figures.append(fig)
                except Exception as e:
                    logger.warning(
                        f"Could not create boundary data plot for {boundary_file}: {e}"
                    )

        # 6. Comprehensive Overview
        logger.info("Creating comprehensive overview...")
        try:
            # Pass atmospheric_file explicitly if available
            atm_file = files["atmospheric"][0] if files["atmospheric"] else None
            fig, axes = plotter.plot_comprehensive_overview(
                figsize=(20, 16), atmospheric_file=atm_file
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "06_comprehensive_overview.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info("Saved comprehensive overview")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create comprehensive overview: {e}")

        # 7. Grid Analysis Overview
        logger.info("Creating grid analysis overview...")
        try:
            atm_file = files["atmospheric"][0] if files["atmospheric"] else None
            fig, axes = plotter.plot_grid_analysis_overview(
                figsize=(16, 12), atmospheric_file=atm_file
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "07_grid_analysis.png", dpi=150, bbox_inches="tight"
                )
                logger.info("Saved grid analysis overview")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create grid analysis overview: {e}")

        # 8. Data Analysis Overview
        logger.info("Creating data analysis overview...")
        try:
            atm_file = files["atmospheric"][0] if files["atmospheric"] else None
            fig, axes = plotter.plot_data_analysis_overview(
                figsize=(16, 12), atmospheric_file=atm_file
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "08_data_analysis.png", dpi=150, bbox_inches="tight"
                )
                logger.info("Saved data analysis overview")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create data analysis overview: {e}")

        # 9. Model Validation
        logger.info("Running model validation...")
        try:
            validation_results = plotter.run_model_validation()
            logger.info(f"Completed {len(validation_results)} validation checks")

            # Print validation summary
            status_counts = {"PASS": 0, "WARNING": 0, "FAIL": 0}
            for result in validation_results:
                status_counts[result.status] += 1
                logger.info(
                    f"  {result.check_name}: {result.status} - {result.message}"
                )

            logger.info(
                f"Validation Summary: {status_counts['PASS']} PASS, "
                f"{status_counts['WARNING']} WARNING, {status_counts['FAIL']} FAIL"
            )

            # Validation summary plot
            fig, axes = plotter.plot_validation_summary(figsize=(14, 10))
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "09_validation_summary.png", dpi=150, bbox_inches="tight"
                )
                logger.info("Saved validation summary")
                plt.close(fig)
            else:
                interactive_figures.append(fig)

        except Exception as e:
            logger.warning(f"Could not run model validation: {e}")

        # 10. Tidal Input Data Plots (TPXO data)
        logger.info("Creating tidal input data plots...")
        try:
            # Plot tidal elevation time series from input TPXO data
            fig, ax = plotter.plot_tidal_inputs_at_points(
                plot_type="elevation", time_hours=24, figsize=(14, 8)
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "10_tidal_input_elevation.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info("Saved tidal input elevation plot")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create tidal input elevation plot: {e}")

        try:
            # Plot tidal velocity magnitude time series from input TPXO data
            fig, ax = plotter.plot_tidal_inputs_at_points(
                plot_type="velocity_magnitude", time_hours=24, figsize=(14, 8)
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "11_tidal_input_velocity.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info("Saved tidal input velocity plot")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create tidal input velocity plot: {e}")

        try:
            # Plot spatial amplitude/phase maps from TPXO for M2 constituent
            fig, axes = plotter.plot_tidal_amplitude_phase_maps(
                constituent="M2", variable="elevation", figsize=(16, 8)
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "12_tidal_amplitude_phase_M2.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info("Saved tidal M2 amplitude/phase maps")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create tidal amplitude/phase maps: {e}")

        # 11. SCHISM Boundary Data Plots (actual data SCHISM uses)
        if files["bctides"]:
            logger.info("Creating SCHISM boundary data plots...")
            try:
                # Plot actual SCHISM boundary elevation data from bctides.in
                fig, ax = plotter.plot_schism_boundary_data(
                    bctides_file=files["bctides"],
                    plot_type="elevation",
                    figsize=(14, 8),
                )
                if save_plots and plot_dir:
                    fig.savefig(
                        plot_dir / "13_schism_boundary_elevation.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    logger.info("Saved SCHISM boundary elevation plot")
                    plt.close(fig)
                else:
                    interactive_figures.append(fig)
            except Exception as e:
                logger.warning(f"Could not create SCHISM boundary elevation plot: {e}")

            try:
                # Plot SCHISM boundary configuration summary
                fig, ax = plotter.plot_schism_boundary_data(
                    bctides_file=files["bctides"], plot_type="summary", figsize=(12, 8)
                )
                if save_plots and plot_dir:
                    fig.savefig(
                        plot_dir / "14_schism_boundary_summary.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    logger.info("Saved SCHISM boundary summary plot")
                    plt.close(fig)
                else:
                    interactive_figures.append(fig)
            except Exception as e:
                logger.warning(f"Could not create SCHISM boundary summary plot: {e}")

        # 12. Comprehensive Tidal Analysis Overview
        logger.info("Creating comprehensive tidal analysis overview...")
        try:
            fig, axes = plotter.plot_tidal_analysis_overview(
                time_hours=24, figsize=(20, 16)
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "15_tidal_analysis_overview.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info("Saved comprehensive tidal analysis overview")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(
                f"Could not create comprehensive tidal analysis overview: {e}"
            )

        # 13. Atmospheric Analysis Overview (Input vs Processed Data)
        logger.info("Creating atmospheric analysis overview...")
        try:
            fig, axes = plotter.plot_atmospheric_analysis_overview(
                time_hours=24, plot_type="wind_speed", variable="air", figsize=(20, 12)
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "16_atmospheric_analysis_overview.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info("Saved atmospheric analysis overview")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create atmospheric analysis overview: {e}")

        # 14. Ocean Boundary Analysis Overview - 2D (Input vs Processed Data)
        logger.info("Creating 2D ocean boundary analysis overview...")
        try:
            fig, axes = plotter.plot_ocean_boundary_analysis_overview(
                time_hours=24,
                plot_type="elevation",
                boundary_type="2d",
                figsize=(20, 12),
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "17_ocean_boundary_2d_overview.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info("Saved 2D ocean boundary analysis overview")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create 2D ocean boundary analysis overview: {e}")

        # 15. Ocean Boundary Analysis Overview - 3D (Input vs Processed Data)
        logger.info("Creating 3D ocean boundary analysis overview...")
        try:
            fig, axes = plotter.plot_ocean_boundary_analysis_overview(
                time_hours=24,
                plot_type="velocity_magnitude",
                boundary_type="3d",
                figsize=(20, 12),
            )
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "18_ocean_boundary_3d_overview.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                logger.info("Saved 3D ocean boundary analysis overview")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create 3D ocean boundary analysis overview: {e}")

        # 16. Additional Atmospheric Plot Types
        logger.info("Creating additional atmospheric plots...")
        for plot_type in ["pressure", "temperature"]:
            try:
                fig, axes = plotter.plot_atmospheric_analysis_overview(
                    time_hours=24, plot_type=plot_type, variable="air", figsize=(20, 12)
                )
                if save_plots and plot_dir:
                    fig.savefig(
                        plot_dir / f"19_atmospheric_{plot_type}_overview.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    logger.info(f"Saved atmospheric {plot_type} overview")
                    plt.close(fig)
                else:
                    interactive_figures.append(fig)
            except Exception as e:
                logger.warning(
                    f"Could not create atmospheric {plot_type} overview: {e}"
                )

        # 17. Additional Ocean Boundary Plot Types
        logger.info("Creating additional ocean boundary plots...")
        for plot_type in ["temperature", "salinity"]:
            try:
                fig, axes = plotter.plot_ocean_boundary_analysis_overview(
                    time_hours=24,
                    plot_type=plot_type,
                    boundary_type="3d",
                    figsize=(20, 12),
                )
                if save_plots and plot_dir:
                    fig.savefig(
                        plot_dir / f"20_ocean_boundary_{plot_type}_overview.png",
                        dpi=150,
                        bbox_inches="tight",
                    )
                    logger.info(f"Saved ocean boundary {plot_type} overview")
                    plt.close(fig)
                else:
                    interactive_figures.append(fig)
            except Exception as e:
                logger.warning(
                    f"Could not create ocean boundary {plot_type} overview: {e}"
                )

        # 18. Quality Assessment
        logger.info("Creating quality assessment...")
        try:
            fig = plotter.plot_quality_assessment(figsize=(10, 10))
            if save_plots and plot_dir:
                fig.savefig(
                    plot_dir / "21_quality_assessment.png", dpi=150, bbox_inches="tight"
                )
                logger.info("Saved quality assessment")
                plt.close(fig)
            else:
                interactive_figures.append(fig)
        except Exception as e:
            logger.warning(f"Could not create quality assessment: {e}")

        # Show all plots at once for interactive mode
        if interactive_figures:
            logger.info(f"Displaying {len(interactive_figures)} plots interactively...")
            logger.info(
                "Close each plot window to see the next one, or close all to continue."
            )
            plt.show()

            # Clean up figures after showing
            for fig in interactive_figures:
                plt.close(fig)

        if save_plots and plot_dir:
            logger.info(f"All plots saved to: {plot_dir}")
            logger.info("Plot files created:")
            for plot_file in sorted(plot_dir.glob("*.png")):
                logger.info(f"  - {plot_file.name}")

    except ImportError as e:
        logger.error(f"Could not import SCHISM plotting components: {e}")
        logger.error("Make sure rompy plotting modules are properly installed")
        raise
    except Exception as e:
        logger.error(f"Error creating plots: {e}")
        raise


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="SCHISM Real Data Model Run and Plotting Demo"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("notebooks/schism/demo_nml_3d_nontidal_velocities.yaml"),
        help="Path to SCHISM configuration file",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save plots instead of displaying"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./schism_demo_output"),
        help="Output directory for model run and plots",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip model generation, only create plots from existing data",
    )

    args = parser.parse_args()

    # Check if config file exists
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        logger.error("Please provide a valid SCHISM configuration file")
        return 1

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    logger.info(f"Using output directory: {args.output_dir}")

    try:
        # Load and modify configuration
        config_dict = load_and_modify_config(args.config, args.output_dir)

        if not args.plot_only:
            # Initialize and run model
            model_run, staging_dir = initialize_model_run(config_dict)

            # Find generated files
            files = find_generated_files(staging_dir)
        else:
            logger.info("Plot-only mode: looking for existing model data...")
            model_run_dir = args.output_dir / "model_run"
            if not model_run_dir.exists():
                logger.error(
                    "No existing model data found. Run without --plot-only first."
                )
                return 1

            # Create a minimal model run object for plotting
            from rompy.model import ModelRun

            model_run = ModelRun(**config_dict)
            files = find_generated_files(model_run_dir)

        # Create plots
        create_plots_from_real_data(
            files, model_run, save_plots=args.save_plots, output_dir=args.output_dir
        )

        logger.info("=== SCHISM Real Data Demonstration Complete ===")
        logger.info(f"Results available in: {args.output_dir}")

        if args.save_plots:
            plot_dir = args.output_dir / "plots"
            if plot_dir.exists():
                plot_count = len(list(plot_dir.glob("*.png")))
                logger.info(f"Created {plot_count} plot files including:")
                logger.info("  - Grid structure and bathymetry plots")
                logger.info("  - Boundary condition visualizations")
                logger.info("  - Atmospheric forcing plots (if available)")
                logger.info(
                    "  - Tidal input data plots (TPXO elevation, velocity, amplitude/phase)"
                )
                logger.info("  - SCHISM boundary data plots (actual bctides.in data)")
                logger.info("  - Comprehensive tidal analysis overview")
                logger.info("  - Atmospheric input vs processed data comparison plots")
                logger.info(
                    "  - Ocean boundary input vs processed data comparison plots (2D/3D)"
                )
                logger.info(
                    "  - Additional atmospheric and ocean boundary parameter plots"
                )
                logger.info("  - Model validation and quality assessment")

        return 0

    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
