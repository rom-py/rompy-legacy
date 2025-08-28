#!/usr/bin/env python3
"""
SCHISM Boundary Conditions Examples Test Runner

This script runs through all the boundary condition configuration examples
using the new ROMPY backend framework to validate their functionality and
demonstrate their usage.

This replaces the previous bash script with a Python implementation that
uses the new backend system for Docker execution with automatic image building.

Usage: python run_boundary_conditions_examples.py [OPTIONS]

Options:
  --all              Run all examples (default)
  --tidal            Run only tidal examples
  --hybrid           Run only hybrid examples
  --river            Run only river examples
  --nested           Run only nested examples
  --single <name>    Run single example by name
  --dry-run          Show what would be run without executing
  --keep-outputs     Keep output directories after run
  --help             Show this help message
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from rompy.backends import DockerConfig
from rompy.core.time import TimeRange

# ROMPY imports
from rompy.model import ModelRun

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ANSI color codes for output
class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    BOLD = "\033[1m"
    NC = "\033[0m"  # No Color


def log_info(message: str) -> None:
    """Log info message with color."""
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def log_success(message: str) -> None:
    """Log success message with color."""
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def log_warning(message: str) -> None:
    """Log warning message with color."""
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def log_error(message: str) -> None:
    """Log error message with color."""
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


def log_header(message: str) -> None:
    """Log header message with formatting."""
    print()
    print("=" * 50)
    print(f"{Colors.BOLD}{message}{Colors.NC}")
    print("=" * 50)


class SchismExampleRunner:
    """Runner for SCHISM boundary condition examples using the backend framework."""

    def __init__(self):
        """Initialize the runner with example configurations."""
        self.schism_version = "v5.13.0"
        self.project_root = self._find_project_root()
        self.examples_dir = (
            self.project_root / "notebooks" / "schism" / "boundary_conditions_examples"
        )
        self.base_output_dir = self.project_root / "boundary_conditions_test_outputs"

        # Example definitions with metadata
        self.examples = {
            # Tidal-only examples
            "basic_tidal": {
                "file": self.examples_dir / "01_tidal_only" / "basic_tidal.yaml",
                "description": "Pure tidal forcing with M2, S2, N2 constituents (elev_type=3, vel_type=3)",
                "category": "tidal",
                "schism_dir": "schism_tidal_basic/basic_tidal_example",
                "exe_suffix": "",
            },
            "extended_tidal": {
                "file": self.examples_dir / "01_tidal_only" / "extended_tidal.yaml",
                "description": "Tidal-only setup with refined timestep and additional namelist parameters",
                "category": "tidal",
                "schism_dir": "schism_tidal_extended/extended_tidal_example",
                "exe_suffix": "",
            },
            "tidal_with_potential": {
                "file": self.examples_dir
                / "01_tidal_only"
                / "tidal_with_potential.yaml",
                "description": "Tidal forcing with earth tidal potential and self-attraction loading",
                "category": "tidal",
                "schism_dir": "schism_tidal_potential/tidal_potential_example",
                "exe_suffix": "",
            },
            "tide_wave": {
                "file": self.examples_dir / "01_tidal_only" / "tide_wave.yaml",
                "description": "Tidal forcing with wave interaction (WWM) for wave-current interaction",
                "category": "tidal",
                "schism_dir": "schism_tide_wave/tide_wave_example",
                "exe_suffix": "_WWM",
            },
            "tidal_with_mdt": {
                "file": self.examples_dir / "01_tidal_only" / "tidal_with_mdt.yaml",
                "description": "Tidal forcing with Mean Dynamic Topography (MDT) correction",
                "category": "tidal",
                "schism_dir": "schism_tidal_with_mdt/tidal_with_mdt_example",
                "exe_suffix": "",
            },
            "tidal_with_mdt_const": {
                "file": self.examples_dir
                / "01_tidal_only"
                / "tidal_with_mdt_const.yaml",
                "description": "Tidal forcing with constant MDT correction",
                "category": "tidal",
                "schism_dir": "schism_tidal_with_mdt_const/tidal_with_mdt_const_example",
                "exe_suffix": "",
            },
            # Hybrid examples
            "hybrid_elevation": {
                "file": self.examples_dir / "02_hybrid" / "hybrid_elevation.yaml",
                "description": "Combined tidal and external elevation data (elev_type=5)",
                "category": "hybrid",
                "schism_dir": "schism_hybrid_elevation/hybrid_elevation_example",
                "exe_suffix": "",
            },
            "full_hybrid": {
                "file": self.examples_dir / "02_hybrid" / "full_hybrid.yaml",
                "description": "Complete hybrid setup: tidal+external for elevation, velocity, temperature, salinity",
                "category": "hybrid",
                "schism_dir": "schism_full_hybrid/full_hybrid_example",
                "exe_suffix": "",
            },
            # River examples
            "simple_river": {
                "file": self.examples_dir / "03_river" / "simple_river.yaml",
                "description": "River inflow (boundary 1) with constant flow/tracers, tidal ocean boundary",
                "category": "river",
                "schism_dir": "schism_simple_river/simple_river_example",
                "exe_suffix": "",
            },
            "multi_river": {
                "file": self.examples_dir / "03_river" / "multi_river.yaml",
                "description": "Multiple river boundaries with different flow rates and tracer properties",
                "category": "river",
                "schism_dir": "schism_multi_river/multi_river_example",
                "exe_suffix": "",
            },
            # Nested examples
            "nested_with_tides": {
                "file": self.examples_dir / "04_nested" / "nested_with_tides.yaml",
                "description": "Nested boundary conditions with relaxation and tidal forcing",
                "category": "nested",
                "schism_dir": "schism_nested_with_tides/nested_with_tides_example",
                "exe_suffix": "",
            },
        }

    def _find_project_root(self) -> Path:
        """Find the project root directory."""
        current_dir = Path(__file__).parent

        # Navigate up to find the rompy root
        while current_dir != current_dir.parent:
            # Look for project indicators
            if (
                (current_dir / "setup.py").exists()
                or (current_dir / "pyproject.toml").exists()
                or (current_dir / ".git").exists()
            ):
                return current_dir
            # Also check if this directory contains the rompy package
            if (current_dir / "rompy").exists() and (current_dir / "rompy").is_dir():
                return current_dir
            current_dir = current_dir.parent

        # If not found, assume current directory
        return Path.cwd()

    def _extract_tidal_data(self) -> bool:
        """Extract tidal data from archive if needed."""
        tidal_archive = (
            self.project_root
            / "tests"
            / "schism"
            / "test_data"
            / "tides"
            / "oceanum-atlas.tar.gz"
        )
        tidal_dir = self.project_root / "tests" / "schism" / "test_data" / "tides"

        if not tidal_archive.exists():
            log_error(f"Tidal data archive not found at {tidal_archive}")
            return False

        # Check if already extracted
        if (tidal_dir / "OCEANUM-atlas").exists():
            log_info("Tidal data already extracted")
            return True

        log_info("Extracting tidal data...")
        import tarfile

        try:
            with tarfile.open(tidal_archive, "r:gz") as tar:
                tar.extractall(path=tidal_dir)
            log_success("Tidal data extracted successfully")
            return True
        except Exception as e:
            log_error(f"Failed to extract tidal data: {e}")
            return False

    def _validate_docker_files(self) -> bool:
        """Validate that required Docker files exist."""
        dockerfile_path = self.project_root / "docker" / "schism" / "Dockerfile"
        context_path = self.project_root / "docker" / "schism"

        if not dockerfile_path.exists():
            log_error(f"Dockerfile not found at {dockerfile_path}")
            return False

        if not context_path.exists():
            log_error(f"Docker context directory not found at {context_path}")
            return False

        log_info("Docker files validated successfully")
        return True

    def _get_examples_to_run(
        self, category: str, single_example: Optional[str] = None
    ) -> List[str]:
        """Get list of examples to run based on category."""
        if single_example:
            if single_example in self.examples:
                return [single_example]
            else:
                log_error(f"Example '{single_example}' not found")
                return []

        if category == "all":
            return list(self.examples.keys())

        return [
            name
            for name, config in self.examples.items()
            if config["category"] == category
        ]

    def _create_docker_config(self, schism_dir: Path, exe_suffix: str) -> DockerConfig:
        """Create Docker configuration for SCHISM execution."""
        # Create the command to run SCHISM
        command = f"cd /tmp/schism && mpirun --oversubscribe --allow-run-as-root -n 8 schism_{self.schism_version}{exe_suffix} 4"

        # Ensure the directory exists for volume validation
        schism_dir.mkdir(parents=True, exist_ok=True)

        return DockerConfig(
            dockerfile=Path("Dockerfile"),
            build_context=self.project_root / "docker" / "schism",
            timeout=3600,  # 1 hour timeout
            cpu=8,
            memory="4g",
            executable=f'bash -c "{command}"',
            volumes=[f"{schism_dir}:/tmp/schism:Z"],
            env_vars={
                "OMPI_ALLOW_RUN_AS_ROOT": "1",
                "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            },
            remove_container=True,
            user="root",
        )

    def _run_example(self, example_name: str, dry_run: bool = False) -> bool:
        """Run a single example."""
        example_config = self.examples[example_name]
        config_file = example_config["file"]
        description = example_config["description"]
        schism_dir_relative = example_config["schism_dir"]
        exe_suffix = example_config["exe_suffix"]

        log_header(f"Running Example: {example_name}")
        log_info(f"Description: {description}")
        log_info(f"Config file: {config_file}")

        if dry_run:
            log_info("DRY RUN: Would run example")
            return True

        # Check if config file exists
        if not config_file.exists():
            log_error(f"Configuration file not found: {config_file}")
            return False

        try:
            # Step 1: Generate SCHISM configuration using ModelRun
            log_info("Generating SCHISM configuration...")

            # Load the YAML configuration
            with open(config_file, "r") as f:
                config_data = yaml.safe_load(f)

            # Create ModelRun from the configuration
            model_run = ModelRun(**config_data)

            # Generate the configuration files
            model_run.generate()

            # Step 2: Find the generated SCHISM directory
            schism_dir = self.project_root / schism_dir_relative
            if not schism_dir.exists():
                log_error(f"Generated SCHISM directory not found: {schism_dir}")
                return False

            # Step 3: Copy station.in file if it exists
            station_file = self.project_root / "notebooks" / "schism" / "station.in"
            if station_file.exists():
                log_info("Copying station.in file to SCHISM directory")
                import shutil

                shutil.copy2(station_file, schism_dir)
            else:
                log_warning("station.in file not found, skipping copy")

            # Step 4: Create Docker configuration and run SCHISM
            log_info("Running SCHISM simulation...")
            docker_config = self._create_docker_config(schism_dir, exe_suffix)

            # Create a temporary ModelRun for execution
            temp_model = ModelRun(
                run_id=f"{example_name}_execution",
                period=TimeRange(
                    start=datetime.now(), end=datetime.now(), interval="1H"
                ),
                output_dir=str(schism_dir),
                delete_existing=False,
            )

            # Run the model using the Docker backend
            success = temp_model.run(backend=docker_config)

            if success:
                log_success(
                    f"SCHISM simulation completed successfully for {example_name}"
                )

                # Check for output files
                outputs_dir = schism_dir / "outputs"
                if outputs_dir.exists() and any(outputs_dir.glob("*.nc")):
                    log_success("Output files generated successfully")
                else:
                    log_warning("No output files found - simulation may have failed")
                    return False
            else:
                log_error(f"SCHISM simulation failed for {example_name}")
                return False

            return True

        except Exception as e:
            log_error(f"Error running example {example_name}: {e}")
            return False

    def run_examples(
        self,
        category: str = "all",
        single_example: Optional[str] = None,
        dry_run: bool = False,
        keep_outputs: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Run examples based on category or single example."""
        log_header("SCHISM Boundary Conditions Examples Test Runner")
        log_info(f"Project root: {self.project_root}")
        log_info(f"Examples directory: {self.examples_dir}")
        log_info(f"SCHISM Version: {self.schism_version}")
        log_info(f"Run category: {category}")
        log_info(f"Dry run: {dry_run}")
        log_info(f"Keep outputs: {keep_outputs}")

        # Get examples to run
        examples_to_run = self._get_examples_to_run(category, single_example)

        if not examples_to_run:
            log_warning("No examples found matching criteria")
            return [], []

        log_info(f"Examples to run: {len(examples_to_run)}")
        for example in examples_to_run:
            log_info(f"  - {example}: {self.examples[example]['description']}")

        if dry_run:
            log_info("Dry run complete - no examples were actually executed")
            return examples_to_run, []

        # Prerequisites check
        if not self._extract_tidal_data():
            return [], examples_to_run

        if not self._validate_docker_files():
            return [], examples_to_run

        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Track results
        successful_runs = []
        failed_runs = []
        start_time = time.time()

        # Run examples
        for example in examples_to_run:
            if self._run_example(example, dry_run):
                successful_runs.append(example)
            else:
                failed_runs.append(example)
            print()  # Add spacing between examples

        # Summary
        end_time = time.time()
        duration = int(end_time - start_time)

        log_header("Test Run Summary")
        log_info(f"Total duration: {duration} seconds")
        log_info(f"Total examples: {len(examples_to_run)}")
        log_success(f"Successful runs: {len(successful_runs)}")

        if successful_runs:
            for example in successful_runs:
                log_success(f"  ✓ {example}")

        if failed_runs:
            log_error(f"Failed runs: {len(failed_runs)}")
            for example in failed_runs:
                log_error(f"  ✗ {example}")

        # Clean up if requested
        if not keep_outputs and not failed_runs:
            log_info("Cleaning up output directories...")
            import shutil

            if self.base_output_dir.exists():
                shutil.rmtree(self.base_output_dir)
            log_info("Cleanup complete")
        else:
            log_info(f"Output directories preserved in: {self.base_output_dir}")

        return successful_runs, failed_runs

    def show_help(self) -> None:
        """Show help message with available examples."""
        print("SCHISM Boundary Conditions Examples Test Runner")
        print()
        print("Usage: python run_boundary_conditions_examples.py [OPTIONS]")
        print()
        print("This script uses the ROMPY backend framework with automatic Docker")
        print("image building from the SCHISM Dockerfile.")
        print()
        print("Options:")
        print("  --all              Run all examples (default)")
        print("  --tidal            Run only tidal examples")
        print("  --hybrid           Run only hybrid examples")
        print("  --river            Run only river examples")
        print("  --nested           Run only nested examples")
        print("  --single <name>    Run single example by name")
        print("  --dry-run          Show what would be run without executing")
        print("  --keep-outputs     Keep output directories after run")
        print("  --help             Show this help message")
        print()
        print("Available examples:")
        for name, config in self.examples.items():
            print(f"  {name:<25} {config['description']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SCHISM Boundary Conditions Examples Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--all",
        action="store_const",
        const="all",
        dest="category",
        help="Run all examples (default)",
    )
    parser.add_argument(
        "--tidal",
        action="store_const",
        const="tidal",
        dest="category",
        help="Run only tidal examples",
    )
    parser.add_argument(
        "--hybrid",
        action="store_const",
        const="hybrid",
        dest="category",
        help="Run only hybrid examples",
    )
    parser.add_argument(
        "--river",
        action="store_const",
        const="river",
        dest="category",
        help="Run only river examples",
    )
    parser.add_argument(
        "--nested",
        action="store_const",
        const="nested",
        dest="category",
        help="Run only nested examples",
    )
    parser.add_argument(
        "--single", type=str, dest="single_example", help="Run single example by name"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--keep-outputs", action="store_true", help="Keep output directories after run"
    )

    parser.set_defaults(category="all")

    args = parser.parse_args()

    # Create runner and execute
    runner = SchismExampleRunner()

    try:
        successful_runs, failed_runs = runner.run_examples(
            category=args.category,
            single_example=args.single_example,
            dry_run=args.dry_run,
            keep_outputs=args.keep_outputs,
        )

        # Exit with error if any runs failed
        if failed_runs:
            sys.exit(1)

        log_success("All boundary condition examples completed successfully!")

    except KeyboardInterrupt:
        log_warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
