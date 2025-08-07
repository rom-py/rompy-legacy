#!/usr/bin/env python3
"""
Virtual Environment Testing Framework for ROMPY Repository Split

This script creates isolated virtual environments for testing each split repository,
handles dependency installation, and validates that all packages work correctly.

Features:
- Automated virtual environment creation for each repository
- Dependency installation with proper inter-repository dependencies
- Comprehensive test execution and reporting
- Integration with CI/CD pipeline
- Parallel testing support
- Detailed logging and error reporting

Usage:
    python setup_split_testing.py --split-repos-dir ../split-repos
    python setup_split_testing.py --split-repos-dir ../split-repos --package rompy-core
    python setup_split_testing.py --split-repos-dir ../split-repos --clean --rebuild
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("setup_split_testing.log")],
)
logger = logging.getLogger(__name__)


class VirtualEnvironmentManager:
    """
    Manages virtual environments for testing split repositories.
    """

    def __init__(self, split_repos_dir: Path, python_executable: str = "python3"):
        """
        Initialize the virtual environment manager.

        Args:
            split_repos_dir: Directory containing all split repositories
            python_executable: Python executable to use for virtual environments
        """
        self.split_repos_dir = Path(split_repos_dir).resolve()
        self.python_executable = python_executable
        self.venv_base_dir = self.split_repos_dir
        self.results = {}
        self.errors = []

        # Package dependency order (rompy-core must be installed first)
        self.package_order = [
            ("rompy-core", "rompy_core"),
            ("rompy-swan", "rompy_swan"),
            ("rompy-schism", "rompy_schism"),
            ("rompy-notebooks", "rompy_notebooks"),
        ]

    def run_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        capture_output: bool = True,
        timeout: int = 300,
    ) -> Tuple[bool, str, str]:
        """
        Run a command and return success status and output.

        Args:
            command: Command to run as list of strings
            cwd: Working directory for command
            capture_output: Whether to capture stdout/stderr
            timeout: Command timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            logger.debug(f"Running command: {' '.join(command)}")
            if cwd:
                logger.debug(f"Working directory: {cwd}")

            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                check=False,
            )

            success = result.returncode == 0
            stdout = result.stdout if capture_output else ""
            stderr = result.stderr if capture_output else ""

            if not success:
                logger.debug(f"Command failed with return code: {result.returncode}")
                logger.debug(f"Stderr: {stderr}")

            return success, stdout, stderr

        except subprocess.TimeoutExpired:
            error_msg = (
                f"Command timed out after {timeout} seconds: {' '.join(command)}"
            )
            logger.error(error_msg)
            return False, "", error_msg
        except Exception as e:
            error_msg = f"Command execution failed: {e}"
            logger.error(error_msg)
            return False, "", error_msg

    def get_venv_python(self, venv_path: Path) -> Path:
        """
        Return the path to the python executable in the virtual environment.
        """
        if (venv_path / "bin" / "python").exists():
            return venv_path / "bin" / "python"
        elif (venv_path / "Scripts" / "python.exe").exists():
            return venv_path / "Scripts" / "python.exe"
        else:
            raise FileNotFoundError(f"Python executable not found in {venv_path}")

    def get_venv_pip(self, venv_path: Path) -> Path:
        """
        Return the path to the pip executable in the virtual environment.
        """
        if (venv_path / "bin" / "pip").exists():
            return venv_path / "bin" / "pip"
        elif (venv_path / "Scripts" / "pip.exe").exists():
            return venv_path / "Scripts" / "pip.exe"
        else:
            raise FileNotFoundError(f"pip executable not found in {venv_path}")

    def upgrade_pip(self, venv_path: Path):
        """
        Upgrade pip in the virtual environment.
        """
        python_exe = self.get_venv_python(venv_path)
        self.run_command([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])

    def create_virtual_environment(self, package_name: str) -> Tuple[bool, Path]:
        """
        Create a virtual environment for a specific package.
        """
        venv_path = self.venv_base_dir / f"{package_name}/.venv"

        logger.info(f"📦 Creating virtual environment for {package_name}")
        logger.debug(f"Virtual environment path: {venv_path}")

        # Remove existing environment if it exists
        if venv_path.exists():
            shutil.rmtree(venv_path)

        # Create new virtual environment
        success, stdout, stderr = self.run_command(
            [str(self.python_executable), "-m", "venv", str(venv_path)]
        )
        if not success:
            error_msg = (
                f"Failed to create virtual environment for {package_name}: {stderr}"
            )
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False, venv_path

        pip_exe = self.get_venv_pip(venv_path)

        # Upgrade pip
        self.upgrade_pip(venv_path)

        # Install build dependencies
        self.run_command([str(pip_exe), "install", "build", "wheel", "setuptools"])

        # Install package in editable mode with test dependencies
        package_dir = self.split_repos_dir / package_name
        pyproject_file = package_dir / "pyproject.toml"
        setup_file = package_dir / "setup.py"

        if pyproject_file.exists() or setup_file.exists():
            success, stdout, stderr = self.run_command(
                [str(pip_exe), "install", "-e", ".[test]"], cwd=package_dir
            )
            if not success:
                error_msg = (
                    f"Failed to install dependencies for {package_name}: {stderr}"
                )
                logger.error(error_msg)
                self.errors.append(error_msg)
                return False, venv_path
        else:
            error_msg = f"No pyproject.toml or setup.py found in {package_dir}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False, venv_path

        logger.info(f"✅ Installed dependencies for {package_name}")
        return True, venv_path

    def install_package_dependencies(self, package_name: str, venv_path: Path) -> bool:
        """
        Install package dependencies in editable mode with test extras.
        """
        # This method is called by setup_package_environment
        # It is intentionally left as a stub for future extension.
        return True

    def install_inter_package_dependencies(
        self, package_name: str, venv_path: Path
    ) -> bool:
        """
        Install inter-package dependencies (e.g., rompy-swan depends on rompy-core).

        Args:
            package_name: Name of the package
            venv_path: Path to virtual environment

        Returns:
            True if successful, False otherwise
        """
        pip_exe = self.get_venv_pip(venv_path)

        # Define inter-package dependencies
        dependencies = {
            "rompy-swan": ["rompy-core"],
            "rompy-schism": ["rompy-core"],
            "rompy-notebooks": ["rompy-core", "rompy-swan", "rompy-schism"],
        }

        if package_name not in dependencies:
            logger.debug(f"No inter-package dependencies for {package_name}")
            return True

        logger.info(f"🔗 Installing inter-package dependencies for {package_name}")

        for dep_name in dependencies[package_name]:
            dep_dir = self.split_repos_dir / dep_name

            if not dep_dir.exists():
                logger.warning(f"Dependency directory not found: {dep_dir}")
                continue

            logger.debug(f"Installing dependency: {dep_name}")
            success, stdout, stderr = self.run_command(
                [str(pip_exe), "install", "-e", str(dep_dir)]
            )

            if not success:
                error_msg = f"Failed to install {dep_name} dependency for {package_name}: {stderr}"
                logger.error(error_msg)
                self.errors.append(error_msg)
                return False

        logger.info(f"✅ Installed inter-package dependencies for {package_name}")
        return True

    def verify_package_installation(
        self, package_name: str, module_name: str, venv_path: Path
    ) -> bool:
        """
        Verify that a package can be imported successfully.

        Args:
            package_name: Name of the package
            module_name: Python module name to import
            venv_path: Path to virtual environment

        Returns:
            True if package can be imported, False otherwise
        """
        python_exe = self.get_venv_python(venv_path)

        logger.info(f"🔍 Verifying installation of {package_name} ({module_name})")

        # Test basic import
        success, stdout, stderr = self.run_command(
            [
                str(python_exe),
                "-c",
                f"import {module_name}; print(f'{module_name} imported successfully')",
            ]
        )

        if not success:
            error_msg = f"Failed to import {module_name}: {stderr}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False

        logger.info(f"✅ Successfully verified {package_name} installation")
        return True

    def run_package_tests(
        self, package_name: str, venv_path: Path
    ) -> Tuple[bool, Dict]:
        """
        Run tests for a specific package.

        Args:
            package_name: Name of the package
            venv_path: Path to virtual environment

        Returns:
            Tuple of (success, test_results)
        """
        package_dir = self.split_repos_dir / package_name
        python_exe = self.get_venv_python(venv_path)
        pip_exe = self.get_venv_pip(venv_path)

        logger.info(f"🧪 Running tests for {package_name}")

        # Install test dependencies
        logger.debug("Installing test dependencies")
        test_deps = ["pytest", "pytest-cov", "pytest-xdist"]
        for dep in test_deps:
            success, stdout, stderr = self.run_command([str(pip_exe), "install", dep])
            if not success:
                logger.warning(f"Failed to install {dep}: {stderr}")

        # Find tests directory
        tests_dir = package_dir / "tests"
        if not tests_dir.exists():
            logger.warning(f"No tests directory found for {package_name}")
            return True, {"status": "no_tests", "message": "No tests directory found"}

        # Run pytest
        start_time = time.time()
        success, stdout, stderr = self.run_command(
            [str(python_exe), "-m", "pytest", str(tests_dir), "-v", "--tb=short"],
            cwd=package_dir,
            timeout=600,
        )  # 10 minute timeout for tests
        end_time = time.time()

        test_results = {
            "status": "passed" if success else "failed",
            "duration": end_time - start_time,
            "stdout": stdout,
            "stderr": stderr,
        }

        if success:
            logger.info(
                f"✅ Tests passed for {package_name} ({test_results['duration']:.1f}s)"
            )
        else:
            logger.error(f"❌ Tests failed for {package_name}")
            logger.debug(f"Test output: {stdout}")
            logger.debug(f"Test errors: {stderr}")

        return success, test_results

    def setup_package_environment(
        self, package_name: str, module_name: str, skip_tests: bool = False
    ) -> Dict:
        """
        Set up complete testing environment for a package.
        """
        logger.info(f"🚀 Setting up testing environment for {package_name}")

        result = {
            "package_name": package_name,
            "module_name": module_name,
            "success": False,
            "steps": {
                "venv_created": False,
                "dependencies_installed": False,
                "inter_deps_installed": False,
                "package_verified": False,
                "tests_run": False,
            },
            "test_results": None,
            "errors": [],
        }

        try:
            # Step 1: Create virtual environment
            venv_success, venv_path = self.create_virtual_environment(package_name)
            result["venv_path"] = str(venv_path)
            result["steps"]["venv_created"] = venv_success

            if not venv_success:
                return result

            # Step 2: Install package dependencies
            deps_success = self.install_package_dependencies(package_name, venv_path)
            result["steps"]["dependencies_installed"] = deps_success

            if not deps_success:
                return result

            # Step 3: Install inter-package dependencies
            inter_deps_success = self.install_inter_package_dependencies(
                package_name, venv_path
            )
            result["steps"]["inter_deps_installed"] = inter_deps_success

            if not inter_deps_success:
                return result

            # Step 4: Verify package installation
            verify_success = self.verify_package_installation(
                package_name, module_name, venv_path
            )
            result["steps"]["package_verified"] = verify_success

            if not verify_success:
                return result

            # Step 5: Run tests (if requested)
            if not skip_tests:
                tests_success, test_results = self.run_package_tests(
                    package_name, venv_path
                )
                result["steps"]["tests_run"] = tests_success
                result["test_results"] = test_results

            result["success"] = True
            logger.info(
                f"✅ Successfully set up testing environment for {package_name}"
            )

        except Exception as e:
            error_msg = f"Unexpected error setting up {package_name}: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            logger.debug(traceback.format_exc())

        return result

    def setup_all_environments(
        self, skip_tests: bool = False, parallel: bool = False
    ) -> Dict:
        """
        Set up testing environments for all packages.

        Args:
            skip_tests: Whether to skip running tests
            parallel: Whether to set up environments in parallel (dependencies permitting)

        Returns:
            Dictionary with all results
        """
        logger.info("🚀 Setting up testing environments for all split repositories")

        all_results = {
            "total_packages": len(self.package_order),
            "successful_setups": 0,
            "failed_setups": 0,
            "package_results": [],
            "overall_success": False,
            "start_time": time.time(),
        }

        if parallel:
            # For parallel execution, we need to respect dependencies
            # rompy-core first, then swan/schism in parallel, then notebooks

            # Phase 1: rompy-core
            core_result = self.setup_package_environment(
                "rompy-core", "rompy_core", skip_tests
            )
            all_results["package_results"].append(core_result)

            if not core_result["success"]:
                logger.error(
                    "❌ rompy-core setup failed, cannot continue with dependent packages"
                )
                all_results["failed_setups"] = len(self.package_order)
                return all_results

            all_results["successful_setups"] += 1

            # Phase 2: swan and schism in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for package_name, module_name in [
                    ("rompy-swan", "rompy_swan"),
                    ("rompy-schism", "rompy_schism"),
                ]:
                    future = executor.submit(
                        self.setup_package_environment,
                        package_name,
                        module_name,
                        skip_tests,
                    )
                    futures.append((future, package_name))

                for future, package_name in futures:
                    try:
                        result = future.result(timeout=1800)  # 30 minute timeout
                        all_results["package_results"].append(result)
                        if result["success"]:
                            all_results["successful_setups"] += 1
                        else:
                            all_results["failed_setups"] += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to complete setup for {package_name}: {e}"
                        )
                        all_results["failed_setups"] += 1

            # Phase 3: notebooks
            notebooks_result = self.setup_package_environment(
                "rompy-notebooks", "rompy_notebooks", skip_tests
            )
            all_results["package_results"].append(notebooks_result)
            if notebooks_result["success"]:
                all_results["successful_setups"] += 1
            else:
                all_results["failed_setups"] += 1

        else:
            # Sequential execution
            for package_name, module_name in self.package_order:
                result = self.setup_package_environment(
                    package_name, module_name, skip_tests
                )
                all_results["package_results"].append(result)

                if result["success"]:
                    all_results["successful_setups"] += 1
                else:
                    all_results["failed_setups"] += 1
                    # Continue with other packages even if one fails
                    logger.warning(
                        f"Setup failed for {package_name}, continuing with other packages"
                    )

        all_results["end_time"] = time.time()
        all_results["total_duration"] = (
            all_results["end_time"] - all_results["start_time"]
        )
        all_results["overall_success"] = all_results["failed_setups"] == 0

        logger.info(
            f"📊 Setup completed: {all_results['successful_setups']}/{all_results['total_packages']} packages successful"
        )

        return all_results

    def generate_report(self, results: Dict) -> str:
        """
        Generate comprehensive report of environment setup and testing.

        Args:
            results: Results dictionary from setup operations

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("ROMPY REPOSITORY SPLIT - TESTING ENVIRONMENT SETUP REPORT")
        report.append("=" * 80)
        report.append(f"Split repositories directory: {self.split_repos_dir}")
        report.append(f"Virtual environments directory: {self.venv_base_dir}")
        report.append(f"Python executable: {self.python_executable}")
        report.append("")

        if "total_packages" in results:
            # All environments report
            report.append(f"Total packages: {results['total_packages']}")
            report.append(f"Successful setups: {results['successful_setups']}")
            report.append(f"Failed setups: {results['failed_setups']}")
            report.append(
                f"Total duration: {results.get('total_duration', 0):.1f} seconds"
            )
            report.append(
                f"Overall success: {'✅ YES' if results['overall_success'] else '❌ NO'}"
            )
            report.append("")

            for pkg_result in results["package_results"]:
                self._add_package_report(report, pkg_result)

        else:
            # Single package report
            self._add_package_report(report, results)

        if self.errors:
            report.append("ERRORS ENCOUNTERED:")
            for error in self.errors:
                report.append(f"  - {error}")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)

    def _add_package_report(self, report: List[str], pkg_result: Dict):
        """Add package-specific information to report."""
        name = pkg_result["package_name"]
        success = "✅" if pkg_result["success"] else "❌"

        report.append(f"Package: {name} {success}")
        report.append(f"  Module: {pkg_result['module_name']}")

        if "venv_path" in pkg_result:
            report.append(f"  Virtual environment: {pkg_result['venv_path']}")

        report.append("  Setup steps:")
        for step, status in pkg_result["steps"].items():
            status_icon = "✅" if status else "❌"
            report.append(f"    - {step.replace('_', ' ').title()}: {status_icon}")

        if pkg_result["test_results"]:
            test_res = pkg_result["test_results"]
            status_icon = "✅" if test_res["status"] == "passed" else "❌"
            report.append(
                f"  Tests: {status_icon} ({test_res.get('duration', 0):.1f}s)"
            )

        if pkg_result["errors"]:
            report.append("  Errors:")
            for error in pkg_result["errors"]:
                report.append(f"    - {error}")

        report.append("")


def main():
    """Main function to handle command-line interface and orchestrate environment setup."""

    parser = argparse.ArgumentParser(
        description="Virtual environment testing framework for ROMPY repository split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Set up all testing environments
  python setup_split_testing.py --split-repos-dir ../split-repos

  # Set up specific package environment
  python setup_split_testing.py --split-repos-dir ../split-repos --package rompy-core

  # Clean and rebuild all environments
  python setup_split_testing.py --split-repos-dir ../split-repos --clean --rebuild

  # Set up environments without running tests
  python setup_split_testing.py --split-repos-dir ../split-repos --skip-tests

  # Parallel setup (faster but more resource intensive)
  python setup_split_testing.py --split-repos-dir ../split-repos --parallel
        """,
    )

    parser.add_argument(
        "--split-repos-dir",
        type=Path,
        required=True,
        help="Directory containing all split repositories",
    )

    parser.add_argument(
        "--package",
        choices=["rompy-core", "rompy-swan", "rompy-schism", "rompy-notebooks"],
        help="Set up environment for specific package only",
    )

    parser.add_argument(
        "--python",
        default="python3",
        help="Python executable to use for virtual environments (default: python3)",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean existing virtual environments before setup",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of virtual environments (implies --clean)",
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests, only set up environments",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Set up environments in parallel where possible",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.split_repos_dir.exists():
        logger.error(f"Split repositories directory not found: {args.split_repos_dir}")
        sys.exit(1)

    # Initialize environment manager
    env_manager = VirtualEnvironmentManager(args.split_repos_dir, args.python)

    try:

        # Set up environments
        if args.package:
            # Single package setup
            logger.info(f"🚀 Setting up testing environment for {args.package}")

            module_map = {
                "rompy-core": "rompy_core",
                "rompy-swan": "rompy_swan",
                "rompy-schism": "rompy_schism",
                "rompy-notebooks": "rompy_notebooks",
            }

            module_name = module_map[args.package]
            results = env_manager.setup_package_environment(
                args.package, module_name, args.skip_tests
            )
        else:
            # All packages setup
            results = env_manager.setup_all_environments(args.skip_tests, args.parallel)

        # Generate and display report
        report = env_manager.generate_report(results)
        print(report)

        # Write report to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"testing_environment_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(report)
        logger.info(f"📄 Report written to: {report_file}")

        # Write JSON results for programmatic access
        json_file = f"testing_environment_results_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 JSON results written to: {json_file}")

        # Exit with appropriate code
        if env_manager.errors or (
            isinstance(results, dict)
            and not results.get("overall_success", results.get("success", False))
        ):
            logger.error("❌ Environment setup completed with errors")
            sys.exit(1)
        else:
            logger.info("🎉 All testing environments set up successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
