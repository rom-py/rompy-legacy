#!/usr/bin/env python3
"""
Comprehensive Test Runner for ROMPY Repository Split

This script runs comprehensive tests across all split repositories using the
pre-configured virtual environments created by setup_split_testing.py.

Features:
- Comprehensive test execution and reporting
- Parallel test execution support
- Integration testing across packages
- Detailed test result analysis
- Coverage reporting
- Performance benchmarking
- CI/CD integration support

Usage:
    python run_split_tests.py --split-repos-dir ../split-repos
    python run_split_tests.py --split-repos-dir ../split-repos --package rompy-core
    python run_split_tests.py --split-repos-dir ../split-repos --coverage --parallel
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import traceback
import re


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('run_split_tests.log')
    ]
)
logger = logging.getLogger(__name__)


class TestRunner:
    """
    Comprehensive test runner for split repositories.
    """

    def __init__(self, split_repos_dir: Path):
        """
        Initialize the test runner.

        Args:
            split_repos_dir: Directory containing all split repositories
        """
        self.split_repos_dir = Path(split_repos_dir).resolve()
        self.venv_base_dir = self.split_repos_dir / ".testing-envs"
        self.results = {}
        self.errors = []

        # Test order (based on dependencies)
        self.test_order = [
            ('rompy', 'rompy'),
            ('rompy-swan', 'rompy_swan'),
            ('rompy-schism', 'rompy_schism'),
            ('rompy-notebooks', 'rompy_notebooks'),
        ]

    def get_venv_python(self, venv_path: Path) -> Path:
        """Get the Python executable path for a virtual environment."""
        if os.name == 'nt':  # Windows
            return venv_path / "Scripts" / "python.exe"
        else:  # Unix-like
            return venv_path / "bin" / "python"

    def get_venv_pip(self, venv_path: Path) -> Path:
        """Get the pip executable path for a virtual environment."""
        if os.name == 'nt':  # Windows
            return venv_path / "Scripts" / "pip.exe"
        else:  # Unix-like
            return venv_path / "bin" / "pip"

    def run_command(self, command: List[str], cwd: Optional[Path] = None,
                   capture_output: bool = True, timeout: int = 600) -> Tuple[bool, str, str]:
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
                check=False
            )

            success = result.returncode == 0
            stdout = result.stdout if capture_output else ""
            stderr = result.stderr if capture_output else ""

            return success, stdout, stderr

        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds: {' '.join(command)}"
            logger.error(error_msg)
            return False, "", error_msg
        except Exception as e:
            error_msg = f"Command execution failed: {e}"
            logger.error(error_msg)
            return False, "", error_msg

    def parse_pytest_output(self, output: str) -> Dict:
        """
        Parse pytest output to extract test statistics.

        Args:
            output: Raw pytest output

        Returns:
            Dictionary with parsed test statistics
        """
        stats = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'warnings': 0,
            'duration': 0.0,
            'success_rate': 0.0
        }

        try:
            # Extract test counts from pytest summary line
            # Example: "= 85 passed, 2 skipped in 15.23s ="
            summary_pattern = r'=+\s*(\d+)\s+passed(?:,\s*(\d+)\s+failed)?(?:,\s*(\d+)\s+skipped)?(?:,\s*(\d+)\s+error)?.*?in\s+([\d.]+)s\s*=+'
            match = re.search(summary_pattern, output)

            if match:
                stats['passed'] = int(match.group(1) or 0)
                stats['failed'] = int(match.group(2) or 0)
                stats['skipped'] = int(match.group(3) or 0)
                stats['errors'] = int(match.group(4) or 0)
                stats['duration'] = float(match.group(5) or 0)
                stats['total_tests'] = stats['passed'] + stats['failed'] + stats['skipped'] + stats['errors']

            # Calculate success rate
            if stats['total_tests'] > 0:
                stats['success_rate'] = (stats['passed'] / stats['total_tests']) * 100

            # Count warnings
            warning_pattern = r'(\d+)\s+warning'
            warning_match = re.search(warning_pattern, output)
            if warning_match:
                stats['warnings'] = int(warning_match.group(1))

        except Exception as e:
            logger.warning(f"Failed to parse pytest output: {e}")

        return stats

    def run_basic_tests(self, package_name: str, venv_path: Path, coverage: bool = False) -> Dict:
        """
        Run basic unit tests for a package.

        Args:
            package_name: Name of the package
            venv_path: Path to virtual environment
            coverage: Whether to collect coverage data

        Returns:
            Dictionary with test results
        """
        package_dir = self.split_repos_dir / package_name
        python_exe = self.get_venv_python(venv_path)
        pip_exe = self.get_venv_pip(venv_path)

        logger.info(f"🧪 Running basic tests for {package_name}")

        # Ensure test dependencies are installed
        test_deps = ["pytest", "pytest-cov", "pytest-xdist", "pytest-html"]
        for dep in test_deps:
            success, stdout, stderr = self.run_command([str(pip_exe), "install", dep])
            if not success:
                logger.warning(f"Failed to install {dep}: {stderr}")

        # Find tests directory
        tests_dir = package_dir / "tests"
        if not tests_dir.exists():
            return {
                "status": "no_tests",
                "message": "No tests directory found",
                "stats": {},
                "duration": 0.0
            }

        # Prepare pytest command
        pytest_cmd = [str(python_exe), "-m", "pytest", str(tests_dir), "-v", "--tb=short"]

        if coverage:
            pytest_cmd.extend([
                "--cov", str(package_dir / "src"),
                "--cov-report", "term-missing",
                "--cov-report", f"html:{package_dir}/htmlcov"
            ])

        # Add HTML report
        pytest_cmd.extend(["--html", str(package_dir / "test_report.html"), "--self-contained-html"])

        # Run pytest
        start_time = time.time()
        success, stdout, stderr = self.run_command(pytest_cmd, cwd=package_dir, timeout=900)
        end_time = time.time()

        # Parse test output
        stats = self.parse_pytest_output(stdout)
        stats['duration'] = end_time - start_time

        result = {
            "status": "passed" if success else "failed",
            "stats": stats,
            "duration": stats['duration'],
            "stdout": stdout,
            "stderr": stderr
        }

        if success:
            logger.info(f"✅ Basic tests passed for {package_name} "
                       f"({stats['passed']}/{stats['total_tests']} tests, {stats['duration']:.1f}s)")
        else:
            logger.error(f"❌ Basic tests failed for {package_name}")

        return result

    def run_import_tests(self, package_name: str, module_name: str, venv_path: Path) -> Dict:
        """
        Run import tests to verify all modules can be imported.

        Args:
            package_name: Name of the package
            module_name: Python module name
            venv_path: Path to virtual environment

        Returns:
            Dictionary with import test results
        """
        python_exe = self.get_venv_python(venv_path)

        logger.info(f"📦 Running import tests for {package_name}")

        import_tests = [
            f"import {module_name}",
            f"from {module_name} import *",
        ]

        # Add package-specific import tests
        if module_name == 'rompy':
            import_tests.extend([
                "from rompy.core import BaseConfig",
                "from rompy.model import ModelRun",
                "from rompy.formatting import configure_logging",
                "from rompy.utils import load_config",
            ])
        elif module_name == 'rompy_swan':
            import_tests.extend([
                "from rompy_swan import SwanConfig",
                "from rompy_swan.components import SwanGrid",
            ])
        elif module_name == 'rompy_schism':
            import_tests.extend([
                "from rompy_schism import SCHISMConfig",
                "from rompy_schism import SCHISMGrid",
            ])

        results = []
        total_time = 0

        for test in import_tests:
            start_time = time.time()
            success, stdout, stderr = self.run_command([
                str(python_exe), "-c", test
            ], timeout=30)
            end_time = time.time()

            duration = end_time - start_time
            total_time += duration

            results.append({
                "test": test,
                "success": success,
                "duration": duration,
                "error": stderr if not success else None
            })

            if success:
                logger.debug(f"  ✅ {test}")
            else:
                logger.error(f"  ❌ {test}: {stderr}")

        passed = sum(1 for r in results if r["success"])
        total = len(results)

        result = {
            "status": "passed" if passed == total else "failed",
            "passed": passed,
            "total": total,
            "duration": total_time,
            "tests": results
        }

        logger.info(f"{'✅' if passed == total else '❌'} Import tests: {passed}/{total} passed")

        return result

    def run_integration_tests(self, package_name: str, venv_path: Path) -> Dict:
        """
        Run integration tests between packages.

        Args:
            package_name: Name of the package
            venv_path: Path to virtual environment

        Returns:
            Dictionary with integration test results
        """
        python_exe = self.get_venv_python(venv_path)

        logger.info(f"🔗 Running integration tests for {package_name}")

        integration_tests = []

        if package_name == 'rompy-swan':
            integration_tests = [
                "import rompy_core; import rompy_swan; print('Core + Swan integration OK')",
                "from rompy_core.core import BaseConfig; from rompy_swan import SwanConfig; print('Config integration OK')",
            ]
        elif package_name == 'rompy-schism':
            integration_tests = [
                "import rompy_core; import rompy_schism; print('Core + Schism integration OK')",
                "from rompy_core.core import BaseConfig; from rompy_schism import SCHISMConfig; print('Config integration OK')",
            ]
        elif package_name == 'rompy-notebooks':
            integration_tests = [
                "import rompy_core; import rompy_swan; import rompy_schism; print('All packages integration OK')",
                "from rompy_core.model import ModelRun; print('ModelRun integration OK')",
            ]

        if not integration_tests:
            return {
                "status": "skipped",
                "message": "No integration tests defined for this package",
                "tests": []
            }

        results = []
        total_time = 0

        for test in integration_tests:
            start_time = time.time()
            success, stdout, stderr = self.run_command([
                str(python_exe), "-c", test
            ], timeout=60)
            end_time = time.time()

            duration = end_time - start_time
            total_time += duration

            results.append({
                "test": test,
                "success": success,
                "duration": duration,
                "output": stdout,
                "error": stderr if not success else None
            })

            if success:
                logger.debug("  ✅ Integration test passed")
            else:
                logger.error(f"  ❌ Integration test failed: {stderr}")

        passed = sum(1 for r in results if r["success"])
        total = len(results)

        result = {
            "status": "passed" if passed == total else "failed",
            "passed": passed,
            "total": total,
            "duration": total_time,
            "tests": results
        }

        logger.info(f"{'✅' if passed == total else '❌'} Integration tests: {passed}/{total} passed")

        return result

    def run_performance_tests(self, package_name: str, venv_path: Path) -> Dict:
        """
        Run basic performance benchmarks.

        Args:
            package_name: Name of the package
            venv_path: Path to virtual environment

        Returns:
            Dictionary with performance test results
        """
        python_exe = self.get_venv_python(venv_path)

        logger.info(f"⚡ Running performance tests for {package_name}")

        # Simple import timing test
        import_timing_test = f"""
import time
start = time.time()
import {package_name.replace('-', '_')}
end = time.time()
print(f"Import time: {{end - start:.4f}} seconds")
"""

        start_time = time.time()
        success, stdout, stderr = self.run_command([
            str(python_exe), "-c", import_timing_test
        ], timeout=30)
        end_time = time.time()

        if success and "Import time:" in stdout:
            match = re.search(r'Import time: ([\d.]+)', stdout)
            import_time = float(match.group(1)) if match else end_time - start_time
        else:
            import_time = end_time - start_time

        result = {
            "status": "passed" if success else "failed",
            "import_time": import_time,
            "total_duration": end_time - start_time,
            "output": stdout,
            "error": stderr if not success else None
        }

        logger.info(f"⚡ Import performance: {import_time:.4f}s")

        return result

    def run_package_tests(self, package_name: str, module_name: str, coverage: bool = False,
                         skip_integration: bool = False, skip_performance: bool = False) -> Dict:
        """
        Run comprehensive tests for a single package.

        Args:
            package_name: Name of the package
            module_name: Python module name
            coverage: Whether to collect coverage data
            skip_integration: Whether to skip integration tests
            skip_performance: Whether to skip performance tests

        Returns:
            Dictionary with all test results
        """
        logger.info(f"🎯 Running comprehensive tests for {package_name}")

        # Check if virtual environment exists
        venv_path = self.venv_base_dir / f"{package_name}-venv"
        if not venv_path.exists():
            error_msg = f"Virtual environment not found for {package_name}: {venv_path}"
            logger.error(error_msg)
            return {
                "package_name": package_name,
                "module_name": module_name,
                "status": "error",
                "error": error_msg,
                "tests": {}
            }

        start_time = time.time()

        test_results = {
            "package_name": package_name,
            "module_name": module_name,
            "venv_path": str(venv_path),
            "start_time": start_time,
            "tests": {}
        }

        try:
            # Run import tests
            test_results["tests"]["import"] = self.run_import_tests(package_name, module_name, venv_path)

            # Run basic unit tests
            test_results["tests"]["unit"] = self.run_basic_tests(package_name, venv_path, coverage)

            # Run integration tests (if not skipped)
            if not skip_integration:
                test_results["tests"]["integration"] = self.run_integration_tests(package_name, venv_path)

            # Run performance tests (if not skipped)
            if not skip_performance:
                test_results["tests"]["performance"] = self.run_performance_tests(package_name, venv_path)

            # Calculate overall status
            test_statuses = [result["status"] for result in test_results["tests"].values()]
            if all(status in ["passed", "skipped"] for status in test_statuses):
                test_results["status"] = "passed"
            elif any(status == "failed" for status in test_statuses):
                test_results["status"] = "failed"
            else:
                test_results["status"] = "error"

        except Exception as e:
            error_msg = f"Unexpected error testing {package_name}: {e}"
            logger.error(error_msg)
            test_results["status"] = "error"
            test_results["error"] = error_msg
            logger.debug(traceback.format_exc())

        test_results["end_time"] = time.time()
        test_results["total_duration"] = test_results["end_time"] - test_results["start_time"]

        status_icon = "✅" if test_results["status"] == "passed" else "❌"
        logger.info(f"{status_icon} Completed tests for {package_name} ({test_results['total_duration']:.1f}s)")

        return test_results

    def run_all_tests(self, coverage: bool = False, parallel: bool = False,
                     skip_integration: bool = False, skip_performance: bool = False) -> Dict:
        """
        Run tests for all packages.

        Args:
            coverage: Whether to collect coverage data
            parallel: Whether to run tests in parallel
            skip_integration: Whether to skip integration tests
            skip_performance: Whether to skip performance tests

        Returns:
            Dictionary with all test results
        """
        logger.info("🚀 Running comprehensive tests for all split repositories")

        all_results = {
            "total_packages": len(self.test_order),
            "start_time": time.time(),
            "package_results": [],
            "summary": {
                "passed": 0,
                "failed": 0,
                "errors": 0
            }
        }

        if parallel:
            # Run tests in parallel (with dependency consideration)
            # Core first, then swan/schism in parallel, then notebooks

            # Phase 1: rompy-core
            core_result = self.run_package_tests('rompy', 'rompy', coverage,
                                               skip_integration, skip_performance)
            all_results["package_results"].append(core_result)
            self._update_summary(all_results["summary"], core_result["status"])

            # Phase 2: swan and schism in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                for package_name, module_name in [('rompy-swan', 'rompy_swan'), ('rompy-schism', 'rompy_schism')]:
                    future = executor.submit(self.run_package_tests, package_name, module_name,
                                           coverage, skip_integration, skip_performance)
                    futures.append((future, package_name))

                for future, package_name in futures:
                    try:
                        result = future.result(timeout=1800)  # 30 minute timeout
                        all_results["package_results"].append(result)
                        self._update_summary(all_results["summary"], result["status"])
                    except Exception as e:
                        logger.error(f"Failed to complete tests for {package_name}: {e}")
                        error_result = {
                            "package_name": package_name,
                            "status": "error",
                            "error": str(e)
                        }
                        all_results["package_results"].append(error_result)
                        self._update_summary(all_results["summary"], "error")

            # Phase 3: notebooks
            notebooks_result = self.run_package_tests('rompy-notebooks', 'rompy_notebooks',
                                                    coverage, skip_integration, skip_performance)
            all_results["package_results"].append(notebooks_result)
            self._update_summary(all_results["summary"], notebooks_result["status"])

        else:
            # Sequential execution
            for package_name, module_name in self.test_order:
                result = self.run_package_tests(package_name, module_name, coverage,
                                              skip_integration, skip_performance)
                all_results["package_results"].append(result)
                self._update_summary(all_results["summary"], result["status"])

        all_results["end_time"] = time.time()
        all_results["total_duration"] = all_results["end_time"] - all_results["start_time"]
        all_results["overall_success"] = all_results["summary"]["failed"] == 0 and all_results["summary"]["errors"] == 0

        logger.info(f"📊 All tests completed: {all_results['summary']['passed']} passed, "
                   f"{all_results['summary']['failed']} failed, {all_results['summary']['errors']} errors "
                   f"({all_results['total_duration']:.1f}s)")

        return all_results

    def _update_summary(self, summary: Dict, status: str):
        """Update test summary with package result."""
        if status == "passed":
            summary["passed"] += 1
        elif status == "failed":
            summary["failed"] += 1
        else:
            summary["errors"] += 1

    def generate_report(self, results: Dict) -> str:
        """
        Generate comprehensive test report.

        Args:
            results: Results dictionary from test runs

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("ROMPY REPOSITORY SPLIT - COMPREHENSIVE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Split repositories directory: {self.split_repos_dir}")
        report.append(f"Test execution time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        if "total_packages" in results:
            # All packages report
            summary = results["summary"]
            report.append(f"Total packages tested: {results['total_packages']}")
            report.append(f"Passed: {summary['passed']}")
            report.append(f"Failed: {summary['failed']}")
            report.append(f"Errors: {summary['errors']}")
            report.append(f"Total duration: {results.get('total_duration', 0):.1f} seconds")
            report.append(f"Overall success: {'✅ YES' if results['overall_success'] else '❌ NO'}")
            report.append("")

            for pkg_result in results["package_results"]:
                self._add_package_test_report(report, pkg_result)

        else:
            # Single package report
            self._add_package_test_report(report, results)

        report.append("=" * 80)
        return "\n".join(report)

    def _add_package_test_report(self, report: List[str], pkg_result: Dict):
        """Add package-specific test information to report."""
        name = pkg_result["package_name"]
        status_icon = "✅" if pkg_result["status"] == "passed" else "❌"

        report.append(f"Package: {name} {status_icon}")
        report.append(f"  Status: {pkg_result['status']}")
        report.append(f"  Duration: {pkg_result.get('total_duration', 0):.1f}s")

        if "tests" in pkg_result:
            for test_type, test_result in pkg_result["tests"].items():
                status = test_result["status"]
                status_icon = "✅" if status == "passed" else "⚠️" if status == "skipped" else "❌"

                report.append(f"  {test_type.title()} Tests: {status_icon}")

                if test_type == "unit" and "stats" in test_result:
                    stats = test_result["stats"]
                    if stats.get("total_tests", 0) > 0:
                        report.append(f"    Tests: {stats['passed']}/{stats['total_tests']} passed")
                        report.append(f"    Success rate: {stats['success_rate']:.1f}%")
                        report.append(f"    Duration: {stats['duration']:.1f}s")

                elif test_type == "import" and "passed" in test_result:
                    report.append(f"    Imports: {test_result['passed']}/{test_result['total']} successful")

                elif test_type == "integration" and "passed" in test_result:
                    if test_result["total"] > 0:
                        report.append(f"    Integration: {test_result['passed']}/{test_result['total']} passed")

                elif test_type == "performance" and "import_time" in test_result:
                    report.append(f"    Import time: {test_result['import_time']:.4f}s")

        if "error" in pkg_result:
            report.append(f"  Error: {pkg_result['error']}")

        report.append("")


def main():
    """Main function to handle command-line interface and orchestrate test execution."""

    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for ROMPY repository split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_split_tests.py --split-repos-dir ../split-repos

  # Run tests for specific package
  python run_split_tests.py --split-repos-dir ../split-repos --package rompy-core

  # Run tests with coverage
  python run_split_tests.py --split-repos-dir ../split-repos --coverage

  # Run tests in parallel (faster)
  python run_split_tests.py --split-repos-dir ../split-repos --parallel

  # Run basic tests only (skip integration and performance)
  python run_split_tests.py --split-repos-dir ../split-repos --basic-only
        """
    )

    parser.add_argument(
        '--split-repos-dir',
        type=Path,
        required=True,
        help='Directory containing all split repositories'
    )

    parser.add_argument(
        '--package',
        choices=['rompy', 'rompy-swan', 'rompy-schism', 'rompy-notebooks'],
        help='Run tests for specific package only'
    )

    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Collect test coverage data'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel where possible'
    )

    parser.add_argument(
        '--basic-only',
        action='store_true',
        help='Run only basic unit and import tests (skip integration and performance)'
    )

    parser.add_argument(
        '--skip-integration',
        action='store_true',
        help='Skip integration tests'
    )

    parser.add_argument(
        '--skip-performance',
        action='store_true',
        help='Skip performance tests'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Convert split_repos_dir to Path
    args.split_repos_dir = Path(args.split_repos_dir)

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.split_repos_dir.exists():
        logger.error(f"Split repositories directory not found: {args.split_repos_dir}")
        sys.exit(1)

    # Handle basic-only flag
    skip_integration = args.skip_integration or args.basic_only
    skip_performance = args.skip_performance or args.basic_only

    # Initialize test runner
    test_runner = TestRunner(args.split_repos_dir)

    try:
        # Run tests
        if args.package:
            # Single package testing
            logger.info(f"🚀 Running tests for {args.package}")

            module_map = {
                'rompy': 'rompy',
                'rompy-swan': 'rompy_swan',
                'rompy-schism': 'rompy_schism',
                'rompy-notebooks': 'rompy_notebooks',
            }

            module_name = module_map[args.package]
            results = test_runner.run_package_tests(args.package, module_name, args.coverage,
                                                   skip_integration, skip_performance)
        else:
            # All packages testing
            results = test_runner.run_all_tests(args.coverage, args.parallel,
                                              skip_integration, skip_performance)

        # Generate and display report
        report = test_runner.generate_report(results)
        print(report)

        # Write report to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"📄 Report written to: {report_file}")

        # Write JSON results for programmatic access
        json_file = f"test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 JSON results written to: {json_file}")

        # Exit with appropriate code
        success = results.get("overall_success", results.get("status") == "passed")
        if not success or test_runner.errors:
            logger.error("❌ Tests completed with failures")
            sys.exit(1)
        else:
            logger.info("🎉 All tests passed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Test Runner for ROMPY Repository Split")
    parser.add_argument("--split-repos-dir", required=True, help="Directory containing split repositories")
    parser.add_argument("--package", choices=["rompy", "rompy-swan", "rompy-schism", "rompy-notebooks"], help="Specific package to test")
    parser.add_argument("--coverage", action="store_true", help="Collect test coverage data")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel where possible")
    parser.add_argument("--basic-only", action="store_true", help="Run only basic unit and import tests (skip integration and performance)")
    parser.add_argument("--skip-integration", action="store_true", help="Skip integration tests")
    parser.add_argument("--skip-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Convert split_repos_dir to Path
    args.split_repos_dir = Path(args.split_repos_dir)

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate arguments
    if not args.split_repos_dir.exists():
        logger.error(f"Split repositories directory not found: {args.split_repos_dir}")
        sys.exit(1)

    # Handle basic-only flag
    skip_integration = args.skip_integration or args.basic_only
    skip_performance = args.skip_performance or args.basic_only

    # Initialize test runner
    test_runner = TestRunner(args.split_repos_dir)

    try:
        # Run tests
        if args.package:
            # Single package testing
            logger.info(f"🚀 Running tests for {args.package}")

            module_map = {
                'rompy': 'rompy',
                'rompy-swan': 'rompy_swan',
                'rompy-schism': 'rompy_schism',
                'rompy-notebooks': 'rompy_notebooks',
            }

            module_name = module_map[args.package]
            results = test_runner.run_package_tests(args.package, module_name, args.coverage,
                                                   skip_integration, skip_performance)
        else:
            # All packages testing
            results = test_runner.run_all_tests(args.coverage, args.parallel,
                                              skip_integration, skip_performance)

        # Generate and display report
        report = test_runner.generate_report(results)
        print(report)

        # Write report to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"test_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"📄 Report written to: {report_file}")

        # Write JSON results for programmatic access
        json_file = f"test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📄 JSON results written to: {json_file}")

        # Exit with appropriate code
        success = results.get("overall_success", results.get("status") == "passed")
        if not success or test_runner.errors:
            logger.error("❌ Tests completed with failures")
            sys.exit(1)
        else:
            logger.info("🎉 All tests passed successfully!")
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
