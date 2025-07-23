#!/usr/bin/env python3
"""
Test script for validating repository splits.

This script creates a virtual environment, installs the split packages,
and runs tests to verify the split was successful.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from typing import Dict, List, Optional


class SplitTester:
    """
    Test harness for validating repository splits.
    """

    def __init__(self, split_dir: str, venv_dir: Optional[str] = None, cleanup: bool = True):
        """
        Initialize the split tester.

        Args:
            split_dir: Directory containing the split repositories
            venv_dir: Directory for virtual environment (None for temp dir)
            cleanup: Whether to cleanup temporary files after testing
        """
        self.split_dir = Path(split_dir).resolve()
        self.cleanup = cleanup
        self.temp_dir = None

        if venv_dir:
            self.venv_dir = Path(venv_dir).resolve()
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="rompy_split_test_")
            self.venv_dir = Path(self.temp_dir) / "venv"

        self.packages = {
            'rompy-core': {
                'dir': self.split_dir / 'rompy-core',
                'import_name': 'rompy_core',
                'description': 'Core rompy library'
            },
            'rompy-swan': {
                'dir': self.split_dir / 'rompy-swan',
                'import_name': 'rompy_swan',
                'description': 'SWAN wave model plugin'
            },
            'rompy-schism': {
                'dir': self.split_dir / 'rompy-schism',
                'import_name': 'rompy_schism',
                'description': 'SCHISM model plugin'
            }
        }

        # Results tracking
        self.results = {}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.cleanup and self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_command(self, cmd: List[str], cwd: str = None, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Run a command and return the result.

        Args:
            cmd: Command and arguments
            cwd: Working directory
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess result
        """
        print(f"🔧 Running: {' '.join(cmd)}")
        if cwd:
            print(f"   Working directory: {cwd}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=capture_output,
                text=True,
                check=False
            )
            return result
        except Exception as e:
            print(f"❌ Command failed with exception: {e}")
            raise

    def create_virtual_environment(self) -> bool:
        """
        Create a virtual environment for testing.

        Returns:
            True if successful, False otherwise
        """
        print("🐍 Creating virtual environment...")

        try:
            # Remove existing venv if it exists
            if self.venv_dir.exists():
                shutil.rmtree(self.venv_dir)

            # Create virtual environment
            venv.create(self.venv_dir, with_pip=True, clear=True)

            # Verify creation
            if not self.venv_dir.exists():
                print("❌ Failed to create virtual environment")
                return False

            print(f"✅ Virtual environment created at: {self.venv_dir}")
            return True

        except Exception as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False

    def get_pip_command(self) -> List[str]:
        """Get the pip command for the virtual environment."""
        if sys.platform == "win32":
            return [str(self.venv_dir / "Scripts" / "pip.exe")]
        else:
            return [str(self.venv_dir / "bin" / "pip")]

    def get_python_command(self) -> List[str]:
        """Get the python command for the virtual environment."""
        if sys.platform == "win32":
            return [str(self.venv_dir / "Scripts" / "python.exe")]
        else:
            return [str(self.venv_dir / "bin" / "python")]

    def upgrade_pip(self) -> bool:
        """Upgrade pip in the virtual environment."""
        print("📦 Upgrading pip...")

        result = self._run_command(
            self.get_pip_command() + ["install", "--upgrade", "pip"]
        )

        if result.returncode == 0:
            print("✅ Pip upgraded successfully")
            return True
        else:
            print(f"❌ Failed to upgrade pip: {result.stderr}")
            return False

    def install_package(self, package_name: str, package_info: Dict) -> bool:
        """
        Install a package in development mode.

        Args:
            package_name: Name of the package
            package_info: Package information dictionary

        Returns:
            True if successful, False otherwise
        """
        package_dir = package_info['dir']

        if not package_dir.exists():
            print(f"❌ Package directory not found: {package_dir}")
            return False

        print(f"📦 Installing {package_name} from {package_dir}...")

        # Install in development mode with test dependencies
        result = self._run_command(
            self.get_pip_command() + ["install", "-e", f"{package_dir}[dev,test]"],
        )

        if result.returncode == 0:
            print(f"✅ {package_name} installed successfully")
            return True
        else:
            print(f"❌ Failed to install {package_name}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False

    def test_package_import(self, package_name: str, package_info: Dict) -> bool:
        """
        Test that a package can be imported.

        Args:
            package_name: Name of the package
            package_info: Package information dictionary

        Returns:
            True if import successful, False otherwise
        """
        import_name = package_info['import_name']

        print(f"🔍 Testing import of {import_name}...")

        result = self._run_command(
            self.get_python_command() + ["-c", f"import {import_name}; print(f'Successfully imported {import_name}')"]
        )

        if result.returncode == 0:
            print(f"✅ {import_name} imports successfully")
            return True
        else:
            print(f"❌ Failed to import {import_name}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False

    def run_package_tests(self, package_name: str, package_info: Dict) -> bool:
        """
        Run tests for a package.

        Args:
            package_name: Name of the package
            package_info: Package information dictionary

        Returns:
            True if tests pass, False otherwise
        """
        package_dir = package_info['dir']

        print(f"🧪 Running tests for {package_name}...")

        # Check if pytest is available and there are tests to run
        test_dir = package_dir / "tests"
        if not test_dir.exists():
            print(f"⚠️  No tests directory found for {package_name}")
            return True  # Not a failure if no tests exist

        # Run pytest
        result = self._run_command(
            self.get_python_command() + ["-m", "pytest", str(test_dir), "-v", "--tb=short"],
            cwd=str(package_dir)
        )

        if result.returncode == 0:
            print(f"✅ Tests passed for {package_name}")
            return True
        else:
            print(f"❌ Tests failed for {package_name}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False

    def test_package(self, package_name: str) -> Dict[str, bool]:
        """
        Test a single package (install, import, and run tests).

        Args:
            package_name: Name of the package to test

        Returns:
            Dictionary with test results
        """
        if package_name not in self.packages:
            raise ValueError(f"Unknown package: {package_name}")

        package_info = self.packages[package_name]
        results = {}

        print(f"\n{'='*60}")
        print(f"🔄 Testing {package_name}: {package_info['description']}")
        print(f"{'='*60}")

        # Install package
        results['install'] = self.install_package(package_name, package_info)

        # Test import (only if install succeeded)
        if results['install']:
            results['import'] = self.test_package_import(package_name, package_info)
        else:
            results['import'] = False

        # Run tests (only if import succeeded)
        if results['import']:
            results['tests'] = self.run_package_tests(package_name, package_info)
        else:
            results['tests'] = False

        return results

    def run_all_tests(self) -> Dict[str, Dict[str, bool]]:
        """
        Run tests for all packages in dependency order.

        Returns:
            Dictionary mapping package names to test results
        """
        print("🚀 Starting repository split validation...")
        print(f"Split directory: {self.split_dir}")
        print(f"Virtual environment: {self.venv_dir}")

        # Verify split directories exist
        missing_dirs = []
        for pkg_name, pkg_info in self.packages.items():
            if not pkg_info['dir'].exists():
                missing_dirs.append(f"{pkg_name} ({pkg_info['dir']})")

        if missing_dirs:
            print(f"\n❌ Missing split directories:")
            for missing in missing_dirs:
                print(f"   - {missing}")
            return {}

        # Create virtual environment
        if not self.create_virtual_environment():
            return {}

        # Upgrade pip
        if not self.upgrade_pip():
            print("⚠️  Warning: Failed to upgrade pip, continuing anyway...")

        # Test packages in dependency order (core first, then plugins)
        test_order = ['rompy-core', 'rompy-swan', 'rompy-schism']
        all_results = {}

        for package_name in test_order:
            if package_name in self.packages:
                all_results[package_name] = self.test_package(package_name)

                # Stop if core package fails
                if package_name == 'rompy-core' and not all(all_results[package_name].values()):
                    print(f"\n❌ Core package {package_name} failed, stopping tests")
                    break

        return all_results

    def print_summary(self, results: Dict[str, Dict[str, bool]]):
        """
        Print a summary of test results.

        Args:
            results: Test results dictionary
        """
        print(f"\n{'='*60}")
        print("📊 TEST SUMMARY")
        print(f"{'='*60}")

        if not results:
            print("❌ No tests were run")
            return

        all_passed = True
        for package_name, package_results in results.items():
            package_info = self.packages[package_name]
            status = "✅ PASS" if all(package_results.values()) else "❌ FAIL"
            print(f"{status} {package_name}: {package_info['description']}")

            for test_type, passed in package_results.items():
                test_status = "✅" if passed else "❌"
                print(f"   {test_status} {test_type}")

            if not all(package_results.values()):
                all_passed = False

        print(f"\n{'='*60}")
        if all_passed:
            print("🎉 ALL TESTS PASSED! The repository split appears successful.")
        else:
            print("❌ SOME TESTS FAILED. The repository split needs attention.")
        print(f"{'='*60}")

        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test script for validating repository splits"
    )
    parser.add_argument(
        "split_dir",
        help="Directory containing the split repositories"
    )
    parser.add_argument(
        "--venv-dir",
        help="Directory for virtual environment (default: temporary directory)"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Don't cleanup temporary files after testing"
    )
    parser.add_argument(
        "--package",
        help="Test only a specific package (rompy-core, rompy-swan, rompy-schism)"
    )

    args = parser.parse_args()

    # Validate split directory
    split_dir = Path(args.split_dir)
    if not split_dir.exists():
        print(f"❌ Split directory does not exist: {split_dir}")
        sys.exit(1)

    # Run tests
    with SplitTester(
        split_dir=args.split_dir,
        venv_dir=args.venv_dir,
        cleanup=not args.no_cleanup
    ) as tester:

        if args.package:
            # Test single package
            if args.package not in tester.packages:
                print(f"❌ Unknown package: {args.package}")
                print(f"Available packages: {', '.join(tester.packages.keys())}")
                sys.exit(1)

            # Create venv first
            if not tester.create_virtual_environment():
                sys.exit(1)
            if not tester.upgrade_pip():
                print("⚠️  Warning: Failed to upgrade pip, continuing anyway...")

            results = {args.package: tester.test_package(args.package)}
        else:
            # Test all packages
            results = tester.run_all_tests()

        # Print summary and exit with appropriate code
        success = tester.print_summary(results)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
