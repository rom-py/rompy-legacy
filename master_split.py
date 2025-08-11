#!/usr/bin/env python3
"""
Master Repository Split Script

Single command to perform complete ROMPY repository split with all fixes and testing.

Usage:
    python master_split.py [--clean] [--no-test] [--retry-setup] [--skip-entry-points-fix]

This script:
1. Splits the repository into rompy-core, rompy-swan, rompy-schism, rompy-notebooks
2. Fixes all dependencies and imports
3. Fixes entry point namespaces and references
4. Verifies entry points configuration
5. Applies manual fixes
6. Sets up testing environments (with automatic retry)
7. Runs comprehensive tests
8. Generates final report

Author: ROMPY Development Team
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, description="", max_retries=0, retry_delay=5):
    """Run a command and return success status with optional retries."""
    logger.info(f"📍 {description}")
    logger.info(f"Running: {' '.join(command)}")

    retries = 0
    while True:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info(f"✅ {description} - SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} - FAILED")
            logger.error(f"Error: {e.stderr}")

            # Check if we should retry
            if retries < max_retries:
                retries += 1
                logger.warning(f"🔄 Retrying ({retries}/{max_retries}) in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return False
        except Exception as e:
            logger.error(f"❌ {description} - CRASHED: {e}")

            # Check if we should retry
            if retries < max_retries:
                retries += 1
                logger.warning(f"🔄 Retrying ({retries}/{max_retries}) in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            return False


def main():
    parser = argparse.ArgumentParser(description="Complete ROMPY repository split")
    parser.add_argument("--clean", action="store_true", help="Clean existing split-repos directory")
    parser.add_argument("--no-test", action="store_true", help="Skip comprehensive testing")
    parser.add_argument("--retry-setup", action="store_true", help="Retry setup steps if they fail")
    parser.add_argument("--skip-entry-points-fix", action="store_true", help="Skip fixing entry points")
    parser.add_argument("--skip-verification", action="store_true", help="Skip entry points verification")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("🚀 Starting Master Repository Split Process")
    logger.info("=" * 80)

    # Step 1: Clean if requested
    if args.clean:
        import shutil
        split_dir = Path("../split-repos")
        if split_dir.exists():
            logger.info("🧹 Cleaning existing split-repos directory...")
            shutil.rmtree(split_dir)

    # Step 2: Repository Split
    if not run_command([
        "python", "split_repository.py",
        "--config", "repo_split_config_with_cookiecutter.yaml"
    ], "Repository Split"):
        return 1

    # Step 3: Fix Dependencies
    if not run_command([
        "python", "fix_split_dependencies.py"
    ], "Fix Dependencies"):
        return 1

    # Step 4: Fix Imports
    if not run_command([
        "python", "fix_split_imports.py",
        "--all",
        "--split-repos-dir", "../split-repos"
    ], "Fix Imports"):
        return 1

    # Step 5: Fix Test Imports
    if not run_command([
        "python", "fix_test_imports.py"
    ], "Fix Test Imports"):
        return 1

    # Step 6: Fix Entry Points
    if not args.skip_entry_points_fix:
        if not run_command([
            "python", "fix_entry_points.py"
        ], "Fix Entry Points"):
            logger.warning("⚠️ Entry points fix failed - this may cause import errors")
    else:
        logger.info("🔍 Skipping entry points fix as requested")

    # Step 7: Verify Entry Points
    if not args.skip_verification:
        verify_success = run_command([
            "python", "verify_entry_points.py"
        ], "Verify Entry Points")
        if not verify_success:
            logger.warning("⚠️ Entry points verification failed - this may cause issues")
    else:
        logger.info("🔍 Skipping entry points verification as requested")

    # Step 8: Apply Manual Fixes
    if not run_command([
        "python", "apply_manual_fixes.py"
    ], "Apply Manual Fixes"):
        return 1

    # Step 8.5: Remove all 'disabled' directories from split packages
    if not run_command([
        "python", "cleanup_split_disabled_dirs.py",
        "--split-repos-dir", "../split-repos"
    ], "Cleanup Disabled Directories"):
        logger.warning("⚠️ Cleanup of disabled directories failed - this may cause test collection errors")

    # Step 8.6: Create universal test_utils in all split packages
    if not run_command([
        "python", "create_universal_test_utils.py",
        "--split-repos-dir", "../split-repos"
    ], "Create Universal test_utils"):
        logger.warning("⚠️ Creation of universal test_utils failed - this may cause test import errors")

    # Step 9: Setup Testing Environments
    packages = ['rompy', 'rompy-swan', 'rompy-schism']

    # Setup retry configuration
    max_retries = 1 if args.retry_setup else 0
    retry_delay = 5

    # Always process core package first, as others depend on it
    core_success = run_command([
        "python", "setup_split_testing.py",
        "--split-repos-dir", "../split-repos",
        "--package", "rompy"
    ], f"Setup Testing Environment for rompy", max_retries=max_retries, retry_delay=retry_delay)

    if not core_success:
        logger.warning("⚠️ Core environment setup failed - this may cause other package setups to fail")

    # Process other packages
    for package in ['rompy-swan', 'rompy-schism']:
        if not run_command([
            "python", "setup_split_testing.py",
            "--split-repos-dir", "../split-repos",
            "--package", package
        ], f"Setup Testing Environment for {package}", max_retries=max_retries, retry_delay=retry_delay):
            logger.warning(f"⚠️ Environment setup failed for {package} - continuing...")

    # Step 10: Run Tests (if requested)
    test_results = {}
    if not args.no_test:
        logger.info("🧪 Running Comprehensive Tests...")

        for package in packages:
            success = run_command([
                "python", "run_split_tests.py",
                "--split-repos-dir", "../split-repos",
                "--package", package,
                "--verbose"
            ], f"Test {package}", max_retries=0)
            test_results[package] = success

    # Final Report
    elapsed_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("🎉 MASTER REPOSITORY SPLIT COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"Total Duration: {elapsed_time:.1f} seconds")
    logger.info("")
    logger.info("SPLIT REPOSITORIES CREATED:")

    for package in packages + ['rompy-notebooks']:
        package_dir = Path(f"../split-repos/{package}")
        status = "✅" if package_dir.exists() else "❌"
        logger.info(f"  {status} {package}")

    if not args.no_test and test_results:
        logger.info("")
        logger.info("TEST RESULTS:")
        for package, success in test_results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"  {package}: {status}")

    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info("1. Review split repositories in: ../split-repos")
    logger.info("2. Each package is independently installable:")
    for package in packages:
        logger.info(f"   cd ../split-repos/{package} && pip install -e .")
    logger.info("3. Set up remote repositories and CI/CD")

    if "rompy" in test_results and not test_results["rompy"]:
        logger.info("")
        logger.info("⚠️ TROUBLESHOOTING SUGGESTIONS:")
        logger.info("- Check if entry points are properly defined in pyproject.toml")
        logger.info("- Review error logs for detailed information")
        logger.info("- Run fix_entry_points.py to update entry point namespaces")
        logger.info("- Run verify_entry_points.py to check entry point configuration")
        logger.info("- Run fix_data_file.py to fix Union type issues")
        logger.info("- Run apply_manual_fixes.py script again to ensure all fixes are applied")
        logger.info("- Try running with --retry-setup flag to automatically retry failed setup steps")

    logger.info("")
    logger.info("🎯 Repository split automation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
