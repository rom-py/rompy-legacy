#!/usr/bin/env python3
"""
Verify Entry Points

This script verifies that entry points are correctly registered and loaded.
It checks:
1. Entry points in pyproject.toml files
2. That SOURCE_TYPES_TS is not empty when loaded
3. Reports on any issues found

Usage:
    python verify_entry_points.py [--split-repos-dir PATH]

Author: ROMPY Development Team
"""

import argparse
import importlib.util
import logging
import re
import sys
from importlib.metadata import entry_points
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_pyproject_entry_points(pyproject_path: Path) -> Dict[str, List[str]]:
    """Check entry points defined in pyproject.toml."""
    if not pyproject_path.exists():
        logger.error(f"❌ pyproject.toml not found: {pyproject_path}")
        return {}

    try:
        with open(pyproject_path, 'r') as f:
            content = f.read()

        # Find all entry point group sections
        entry_point_groups = {}
        pattern = r'\[project\.entry-points\."([^"]+)"\](.*?)(?=\[project\.|\Z)'
        for match in re.finditer(pattern, content, re.DOTALL):
            group_name = match.group(1)
            group_content = match.group(2)

            # Extract entry points from this group
            entry_points_list = []
            for line in group_content.strip().split('\n'):
                line = line.strip()
                if '=' in line:
                    entry_name = line.split('=')[0].strip().strip('"\'')
                    entry_points_list.append(entry_name)

            entry_point_groups[group_name] = entry_points_list

        return entry_point_groups

    except Exception as e:
        logger.error(f"❌ Error reading pyproject.toml: {e}")
        return {}


def check_installed_entry_points() -> Dict[str, List[str]]:
    """Check entry points currently registered in the environment."""
    try:
        all_entry_points = entry_points()
        entry_point_groups = {}

        for ep in all_entry_points:
            group_name = ep.group
            if group_name not in entry_point_groups:
                entry_point_groups[group_name] = []
            entry_point_groups[group_name].append(ep.name)

        return entry_point_groups
    except Exception as e:
        logger.error(f"❌ Error checking installed entry points: {e}")
        return {}


def test_load_entry_points_function(split_repos_dir: Path) -> bool:
    """Test the load_entry_points function by importing it and calling it."""
    # Add split_repos_dir to sys.path to allow importing
    sys.path.insert(0, str(split_repos_dir / "rompy-core" / "src"))

    try:
        # Try to import the utils module
        logger.info("📦 Attempting to import rompy_core.utils...")
        import rompy_core.utils

        # Get the load_entry_points function
        load_entry_points = rompy_core.utils.load_entry_points

        # Test with rompy_core.source
        logger.info("🔍 Testing load_entry_points with rompy_core.source...")
        sources = load_entry_points("rompy_core.source")
        logger.info(f"  ✅ load_entry_points('rompy_core.source') returned {len(sources)} entries")

        # Test with rompy_core.source, etype=timeseries
        logger.info("🔍 Testing load_entry_points with rompy_core.source, etype=timeseries...")
        sources_ts = load_entry_points("rompy_core.source", etype="timeseries")
        logger.info(f"  ✅ load_entry_points('rompy_core.source', etype='timeseries') returned {len(sources_ts)} entries")

        # Check if SOURCE_TYPES_TS is empty
        if len(sources_ts) == 0:
            logger.warning("⚠️ SOURCE_TYPES_TS is empty! This will cause errors in data.py")
            logger.warning("   Make sure at least one entry point with ':timeseries' is defined")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Error testing load_entry_points: {e}")
        return False
    finally:
        # Remove split_repos_dir from sys.path
        if str(split_repos_dir / "rompy-core" / "src") in sys.path:
            sys.path.remove(str(split_repos_dir / "rompy-core" / "src"))


def test_source_types_ts_in_data(split_repos_dir: Path) -> bool:
    """Test that SOURCE_TYPES_TS is properly defined and used in data.py."""
    data_py_path = split_repos_dir / "rompy-core" / "src" / "rompy_core" / "core" / "data.py"

    if not data_py_path.exists():
        logger.error(f"❌ data.py not found: {data_py_path}")
        return False

    try:
        with open(data_py_path, 'r') as f:
            content = f.read()

        # Check if SOURCE_TYPES_TS is defined
        if "SOURCE_TYPES_TS = load_entry_points" not in content:
            logger.error("❌ SOURCE_TYPES_TS is not defined in data.py")
            return False

        # Check how SOURCE_TYPES_TS is used in Union
        union_pattern = r"source:\s+Union\[([^\]]+)\]"
        union_match = re.search(union_pattern, content)

        if not union_match:
            logger.error("❌ Union type not found in data.py")
            return False

        union_contents = union_match.group(1)
        logger.info(f"📊 Union type in data.py: Union[{union_contents}]")

        # Check if AnyPath is in the Union
        if "AnyPath" in union_contents and "SOURCE_TYPES_TS" in union_contents:
            logger.warning("⚠️ Both SOURCE_TYPES_TS and AnyPath are in the Union type")
            logger.warning("   This may cause errors if SOURCE_TYPES_TS is empty")
            return False

        return True

    except Exception as e:
        logger.error(f"❌ Error checking data.py: {e}")
        return False


def main():
    """Verify entry points are correctly registered and loaded."""
    parser = argparse.ArgumentParser(description="Verify entry points")
    parser.add_argument("--split-repos-dir", default="../split-repos",
                      help="Path to the directory containing split repositories")
    args = parser.parse_args()

    split_repos_dir = Path(args.split_repos_dir).resolve()
    if not split_repos_dir.exists():
        logger.error(f"Split repositories directory not found: {split_repos_dir}")
        return 1

    logger.info("🔍 Verifying entry points...")
    logger.info("=" * 80)

    # Check each package's pyproject.toml
    packages = ['rompy-core', 'rompy-swan', 'rompy-schism']
    all_verified = True

    for package in packages:
        logger.info(f"\n📦 Checking {package}...")
        package_dir = split_repos_dir / package

        if not package_dir.exists():
            logger.warning(f"⚠️ Package directory not found: {package_dir}")
            all_verified = False
            continue

        # Check pyproject.toml
        pyproject_path = package_dir / "pyproject.toml"
        entry_point_groups = check_pyproject_entry_points(pyproject_path)

        logger.info(f"📊 Entry point groups defined in {package}:")
        for group, entries in entry_point_groups.items():
            logger.info(f"  - {group}: {len(entries)} entries")
            # Check for timeseries entry points
            timeseries_entries = [e for e in entries if ":timeseries" in e]
            if timeseries_entries:
                logger.info(f"    ✅ Found {len(timeseries_entries)} timeseries entries: {timeseries_entries}")
            elif package == "rompy-core" and "rompy_core.source" in group:
                logger.warning(f"    ⚠️ No timeseries entries found in {group}")
                all_verified = False

    # Test the load_entry_points function
    logger.info("\n🔧 Testing load_entry_points function...")
    load_success = test_load_entry_points_function(split_repos_dir)
    if not load_success:
        all_verified = False

    # Test SOURCE_TYPES_TS in data.py
    logger.info("\n🔧 Testing SOURCE_TYPES_TS in data.py...")
    data_success = test_source_types_ts_in_data(split_repos_dir)
    if not data_success:
        all_verified = False

    # Summary
    logger.info("\n" + "=" * 80)
    if all_verified:
        logger.info("✅ All entry points verified successfully!")
        return 0
    else:
        logger.error("❌ Some entry point verifications failed")
        logger.info("\nTo fix entry point issues:")
        logger.info("1. Ensure pyproject.toml has entry points with ':timeseries' in rompy_core.source")
        logger.info("2. Run: python fix_entry_points.py")
        logger.info("3. Run: python fix_data_file.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())
