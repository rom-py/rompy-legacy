#!/usr/bin/env python3
"""
Fix Entry Points

This script updates entry point namespaces in all split repositories from the old 'rompy.*'
to the new package-specific namespaces (e.g., 'rompy_core.*', 'rompy_swan.*', etc).

Usage:
    python fix_entry_points.py [--split-repos-dir PATH]

Author: ROMPY Development Team
"""

import argparse
import logging
import sys
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Entry point namespace mapping
PACKAGE_MAPPINGS = {
    'rompy-core': {
        'old_prefix': 'rompy.',
        'new_prefix': 'rompy_core.',
        'module_prefix': 'rompy_core',
    },
    'rompy-swan': {
        'old_prefix': 'rompy.',
        'new_prefix': 'rompy_swan.',
        'module_prefix': 'rompy_swan',
    },
    'rompy-schism': {
        'old_prefix': 'rompy.',
        'new_prefix': 'rompy_schism.',
        'module_prefix': 'rompy_schism',
    },
}

# Python code reference patterns
CODE_PATTERNS = [
    (r'load_entry_points\(["\']rompy\.', 'load_entry_points("{module_prefix}.'),
    (r'load_entry_points\(["\']rompy_core\.', 'load_entry_points("{module_prefix}.'),
    (r'load_entry_points\(["\']rompy_swan\.', 'load_entry_points("{module_prefix}.'),
    (r'load_entry_points\(["\']rompy_schism\.', 'load_entry_points("{module_prefix}.'),
]

def fix_entry_points_in_pyproject(package_dir, package_info):
    """Update entry point namespaces in pyproject.toml file."""
    pyproject_path = package_dir / "pyproject.toml"
    if not pyproject_path.exists():
        logger.warning(f"pyproject.toml not found in {package_dir}")
        return False

    logger.info(f"Updating entry points in {pyproject_path}")

    try:
        with open(pyproject_path, 'r') as f:
            content = f.read()

        # Look for project.entry-points sections and update them
        old_prefix = package_info['old_prefix']
        new_prefix = package_info['new_prefix']

        # Update entry point namespaces
        entry_point_pattern = r'(\[project\.entry-points\.")(rompy\.)([^"]+)("\])'
        updated_content = re.sub(
            entry_point_pattern,
            r'\1' + new_prefix + r'\3\4',
            content
        )

        # Write the updated content back to the file
        with open(pyproject_path, 'w') as f:
            f.write(updated_content)

        changes_made = content != updated_content
        if changes_made:
            logger.info(f"✅ Updated entry points in {pyproject_path}")
        else:
            logger.info(f"ℹ️ No changes needed in {pyproject_path}")

        return True
    except Exception as e:
        logger.error(f"Error updating {pyproject_path}: {e}")
        return False

def fix_references_in_code(package_dir, package_info):
    """Update entry point references in Python code."""
    src_dir = package_dir / "src"
    if not src_dir.exists():
        logger.warning(f"src directory not found in {package_dir}")
        return False

    # Find all Python files
    python_files = list(src_dir.glob("**/*.py"))
    logger.info(f"Found {len(python_files)} Python files to check in {package_dir.name}")

    updated_files = 0

    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()

            original_content = content

            # Replace load_entry_points references
            for pattern, replacement in CODE_PATTERNS:
                formatted_replacement = replacement.format(module_prefix=package_info['module_prefix'])
                content = re.sub(pattern, formatted_replacement, content)

            if content != original_content:
                with open(py_file, 'w') as f:
                    f.write(content)
                logger.info(f"  ✅ Updated references in {py_file.relative_to(package_dir)}")
                updated_files += 1

        except Exception as e:
            logger.error(f"Error processing {py_file}: {e}")

    logger.info(f"Updated {updated_files} Python files in {package_dir.name}")
    return True

def verify_entry_points(package_dir):
    """Verify entry points are properly defined in pyproject.toml."""
    if package_dir.name != "rompy-core":
        return True

    pyproject_file = package_dir / "pyproject.toml"
    if not pyproject_file.exists():
        logger.warning(f"⚠️ pyproject.toml not found: {pyproject_file}")
        return False

    try:
        with open(pyproject_file, 'r') as f:
            content = f.read()

        # Verify at least one timeseries source is defined
        if ":timeseries" in content:
            logger.info(f"✅ Verified timeseries source entry points are defined in pyproject.toml")
            return True
        else:
            logger.warning(f"⚠️ No timeseries source entry points found in pyproject.toml")
            logger.warning(f"   This will cause errors with SOURCE_TYPES_TS in data.py")
            return False

    except Exception as e:
        logger.error(f"❌ Error verifying entry points: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fix entry point namespaces in split repositories")
    parser.add_argument("--split-repos-dir", default="../split-repos",
                      help="Path to the directory containing split repositories")
    args = parser.parse_args()

    split_repos_dir = Path(args.split_repos_dir).resolve()
    if not split_repos_dir.exists():
        logger.error(f"Split repositories directory not found: {split_repos_dir}")
        return 1

    logger.info(f"🔍 Scanning split repositories in {split_repos_dir}")

    success = True
    for package_name, package_info in PACKAGE_MAPPINGS.items():
        package_dir = split_repos_dir / package_name
        if not package_dir.exists():
            logger.warning(f"Package directory not found: {package_dir}")
            continue

        logger.info(f"\n📦 Processing {package_name}")

        # Fix entry points in pyproject.toml
        success = fix_entry_points_in_pyproject(package_dir, package_info) and success

        # Fix entry point references in code
        success = fix_references_in_code(package_dir, package_info) and success

        # For rompy-core, verify entry points are properly defined
        if package_name == "rompy-core":
            success = verify_entry_points(package_dir) and success

    logger.info("\n" + "=" * 80)
    if success:
        logger.info("✅ Entry point namespace fixes completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Run: python master_split.py --retry-setup")
        logger.info("\nIMPORTANT: Entry points must be correctly defined in pyproject.toml")
        logger.info("for SOURCE_TYPES_TS to be properly populated. If you encounter Union")
        logger.info("type errors, make sure there is at least one ':timeseries' entry point.")
        return 0
    else:
        logger.error("⚠️ Some fixes were not applied successfully")
        return 1

if __name__ == "__main__":
    sys.exit(main())
