#!/usr/bin/env python3
"""
Fix Data File

This script fixes issues in the data.py file in rompy-core, specifically:
- Removes AnyPath from Union type hints
- Ensures proper use of SOURCE_TYPES_TS
- Does not add any fallbacks; if SOURCE_TYPES_TS is empty, let the error happen

Usage:
    python fix_data_file.py [--split-repos-dir PATH]

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


def fix_data_py(split_repos_dir: Path) -> bool:
    """Fix the Union type in rompy-core/src/rompy_core/core/data.py."""
    data_file = split_repos_dir / "rompy-core" / "src" / "rompy_core" / "core" / "data.py"

    if not data_file.exists():
        logger.error(f"❌ Core data.py not found: {data_file}")
        return False

    try:
        # Read the file
        with open(data_file, 'r') as f:
            content = f.read()

        # Replace the incorrect Union syntax
        content = content.replace(
            "source: Union[SOURCE_TYPES_TS, AnyPath] = Field(",
            "source: Union[SOURCE_TYPES_TS] = Field("
        )

        # Write the updated content back to the file
        with open(data_file, 'w') as f:
            f.write(content)

        logger.info("✅ Fixed Union type in data.py")
        return True

    except Exception as e:
        logger.error(f"❌ Error fixing data.py: {e}")
        return False


def main():
    """Fix the data.py file in rompy-core."""
    parser = argparse.ArgumentParser(description="Fix data.py file")
    parser.add_argument("--split-repos-dir", default="../split-repos",
                      help="Path to the directory containing split repositories")
    args = parser.parse_args()

    split_repos_dir = Path(args.split_repos_dir).resolve()
    if not split_repos_dir.exists():
        logger.error(f"Split repositories directory not found: {split_repos_dir}")
        return 1

    logger.info("🔧 Fixing data.py file...")
    success = fix_data_py(split_repos_dir)

    if success:
        logger.info("✅ Successfully fixed data.py file")
        logger.info("\nNOTE: This fix removes AnyPath from the Union type.")
        logger.info("If SOURCE_TYPES_TS is empty, you will still get an error.")
        logger.info("This is expected - you need to have properly defined entry points.")
        return 0
    else:
        logger.error("❌ Failed to fix data.py file")
        return 1


if __name__ == "__main__":
    sys.exit(main())
