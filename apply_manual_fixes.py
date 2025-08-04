#!/usr/bin/env python3
"""
Comprehensive script to apply all manual fixes discovered during repository split testing.
This script applies all the critical fixes needed to make the split repositories fully functional.
"""

import os
import sys
from pathlib import Path


def fix_data_file():
    """Fix the Union type in rompy-core/src/rompy_core/core/data.py"""
    print("🔧 Fixing data.py Union type...")

    split_dir = Path("../split-repos").resolve()
    data_file = split_dir / "rompy-core" / "src" / "rompy_core" / "core" / "data.py"

    if not data_file.exists():
        print(f"⚠️  Data file not found: {data_file}")
        return False

    try:
        with open(data_file, "r") as f:
            content = f.read()

        # Remove AnyPath from Union type
        content = content.replace(
            "source: Union[SOURCE_TYPES_TS, AnyPath] = Field(",
            "source: Union[SOURCE_TYPES_TS] = Field(",
        )

        # Check if there are any other instances we need to fix
        if "source: Union[SOURCE_TYPES_TS + (AnyPath,)]" in content:
            content = content.replace(
                "source: Union[SOURCE_TYPES_TS + (AnyPath,)] = Field(",
                "source: Union[SOURCE_TYPES_TS] = Field(",
            )

        with open(data_file, "w") as f:
            f.write(content)

        print("✅ Fixed data.py Union type")
        return True

    except Exception as e:
        print(f"❌ Error fixing data.py: {e}")
        return False


def apply_core_exports_fix():
    """Fix the commented exports in rompy-core/src/rompy_core/core/__init__.py"""
    print("🔧 Fixing core exports...")

    split_dir = Path("../split-repos").resolve()
    core_init = split_dir / "rompy-core" / "src" / "rompy_core" / "core" / "__init__.py"

    if not core_init.exists():
        print(f"⚠️  Core __init__.py not found: {core_init}")
        return False

    try:
        with open(core_init, "r") as f:
            content = f.read()

        # Uncomment the exports
        content = content.replace(
            "# from .config import BaseConfig", "from .config import BaseConfig"
        )
        content = content.replace(
            "# from .data import DataBlob, DataGrid, DataPoint",
            "from .data import DataBlob, DataGrid, DataPoint",
        )
        # content = content.replace('# from .filters import *', 'from .filters import *')
        content = content.replace(
            "# from .grid import BaseGrid, RegularGrid",
            "from .grid import BaseGrid, RegularGrid",
        )
        content = content.replace(
            "# from .spectrum import LogFrequency", "from .spectrum import LogFrequency"
        )
        content = content.replace(
            "# from .time import TimeRange", "from .time import TimeRange"
        )
        content = content.replace("# from .types import *", "from .types import *")

        with open(core_init, "w") as f:
            f.write(content)

        print("✅ Fixed core exports")
        return True

    except Exception as e:
        print(f"❌ Error fixing core exports: {e}")
        return False


def apply_utils_load_config_fix():
    """Add load_config function to rompy-core utils.py"""
    print("🔧 Adding load_config to utils...")

    split_dir = Path("../split-repos").resolve()
    utils_file = split_dir / "rompy-core" / "src" / "rompy_core" / "utils.py"

    if not utils_file.exists():
        print(f"⚠️  Utils file not found: {utils_file}")
        return False

    try:
        with open(utils_file, "r") as f:
            content = f.read()

        # Add load_config function if not already present
        if "def load_config(" not in content:
            load_config_function = '''

def load_config(*args, **kwargs):
    """Load configuration from file, string, or environment variable.

    This is a lazy import wrapper to avoid circular imports.
    """
    from rompy_core.cli import load_config as _load_config
    return _load_config(*args, **kwargs)
'''
            content += load_config_function

            with open(utils_file, "w") as f:
                f.write(content)

            print("✅ Added load_config to utils")
            return True
        else:
            print("✅ load_config already exists in utils")
            return True

    except Exception as e:
        print(f"❌ Error adding load_config: {e}")
        return False


def apply_swan_components_fix():
    """Add SwanGrid import to rompy-swan components/__init__.py"""
    print("🔧 Adding SwanGrid to swan components...")

    split_dir = Path("../split-repos").resolve()
    components_init = (
        split_dir / "rompy-swan" / "src" / "rompy_swan" / "components" / "__init__.py"
    )

    if not components_init.exists():
        print(f"⚠️  Swan components __init__.py not found: {components_init}")
        return False

    try:
        with open(components_init, "r") as f:
            content = f.read()

        # Add SwanGrid import if not already present
        if "from rompy_swan.grid import SwanGrid" not in content:
            content = content.strip() + "\nfrom rompy_swan.grid import SwanGrid\n"

            with open(components_init, "w") as f:
                f.write(content)

            print("✅ Added SwanGrid to swan components")
            return True
        else:
            print("✅ SwanGrid already imported in swan components")
            return True

    except Exception as e:
        print(f"❌ Error adding SwanGrid import: {e}")
        return False


def apply_source_unpacking_fix():
    """Fix the unpacking error in rompy-core source.py"""
    print("🔧 Fixing source.py unpacking error...")

    split_dir = Path("../split-repos").resolve()
    source_file = split_dir / "rompy-core" / "src" / "rompy_core" / "core" / "source.py"

    if not source_file.exists():
        print(f"⚠️  Source file not found: {source_file}")
        return False

    try:
        with open(source_file, "r") as f:
            content = f.read()

        # Fix the unpacking issue
        content = content.replace(
            "SourceDataset, SourceTimeseriesDataFrame = None",
            "SourceDataset, SourceTimeseriesDataFrame = None, None",
        )

        with open(source_file, "w") as f:
            f.write(content)

        print("✅ Fixed source.py unpacking error")
        return True

    except Exception as e:
        print(f"❌ Error fixing source unpacking: {e}")
        return False


def apply_schism_circular_import_fix():
    """Fix circular import in rompy-schism namelists/__init__.py"""
    print("🔧 Fixing schism circular import...")

    split_dir = Path("../split-repos").resolve()
    namelists_init = (
        split_dir
        / "rompy-schism"
        / "src"
        / "rompy_schism"
        / "namelists"
        / "__init__.py"
    )

    if not namelists_init.exists():
        print(f"⚠️  Schism namelists __init__.py not found: {namelists_init}")
        return False

    try:
        with open(namelists_init, "r") as f:
            content = f.read()

        # Fix the circular import
        content = content.replace(
            "from rompy_schism import NML", "from .schism import NML"
        )

        with open(namelists_init, "w") as f:
            f.write(content)

        print("✅ Fixed schism circular import")
        return True

    except Exception as e:
        print(f"❌ Error fixing schism circular import: {e}")
        return False


def apply_version_attributes():
    """Add __version__ attributes to split packages"""
    print("🔧 Adding version attributes...")

    split_dir = Path("../split-repos").resolve()

    packages = [("rompy-swan", "rompy_swan"), ("rompy-schism", "rompy_schism")]

    success_count = 0

    for package_name, module_name in packages:
        init_file = split_dir / package_name / "src" / module_name / "__init__.py"

        if not init_file.exists():
            print(f"⚠️  {package_name} __init__.py not found: {init_file}")
            continue

        try:
            with open(init_file, "r") as f:
                content = f.read()

            # Add version attribute if not present
            if "__version__ = " not in content:
                # Add version at the end
                content = content.rstrip() + '\n\n__version__ = "0.5.1"\n'

                with open(init_file, "w") as f:
                    f.write(content)

                print(f"✅ Added __version__ to {package_name}")
                success_count += 1
            else:
                print(f"✅ __version__ already exists in {package_name}")
                success_count += 1

        except Exception as e:
            print(f"❌ Error adding version to {package_name}: {e}")

    return success_count == len(packages)


def apply_dependency_fixes():
    """Remove rompy-core from swan/schism dependencies to avoid circular deps"""
    print("🔧 Fixing circular dependencies...")

    split_dir = Path("../split-repos").resolve()

    packages = ["rompy-swan", "rompy-schism"]
    success_count = 0

    for package_name in packages:
        pyproject_file = split_dir / package_name / "pyproject.toml"

        if not pyproject_file.exists():
            print(f"⚠️  {package_name} pyproject.toml not found: {pyproject_file}")
            continue

        try:
            with open(pyproject_file, "r") as f:
                content = f.read()

            # Remove rompy-core from dependencies list
            lines = content.split("\n")
            new_lines = []
            in_dependencies = False

            for line in lines:
                if line.strip().startswith("dependencies = ["):
                    in_dependencies = True
                    new_lines.append(line)
                elif in_dependencies and line.strip() == "]":
                    in_dependencies = False
                    new_lines.append(line)
                elif in_dependencies and "rompy-core" in line:
                    # Skip this line (remove rompy-core dependency)
                    continue
                else:
                    new_lines.append(line)

            new_content = "\n".join(new_lines)

            with open(pyproject_file, "w") as f:
                f.write(new_content)

            print(f"✅ Fixed dependencies in {package_name}")
            success_count += 1

        except Exception as e:
            print(f"❌ Error fixing dependencies in {package_name}: {e}")

    return success_count == len(packages)


def apply_test_script_fixes():
    """Fix test script import expectations"""
    print("🔧 Fixing test script expectations...")

    test_script = Path("run_split_tests.py")

    if not test_script.exists():
        print(f"⚠️  Test script not found: {test_script}")
        return False

    try:
        with open(test_script, "r") as f:
            content = f.read()

        # Fix the test expectations
        content = content.replace(
            '"from rompy_schism import SchismConfig"',
            '"from rompy_schism import SCHISMConfig"',
        )
        content = content.replace(
            '"from rompy_schism.components import SchismGrid"',
            '"from rompy_schism import SCHISMGrid"',
        )
        content = content.replace(
            "from rompy_schism import SchismConfig",
            "from rompy_schism import SCHISMConfig",
        )

        with open(test_script, "w") as f:
            f.write(content)

        print("✅ Fixed test script expectations")
        return True

    except Exception as e:
        print(f"❌ Error fixing test script: {e}")
        return False


def main():
    """Apply all manual fixes to make split repositories functional"""

    print("🚀 Applying comprehensive manual fixes to split repositories...")
    print("=" * 80)

    # Check if split repositories exist
    split_dir = Path("../split-repos").resolve()
    if not split_dir.exists():
        print(f"❌ Split repositories directory not found: {split_dir}")
        print("Please run the repository split first!")
        sys.exit(1)

    fixes = [
        ("Data File Fix", fix_data_file),
        ("Core Exports", apply_core_exports_fix),
        ("Utils load_config", apply_utils_load_config_fix),
        ("Swan Components", apply_swan_components_fix),
        ("Source Unpacking", apply_source_unpacking_fix),
        ("Schism Circular Import", apply_schism_circular_import_fix),
        ("Version Attributes", apply_version_attributes),
        ("Dependency Fixes", apply_dependency_fixes),
        ("Test Script Fixes", apply_test_script_fixes),
    ]

    results = []

    for fix_name, fix_function in fixes:
        print(f"\n--- {fix_name} ---")
        try:
            success = fix_function()
            results.append((fix_name, success))
        except Exception as e:
            print(f"❌ Critical error in {fix_name}: {e}")
            results.append((fix_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("MANUAL FIXES SUMMARY")
    print("=" * 80)

    successful_fixes = 0
    for fix_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{fix_name:.<30} {status}")
        if success:
            successful_fixes += 1

    print(f"\nCompleted: {successful_fixes}/{len(fixes)} fixes successful")

    if successful_fixes == len(fixes):
        print("\n🎉 All manual fixes applied successfully!")
        print("\nNext steps:")
        print(
            "1. Run: python setup_split_testing.py --split-repos-dir ../split-repos --package rompy-core"
        )
        print(
            "2. Run: python setup_split_testing.py --split-repos-dir ../split-repos --package rompy-swan"
        )
        print(
            "3. Run: python setup_split_testing.py --split-repos-dir ../split-repos --package rompy-schism"
        )
        print(
            "4. Test: python run_split_tests.py --split-repos-dir ../split-repos --package rompy-core"
        )

        return 0
    else:
        print(
            f"\n⚠️  {len(fixes) - successful_fixes} fixes failed. Review the errors above."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
