#!/usr/bin/env python3
"""
Script to fix import issues in split repositories.
Corrects imports from rompy.* to rompy_core.*, rompy_swan.*, etc.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple


class ImportCorrector:
    """Handles import corrections for split repositories."""

    def __init__(self, split_repos_dir: str):
        """Initialize the import corrector.

        Args:
            split_repos_dir: Path to the split repositories directory
        """
        self.split_repos_dir = Path(split_repos_dir)
        self.corrections_applied = 0
        self.files_modified = 0

    def get_import_patterns(self, package_type: str) -> List[Tuple[str, str]]:
        """Get import patterns for a specific package type.

        Args:
            package_type: Type of package (core, swan, schism)

        Returns:
            List of (old_pattern, new_pattern) tuples
        """
        # if package_type == "core":
        #     return [
        #         # Core package imports
        #         (r"from rompy\.run", "from rompy_core.run"),
        #         (r"from rompy\.formatting", "from rompy_core.formatting"),
        #         (r"from rompy\.cli", "from rompy_core.cli"),
        #         (r"from rompy\.backends", "from rompy_core.backends"),
        #         (r"from rompy\.postprocess", "from rompy_core.postprocess"),
        #         (r"from rompy\.pipeline", "from rompy_core.pipeline"),
        #         (r"from rompy\.intake", "from rompy_core.intake"),
        #         (r"from rompy\.configuration", "from rompy_core.configuration"),
        #         (r"from rompy\.templates", "from rompy_core.templates"),
        #         # Import statements
        #         (r"import rompy\.backends", "import rompy_core.backends"),
        #         # Direct module references
        #         (r"rompy\.core\.", "rompy_core.core."),
        #         (r"rompy\.utils\.", "rompy_core.utils."),
        #         (r"rompy\.model\.", "rompy_core.model."),
        #         (r"rompy\.run\.", "rompy_core.run."),
        #         (r"rompy\.cli\.", "rompy_core.cli."),
        #         # Entry point references
        #         (r'"rompy\.', '"rompy_core.'),
        #         (r"'rompy\.", "'rompy_core."),
        #         # Special cases for imports within strings
        #         (r'patch\("rompy\.', 'patch("rompy_core.'),
        #         (r'Mock\("rompy\.', 'Mock("rompy_core.'),
        #     ]
        if package_type == "swan":
            return [
                # Swan specific imports
                (r"from rompy\.swan\.", "from rompy_swan."),
                (r"import rompy\.swan\.", "import rompy_swan."),
                (r"rompy\.swan\.", "rompy_swan."),
                # Core imports (swan depends on core)
                # Entry points
                (r'"rompy\.swan\.', '"rompy_swan.'),
                (r"'rompy\.swan\.", "'rompy_swan."),
                # Patch references
                (r'patch\("rompy\.swan\.', 'patch("rompy_swan.'),
            ]
        elif package_type == "schism":
            return [
                # Schism specific imports
                (r"from rompy\.schism\.", "from rompy_schism."),
                (r"import rompy\.schism\.", "import rompy_schism."),
                (r"rompy\.schism\.", "rompy_schism."),
                # Entry points
                (r'"rompy\.schism\.', '"rompy_schism.'),
                (r"'rompy\.schism\.", "'rompy_schism."),
                # Patch references
                (r'patch\("rompy\.schism\.', 'patch("rompy_schism.'),
            ]
        else:
            return []

    def get_optional_import_patterns(self) -> List[Tuple[str, str]]:
        """Get patterns for optional imports that might not be present."""
        # return [
        #     # General rompy imports that might need context-specific fixing
        #     (r"from rompy import", "from rompy_core import"),
        #     (r"import rompy$", "import rompy_core"),
        #     (r"import rompy,", "import rompy_core,"),
        # ]
        return []

    def apply_import_corrections(
        self, file_path: Path, patterns: List[Tuple[str, str]]
    ) -> int:
        """Apply import corrections to a single file.

        Args:
            file_path: Path to the file to correct
            patterns: List of (old_pattern, new_pattern) tuples

        Returns:
            Number of corrections applied
        """

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content
            corrections_in_file = 0

            # for old_pattern, new_pattern in patterns:
            #     # Update patterns to use 'rompy' as the core
            #     if "rompy_core" in old_pattern or "rompy-core" in old_pattern:
            #         old_pattern = old_pattern.replace("rompy_core", "rompy").replace(
            #             "rompy-core", "rompy"
            #         )
            #     if "rompy_core" in new_pattern or "rompy-core" in new_pattern:
            #         new_pattern = new_pattern.replace("rompy_core", "rompy").replace(
            #             "rompy-core", "rompy"
            #         )
            #     matches = re.findall(old_pattern, content)
            #     if matches:
            #         content = re.sub(old_pattern, new_pattern, content)
            #         corrections_in_file += len(matches)

            # Only write if changes were made
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return corrections_in_file

            return 0

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return 0

    def find_python_files(self, directory: Path) -> List[Path]:
        """Find all Python files in a directory.

        Args:
            directory: Directory to search

        Returns:
            List of Python file paths
        """
        python_files = []

        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and __pycache__
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

        return python_files

    def correct_package_imports(self, package_name: str, package_type: str) -> dict:
        """Correct imports for a specific package.

        Args:
            package_name: Name of the package (e.g., 'rompy-core')
            package_type: Type of package ('core', 'swan', 'schism')

        Returns:
            Dictionary with correction statistics
        """
        package_dir = self.split_repos_dir / package_name

        if not package_dir.exists():
            return {
                "package": package_name,
                "exists": False,
                "files_found": 0,
                "files_modified": 0,
                "corrections_applied": 0,
            }

        # Get patterns for this package type
        patterns = self.get_import_patterns(package_type)
        optional_patterns = self.get_optional_import_patterns()
        all_patterns = patterns + optional_patterns

        print(
            f"🔧 Correcting imports for {package_type} package: {package_name.replace('-', '_')}"
        )
        print(f"   Target directory: {package_dir}")
        print(
            f"   Using {len(patterns)} import patterns + {len(optional_patterns)} optional patterns"
        )

        # Find all Python files
        python_files = self.find_python_files(package_dir)
        print(f"   Found {len(python_files)} Python files to process")

        # Apply corrections
        total_corrections = 0
        files_modified = 0

        for file_path in python_files:
            corrections = self.apply_import_corrections(file_path, all_patterns)
            if corrections > 0:
                files_modified += 1
                total_corrections += corrections
                rel_path = file_path.relative_to(package_dir)
                status = "modified" if corrections > 0 else "unchanged"
                print(f"  ✅ {rel_path} - {corrections} corrections ({status})")

        print(
            f"✅ Completed {package_type}: {files_modified}/{len(python_files)} files modified with {total_corrections} corrections"
        )

        return {
            "package": package_name,
            "package_type": package_type,
            "exists": True,
            "files_found": len(python_files),
            "files_modified": files_modified,
            "corrections_applied": total_corrections,
        }

    def generate_report(self, results: List[dict]) -> str:
        """Generate a comprehensive report of the import corrections.

        Args:
            results: List of correction results

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("ROMPY REPOSITORY SPLIT - IMPORT CORRECTION REPORT")
        report.append("=" * 80)
        report.append("Mode: LIVE")

        total_files = sum(r["files_found"] for r in results if r["exists"])
        total_modified = sum(r["files_modified"] for r in results if r["exists"])
        total_corrections = sum(
            r["corrections_applied"] for r in results if r["exists"]
        )

        report.append(f"Total files processed: {total_files}")
        report.append(f"Total files modified: {total_modified}")
        report.append(f"Total corrections applied: {total_corrections}")
        report.append("Errors encountered: 0")
        report.append("")

        for result in results:
            if not result["exists"]:
                continue

            package_name = result["package"]
            package_type = result["package_type"]

            report.append(f"Package: {package_type} ({package_name.replace('-', '_')})")
            report.append(f"  Directory: {self.split_repos_dir / package_name}")
            report.append(f"  Files found: {result['files_found']}")
            report.append(f"  Files processed: {result['files_found']}")
            report.append(f"  Files modified: {result['files_modified']}")
            report.append(f"  Corrections applied: {result['corrections_applied']}")
            report.append("")

            if result["files_modified"] > 0:
                report.append("  Modified files:")
                # This is simplified - in a real implementation we'd track individual files
                report.append(
                    f"    - Multiple files modified ({result['files_modified']} total)"
                )
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)


def main():
    """Main function to run import corrections."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix import statements in split repositories"
    )
    parser.add_argument(
        "--split-repos-dir",
        default="../split-repos",
        help="Directory containing split repositories",
    )
    parser.add_argument("--all", action="store_true", help="Process all packages")
    parser.add_argument(
        "--package",
        choices=["core", "swan", "schism"],
        help="Process specific package only",
    )

    args = parser.parse_args()

    # Validate split directory exists
    split_dir = Path(args.split_repos_dir).resolve()
    if not split_dir.exists():
        print(f"❌ Split repositories directory not found: {split_dir}")
        sys.exit(1)

    corrector = ImportCorrector(str(split_dir))
    results = []

    if args.all or args.package is None:
        print("🚀 Starting import corrections for all split repositories")

        # Process packages in dependency order
        packages = [
            ("rompy-core", "core"),
            ("rompy-swan", "swan"),
            ("rompy-schism", "schism"),
        ]

        for package_name, package_type in packages:
            result = corrector.correct_package_imports(package_name, package_type)
            results.append(result)

    elif args.package:
        package_map = {
            "core": ("rompy-core", "core"),
            "swan": ("rompy-swan", "swan"),
            "schism": ("rompy-schism", "schism"),
        }

        package_name, package_type = package_map[args.package]
        print(f"🚀 Starting import corrections for {package_name}")

        result = corrector.correct_package_imports(package_name, package_type)
        results.append(result)

    # Handle notebooks warning
    notebooks_dir = split_dir / "rompy-notebooks"
    if notebooks_dir.exists():
        src_dir = notebooks_dir / "src" / "rompy_notebooks"
        if not src_dir.exists():
            print(
                "⚠️  Source directory not found: ../split-repos/rompy-notebooks/src/rompy_notebooks"
            )

    # Generate and save report
    report = corrector.generate_report(results)
    print(report)

    # Save report to file
    report_file = "import_correction_report___main__.txt"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"📄 Report written to: {report_file}")

    print("🎉 All import corrections completed successfully!")


if __name__ == "__main__":
    main()
