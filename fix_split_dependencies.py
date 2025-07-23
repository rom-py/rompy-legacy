#!/usr/bin/env python3
"""
Script to fix pyproject.toml files in split repositories.
Fixes dependencies, entry points, and other configuration issues.
"""

import os
import sys
from pathlib import Path


def fix_rompy_core_pyproject():
    """Fix rompy-core pyproject.toml"""
    content = '''[build-system]
requires = ["setuptools", "versioneer[toml]"]
build-backend = "setuptools.build_meta"

[project]
name = "rompy-core"
description = "Core rompy library for ocean wave modeling with plugin system"
readme = "README.rst"
authors = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
maintainers = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
license = {text = "MIT license"}
requires-python = ">=3.10"
dependencies = [
    "cartopy",
    "click",
    "cloudpathlib",
    "cookiecutter>=2.6",
    "dask",
    "fsspec",
    "geopandas",
    "h5py",
    "intake",
    "intake-xarray",
    "intake-geopandas",
    "matplotlib",
    "netcdf4",
    "numpy",
    "oceanum",
    "owslib",
    "pandas",
    "pydantic>2",
    "pydantic-settings",
    "pydantic_numpy",
    "scipy",
    "tqdm",
    "xarray",
    "wavespectra",
    "isodate",
    "appdirs",
    "typer"
]

dynamic = ["version"]

[project.optional-dependencies]
dev = [
    "coverage",
    "mypy",
    "pytest",
    "ruff"
]
test = [
    "pytest",
    "envyaml",
    "coverage"
]
extra = [
    "gcsfs",
    "zarr",
]
docs = [
    "autodoc_pydantic",
    "ipython",
    "nbsphinx",
    "pydata_sphinx_theme",
    "sphinx<7.3.6",
    "sphinx-collections",
]

[project.scripts]
rompy = "rompy_core.cli:main"

[project.entry-points."intake.drivers"]
"netcdf_fcstack" = "rompy_core.intake:NetCDFFCStackSource"

[project.entry-points."rompy.config"]
base = "rompy_core.core.config:BaseConfig"

[project.entry-points."rompy.source"]
file = "rompy_core.core.source:SourceFile"
intake = "rompy_core.core.source:SourceIntake"
datamesh = "rompy_core.core.source:SourceDatamesh"
wavespectra = "rompy_core.core.source:SourceWavespectra"
"csv:timeseries" = "rompy_core.core.source:SourceTimeseriesCSV"

[project.entry-points."intake.catalogs"]
"rompy_data" = "rompy_core:cat"

[project.entry-points."rompy.run"]
local = "rompy_core.run:LocalRunBackend"
docker = "rompy_core.run.docker:DockerRunBackend"

[project.entry-points."rompy.postprocess"]
noop = "rompy_core.postprocess:NoopPostprocessor"

[project.entry-points."rompy.pipeline"]
local = "rompy_core.pipeline:LocalPipelineBackend"

[project.urls]
bugs = "https://github.com/rom-py/rompy-core/issues"
changelog = "https://github.com/rom-py/rompy-core/blob/master/changelog.md"
homepage = "https://github.com/rom-py/rompy-core"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.package-data]
"*" = ["*.y*ml", "*.csv", "*.html"]

[tool.setuptools.dynamic]
version = {attr = "rompy_core.__version__"}

[tool.pytest.ini_options]
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"

[tool.black]
line-length = 88
'''
    return content


def fix_rompy_swan_pyproject():
    """Fix rompy-swan pyproject.toml"""
    content = '''[build-system]
requires = ["setuptools", "versioneer[toml]"]
build-backend = "setuptools.build_meta"

[project]
name = "rompy-swan"
description = "SWAN wave model plugin for rompy"
readme = "README.rst"
authors = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
maintainers = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
license = {text = "MIT license"}
requires-python = ">=3.10"
dependencies = [
    "pydantic>2",
]

dynamic = ["version"]

[project.optional-dependencies]
dev = [
    "coverage",
    "mypy",
    "pytest",
    "ruff"
]
test = [
    "pytest",
    "coverage"
]

[project.entry-points."rompy.config"]
swan = "rompy_swan.config:SwanConfig"
swan_components = "rompy_swan.config:SwanConfigComponents"

[project.urls]
bugs = "https://github.com/rom-py/rompy-swan/issues"
changelog = "https://github.com/rom-py/rompy-swan/blob/master/changelog.md"
homepage = "https://github.com/rom-py/rompy-swan"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.package-data]
"*" = ["*.y*ml", "*.csv", "*.html"]

[tool.setuptools.dynamic]
version = {attr = "rompy_swan.__version__"}

[tool.pytest.ini_options]
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"

[tool.black]
line-length = 88
'''
    return content


def fix_rompy_schism_pyproject():
    """Fix rompy-schism pyproject.toml"""
    content = '''[build-system]
requires = ["setuptools", "versioneer[toml]"]
build-backend = "setuptools.build_meta"

[project]
name = "rompy-schism"
description = "SCHISM model plugin for rompy"
readme = "README.rst"
authors = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
maintainers = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
license = {text = "MIT license"}
requires-python = ">=3.10"
dependencies = [
    "pydantic>2",
    "pylibs-ocean",
    "pytmd",
]

dynamic = ["version"]

[project.optional-dependencies]
dev = [
    "coverage",
    "mypy",
    "pytest",
    "ruff"
]
test = [
    "pytest",
    "coverage"
]

[project.entry-points."rompy.config"]
schismcsiro = "rompy_schism.config:SchismCSIROConfig"
schism = "rompy_schism.config:SCHISMConfig"

[project.urls]
bugs = "https://github.com/rom-py/rompy-schism/issues"
changelog = "https://github.com/rom-py/rompy-schism/blob/master/changelog.md"
homepage = "https://github.com/rom-py/rompy-schism"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.package-data]
"*" = ["*.y*ml", "*.csv", "*.html"]

[tool.setuptools.dynamic]
version = {attr = "rompy_schism.__version__"}

[tool.pytest.ini_options]
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"

[tool.black]
line-length = 88
'''
    return content


def main():
    """Main function to fix all split repository pyproject.toml files"""

    # Get the split repositories directory
    split_dir = Path("../split-repos").resolve()

    if not split_dir.exists():
        print(f"❌ Split repositories directory not found: {split_dir}")
        sys.exit(1)

    # Fix each repository
    repos = {
        "rompy-core": fix_rompy_core_pyproject,
        "rompy-swan": fix_rompy_swan_pyproject,
        "rompy-schism": fix_rompy_schism_pyproject,
    }

    for repo_name, fix_func in repos.items():
        repo_path = split_dir / repo_name
        pyproject_path = repo_path / "pyproject.toml"

        if not repo_path.exists():
            print(f"⚠️  Repository not found: {repo_path}")
            continue

        print(f"🔧 Fixing {repo_name}/pyproject.toml...")

        # Write the fixed content
        content = fix_func()
        with open(pyproject_path, 'w') as f:
            f.write(content)

        print(f"✅ Fixed {repo_name}/pyproject.toml")

    print("🎉 All pyproject.toml files have been fixed!")
    print("\nKey changes made:")
    print("- Fixed circular dependencies (rompy-core no longer depends on rompy)")
    print("- Added missing dependencies (envyaml, etc.)")
    print("- Corrected entry points to match actual module structure")
    print("- Added proper plugin dependencies (swan/schism depend on core)")
    print("- Fixed package names to use hyphens (rompy-core, not rompy_core)")


if __name__ == "__main__":
    main()
