> **⚠️ DEPRECATION NOTICE: This monorepo is no longer maintained. ROMPY has been split into dedicated repositories. Please use the new repos below:**
>
> **The final PyPI release of this version of rompy is v0.5.4. Any install of rompy >=0.6 will install the new split version of the code, which contains only the core rompy functionality.**
>
> - [rompy](https://github.com/rom-py/rompy)
> - [rompy-swan](https://github.com/rom-py/rompy-swan)
> - [rompy-schism](https://github.com/rom-py/rompy-schism)
> - [rompy-notebooks](https://github.com/rom-py/rompy-notebooks)
>
> **All new development and issues should be directed to the split repositories above. This monorepo is archived and will not receive updates.**
> 
> See [MIGRATION_PLAN.md]

---
title: "Relocatable Ocean Modelling in PYthon (rompy)"
---

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15093426.svg)](https://doi.org/10.5281/zenodo.15093426)
[![GitHub Pages](https://github.com/rom-py/rompy/actions/workflows/sphinx_docs_to_gh_pages.yaml/badge.svg)](https://rom-py.github.io/rompy/)
[![PyPI version](https://img.shields.io/pypi/v/rompy.svg)](https://pypi.org/project/rompy/)
[![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/rom-py/rompy/python-publish.yml)](https://github.com/rom-py/rompy/actions)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rompy)](https://pypistats.org/packages/rompy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rompy)](https://pypi.org/project/rompy/)

# Introduction

Relocatable Ocean Modelling in PYthon (rompy) is a modular Python library that aims to streamline the setup, configuration, execution, and analysis of coastal ocean models. Rompy combines templated model configuration with powerful xarray-based data handling and pydantic validation, enabling users to efficiently generate model control files and input datasets for a variety of ocean and wave models. The architecture centers on high-level execution control (`ModelRun`) and flexible configuration objects, supporting both persistent scientific model state and runtime backend selection. Rompy provides unified interfaces for grids, data sources, boundary conditions, and spectra, with extensible plugin support for new models and execution environments. Comprehensive documentation, example Jupyter notebooks, and a robust logging/formatting framework make rompy accessible for both research and operational workflows. Current model support includes SWAN and SCHISM, with ongoing development for additional models and cloud/HPC backends.

Key Features:
- Modular architecture with clear separation of configuration and execution logic
- Templated, reproducible model configuration using pydantic and xarray
- Unified interfaces for grids, data, boundaries, and spectra
- Extensible plugin system for models, data sources, backends, and postprocessors
- Robust logging and formatting for consistent output and diagnostics
- Example notebooks and comprehensive documentation for rapid onboarding
- Support for local, Docker, and HPC execution backends

rompy is under active development—features, model support, and documentation are continually evolving. Contributions and feedback are welcome!

# Documentation

See https://rom-py.github.io/rompy/

# Code Formatting and Pre-commit Hooks

This repository enforces Python code formatting using [black](https://github.com/psf/black) via the pre-commit framework.

To set up pre-commit hooks locally (required for all contributors)::

    pip install pre-commit
    pre-commit install

This will automatically check code formatting before each commit. To format your code manually, run::

    pre-commit run --all-files

All code must pass black formatting before it can be committed or merged.

# Versioning and Release

This project uses [tbump](https://github.com/dmerejkowsky/tbump) for version management.

To bump the version, run::

    tbump <new_version>

This will update the version in `src/rompy/__init__.py`, commit the change, and optionally create a git tag.

tbump is included in the development requirements (`requirements_dev.txt`).

For more advanced configuration, see `tbump.toml` in the project root.
