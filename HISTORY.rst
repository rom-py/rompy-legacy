=======
History
=======

Relocatable Ocean Modelling in PYthon (rompy) is a modular Python library that
aims to streamline the setup, configuration, execution, and analysis of coastal
ocean models. Rompy combines templated model configuration with xarray-based
data handling and pydantic validation, enabling users to efficiently generate
model control files and input datasets for a variety of ocean and wave models.
The architecture centers on high-level execution control (`ModelRun`) and
flexible configuration objects, supporting both persistent scientific model
state and runtime backend selection. Rompy provides unified interfaces for
grids, data sources, boundary conditions, and spectra, with extensible plugin
support for new models and execution environments. Comprehensive documentation,
example Jupyter notebooks, and a robust logging/formatting framework make rompy
accessible for both research and operational workflows. Current model support
includes SWAN and SCHISM, with ongoing development for additional models and
cloud/HPC backends.

Key Features: - Modular architecture with clear separation of configuration and
execution logic - Templated, reproducible model configuration using pydantic
and xarray - Unified interfaces for grids, data, boundaries, and spectra -
Extensible plugin system for models, data sources, backends, and postprocessors
- Robust logging and formatting for consistent output and diagnostics - Example
  notebooks and comprehensive documentation for rapid onboarding - Support for
  local, Docker, and HPC execution backends

rompy is under active development—features, model support, and documentation
are continually evolving. Contributions and feedback are welcome!


********
Releases
********

0.5.0 (2025-07-13)
___________________

New Features
------------
* Improved logging for SCHISM model components.
* Added string formatting methods for SCHISM components.
* Added backend testing and execution scripts.
* Added backend config examples and quickstart test script.
* Added backend demo notebook and documentation.
* Added support for multiple include_modules in docker.
* Added docker backend test and setup.

Bug Fixes
---------
* Fixed backend demo notebook.
* Fixed merge issues and improved PyLibs import handling.
* Fixed mounting workspace in docker.
* Fixed issues from merge and removed redundant code.

Internal Changes
----------------
* Refactored backend approach using strong typing.
* Improved backend docs and tutorial.
* Consolidated documentation and removed legacy sections.
* Improved INPUT file diagnostics in Docker container.


0.4.0 (2025-07-10)
___________________

New Features
------------
* Refactored SCHISM boundary conditions, unified SCHISMDataTides and SCHISMDataOcean into SCHISMDataBoundaryConditions.
* Added support for pyTMD for tidal forcing.
* Added tidal database and updated yaml of tidal runs.
* Added boundary condition examples and documentation.
* Added plotting utilities and improved grid plotting.
* Added support for v5.12 vegetation model.
* Added MDT specification for TidalDataset for Z0.

Bug Fixes
---------
* Fixed duplicated subfolder for oceanum-atlas in database.json.
* Fixed test cases for pyTMD compatibility.
* Fixed decorators and case handling for tidal API.
* Fixed missing station.in file for tidal examples.
* Fixed case test with new tidal API.

Internal Changes
----------------
* Restructured SCHISM boundary condition naming.
* Overwrote pre-refactor config files with post-refactor versions.
* Cleaned up code and tests.
* Updated enum types and documentation.



0.3.0 (2023-03-29)
___________________

Major refactor: redefinition of the entire codebase using pydantic models.
Separation of concerns between runtime information and model configuration.
Added model_type field and pydantic basegrid.
Added methods to store and dump original inputs to RompyBaseModel.
Added json support to CLI.
Added iso timedelta for JSON serialization.
Added DataPoint object and timeseries-based sources.
Added validator for hotfiles against timestep.

Bug Fixes
---------
Fixed issue with create_model in older versions of pydantic.
Fixed validator of IDLA to fix serialization issue.
Fixed path definition in new test.
Fixed schism serialization issues.
Fixed bug in string output of regular grid for SWAN.

Internal Changes
----------------
Removed convenience imports from core.
Promoted appdirs dependency from schism to main list.
Reordered imports.
Refactored intake source to prevent recursion with dask.
Cleaned up debug messages and removed redundant code.

0.1.0 (2023-MM-DD)
___________________

Initial release of rompy with basic functionality for coastal ocean model configuration and execution.
Provided example Jupyter notebooks for setup, evaluation, and visualization.
Basic support for SWAN and SCHISM models.


.. _`CSIRO`: https://www.csiro.au/en/
.. _`Oceanum`: https://oceanum.science/
