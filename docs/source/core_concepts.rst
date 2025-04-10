.. -*- mode: rst -*-

=================
Core Concepts
=================

Rompy provides a framework for generating ocean model input files and managing simulation setup. It revolves around two primary concepts: defining *what* to run (Configuration) and *how/when* to run it (Runtime).

Model Runtime (`ModelRun`)
--------------------------
The :py:class:`~rompy.model.ModelRun` class orchestrates the entire process for a specific simulation instance. It defines:

*   **`run_id`**: A unique identifier for the simulation run.
*   **`output_dir`**: The base directory where simulation files will be generated.
*   **`period`**: A :py:class:`~rompy.core.time.TimeRange` object specifying the start time, duration/end time, and interval for the simulation.
*   **`config`**: An instance of a model-specific configuration class (subclass of :py:class:`~rompy.core.config.BaseConfig`).

When executed (e.g., by calling `run()`), `ModelRun` combines its runtime parameters with the `config` object and uses a templating engine (`cookiecutter`) to generate the necessary model input files within a structured directory (`output_dir/run_id`).

.. autosummary::
    :nosignatures:
    :toctree: _generated/

    rompy.model.ModelRun

Model Configuration (`BaseConfig` and Subclasses)
-------------------------------------------------
The :py:class:`~rompy.core.config.BaseConfig` class, and its model-specific subclasses (e.g., :py:class:`~rompy.swan.config.SwanConfigComponents`, :py:class:`~rompy.schism.config.SCHISMConfig`, :py:class:`~rompy_xbeach.config.Config`), define the static aspects of a model setup using `Pydantic`. This includes:

*   Model parameters (physics options, numerical schemes).
*   Spatial grid definitions.
*   Input forcing specifications.
*   Output requirements.
*   A reference to the `cookiecutter` template used for generating input files.

These configuration objects ensure that settings are type-checked, validated, and can be easily created from Python dictionaries or loaded from YAML/JSON files, promoting a declarative approach to model setup. Model-specific configurations translate these structured settings into the syntax required by the target model (e.g., SWAN command components, SCHISM namelists, XBeach parameter files).

.. autosummary::
    :nosignatures:
    :toctree: _generated/

    rompy.core.config.BaseConfig

Supporting Objects
------------------

Several core objects support the `ModelRun` and `Config` classes:

**Grid Definitions (`rompy.core.grid`, model-specific grid modules)**
Represent the spatial domain and discretization of the model.

*   **Core Grids:**
    *   :py:class:`~rompy.core.grid.BaseGrid`: Abstract base defining minimal interface (coordinate properties, `bbox()`, `boundary()` methods).
    *   :py:class:`~rompy.core.grid.RegularGrid`: Concrete implementation for standard rectangular grids (origin, rotation, spacing, dimensions).
*   **Model-Specific Grids:**
    *   Plugins define their own grid types inheriting from `BaseGrid` or `RegularGrid` to add model-specific parameters or methods (e.g., :py:class:`~rompy.swan.grid.SwanGrid`, :py:class:`~rompy.schism.grid.SCHISMGrid`).
    *   The XBeach plugin :py:class:`~rompy_xbeach.grid.RegularGrid` extends the core `RegularGrid` with CRS handling and an `alfa` (rotation) parameter specific to XBeach conventions.

.. autosummary::
    :nosignatures:
    :toctree: _generated/

    rompy.core.grid.BaseGrid
    rompy.core.grid.RegularGrid
    rompy.swan.grid.SwanGrid
    rompy.schism.grid.SCHISMGrid
    rompy_xbeach.grid.RegularGrid

**Data Handling and Forcing (`rompy.core.source`, `rompy.core.data`, `rompy.core.boundary`, `rompy.core.filters`, model-specific data modules)**
Manages acquisition, processing, and formatting of model input data (e.g., bathymetry, wind, boundary conditions). This uses a layered approach:

*   **Source Objects (The "Where"):** Define *where* the raw data comes from.
    *   :py:class:`~rompy.core.source.SourceBase`: Abstract base class.
    *   Core implementations handle origins like local files (:py:class:`~rompy.core.source.SourceFile`), intake catalogs (:py:class:`~rompy.core.source.SourceIntake`), Datamesh (:py:class:`~rompy.core.source.SourceDatamesh`), existing datasets (:py:class:`~rompy.core.source.SourceDataset`), spectral files (:py:class:`~rompy.core.source.SourceWavespectra`), CSV/DataFrames (:py:class:`~rompy.core.source.SourceTimeseriesCSV`, :py:class:`~rompy.core.source.SourceTimeseriesDataFrame`).
    *   Plugins can define additional sources tailored to specific model needs or data types (e.g., :py:class:`~rompy_xbeach.source.SourceGeotiff` for geospatial rasters, :py:class:`~rompy_xbeach.source.SourceCRSOceantide` for tidal constituents). These often add CRS awareness.
    *   The `open()` method returns an `xarray.Dataset`.

    .. autosummary::
        :nosignatures:
        :toctree: _generated/

        rompy.core.source.SourceBase
        rompy.core.source.SourceFile
        rompy.core.source.SourceIntake
        rompy.core.source.SourceDatamesh
        rompy.core.source.SourceDataset
        rompy.core.source.SourceWavespectra
        rompy.core.source.SourceTimeseriesCSV
        rompy.core.source.SourceTimeseriesDataFrame
        rompy_xbeach.source.SourceGeotiff
        rompy_xbeach.source.SourceCRSOceantide

*   **Data Objects (The "What" and "How"):** Define *what* data is needed and *how* it should be processed.
    *   :py:class:`~rompy.core.data.DataGrid`: Central class for gridded data. Holds a `Source` object, specifies `variables`, `coords` mapping, and `filters`. Manages automatic spatial/temporal cropping based on the model `grid` and `period` via the `crop_data` flag and buffers. It manages the 'What' (variables, coordinates) and 'How' (filters, source) aspects of data preparation. The `ds` property provides the processed `xarray.Dataset`.
    *   :py:class:`~rompy.core.boundary.DataBoundary`: Specializes `DataGrid` for boundary conditions. Adds `spacing` and `sel_method` for selecting points along the model boundary.
    *   :py:class:`~rompy.core.data.DataPoint`: Simplified version for timeseries/point data.
    *   :py:class:`~rompy.core.data.DataBlob`: Basic file/directory handler (copy or link).

    .. autosummary::
        :nosignatures:
        :toctree: _generated/

        rompy.core.data.DataGrid
        rompy.core.boundary.DataBoundary
        rompy.core.data.DataPoint
        rompy.core.data.DataBlob

*   **Filter Object (Processing):**
    *   :py:class:`~rompy.core.filters.Filter`: Applies transformations like sorting, subsetting, cropping, renaming, and deriving variables to the dataset loaded by the `Source`. Automatically updated by `DataGrid` if `crop_data` is enabled.

    .. autosummary::
        :nosignatures:
        :toctree: _generated/

        rompy.core.filters.Filter

*   **Model-Specific Data Objects (Formatting):**
    *   While core `DataGrid` and `DataBoundary` handle sourcing and filtering data into a standardized `xarray.Dataset`, model-specific subclasses handle the final step: **formatting and writing** this data into the files required by the target model. They override the `get()` method to perform this task. Examples include:
        *   :py:class:`~rompy.swan.data.SwanDataGrid`: Writes processed data into SWAN ASCII grid files.
        *   :py:class:`~rompy.schism.data.SfluxAir`: Writes atmospheric data into SCHISM's sflux NetCDF format.
        *   :py:class:`~rompy_xbeach.data.XBeachBathy`: Handles geospatial rasters (via `SourceGeotiff`), interpolates using specified methods (like :py:class:`~rompy_xbeach.interpolate.RegularGridInterpolator`), potentially extends the grid seaward (e.g., using :py:class:`~rompy_xbeach.data.SeawardExtensionLinear`), and writes the final bathymetry into XBeach's specific format (xdata.txt, ydata.txt, bathy.txt).
        *   :py:class:`~rompy_xbeach.boundary.BoundaryStationSpectraJons`: Selects spectral data from stations, calculates JONSWAP parameters, and writes boundary conditions to XBeach JONS format files (either a single file or a filelist).
        *   :py:class:`~rompy_xbeach.forcing.WindGrid`: Selects gridded wind data (potentially calculating speed/direction from U/V components using :py:class:`~rompy_xbeach.forcing.WindVector` or :py:class:`~rompy_xbeach.forcing.WindScalar`) and writes it to the XBeach time-varying wind file format.

    .. autosummary::
        :nosignatures:
        :toctree: _generated/

        rompy.swan.data.SwanDataGrid
        rompy.schism.data.SfluxAir
        rompy.schism.data.SCHISMDataOcean
        rompy.swan.boundary.Boundnest1
        rompy_xbeach.data.XBeachBathy
        rompy_xbeach.boundary.BoundaryStationSpectraJons
        rompy_xbeach.forcing.WindGrid

**Time Definition:**
Specifies simulation periods and intervals.
.. autosummary::
    :nosignatures:
    :toctree: _generated/

    rompy.core.time.TimeRange

Workflow Summary
----------------

1.  **Define Configuration:** Create a model-specific `Config` object (e.g., `SwanConfigComponents`, `SCHISMConfig`, `rompy_xbeach.config.Config`) defining the grid, physics, data requirements (using `DataGrid`, `DataBoundary`, etc., each containing a `Source`), outputs, etc.
2.  **Define Runtime:** Create a `ModelRun` instance, specifying the `run_id`, `output_dir`, simulation `period`, and the `config`.
3.  **Generate:** Call `model.generate()`. This triggers the `get()` method on each data object within the `config`. The `get()` method:
    *   Optionally updates its internal `crop` filter based on the `ModelRun`'s `period` and the `Config`'s `grid`.
    *   Accesses its `ds` property, which loads data via the `Source` object and applies all defined `filters`.
    *   Writes the processed data to the model-specific format in the staging directory (`output_dir/run_id`).
    *   The `cookiecutter` template is rendered using `runtime` and `config` data, embedding paths to the generated input files.
4.  **Model Plugins:** Model plugins (like `rompy-swan`, `rompy-schism`, `rompy-xbeach`) provide the specific `Config`, `Grid`, and `Data` subclasses needed for their respective models, fitting seamlessly into this core workflow.
5.  **Execute (External):** Run the ocean model executable using the files in the staging directory.
6.  **Analyze:** Analyze model output.