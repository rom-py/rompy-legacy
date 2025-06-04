. -*- mode: rst -*-

=================================
Welcome to rompy's documentation!
=================================

*Taking the pain out of ocean model setup*

**Status:** This library is under active development. While functional, interfaces may evolve.

.. figure:: /_static/logo.svg
    :align: center
    :alt: rompy logo
    :width: 400px

Introduction
------------

Relocatable Ocean Modelling in PYthon (rompy) is a Python framework designed to streamline the configuration, execution, and analysis of coastal and ocean numerical models. It addresses the often complex and model-specific nature of setting up simulations by providing:

*   **Structured Configuration:** Leverages `Pydantic` models for defining model settings (including spatial grids, physics, forcing sources, outputs) in a clear, type-safe, and validated manner. Configurations can be defined programmatically in Python or declaratively via YAML/JSON files.
*   **Templated Setup:** Utilizes the `cookiecutter` engine to generate model-specific input files and directory structures based on the defined configuration and runtime parameters (e.g., simulation period).
*   **Abstracted Data Handling:** Integrates with `xarray`, `intake`, `fsspec`, and `oceanum`'s Datamesh to provide flexible ways to source, filter, and process input forcing data (bathymetry, wind, boundary conditions, etc.) required by the models.
*   **Workflow Orchestration:** The central `ModelRun` class manages the simulation lifecycle, combining runtime information (like the simulation period) with a model-specific `Config` object to generate a complete, ready-to-run model setup.
*   **Extensibility:** Designed with a plugin architecture using Python's `entry_points`, allowing users and developers to easily add support for new models or data sources.

`rompy` facilitates setup for models including:

*   **SWAN:** A detailed, component-based configuration mirroring SWAN's command structure (via the `rompy-swan` package).
*   **SCHISM:** Support including both a minimal configuration and a comprehensive namelist-based approach (via the `rompy-schism` package).
*   **XBeach:** Configuration and input generation (via the `rompy-xbeach <https://github.com/rom-py/rompy-xbeach>`_ plugin).

The goal is to provide a unified, Pythonic interface for diverse ocean models, promoting reproducibility, efficiency, and automation in modelling workflows.

.. toctree::
    :hidden:
    :maxdepth: 2

    Home <self>
    quickstart
    core_concepts
    models
    demo
    api
    # relational_diagrams (Keep if relevant)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`