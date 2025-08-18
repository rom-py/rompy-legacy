======
SCHISM
======

Grids
------

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.grid.SCHISMGrid

Data
-----

Supporting objects for SCHISM data files.jects

.. autosummary::
   :nosignatures:
   :toctree: _generated/
   rompy.schism.data.SfluxSource
   rompy.schism.data.SfluxAir
   rompy.schism.data.SfluxRad
   rompy.schism.data.SfluxPrc
   rompy.schism.data.SCHISMDataBoundary

Main objects

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.data.SCHISMDataSflux
   rompy.schism.data.SCHISMDataWave
   rompy.schism.data.SCHISMDataBoundaryConditions
   rompy.schism.data.HotstartConfig
   rompy.schism.data.SCHISMData


Boundary Conditions
-------------------

The boundary conditions module provides a unified interface for configuring all types of SCHISM boundary conditions including tidal, ocean, river, and nested model boundaries.

.. toctree::
   :maxdepth: 2

   boundary_conditions

Hotstart Configuration
----------------------

The hotstart system provides integrated initial condition file generation, allowing you to create hotstart.nc files from the same ocean data sources used for boundary conditions.

.. toctree::
   :maxdepth: 2

   hotstart

Backend Framework
-----------------

The backend framework provides unified execution of SCHISM simulations using Docker containers with automatic image building and comprehensive testing capabilities.

.. toctree::
   :maxdepth: 2

   backend_framework
   backend_tutorial

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.data.SCHISMDataBoundaryConditions
   rompy.schism.data.BoundarySetupWithSource
   rompy.schism.boundary_conditions.create_tidal_only_boundary_config
   rompy.schism.boundary_conditions.create_hybrid_boundary_config
   rompy.schism.boundary_conditions.create_river_boundary_config
   rompy.schism.boundary_conditions.create_nested_boundary_config

Plotting Classes
~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.plotting.SchismPlotter
   rompy.schism.plotting.core.PlotConfig
   rompy.schism.plotting.core.BasePlotter
   rompy.schism.plotting.grid.GridPlotter
   rompy.schism.plotting.data.DataPlotter
   rompy.schism.plotting.overview.OverviewPlotter
   rompy.schism.plotting.validation.ModelValidator
   rompy.schism.plotting.validation.ValidationPlotter


Config Minimal
---------------

This object has been implemented to meet the initial operational requirements of CSIRO. It is likely that this will be superceded by the full implementation.

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.config.SchismCSIROConfig

Full Namelist Implementation
-----------------------------

This object implements a set of models for each namelist and assembles a config object using this group of models.  This is curently only partly implemented.

PARAM
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.param.Core
   rompy.schism.namelists.param.Opt
   rompy.schism.namelists.param.Schout
   rompy.schism.namelists.param.Vertical
   rompy.schism.namelists.param.Param

ICE
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.ice.Ice_in
   rompy.schism.namelists.ice.Ice

MICE
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.mice.Mice_in
   rompy.schism.namelists.mice.Mice

ICM
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.icm.Bag
   rompy.schism.namelists.icm.Core
   rompy.schism.namelists.icm.Ero
   rompy.schism.namelists.icm.Marco
   rompy.schism.namelists.icm.Ph_icm
   rompy.schism.namelists.icm.Poc
   rompy.schism.namelists.icm.Sav
   rompy.schism.namelists.icm.Sfm
   rompy.schism.namelists.icm.Silica
   rompy.schism.namelists.icm.Stem
   rompy.schism.namelists.icm.Veg
   rompy.schism.namelists.icm.Zb
   rompy.schism.namelists.icm.Icm

SEDIMENT
~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.sediment.Sed_opt
   rompy.schism.namelists.sediment.Sed_core
   rompy.schism.namelists.sediment.Sediment


COSINE
~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.cosine.Core
   rompy.schism.namelists.cosine.Marco
   rompy.schism.namelists.cosine.Misc
   rompy.schism.namelists.cosine.Cosine


WWMINPUT
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.wwminput.Coupl
   rompy.schism.namelists.wwminput.Engs
   rompy.schism.namelists.wwminput.Grid
   rompy.schism.namelists.wwminput.History
   rompy.schism.namelists.wwminput.Hotfile
   rompy.schism.namelists.wwminput.Init
   rompy.schism.namelists.wwminput.Nesting
   rompy.schism.namelists.wwminput.Nums
   rompy.schism.namelists.wwminput.Petscoptions
   rompy.schism.namelists.wwminput.Proc
   rompy.schism.namelists.wwminput.Station
   rompy.schism.namelists.wwminput.Wwminput


NML
~~~~~

This is the full namelist object that is the container for all the other namelist objects.

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.NML






Config Object
~~~~~~~~~~~~~~


.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.config.SCHISMConfig
