======
SCHISM
======

Grids
------

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.grid.SCHISMGrid2D
   rompy.schism.grid.SCHISMGrid3D

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
   rompy.schism.data.SCHISMDataOcean
   rompy.schism.data.SCHISMDataTides
   rompy.schism.data.SCHISMData




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

Param
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.param.PARAM

Ice
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.ice.ICE

ICM
~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.icm.ICM

SEDIMENT
~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.sediment.SEDIMENT


COSINE
~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.namelists.cosine.COSINE


Config Object
~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.schism.config.SCHISMConfig
