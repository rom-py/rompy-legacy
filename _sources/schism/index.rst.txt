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

   rompy.schism.namelists.ice.Mice_in
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
