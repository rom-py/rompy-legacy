===================
API  Documentation
===================


Base Classes
==============

This is a reference API class listing.

.. autosummary::
   :toctree: classes/rompy

   rompy.core

.. autopydantic_model:: rompy.core.model.ModelRun
   :members:

.. autopydantic_model:: rompy.core.time.TimeRange
   :members:

.. autopydantic_model:: rompy.core.config.BaseConfig
   :members:

.. autopydantic_model:: rompy.core.grid.BaseGrid
   :members:

.. autopydantic_model:: rompy.core.data.DataBlob
   :members:

.. autopydantic_model:: rompy.core.data.DataGrid
   :members:

Swan Classes
==============

.. autosummary::
   :toctree: classes/swan

   rompy.swan

.. autopydantic_model:: rompy.swan.config.SwanConfig
   :members:

.. autopydantic_model:: rompy.swan.grid.SwanGrid
   :members:

.. autopydantic_model:: rompy.swan.config.ForcingData
   :members:

.. autopydantic_model:: rompy.swan.data.SwanDataGrid
   :members:

.. autopydantic_model:: rompy.swan.config.SwanSpectrum
   :members:

.. autopydantic_model:: rompy.swan.config.SwanPhysics
   :members:

.. autopydantic_model:: rompy.swan.config.GridOutput
   :members:

.. autopydantic_model:: rompy.swan.config.SpecOutput
   :members:
