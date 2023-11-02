====
SWAN
====

TODO: Ensure the `model_type` is shown next to each class in the autosummaries.

TODO: Fix broken links to classes and modules.

Grid
----

.. autosummary::
   :nosignatures:
   :toctree: _generated/

   rompy.swan.grid.SwanGrid


Data
----

.. autosummary::
    :nosignatures:
    :toctree: _generated/

    rompy.swan.data.SwanDataGrid
    rompy.swan.boundary.Boundnest1


Components
----------

SWAN command instructions are described in Rompy by a set of pydantic models defined as
`components`. Each component defines a full command instruction such as `PROJECT`,
`CGRID`, `GEN3`, NUMERIC, etc. Inputs to the components may include other pydantic
models called `subcomponents` to handle more complex arguments.

Components are subclasses of :py:class:`rompy.swan.components.base.BaseComponent`.
The base component class implements the following attribues:

* The **model_type** field that should be overwritten in each component subclass. The
  `model_type` field is defined as a `Literal`_ type and is used to discriminate the
  exact components to use in fields defined by a `Union` type of two or more components
  in a declarative framework (i.e., instantiating with a dict from yaml or json file).

* The **cmd()** method that must be overwritten in each component subclass. The `cmd()`
  method should return either a string or a list of strings to fully define a SWAN
  command line instruction. A list of strings defines multiple command line
  instructions that are executed in sequence such as the INPGRID/READGRID components.

* The **render()** method that constructs the command line instruction from the content
  returned from the `cmd()` method. The `render()` method is typically called inside
  the `__call__` method of the config class to construct the specific command line
  instruction from that component, taking care of maximum line size, line break and
  line continuation.


Components are defined within the :py:mod:`rompy.swan.components` subpackage and
render an entire SWAN command line specification. The following modules are available:

.. toctree::
    :hidden:

    components/startup
    components/cgrid
    components/inpgrid
    components/boundary
    components/physics
    components/numerics
    components/output
    components/lockup
    components/group

* :doc:`components/startup`
* :doc:`components/cgrid`
* :doc:`components/inpgrid`
* :doc:`components/boundary`
* :doc:`components/physics`
* :doc:`components/numerics`
* :doc:`components/output`
* :doc:`components/lockup`
* :doc:`components/group`


Subcomponents
-------------

Subcomponents are defined within the :py:mod:`rompy.swan.subcomponents` subpackage
and render part of a SWAN command line specification. They typically define specific
arguments to one or more component. The following modules are available:

.. toctree::
    :hidden:

    subcomponents/base
    subcomponents/startup
    subcomponents/spectrum
    subcomponents/time
    subcomponents/cgrid
    subcomponents/readgrid
    subcomponents/boundary
    subcomponents/physics
    subcomponents/numerics
    subcomponents/output

* :doc:`subcomponents/base`
* :doc:`subcomponents/startup`
* :doc:`subcomponents/spectrum`
* :doc:`subcomponents/time`
* :doc:`subcomponents/cgrid`
* :doc:`subcomponents/readgrid`
* :doc:`subcomponents/boundary`
* :doc:`subcomponents/physics`
* :doc:`subcomponents/numerics`
* :doc:`subcomponents/output`


Interface
---------

Interface classes provide an interface between swan components and higher level objects
such as `TimeRange`, `Data` and `Grid` objects. They are used inside the `__call__`
method of the config classes to pass instances of these objects to the appropriate
components and define consistent parameters to the config after instantiating them.


.. autosummary::
    :nosignatures:
    :toctree: _generated/

    rompy.swan.interface.DataInterface
    rompy.swan.interface.BoundaryInterface
    rompy.swan.interface.OutputInterface
    rompy.swan.interface.LockupInterface


Types
-----

SWAN types provide valid values for a specific SWAN command line argument.

.. toctree::
    :hidden:

    types

* :doc:`types`


.. _`Literal`: https://docs.python.org/3/library/typing.html#typing.Literal