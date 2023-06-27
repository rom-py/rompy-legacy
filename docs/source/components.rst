.. currentmodule:: rompy

================
Model Components
================

The SWAN command instructions are described in Rompy by a set of pydantic models
defined as `components`. Each component defines a full command instruction such
as `PROJECT`, `CGRID`, `GEN3`, etc. Inputs to the components may include other
pydantic models called `subcomponents` to handle more complex arguments.

Components are subclasses of :py:class:`rompy.swan.components.base.BaseComponent`.
The base component class implements the following attribues:

* The **model_type** field that must be overwritten in each component subclass. The
  `model_type` field is defined as a `Literal`_ type and is used to specify the exact
  component class in a declarative framework (i.e., when using a dict from a yaml or
  json file to prescribe the model config).

* The **cmd()** method that must be overwritten in each component subclass. The `cmd()`
  method should return either a string or a list of strings to fully define a SWAN
  command line instruction. A list of strings defines multiple command line
  instructions that are executed in sequence such as the [INPGRID, READGRID] components.

* The **render()** method that constructs the command line instruction from the content
  returned from the `cmd()` method. The `render()` method is typically called inside
  the model template to construct the specific command line instruction from that
  component, taking care of maximum line size, line break and line continuation.

TODO: Ensure the `model_type` is shown next to each class in the autosummaries.

TODO: Fix broken links to classes and modules.

Components
----------

Components are defined within the :py:mod:`rompy.swan.components` subpackage and
render an entire SWAN command line specification. The following modules are available:

* :doc:`components/startup`
* :doc:`components/cgrid`
* :doc:`components/inpgrid`
* :doc:`components/boundary`
* :doc:`components/physics`
* :doc:`components/numerics`
* :doc:`components/output`
* :doc:`components/lockup`


Subcomponents
-------------

Subcomponents are defined within the :py:mod:`rompy.swan.subcomponents` subpackage
and render part of a SWAN command line specification. They typically define specific
arguments to one or more component. The following modules are available:

* :doc:`subcomponents/startup`
* :doc:`subcomponents/spectrum`
* :doc:`subcomponents/cgrid`
* :doc:`subcomponents/time`
* :doc:`subcomponents/readgrid`
* :doc:`subcomponents/boundary`
* :doc:`subcomponents/physics`

.. _`Literal`: https://docs.python.org/3/library/typing.html#typing.Literal