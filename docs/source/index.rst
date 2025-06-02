=================================
Welcome to rompy's documentation!
=================================

*Taking the pain out of ocean model setup*

**This library is in early prototype stage and the interfaces are likely to change**

This library takes an opinionated approach to combining the functionality of the cookie-cutter library (https://github.com/cookiecutter/cookiecutter) with the XArray ecosystem (http://xarray.pydata.org/en/stable/related-projects.html) and intake (https://github.com/intake/intake) for data to aid in the configuration and evaluation of coastal scale numerical models.

There are two base classes BaseModel and BaseGrid. BaseModel implements the cookie-cutter code and model configuration packaging. BaseGrid defines a loose definition of the grid as two arrays of x, y points that establish the models geographic extents, bounding box and convex hull.

At present only one example model has been implemented - the SwanModel (http://swanmodel.sourceforge.net/). An example cookie-cutter template for swan is provided in the ```rompy/templates``` folder.

A model implementation will generally consist of the following components:

1. A model class that inherits from BaseModel and implements the minimal interface. At present only a private ```_get_grid()``` method.
2. A grid class that inherits from BaseGrid and implements the minimal interface of either loading the grid from file or a model specific grid specification string
3. An XArray accessor that has methods that translate an XArray dataset into a model specific input file format (usually some bespoke text file format). This allows convenient namespacing of methods from an XArray dataset e.g.:

   ``` ds.swan.to_inpgrid(filename) ```

The final main component of the library is an intake driver that builds on the intake-xarray.DataSourceMixin and allows for the stacking of multiple model forecast datasets that are typically published in netCDF format on THREDDS/OpenDAP servers. The unique feature of the driver include:

1. The ability to use format strings in the urlpath and pass a dictionary of values for the format keys. The product of the dictionary values is expanded to a set of URLs that are scanned checked for existence using the fsspec library. This allows for scanning of both local filesystems and http servers in a targetted fashion, for example a specific date range of interest.
2. The subset of urls identified are opened with XArray with a preprocessing function that takes a dictionary of filters for common operations that are applied during pre-processing - allowing this to be parameterised in the intake catalog yaml entry for a specific dataset.
3. The result is either a stack of model forcasts normalised to an initialisation and lead time (hindcast=false), or a pseudo-reanalysis that selects the shortest lead-time for each time point in the stack.


.. toctree::
    :hidden:
    :maxdepth: 4

    Home <self>
    quickstart
    core_concepts
    formatting_and_logging
    cli
    models
    demo
    api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
