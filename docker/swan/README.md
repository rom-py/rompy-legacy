# SWAN model docker


This image builds the [SWAN wave model](https://swanmodel.sourceforge.io/) with netcdf and mpich support using gfortran. The image uses the cmake configuration files available in the new [gitlab repository](https://gitlab.tudelft.nl/citg/wavemodels/swan) but installs a local copy of the source code given previous releases are currently not available in either sourceforge or gitlab swan repositories.

The following structure is expected in order to build an specific version of SWAN, for instance version 4141:

src-4141

├── `swan4141.tar.gz`

├── `patch41.41.A`

├── `patch41.41.B`

└── ...

where `swan4141.tar.gz` is the tarball for this release downloaded from sourceforge and the patch files `A`, `B`, etc are the patch files to apply (patching is currently not implemented in the docker build).

SWAN is instaled inside the docker at */usr/local/bin/swan.exe*.

Versions for SWAN and some other building tools are specified in the *.env* file next to *compose.yaml*.