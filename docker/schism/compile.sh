#!/bin/bash

set -e

# take version from commandline 
# if no version is provided, use 5.11.1
version=${1:-v5.11.1}

#check if tar file is already downloaded and download if not
if [ ! -f ${version}.tar.gz ]; then
  echo "Downloading schism version ${version}"
  wget https://github.com/schism-dev/schism/archive/refs/tags/${version}.tar.gz 
else
  echo "Using schism version ${version} from local disk"
fi
tar -xvf ${version}.tar.gz 
directory=$(echo $version | sed 's/v//')
mv schism-${directory} schism 
cp Make.defs.local /usr/src/schism/mk/Make.defs.local 
cp include_modules /usr/src/schism/mk/include_modules
cd /usr/src/schism/src/ 
make clean 
make psc 
cp pschism_git_HYDRO_VL_WWM /usr/local/bin/schism_${version} 
cd /usr/src/schism/src/Utility/Combining_Scripts/ 
gfortran -ffree-line-length-none -cpp -O2 -o combine_output11 ../UtilLib/argparse.f90 ../UtilLib/schism_geometry.f90 combine_output11.f90 -I/usr/include -L/usr/lib/ -lnetcdf -lnetcdff 
cp combine_output11 /usr/local/bin/combine_output11_${version} 
gfortran -ffree-line-length-none -cpp -O2 -o combine_hotstart7 ../UtilLib/argparse.f90 ../UtilLib/schism_geometry.f90 combine_hotstart7.f90 -I/usr/include -L/usr/lib/ -lnetcdf -lnetcdff 
cp combine_hotstart7 /usr/local/bin/combine_hotstart7_${version}
cd 
rm -rf /usr/src/schism
rm -rf /usr/src/${version}.tar.gz
