#!/bin/bash
#
set -e

# test=nml_3d_tidal_velocities
# test=nml
test=nml_3d_nontidal_velocities_enhanced
version=v5.11.1


rm -fr *png

echo ""
echo "-------------- Preparing workspace -------------- "
# rompy schism demo.yaml
# rompy schism demo_tides_only.yaml
echo rompy schism demo_${test}.yaml
rompy schism demo_${test}.yaml

# Make sure the directory exists
mkdir -p ./schism_declaritive/test_schism_${test}


echo ""
echo "--------------- Exploring Docker container ------------------"
# First inspect the directory structure in the container
docker run -v ./schism_declaritive/test_schism_${test}:/tmp/schism:Z schism bash -c "ls -la /tmp/schism/ && echo 'Current working dir:' "

echo ""
echo "--------------- Running SCHISM ------------------"
# docker run  -v ./schism_declaritive/${test}:/tmp/schism  schism  mpiexec -np 8 --allow-run-as-root schism_${version} 1
docker run -v ./schism_declaritive/test_schism_${test}:/tmp/schism:Z schism bash -c "cd /tmp/schism && mpirun --allow-run-as-root -n 8 schism_${version} 6"

echo "------------ SCHISM run complete ----------------"


# echo ""
# echo "--------------- Preparing plots ------------------"
# python ../../rompy/schism/utils.py ./schism_declaritive/test_schism_${test}/outputs/schout_1.nc
