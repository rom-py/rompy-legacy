FROM  ubuntu:20.04

LABEL maintainer "CSIRO <csiro@domain.com>"

ARG MPICH_VERSION
ARG HDF5_MAJOR_VERSION
ARG HDF5_MINOR_VERSION
ARG NETCDF_C_VERSION
ARG NETCDF_FORTRAN_VERSION
ARG SWAN_VERSION

ENV SWAN_VERSION=$SWAN_VERSION
ENV HDF5_VERSION=$HDF5_MAJOR_VERSION.$HDF5_MINOR_VERSION
ENV BUILD_DIR=/tmp/build
ENV INSTALL_DIR=/usr/local
ENV DEBIAN_FRONTEND noninteractive

RUN echo "-------------------- Install system libraries --------------------" & \
    apt update && apt install -y --no-install-recommends \
        autoconf \
        automake \
        ca-certificates \
        cmake \
        make \
        gcc \
        gfortran \
        git \
        g++ \ 
        ninja-build \
        patch \
        wget \
        zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*


#==========================================
# Model build dependencies
#==========================================

WORKDIR $BUILD_DIR
RUN echo "-------------------- Build MPICH --------------------" && \
    wget http://www.mpich.org/static/downloads/$MPICH_VERSION/mpich-$MPICH_VERSION.tar.gz && \
    tar zxvf mpich-$MPICH_VERSION.tar.gz && \
    cd $BUILD_DIR/mpich-$MPICH_VERSION && \
    ./configure --prefix=$INSTALL_DIR --enable-shared=no --enable-static=yes && \
    make && \
    make install && \
    ldconfig $INSTALL_DIR/lib && \
    cd .. && \
    rm -rf $BUILD_DIR/mpich-$MPICH_VERSION

WORKDIR $BUILD_DIR
RUN echo "-------------------- Build HDF5 --------------------" && \
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-$HDF5_MAJOR_VERSION/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz && \
    tar zxvf hdf5-$HDF5_VERSION.tar.gz && \
    cd $BUILD_DIR/hdf5-$HDF5_VERSION && \
    ./configure --prefix=$INSTALL_DIR --enable-fortran --enable-cxx --enable-hl && \
    make && \
    make install && \
    ldconfig $INSTALL_DIR/lib && \
    cd .. && \
    rm -rf $BUILD_DIR/hdf5-$HDF5_VERSION

WORKDIR $BUILD_DIR
RUN echo "-------------------- Build NETCDF4-C --------------------" && \
    wget https://downloads.unidata.ucar.edu/netcdf-c/$NETCDF_C_VERSION/netcdf-c-$NETCDF_C_VERSION.tar.gz && \
    tar zxvf netcdf-c-$NETCDF_C_VERSION.tar.gz && \
    cd $BUILD_DIR/netcdf-c-$NETCDF_C_VERSION && \
    ./configure --prefix=$INSTALL_DIR --disable-dap --enable-shared --enable-static --disable-v2 && \
    make && \
    make install && \
    nc-config --all && \
    ldconfig $INSTALL_DIR/lib && \
    cd .. && \
    rm -rf $BUILD_DIR/netcdf-c-$NETCDF_C_VERSION

WORKDIR $BUILD_DIR
RUN echo "-------------------- Build NETCDF4-FORTRAN --------------------" && \
    wget https://downloads.unidata.ucar.edu/netcdf-fortran/$NETCDF_FORTRAN_VERSION/netcdf-fortran-$NETCDF_FORTRAN_VERSION.tar.gz && \
    tar zxvf netcdf-fortran-$NETCDF_FORTRAN_VERSION.tar.gz && \
    cd $BUILD_DIR/netcdf-fortran-$NETCDF_FORTRAN_VERSION && \
    ./configure --prefix=$INSTALL_DIR --enable-shared --enable-static && \
    make && \
    make install && \
    nf-config --all && \
    ldconfig $INSTALL_DIR/lib && \
    cd .. && \
    rm -rf $BUILD_DIR/netcdf-fortran-$NETCDF_FORTRAN_VERSION


#==========================================
# Model build
#==========================================
# cmake configuration files are provided in the new swan gitlab but tags / releases
# are currently not available in that repo. Previous releases of the model are also
# not available in sourceforge. Hense this will use the cmake configs from gitlab to
# define the build environment but wil use tagged versions of the source code copied
# from sourceforge and stored in rompy so versions can be tracked.

WORKDIR $BUILD_DIR
ADD src-$SWAN_VERSION/swan$SWAN_VERSION.tar.gz .
RUN echo "-------------------- Build SWAN --------------------" && \
    git clone --depth 1 https://gitlab.tudelft.nl/citg/wavemodels/swan.git && \
    cd swan && \
    mv src src-git && \
    mv ../swan$SWAN_VERSION src && \
    cp -r src-git/hcat src-git/CMakeLists.txt src-git/*.cmake src/ && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DNETCDF=ON -DMPI=ON && \
    cmake --build . && \
    cmake --install . --prefix $INSTALL_DIR && \
    cd ../../ && \
    rm -rf $BUILD_DIR/swan


#==========================================
# Test model with mpi and netcdf output
#==========================================
ADD tests/refrac.tar.gz $BUILD_DIR
WORKDIR $BUILD_DIR/refrac
RUN ln -s a11refr.swn INPUT && \
    sed -i 's/a11ref01.spc/a11ref01.nc/g' INPUT && \
    mpiexec -n 2 swan.exe && \
    ncdump -h a11ref01.nc && \
    cd .. && \
    rm -rf $BUILD_DIR/refrac


WORKDIR /home
RUN rm -rf $BUILD_DIR