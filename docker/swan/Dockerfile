FROM  ubuntu:20.04

LABEL maintainer="CSIRO <csiro@domain.com>"

# Using latest SWAN from Git - no version pinning needed
ENV BUILD_DIR=/tmp/build
ENV INSTALL_DIR=/usr/local
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "-------------------- Install system libraries --------------------" && \
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
        libhdf5-dev \
        libnetcdf-dev \
        libnetcdff-dev \
        libopenmpi-dev \
        ninja-build \
        openmpi-bin \
        patch \
        wget \
        zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*


#==========================================
# Model build dependencies
#==========================================

# Using pre-installed packages for OpenMPI, HDF5, and NetCDF instead of building from source


#==========================================
# Model build
#==========================================
# Using the latest SWAN source and build system from the official GitLab repository.
# This provides the most up-to-date CMake build configuration and source code.

WORKDIR $BUILD_DIR
RUN echo "-------------------- Build SWAN --------------------" && \
    git clone --depth 1 https://gitlab.tudelft.nl/citg/wavemodels/swan.git && \
    cd swan && \
    mkdir build && \
    cd build && \
    cmake .. -DNETCDF=ON -DMPI=OFF && \
    make VERBOSE=1 && \
    echo "=== Build directory contents ===" && \
    find . -type f -executable | head -20 && \
    echo "=== Looking for SWAN executables ===" && \
    find . -name "*swan*" -type f && \
    echo "=== Looking for any executables ===" && \
    find . -type f -perm /u+x | grep -v "\.so" | head -10 && \
    echo "=== Trying to install manually ===" && \
    find . -name "*swan*" -type f -executable -exec cp {} $INSTALL_DIR/bin/ \; && \
    echo "=== Final installed files ===" && \
    ls -la $INSTALL_DIR/bin/ && \
    cd ../../ && \
    rm -rf $BUILD_DIR/swan


# Skip the built-in test for now - will be handled by the container test


WORKDIR /home
RUN rm -rf $BUILD_DIR