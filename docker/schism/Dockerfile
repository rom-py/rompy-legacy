FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /tmp/schism

RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y \
        git vim sudo tar unzip subversion wget \
        gcc g++ gfortran lftp openmpi-bin libopenmpi-dev \
        libnetcdf-dev libnetcdff-dev make cmake python3 python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY compile.sh v5.*.tar.gz /usr/src/
COPY include_modules_wwm /usr/src/include_modules
COPY Make.defs.gfortran /usr/src/Make.defs.local 
RUN  ln -s /usr/bin/mpirun /usr/local/bin/mpiexec
ENV  LC_ALL="C.UTF-8"
ENV  LANG="C.UTF-8"

RUN  cd /usr/src && ./compile.sh "v5.11.1"
RUN  cd /usr/src && ./compile.sh "v5.9.0"

RUN  ln -s /usr/bin/python3 /usr/bin/python
RUN  chmod +x /usr/src/compile.sh
