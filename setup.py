#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------

import sys

from setuptools import find_packages, setup

import versioneer

requires = [
    line.strip()
    for line in open("requirements.txt").readlines()
    if not line.startswith("#")
]
extras_require = {}
extras_require["complete"] = sorted(set(sum(extras_require.values(), [])))

# Only include pytest-runner in setup_requires if we're invoking tests
if {"pytest", "test", "ptr"}.intersection(sys.argv):
    setup_requires = ["pytest-runner"]
else:
    setup_requires = []

setup(
    name="rompy",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Relocatable Ocean Modelling in PYthon (rompy)",
    url="https://github.com/rom-py/rompy",
    maintainer="Paul Branson",
    maintainer_email="paul.branson@csiro.au",
    license="BSD",
    package_data={"": ["*.csv", "*.yml", "*.yaml", "*.html"]},
    include_package_data=True,
    install_requires=requires,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "rompy=rompy.cli:main",
        ],
        "intake.drivers": ["netcdf_fcstack = rompy.intake:NetCDFFCStackSource"],
        "intake.catalogs": ["rompy_data = rompy:cat"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    tests_require=["pytest"],
    extras_require=extras_require,
    zip_safe=False,
)
