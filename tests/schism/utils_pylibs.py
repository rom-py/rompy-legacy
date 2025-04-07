"""
Utility functions for SCHISM tests that work with PyLibs.
This is a PySchism-free version of utils.py.
"""

import logging
import os
from pathlib import Path
import tarfile

# Initialize logger
logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def compare_files(file1, file2):
    """Compare two files line by line, ignoring lines that start with $."""
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            for line1, line2 in zip(f1, f2):
                if line1[0] != "$" and line2[0] != "$":
                    assert line1.rstrip() == line2.rstrip()


def untar_file(file, dest=Path("./")):
    """Extract a tar file."""
    dest = Path(dest)
    dest.mkdir(exist_ok=True, parents=True)
    with tarfile.open(file) as tar:
        tar.extractall(path=dest)


# Function to step through the namelist and compare the values
def compare_nmls_values(nml1, nml2, raise_missing=False):
    """Compare two namelists recursively."""
    for key in nml1.keys():
        if key not in nml2:
            if raise_missing:
                raise ValueError(f"Missing key {key} in nml2")
            else:
                continue
        
        value = nml1[key]
        if isinstance(value, dict):
            # If dictionary size is 2, extract value from 'default' key
            if len(value) == 2 and "default" in value:
                var = value["default"]
                print(key, var, nml2[key])
                if var != nml2[key]["default"]:
                    print(key, var, nml2[key]["default"])
            else:
                compare_nmls_values(value, nml2[key], raise_missing=raise_missing)
        else:
            if value != nml2[key]:
                print(f"Value mismatch for {key}: {value} != {nml2[key]}")
                assert value == nml2[key]


def compare_nmls(nml1, nml2, raise_missing=False):
    """Compare two namelists."""
    from rompy.schism.namelists.generate_models import nml_to_dict
    
    # Convert namelists to dictionaries if they're not already
    if not isinstance(nml1, dict):
        nml1 = nml_to_dict(nml1)
    if not isinstance(nml2, dict):
        nml2 = nml_to_dict(nml2)
    
    compare_nmls_values(nml1, nml2, raise_missing=raise_missing)
