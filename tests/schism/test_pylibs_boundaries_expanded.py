"""
Extended test script for finding boundary information in PyLibs grid objects.
This script explores PyLibs-specific methods for accessing boundary information.
"""

import os
import logging
import sys
import numpy as np

from pylib import read_schism_hgrid

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def explore_boundary_methods(hgrid):
    """Explore PyLibs-specific boundary methods and attributes."""
    results = {}

    # Check if compute_bnd method exists and call it
    if hasattr(hgrid, "compute_bnd"):
        try:
            logger.info("Calling compute_bnd() to ensure boundaries are initialized")
            hgrid.compute_bnd()
            results["compute_bnd"] = "Called successfully"
        except Exception as e:
            logger.error(f"Error calling compute_bnd(): {str(e)}")
            results["compute_bnd"] = f"ERROR: {str(e)}"

    # Check boundary count attributes
    boundary_attrs = ["nlb", "nob"]
    for attr in boundary_attrs:
        if hasattr(hgrid, attr):
            value = getattr(hgrid, attr)
            logger.info(f"Found boundary attribute {attr} = {value}")
            results[attr] = value

    # Check node boundary lists
    node_boundary_attrs = ["nlbn", "nobn"]
    for attr in node_boundary_attrs:
        if hasattr(hgrid, attr):
            value = getattr(hgrid, attr)
            if value is not None:
                logger.info(
                    f"Found node boundary list {attr} with shape {np.shape(value)}"
                )
                if isinstance(value, list):
                    for i, boundary in enumerate(value):
                        logger.info(f"  Boundary {i} has {len(boundary)} nodes")
                    results[attr] = value
                elif hasattr(value, "shape"):  # numpy array
                    results[attr] = value
                    logger.info(f"  Boundary array with shape {value.shape}")
            else:
                logger.info(f"Attribute {attr} is None")

    # Check boundary index arrays
    boundary_index_attrs = ["ilbn", "iobn"]
    for attr in boundary_index_attrs:
        if hasattr(hgrid, attr):
            value = getattr(hgrid, attr)
            if value is not None:
                logger.info(
                    f"Found boundary index array {attr} with shape {np.shape(value)}"
                )
                results[attr] = value
            else:
                logger.info(f"Attribute {attr} is None")

    # Try to create boundaries if none found
    if all(getattr(hgrid, attr, None) is None for attr in node_boundary_attrs):
        if hasattr(hgrid, "create_bnd"):
            try:
                logger.info(
                    "No boundaries found, trying to create them with create_bnd()"
                )
                hgrid.create_bnd()
                # Check boundary attributes again
                for attr in node_boundary_attrs:
                    if hasattr(hgrid, attr):
                        value = getattr(hgrid, attr)
                        if value is not None:
                            logger.info(
                                f"After create_bnd(), found {attr} with shape {np.shape(value)}"
                            )
                            if isinstance(value, list):
                                for i, boundary in enumerate(value):
                                    logger.info(
                                        f"  Boundary {i} has {len(boundary)} nodes"
                                    )
                                results[f"after_create_{attr}"] = value
            except Exception as e:
                logger.error(f"Error calling create_bnd(): {str(e)}")

    return results


def main():
    """Run extended boundary tests on a grid file."""
    if len(sys.argv) < 2:
        print("Usage: python test_pylibs_boundaries_expanded.py <path_to_grid_file>")
        return

    grid_path = sys.argv[1]
    if not os.path.exists(grid_path):
        print(f"Error: Grid file not found: {grid_path}")
        return

    logger.info(f"Loading grid file: {grid_path}")
    try:
        hgrid = read_schism_hgrid(grid_path)
        logger.info("Grid loaded successfully")

        # Explore PyLibs-specific boundary methods
        results = explore_boundary_methods(hgrid)

        print("\nBOUNDARY EXTRACTION TEST RESULTS:")
        print("==================================")
        print(f"Grid file: {grid_path}")

        if "nobn" in results and results["nobn"] is not None:
            print("\nSUCCESS: Found open boundary nodes in 'nobn' attribute")
            if isinstance(results["nobn"], list):
                for i, boundary in enumerate(results["nobn"]):
                    print(f"  Boundary {i}: {len(boundary)} nodes")
            recommended_method = "hgrid.nobn"
        elif "after_create_nobn" in results:
            print("\nSUCCESS: Found open boundary nodes after calling create_bnd()")
            for i, boundary in enumerate(results["after_create_nobn"]):
                print(f"  Boundary {i}: {len(boundary)} nodes")
            recommended_method = "hgrid.create_bnd() followed by hgrid.nobn"
        elif "nlbn" in results and results["nlbn"] is not None:
            print("\nSUCCESS: Found land boundary nodes in 'nlbn' attribute")
            if isinstance(results["nlbn"], list):
                for i, boundary in enumerate(results["nlbn"]):
                    print(f"  Boundary {i}: {len(boundary)} nodes")
            recommended_method = "hgrid.nlbn"
        elif "after_create_nlbn" in results:
            print("\nSUCCESS: Found land boundary nodes after calling create_bnd()")
            for i, boundary in enumerate(results["after_create_nlbn"]):
                print(f"  Boundary {i}: {len(boundary)} nodes")
            recommended_method = "hgrid.create_bnd() followed by hgrid.nlbn"
        else:
            print(
                "\nFAILED: Could not find boundary information in this grid implementation"
            )
            recommended_method = "None"

        print(f"\nRecommended boundary access method: {recommended_method}")

    except Exception as e:
        logger.error(f"Error processing grid: {str(e)}")


if __name__ == "__main__":
    main()
