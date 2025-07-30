"""
Test script to determine which boundary extraction method works with PyLibs.
"""

import os
import logging
import sys
from pathlib import Path

from pylib import read_schism_hgrid

# Configure logging to show detailed information
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_grid_object(hgrid_obj):
    """Analyze a PyLibs grid object to determine how to access boundaries."""
    logger.info(f"Grid object type: {type(hgrid_obj).__name__}")
    logger.info(
        f"Grid object attributes: {[attr for attr in dir(hgrid_obj) if not attr.startswith('__')]}"
    )

    # Test each boundary access pattern
    if hasattr(hgrid_obj, "get_open_boundary_nodes"):
        try:
            boundaries = hgrid_obj.get_open_boundary_nodes()
            logger.info(
                f"Method 1 SUCCESS: get_open_boundary_nodes() returned {len(boundaries)} boundaries"
            )
            for i, nodes in enumerate(boundaries):
                logger.info(f"  Boundary {i} has {len(nodes)} nodes")
            return "Method 1: get_open_boundary_nodes()"
        except Exception as e:
            logger.error(f"Method 1 FAILED: get_open_boundary_nodes() - {str(e)}")
    else:
        logger.info("Method 1 NOT AVAILABLE: get_open_boundary_nodes()")

    if hasattr(hgrid_obj, "open_boundaries"):
        try:
            boundaries = hgrid_obj.open_boundaries
            if boundaries:
                logger.info(
                    f"Method 2 SUCCESS: open_boundaries attribute has {len(boundaries)} boundaries"
                )
                for i, nodes in enumerate(boundaries):
                    logger.info(f"  Boundary {i} has {len(nodes)} nodes")
                return "Method 2: open_boundaries attribute"
            else:
                logger.info(
                    "Method 2 EMPTY: open_boundaries attribute exists but is empty"
                )
        except Exception as e:
            logger.error(f"Method 2 FAILED: open_boundaries - {str(e)}")
    else:
        logger.info("Method 2 NOT AVAILABLE: open_boundaries")

    if hasattr(hgrid_obj, "boundaries"):
        boundaries_obj = hgrid_obj.boundaries
        logger.info(f"Boundaries object type: {type(boundaries_obj).__name__}")
        logger.info(
            f"Boundaries attributes: {[attr for attr in dir(boundaries_obj) if not attr.startswith('__')]}"
        )

        if hasattr(boundaries_obj, "open_bnds"):
            try:
                boundaries = boundaries_obj.open_bnds
                logger.info(
                    f"Method 3a SUCCESS: boundaries.open_bnds has {len(boundaries)} boundaries"
                )
                for i, boundary in enumerate(boundaries):
                    if hasattr(boundary, "nodes"):
                        logger.info(f"  Boundary {i} has {len(boundary.nodes)} nodes")
                    else:
                        logger.info(f"  Boundary {i} has {len(boundary)} nodes")
                return "Method 3a: boundaries.open_bnds"
            except Exception as e:
                logger.error(f"Method 3a FAILED: boundaries.open_bnds - {str(e)}")
        else:
            logger.info("Method 3a NOT AVAILABLE: boundaries.open_bnds")

        if hasattr(boundaries_obj, "get_boundary_nodes"):
            try:
                boundaries = boundaries_obj.get_boundary_nodes()
                logger.info(
                    f"Method 3b SUCCESS: boundaries.get_boundary_nodes() returned {len(boundaries)} boundaries"
                )
                for i, nodes in enumerate(boundaries):
                    logger.info(f"  Boundary {i} has {len(nodes)} nodes")
                return "Method 3b: boundaries.get_boundary_nodes()"
            except Exception as e:
                logger.error(
                    f"Method 3b FAILED: boundaries.get_boundary_nodes() - {str(e)}"
                )
        else:
            logger.info("Method 3b NOT AVAILABLE: boundaries.get_boundary_nodes()")
    else:
        logger.info("Method 3 NOT AVAILABLE: boundaries attribute")

    if hasattr(hgrid_obj, "mesh"):
        mesh = hgrid_obj.mesh
        logger.info(f"Mesh object type: {type(mesh).__name__}")
        logger.info(
            f"Mesh attributes: {[attr for attr in dir(mesh) if not attr.startswith('__')]}"
        )

        if hasattr(mesh, "boundaries") and mesh.boundaries:
            try:
                boundaries = mesh.boundaries
                logger.info(
                    f"Method 4 SUCCESS: mesh.boundaries has {len(boundaries)} boundaries"
                )
                for i, boundary in enumerate(boundaries):
                    if hasattr(boundary, "nodes"):
                        logger.info(f"  Boundary {i} has {len(boundary.nodes)} nodes")
                    else:
                        logger.info(f"  Boundary {i} has {len(boundary)} nodes")
                return "Method 4: mesh.boundaries"
            except Exception as e:
                logger.error(f"Method 4 FAILED: mesh.boundaries - {str(e)}")
        else:
            logger.info("Method 4 NOT AVAILABLE: mesh.boundaries")
    else:
        logger.info("Method 4 NOT AVAILABLE: mesh attribute")

    logger.warning("No working boundary extraction method found")
    return "No working method"


def main():
    """Run test on a grid file to identify boundary extraction methods."""
    if len(sys.argv) < 2:
        print("Usage: python test_pylibs_boundaries.py <path_to_grid_file>")
        return

    grid_path = sys.argv[1]
    if not os.path.exists(grid_path):
        print(f"Error: Grid file not found: {grid_path}")
        return

    logger.info(f"Loading grid file: {grid_path}")
    try:
        hgrid_obj = read_schism_hgrid(grid_path)
        logger.info("Grid loaded successfully")

        result = analyze_grid_object(hgrid_obj)
        logger.info(f"CONCLUSION: {result}")

        print("\n\nBOUNDARY EXTRACTION TEST RESULTS:")
        print("==================================")
        print(f"Grid file: {grid_path}")
        print(f"Working method: {result}")

    except Exception as e:
        logger.error(f"Error loading grid: {str(e)}")


if __name__ == "__main__":
    main()
