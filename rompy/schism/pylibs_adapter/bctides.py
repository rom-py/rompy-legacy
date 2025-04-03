"""
Bctides adapter for PyLibs.

This module provides a Bctides class that uses PyLibs functionality under the hood
while maintaining the same interface as the original PySchism Bctides class.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
from functools import cached_property
from typing import Union, List, Optional

# Initialize logger first
logger = logging.getLogger(__name__)

import numpy as np

# Import PyLibs functions as needed
try:
    from pylib import read_schism_hgrid
except ImportError as e:
    logger.warning(f"Could not import read_schism_hgrid: {e}")
    read_schism_hgrid = None

# Create stubs for PyLibs script functions to avoid loading problematic modules
# These will be properly implemented when actually used
def stub_gen_bctides(*args, **kwargs):
    logger.warning("Using stub implementation of gen_bctides")
    return None

def stub_extract_fes14_h(*args, **kwargs):
    logger.warning("Using stub implementation of extract_fes14_h")
    return None

def stub_extract_fes14_uv(*args, **kwargs):
    logger.warning("Using stub implementation of extract_fes14_uv")
    return None

# Use the stubs instead of the actual implementations for now
gen_bctides = stub_gen_bctides
extract_fes14_h = stub_extract_fes14_h
extract_fes14_uv = stub_extract_fes14_uv


class Tides:
    """Adapter for tides functionality provided by PyLibs."""

    def __init__(
        self,
        constituents: Union[str, list] = "major",
        tidal_database: str = "tpxo"
    ):
        """Initialize Tides adapter.
        
        Parameters
        ----------
        constituents : str or list
            Tidal constituents to use. Can be "major", "all", or a list of specific constituents.
        tidal_database : str
            Tidal database to use. Options are "tpxo" or "fes2014".
        """
        self.tidal_database = tidal_database
        
        # Define constituent sets
        self.major_constituents = ("Q1", "O1", "P1", "K1", "N2", "M2", "S2", "K2")
        self.minor_constituents = ("Mm", "Mf", "M4", "MN4", "MS4", "2N2", "S1")
        self.constituents = self.major_constituents + self.minor_constituents
        
        # Set active constituents
        self.active_constituents = {}
        
        if isinstance(constituents, str):
            constituents = [constituents]
            
        if "all" in constituents:
            for const in self.constituents:
                self.active_constituents[const] = {"potential": True, "forcing": True}
        elif "major" in constituents:
            for const in self.major_constituents:
                self.active_constituents[const] = {"potential": True, "forcing": True}
        else:
            for const in constituents:
                if const in self.constituents:
                    self.active_constituents[const] = {"potential": True, "forcing": True}
    
    def get_active_forcing_constituents(self):
        """Get active forcing constituents."""
        return [c for c, v in self.active_constituents.items() if v["forcing"]]
    
    def get_active_potential_constituents(self):
        """Get active potential constituents."""
        return [c for c, v in self.active_constituents.items() if v["potential"]]
    
    def get_elevation(self, constituent, vertices):
        """Get tidal elevation amplitude and phase for a constituent.
        
        Uses PyLibs to extract tidal information based on the tidal database.
        
        Parameters
        ----------
        constituent : str
            Tidal constituent name
        vertices : array-like
            Coordinates of vertices (Nx2 array with lon, lat)
            
        Returns
        -------
        tuple
            (amplitude, phase) arrays
        """
        lons, lats = vertices[:, 0], vertices[:, 1]
        
        if self.tidal_database.lower() == 'fes2014':
            # Use PyLibs' extract_fes14_h function
            amp, phase = extract_fes14_h(lons, lats, constituent)
        else:
            # For TPXO, implement or call appropriate PyLibs function
            # This is a placeholder - you'd implement the actual call to PyLibs
            logger.warning(f"Using placeholder for {self.tidal_database} database extraction")
            amp = np.ones(len(lons)) * 0.1  # Placeholder
            phase = np.zeros(len(lons))  # Placeholder
            
        return amp, phase
    
    def get_velocity(self, constituent, vertices):
        """Get tidal velocity amplitude and phase for a constituent.
        
        Parameters
        ----------
        constituent : str
            Tidal constituent name
        vertices : array-like
            Coordinates of vertices (Nx2 array with lon, lat)
            
        Returns
        -------
        tuple
            (u_amplitude, u_phase, v_amplitude, v_phase) arrays
        """
        lons, lats = vertices[:, 0], vertices[:, 1]
        
        if self.tidal_database.lower() == 'fes2014':
            # Use PyLibs' extract_fes14_uv function
            uamp, uphase, vamp, vphase = extract_fes14_uv(lons, lats, constituent)
        else:
            # For TPXO, implement or call appropriate PyLibs function
            # This is a placeholder - you'd implement the actual call to PyLibs
            logger.warning(f"Using placeholder for {self.tidal_database} database extraction")
            uamp = np.ones(len(lons)) * 0.05  # Placeholder
            uphase = np.zeros(len(lons))  # Placeholder
            vamp = np.ones(len(lons)) * 0.05  # Placeholder
            vphase = np.zeros(len(lons))  # Placeholder
            
        return uamp, uphase, vamp, vphase
    
    def __call__(self, start_date, rnday, constituent):
        """Calculate tidal factors for the given time period and constituent.
        
        This is a simplified implementation that returns placeholder values.
        You would implement the actual calculation using PyLibs functionality.
        
        Parameters
        ----------
        start_date : datetime
            Start date for the calculation
        rnday : float or int
            Run length in days
        constituent : str
            Tidal constituent name
            
        Returns
        -------
        tuple
            Tidal factors (species_type, amplitude, frequency, nodal_factor, greenwich_factor)
        """
        # Placeholder implementation
        species_type = 1  # Placeholder
        amplitude = 0.1  # Placeholder
        frequency = 1.0  # Placeholder
        nodal_factor = 1.0  # Placeholder
        greenwich_factor = 0.0  # Placeholder
        
        return species_type, amplitude, frequency, nodal_factor, greenwich_factor


class Bctides:
    """Bctides adapter that uses PyLibs under the hood.
    
    This class maintains the same interface as the original PySchism Bctides class
    but uses PyLibs for the actual functionality.
    """

    def __init__(
        self,
        hgrid,
        flags: list = None,
        constituents: Union[str, list] = "major",
        database: str = "tpxo",
        add_earth_tidal: bool = True,
        cutoff_depth: float = 50.0,
        ethconst: list = None,
        vthconst: list = None,
        tthconst: list = None,
        sthconst: list = None,
        tobc: list = None,
        sobc: list = None,
        relax: list = None,
    ):
        """Initialize Bctides adapter.
        
        Parameters
        ----------
        hgrid : SchismHGrid, str, or Path
            SCHISM horizontal grid object or path to grid file
        flags : list of list, optional
            Boundary types for each open boundary [iettype, ifltype, itetype, isatype]
        constituents : str or list, optional
            Tidal constituents to include ('major', 'all', or list of constituents)
        database : str, optional
            Tidal database to use (e.g., 'tpxo', 'fes')
        add_earth_tidal : bool, optional
            Whether to add earth tidal potential
        cutoff_depth : float, optional
            Cutoff depth for tidal potential (meters)
        ethconst : list, optional
            Constant elevation values for each boundary
        vthconst : list, optional
            Constant discharge values for each boundary
        tthconst : list, optional
            Constant temperature values for each boundary
        sthconst : list, optional
            Constant salinity values for each boundary
        tobc : list, optional
            Temperature nudging factors (0: no nudging, 1: full nudging)
        sobc : list, optional
            Salinity nudging factors (0: no nudging, 1: full nudging)
        relax : list, optional
            Relaxation constants for boundaries
        """
        self.hgrid = hgrid
        if isinstance(hgrid, str) or isinstance(hgrid, Path):
            # If hgrid is a path, read it using PyLibs
            self.hgrid_path = hgrid
            self.hgrid_obj = read_schism_hgrid(hgrid)
        else:
            # Otherwise, assume it's already a grid object
            self.hgrid_obj = hgrid
            self.hgrid_path = None
        
        self.flags = flags if flags is not None else [[5, 3, 0, 0]]
        self.add_earth_tidal = add_earth_tidal
        self.cutoff_depth = cutoff_depth
        self.tides = Tides(constituents=constituents, tidal_database=database)
        
        # Boundary constants
        self.ethconst = ethconst if ethconst is not None else []
        self.vthconst = vthconst if vthconst is not None else []
        self.tthconst = tthconst if tthconst is not None else []
        self.sthconst = sthconst if sthconst is not None else []
        self.tobc = tobc if tobc is not None else [1]
        self.sobc = sobc if sobc is not None else [1]
        self.relax = relax if relax is not None else []
        
        # Additional properties
        self._start_date = datetime.now()
        self._rnday = 30  # Default run length in days

    @property
    def start_date(self):
        """Get start date for tidal calculations."""
        return self._start_date
    
    @property
    def rnday(self):
        """Get run length in days."""
        return self._rnday
    
    @property
    def gdf(self):
        """Get open boundaries as a GeoDataFrame-like object."""
        class BoundaryTuple:
            def __init__(self, indexes):
                self.indexes = indexes
        
        class BoundaryCollection:
            def __init__(self, boundaries):
                self.boundaries = boundaries
                
            def __len__(self):
                return len(self.boundaries)
            
            def itertuples(self):
                return self.boundaries
        
        # Get boundaries from the PyLibs grid object
        boundaries = []
        
        # First compute all grid geometry information with compute_all()
        if hasattr(self.hgrid_obj, 'compute_all'):
            logger.info("Computing all grid geometry using PyLibs compute_all() method")
            try:
                self.hgrid_obj.compute_all()
            except Exception as e:
                logger.warning(f"Warning during compute_all(): {str(e)}")
        
        # Then compute boundary information with compute_bnd()
        if hasattr(self.hgrid_obj, 'compute_bnd'):
            logger.info("Computing boundary information using PyLibs compute_bnd() method")
            try:
                # This will populate nob, iobn, and other boundary attributes
                self.hgrid_obj.compute_bnd()
            except Exception as e:
                error_msg = f"Failed to compute grid boundaries using PyLibs: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # After compute_bnd(), check for the standard PyLibs boundary attributes
        # In PyLibs, nob = number of open boundaries, iobn = node indices for each boundary
        if hasattr(self.hgrid_obj, 'nob') and self.hgrid_obj.nob is not None:
            logger.info(f"Grid has {self.hgrid_obj.nob} open boundaries")
            
            if hasattr(self.hgrid_obj, 'iobn') and self.hgrid_obj.iobn is not None:
                # Create boundary tuples from the SCHISM open boundary information
                for i in range(self.hgrid_obj.nob):
                    boundary_nodes = self.hgrid_obj.iobn[i]
                    boundaries.append(BoundaryTuple(boundary_nodes))
                    logger.info(f"Added open boundary {i} with {len(boundary_nodes)} nodes")
            else:
                error_msg = "Grid is missing iobn attribute after compute_bnd(). Cannot extract boundary nodes."
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # Either compute_bnd() wasn't available or the grid has no defined boundaries
            error_msg = "Grid has no open boundaries (nob attribute is missing or zero)."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        return BoundaryCollection(boundaries)
    
    @property
    def ntip(self):
        """Get number of tidal potential constituents."""
        return len(self.tides.get_active_potential_constituents()) if self.add_earth_tidal else 0
    
    @property
    def nbfr(self):
        """Get number of boundary forcing constituents."""
        return len(self.tides.get_active_forcing_constituents())
        
    def write(self, path, start_date=None, rnday=None, overwrite=False):
        """Write bctides.in file using PyLibs.
        
        Parameters
        ----------
        path : str or Path
            Path to output file
        start_date : datetime, optional
            Start date for the run
        rnday : float or int, optional
            Run length in days
        overwrite : bool, optional
            Whether to overwrite existing file (not used in PyLibs implementation
            but maintained for compatibility)
        """
        if start_date is not None:
            self._start_date = start_date
        if rnday is not None:
            self._rnday = rnday
            
        # Use PyLibs' gen_bctides function if available, or implement your own
        # This is a placeholder for the actual implementation
        
        # Check if path is a directory - if so, append 'bctides.in'
        path_obj = Path(path)
        if path_obj.is_dir():
            output_file = path_obj / 'bctides.in'
        else:
            output_file = path_obj
        
        logger.info(f"Writing bctides file to {output_file}")
        
        # Build the bctides.in file content as a list of lines
        content = self._build_bctides_content()
        
        # Write the file
        with open(output_file, 'w') as f:
            f.write('\n'.join(content))
            
        return str(output_file)
    
    def _build_bctides_content(self):
        """Build bctides.in file content.
        
        Returns
        -------
        list
            Lines for the bctides.in file
        """
        # This is a simplified implementation that follows the same format as
        # the original PySchism Bctides.write method
        f = [
            f"!{str(self.start_date)} UTC",
        ]

        # Get earth tidal potential and frequency
        if self.add_earth_tidal:
            f.append(
                f"{self.ntip} {self.cutoff_depth} !number of earth tidal potential, cut-off depth for applying tidal potential"
            )
            for constituent in self.tides.get_active_potential_constituents():
                forcing = self.tides(self.start_date, self.rnday, constituent)
                f.append(
                    " ".join(
                        [
                            f"{constituent}\n",
                            f"{forcing[0]:G}",
                            f"{forcing[1]:G}",
                            f"{forcing[2]:G}",
                            f"{forcing[3]:G}",
                            f"{forcing[4]:G}",
                        ]
                    )
                )
        else:
            f.append(
                f"0 {self.cutoff_depth} !number of earth tidal potential, cut-off depth for applying tidal potential"
            )

        # Get tidal boundary
        f.append(f"{self.nbfr:d} !nbfr")
        if self.nbfr > 0:
            for constituent in self.tides.get_active_forcing_constituents():
                forcing = self.tides(self.start_date, self.rnday, constituent)
                f.append(
                    " ".join(
                        [
                            f"{constituent}\n",
                            f"{forcing[2]:G}",
                            f"{forcing[3]:G}",
                            f"{forcing[4]:G}",
                        ]
                    )
                )

        # Get amplitude and phase for each open boundary
        boundaries = self.gdf
        f.append(f"{len(boundaries)} !nope")
        
        if len(boundaries) != len(self.flags):
            raise ValueError(
                f"Number of open boundary {len(boundaries)} is not consistent with number of given bctypes {len(self.flags)}!"
            )
            
        for ibnd, (boundary, flag) in enumerate(zip(boundaries.itertuples(), self.flags)):
            logger.info(f"Processing boundary {ibnd+1}:")
            # Number of nodes and flags
            line = [
                f"{len(boundary.indexes)}",
                *[str(digit) for digit in flag],
                f"!open bnd {ibnd+1}",
            ]
            f.append(" ".join(line))

            # Process boundary conditions based on flag types
            iettype, ifltype, itetype, isatype = [i for i in flag]

            # Handle elevation boundary
            self._process_elevation_boundary(f, iettype, ibnd, boundary)
            
            # Handle velocity boundary
            self._process_velocity_boundary(f, ifltype, ibnd, boundary)
            
            # Handle temperature boundary
            self._process_temperature_boundary(f, itetype, ibnd)
            
            # Handle salinity boundary
            self._process_salinity_boundary(f, isatype, ibnd)

        return f
    
    def _process_elevation_boundary(self, f, iettype, ibnd, boundary):
        """Process elevation boundary conditions based on flag type."""
        logger.info(f"Elevation type: {iettype}")
        
        if iettype == 1:
            logger.warning("Time history of elevation is read in from elev.th (ASCII)!")
        elif iettype == 2:
            if ibnd < len(self.ethconst):
                f.append(f"{self.ethconst[ibnd]}")
            else:
                raise ValueError(f"Missing ethconst value for boundary {ibnd+1}")
        elif iettype == 4:
            logger.warning("Time history of elevation is read in from elev2D.th.nc (netcdf)")
        elif iettype == 3 or iettype == 5:
            if iettype == 5:
                logger.warning("Combination of 3 and 4, time history of elevation is read in from elev2D.th.nc!")
            
            # Process tidal constituents
            for constituent in self.tides.get_active_forcing_constituents():
                f.append(f"{constituent}")
                
                # Extract coordinates for boundary nodes
                # In a complete implementation, you would get actual coordinates from hgrid
                vertices = np.zeros((len(boundary.indexes), 2))  # Placeholder
                
                # Get amplitude and phase from tides object
                amp, phase = self.tides.get_elevation(constituent, vertices)
                
                # Write amp/phase for each node
                for i in range(len(boundary.indexes)):
                    f.append(f"{amp[i]: .6f} {phase[i]: .6f}")
        elif iettype == 0:
            logger.warning("Elevations are not specified for this boundary (discharge must be specified)")
        else:
            raise ValueError(f"Invalid type {iettype} for elevation!")
    
    def _process_velocity_boundary(self, f, ifltype, ibnd, boundary):
        """Process velocity boundary conditions based on flag type."""
        logger.info(f"Velocity type: {ifltype}")
        
        if ifltype == 0:
            logger.info("Velocity is not specified, no input needed!")
        elif ifltype == 1:
            logger.warning("Time history of discharge is read in from flux.th (ASCII)!")
        elif ifltype == 2:
            if ibnd < len(self.vthconst):
                f.append(f"{self.vthconst[ibnd]}")
            else:
                raise ValueError(f"Missing vthconst value for boundary {ibnd+1}")
        elif ifltype == 3 or ifltype == 5:
            if ifltype == 5:
                logger.warning("Combination of 3 and 4, time history of velocity is read in from uv.3D.th.nc!")
            
            # Process tidal constituents
            for constituent in self.tides.get_active_forcing_constituents():
                f.append(f"{constituent}")
                
                # Extract coordinates for boundary nodes
                # In a complete implementation, you would get actual coordinates from hgrid
                vertices = np.zeros((len(boundary.indexes), 2))  # Placeholder
                
                # Get amplitude and phase from tides object
                uamp, uphase, vamp, vphase = self.tides.get_velocity(constituent, vertices)
                
                # Write amp/phase for each node
                for i in range(len(boundary.indexes)):
                    f.append(f"{uamp[i]: .6f} {uphase[i]: .6f} {vamp[i]: .6f} {vphase[i]: .6f}")
        elif ifltype == 4 or ifltype == -4:
            logger.warning("Time history of velocity is read in from uv3D.th.nc (netcdf)")
        else:
            raise ValueError(f"Invalid type {ifltype} for velocity!")
    
    def _process_temperature_boundary(self, f, itetype, ibnd):
        """Process temperature boundary conditions based on flag type."""
        logger.info(f"Temperature type: {itetype}")
        
        if itetype == 0:
            logger.info("Temperature is not specified, no input needed!")
        elif itetype == 1:
            logger.warning("Time history of temperature is read in from temp.th (ASCII)!")
        elif itetype == 2:
            if ibnd < len(self.tthconst):
                f.append(f"{self.tthconst[ibnd]}")
            else:
                raise ValueError(f"Missing tthconst value for boundary {ibnd+1}")
        elif itetype == 3:
            logger.warning("Time history of temperature is read in from temp3D.th.nc (netcdf)")
            if ibnd < len(self.tobc):
                f.append(f"{self.tobc[ibnd]}")
            else:
                raise ValueError(f"Missing tobc value for boundary {ibnd+1}")
        elif itetype == 4:
            logger.warning("Nudging is used at this boundary for temperature")
            if ibnd < len(self.tobc):
                f.append(f"{self.tobc[ibnd]}")
            else:
                raise ValueError(f"Missing tobc value for boundary {ibnd+1}")
        else:
            raise ValueError(f"Invalid type {itetype} for temperature!")
    
    def _process_salinity_boundary(self, f, isatype, ibnd):
        """Process salinity boundary conditions based on flag type."""
        logger.info(f"Salinity type: {isatype}")
        
        if isatype == 0:
            logger.info("Salinity is not specified, no input needed!")
        elif isatype == 1:
            logger.warning("Time history of salinity is read in from salt.th (ASCII)!")
        elif isatype == 2:
            if ibnd < len(self.sthconst):
                f.append(f"{self.sthconst[ibnd]}")
            else:
                raise ValueError(f"Missing sthconst value for boundary {ibnd+1}")
        elif isatype == 3:
            logger.warning("Time history of salinity is read in from salt3D.th.nc (netcdf)")
            if ibnd < len(self.sobc):
                f.append(f"{self.sobc[ibnd]}")
            else:
                raise ValueError(f"Missing sobc value for boundary {ibnd+1}")
        elif isatype == 4:
            logger.warning("Nudging is used at this boundary for salinity")
            if ibnd < len(self.sobc):
                f.append(f"{self.sobc[ibnd]}")
            else:
                raise ValueError(f"Missing sobc value for boundary {ibnd+1}")
        else:
            raise ValueError(f"Invalid type {isatype} for salinity!")
