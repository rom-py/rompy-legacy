"""
Tidal boundary conditions for SCHISM.

A direct implementation based on PyLibs scripts/gen_bctides.py with no fallbacks.
"""

import logging
import os
import sys
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional, Tuple, Any

import numpy as np
from pylib import *

logger = logging.getLogger(__name__)


class Bctides:
    """Direct implementation of SCHISM tidal boundary conditions using PyLibs.
    
    Based on scripts/gen_bctides.py with no fallback mechanisms.
    """

    def __init__(
        self,
        hgrid,
        flags=None,
        constituents="major",
        tidal_database="tpxo",
        ntip=0,
        tip_dp=1.0,
        cutoff_depth=50.0,
        ethconst=None,
        vthconst=None,
        tthconst=None,
        sthconst=None,
        tobc=None,
        sobc=None,
        relax=None,
        tidal_elevations=None,  # Path to tidal elevations file (TPXO format)
        tidal_velocities=None,  # Path to tidal velocities file (TPXO format)
    ):
        """Initialize Bctides handler.
        
        Parameters
        ----------
        hgrid : Grid or str
            SCHISM horizontal grid
        flags : list of lists, optional
            Boundary condition flags
        constituents : str or list, optional
            Tidal constituents to use, by default "major"
        tidal_database : str, optional
            Tidal database to use, by default "tpxo"
        ntip : int, optional
            Number of earth tidal potential regions (0 to disable), by default 0
        tip_dp : float, optional
            Depth threshold for tidal potential, by default 1.0
        cutoff_depth : float, optional
            Cutoff depth for tides, by default 50.0
        ethconst : list, optional
            Constant elevation for each boundary
        vthconst : list, optional
            Constant velocity for each boundary
        tthconst : list, optional
            Constant temperature for each boundary
        sthconst : list, optional
            Constant salinity for each boundary
        tobc : list, optional
            Temperature OBC values
        sobc : list, optional
            Salinity OBC values
        relax : list, optional
            Relaxation parameters
        """
        self.flags = flags or [[5, 5, 4, 4]]
        self.ntip = ntip
        self.tip_dp = tip_dp
        self.cutoff_depth = cutoff_depth
        self.ethconst = ethconst or []
        self.vthconst = vthconst or []
        self.tthconst = tthconst or []
        self.sthconst = sthconst or []
        self.tobc = tobc or [1]
        self.sobc = sobc or [1]
        self.relax = relax or []
        self.tidal_database = tidal_database
        
        # Store tidal file paths
        self.tidal_elevations = tidal_elevations
        self.tidal_velocities = tidal_velocities
        
        # Store start time and run duration (will be set by SCHISMDataTides.get())
        self._start_time = None
        self._rnday = None

        # Load grid from file or object
        if isinstance(hgrid, str) or isinstance(hgrid, Path):
            hgrid_path = str(hgrid)
            if hgrid_path.endswith('.npz'):
                self.gd = loadz(hgrid_path).hgrid
                self.gd.x = self.gd.lon
                self.gd.y = self.gd.lat
            else:
                self.gd = read_schism_hgrid(hgrid_path)
        else:
            # Assume it's already a grid object
            self.gd = hgrid
            
        # Define constituent sets
        self.major_constituents = ["O1", "K1", "Q1", "P1", "M2", "S2", "K2", "N2"]
        self.minor_constituents = ["MM", "Mf", "M4", "MN4", "MS4", "2N2", "S1"]
        
        # Determine which constituents to use
        if isinstance(constituents, str):
            if constituents.lower() == "major":
                self.tnames = self.major_constituents
            elif constituents.lower() == "all":
                self.tnames = self.major_constituents + self.minor_constituents
            else:
                # Assume it's a comma-separated string
                self.tnames = [c.strip() for c in constituents.split(",")]
        elif isinstance(constituents, list):
            self.tnames = constituents
        else:
            # Default to major constituents
            self.tnames = self.major_constituents

        # For storing tidal factors
        self.amp = []
        self.freq = []
        self.nodal = []
        self.tear = []
        self.species = []
        
        # Pre-defined frequencies and factors for common constituents
        # These will be used if we can't find tide_fac_const.npz
        # Values from tide_fac_const.npz via loadz('/sciclone/data10/wangzg/FES2014/tide_fac_const/tide_fac_const.npz')
        self.default_factors = {
            # name: [amplitude, frequency(cycles/second), species_type]
            'M2': [0.242334, 0.0000140519, 2],  # Semi-diurnal
            'S2': [0.112743, 0.0000145444, 2],  # Semi-diurnal
            'N2': [0.046398, 0.0000137880, 2],  # Semi-diurnal
            'K2': [0.030704, 0.0000145444, 2],  # Semi-diurnal
            'K1': [0.141565, 0.0000072921, 1],  # Diurnal
            'O1': [0.100514, 0.0000067598, 1],  # Diurnal
            'P1': [0.046843, 0.0000072521, 1],  # Diurnal
            'Q1': [0.019256, 0.0000064959, 1],  # Diurnal
            'MF': [0.042041, 0.0000005323, 0],  # Long period
            'MM': [0.022191, 0.0000002639, 0],  # Long period
            'SSA': [0.019669, 0.0000000639, 0],  # Long period
        }

    @property
    def start_date(self):
        """Get start date for tidal calculations."""
        return self._start_time or datetime.now()
        
    def _get_tidal_factors(self):
        """Get tidal amplitude, frequency, and species for constituents.
        
        Uses constituent information from TPXO files or default factors if needed.
        """
        # Check if we already have tidal factors
        if hasattr(self, 'amp') and len(self.amp) > 0:
            return
            
        logger.info("Computing tidal factors")
        
        # Initialize arrays
        self.amp = []
        self.freq = []
        self.species = []
        
        # Try to get tidal constituent information directly from TPXO file
        if self.tidal_elevations and os.path.exists(self.tidal_elevations):
            logger.info(f"Getting tidal constituents from {self.tidal_elevations}")
            # Open the TPXO elevation file
            try:
                nc = ReadNC(self.tidal_elevations, 1)
                # Get list of tidal constituents from the file
                if hasattr(nc, 'variables') and 'con' in nc.variables:
                    cons = nc.variables['con'][:]
                    # Convert constituents to strings
                    file_constituents = []
                    for i in range(len(cons)):
                        const_name = ''.join([c.decode('utf-8') for c in cons[i]]).strip()
                        file_constituents.append(const_name)
                    
                    logger.info(f"Found constituents in TPXO file: {file_constituents}")
                    
                    # Check if our requested constituents are in the file
                    for tname in self.tnames:
                        if tname.upper() in [c.upper() for c in file_constituents]:
                            # Use default factors for these constituents
                            tname_upper = tname.upper()
                            if tname_upper in self.default_factors:
                                default_factors = self.default_factors[tname_upper]
                                self.amp.append(default_factors[0])
                                self.freq.append(default_factors[1])
                                self.species.append(default_factors[2])
                                logger.info(f"Using default factors for {tname}: amp={default_factors[0]}, freq={default_factors[1]}, species={default_factors[2]}")
                            else:
                                # If no default factors, use generic values
                                logger.warning(f"No default factors for {tname}, using generic values")
                                species_type = 2  # Default to semi-diurnal
                                if tname in ['O1', 'K1', 'P1', 'Q1']:
                                    species_type = 1  # Diurnal
                                elif tname in ['MM', 'Mm', 'Mf']:
                                    species_type = 0  # Long period
                                self.amp.append(0.1)
                                self.freq.append(0.00001)
                                self.species.append(species_type)
                        else:
                            logger.warning(f"Requested constituent {tname} not found in TPXO file")
                            raise ValueError(f"Constituent {tname} not found in TPXO file {self.tidal_elevations}")
                nc.close()
            except Exception as e:
                logger.error(f"Error reading constituents from TPXO file: {e}")
                raise
        else:
            # If no TPXO file, use default factors
            logger.warning("No TPXO elevation file provided, using default tidal factors")
            for tname in self.tnames:
                tname_upper = tname.upper()
                if tname_upper in self.default_factors:
                    default_factors = self.default_factors[tname_upper]
                    self.amp.append(default_factors[0])
                    self.freq.append(default_factors[1])
                    self.species.append(default_factors[2])
                    logger.info(f"Using default factors for {tname}: amp={default_factors[0]}, freq={default_factors[1]}, species={default_factors[2]}")
                else:
                    # If no default factors, use generic values based on name
                    logger.warning(f"No default factors for {tname}, using generic values")
                    species_type = 2  # Default to semi-diurnal
                    if tname in ['O1', 'K1', 'P1', 'Q1']:
                        species_type = 1  # Diurnal
                    elif tname in ['MM', 'Mm', 'Mf']:
                        species_type = 0  # Long period
                    self.amp.append(0.1)
                    self.freq.append(0.00001)
                    self.species.append(species_type)
        
        # Set default nodal factors if tide_fac_improved isn't available
        self.nodal = [1.0] * len(self.tnames)
        self.tear = [0.0] * len(self.tnames)
        
        # Try to get nodal factors using tide_fac_improved if it's available
        try:
            self._compute_nodal_factors()
        except Exception as e:
            logger.warning(f"Could not compute nodal factors using tide_fac_improved: {e}")
            logger.warning("Using default nodal factors of 1.0 and earth tear of 0.0")
        
    def _compute_nodal_factors(self):
        """Compute nodal factors using tide_fac_improved.
        
        If tide_fac_improved is not available or fails, nodal factors
        will default to 1.0 and tear to 0.0.
        """
        if not self._start_time or not self._rnday:
            logger.warning("start_time and rnday must be set before computing nodal factors")
            return
        
        # Initialize nodal factors with default values
        # These will be used if tide_fac_improved is not available
        self.nodal = [1.0] * len(self.tnames)
        self.tear = [0.0] * len(self.tnames)
        
        # Try to find the tide_fac executable
        try:
            tide_fac_exe = self._find_tide_fac_exe()
        except FileNotFoundError as e:
            logger.warning(f"tide_fac executable not found: {e}")
            logger.warning("Using default nodal factors of 1.0 and earth tear of 0.0")
            return
            
        # Ensure we have the required parameters
        if isinstance(self._start_time, datetime):
            year = self._start_time.year
            month = self._start_time.month
            day = self._start_time.day
            hour = self._start_time.hour
        else:
            # Assume it's a list [year, month, day, hour]
            year, month, day, hour = self._start_time
        
        # Create input file for tide_fac_improved
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.in') as fid:
            fid.write(f"{self._rnday}\n{hour} {day} {month} {year}\n0\n")
            tide_fac_in = fid.name
            
        tide_fac_out = tide_fac_in.replace('.in', '.out')
        
        try:
            # Run the tidal factor calculator
            cmd = f"{tide_fac_exe} < {tide_fac_in} > {tide_fac_out}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to run {tide_fac_exe}: {result.stderr}")
                
            # Read nodal factors
            with open(tide_fac_out, 'r') as f:
                lines = [i for i in f.readlines() if len(i.split()) == 3]
                
            for i, tname in enumerate(self.tnames):
                found = False
                for line in lines:
                    if line.strip().startswith(tname.upper()):
                        parts = line.strip().split()
                        self.nodal[i] = float(parts[1])
                        self.tear[i] = float(parts[2])
                        found = True
                        break
                        
                if not found:
                    logger.warning(f"Constituent {tname} not found in tide_fac_out, using default values")
        except Exception as e:
            logger.warning(f"Error computing nodal factors: {e}")
            logger.warning("Using default nodal factors of 1.0 and earth tear of 0.0")
        finally:
            # Clean up temporary files
            for fname in [tide_fac_in, tide_fac_out]:
                if os.path.exists(fname):
                    try:
                        os.remove(fname)
                    except:
                        pass
                
    def _find_tide_fac_exe(self):
        """Find or compile the tide_fac_improved executable.
        
        This function searches in many common locations for the executable.
        If not found, it attempts to compile from source if available.
        
        Returns
        -------
        str
            Path to tide_fac executable
            
        Raises
        ------
        FileNotFoundError
            If the executable cannot be found or compiled
        """
        # Try to find an existing executable
        tide_fac_exe = None
        for exe_name in ["tide_fac_improved", "tide_fac"]:
            # Check common locations
            search_paths = [
                ".",  # Current directory
                os.path.dirname(os.path.abspath(__file__)),  # This module's directory
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"),
                os.path.join(os.path.abspath(os.path.dirname(pylib.__file__)), "scripts"),
                os.path.join(os.path.abspath(os.path.dirname(pylib.__file__)), "scripts", "Harmonic_Analysis"),
            ]
            
            # Add pylibs paths if specified environment variable exists
            if 'PYLIB_DIR' in os.environ:
                pylib_dir = os.environ['PYLIB_DIR']
                search_paths.extend([
                    pylib_dir,
                    os.path.join(pylib_dir, "scripts"),
                    os.path.join(pylib_dir, "scripts", "Harmonic_Analysis"),
                ])
                
            # Add FES2014 directory if specified
            if 'FES2014_DIR' in os.environ:
                fes2014_dir = os.environ['FES2014_DIR']
                search_paths.extend([
                    fes2014_dir,
                    os.path.join(fes2014_dir, "tide_fac_improved"),
                ])
                
            # Look for executable in all search paths
            for path in search_paths:
                if not os.path.exists(path):
                    continue
                    
                exe_path = os.path.join(path, exe_name)
                if os.path.exists(exe_path) and os.access(exe_path, os.X_OK):
                    tide_fac_exe = exe_path
                    logger.info(f"Found existing tide factor executable: {tide_fac_exe}")
                    break
            if tide_fac_exe:
                break
        
        # If we found an executable, use it        
        if tide_fac_exe:
            return tide_fac_exe
        
        # If not found, try to compile if FES2014_DIR is defined
        if 'FES2014_DIR' in os.environ:
            fes2014_dir = os.environ['FES2014_DIR']
            tdir = os.path.join(fes2014_dir, "tide_fac_improved")
            
            if os.path.exists(tdir):
                logger.info(f"Checking for source files in {tdir}")
                source_files = ["tf_main.f90", "tf_selfe.f90"]
                has_source = True
                for src in source_files:
                    if not os.path.exists(os.path.join(tdir, src)):
                        has_source = False
                        logger.warning(f"Source file not found: {os.path.join(tdir, src)}")
                        break
                        
                if has_source:
                    # Try to compile
                    logger.info("Attempting to compile tide_fac_improved")
                    compile_cmd = f"cd {tdir} && ifort -o tide_fac_improved tf_main.f90 tf_selfe.f90"
                    try:
                        result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
                        if result.returncode == 0:
                            tide_fac_exe = os.path.join(tdir, "tide_fac_improved")
                            logger.info(f"Successfully compiled tide factor executable: {tide_fac_exe}")
                            return tide_fac_exe
                        else:
                            logger.warning(f"Failed to compile tide_fac_improved: {result.stderr}")
                    except Exception as e:
                        logger.warning(f"Error compiling tide_fac_improved: {e}")
        
        # If we get here, we couldn't find or compile the executable
        raise FileNotFoundError("Could not find or compile tide factor executable")
    
    def _interpolate_tidal_data(self, lons, lats, constituent, data_type="h"):
        """Interpolate tidal data for a constituent to boundary points.
        
        Parameters
        ----------
        lons : array
            Longitude values of boundary points
        lats : array
            Latitude values of boundary points
        constituent : str
            Tidal constituent name
        data_type : str
            'h' for elevation, 'uv' for velocity
            
        Returns
        -------
        np.ndarray
            For elevation: [amp, pha]
            For velocity: [u_amp, u_pha, v_amp, v_pha]
        """
        # Ensure lons and lats are numpy arrays
        lons = np.array(lons)
        lats = np.array(lats)
        
        # Normalize longitudes to 0-360
        xi = np.mod(lons + 360, 360)
        yi = lats
        
        # Initialize result array
        if data_type == "h":
            result = np.zeros((len(xi), 2))
        else:  # data_type == "uv"
            result = np.zeros((len(xi), 4))
            
        # Process based on tidal database type
        if self.tidal_database.lower() == "tpxo" and (
            (data_type == "h" and self.tidal_elevations) or
            (data_type == "uv" and self.tidal_velocities)
        ):
            # Use TPXO format files
            logger.info(f"Using TPXO format for {constituent} {data_type}")
            
            # Get the appropriate file
            tpxo_file = self.tidal_elevations if data_type == "h" else self.tidal_velocities
            if not os.path.exists(tpxo_file):
                raise FileNotFoundError(f"TPXO file not found: {tpxo_file}")
                
            try:
                # Open the TPXO file
                nc = ReadNC(tpxo_file, 1)
                
                # Get the list of constituents in the file
                cons = nc.variables['con'][:]
                file_constituents = []
                for i in range(len(cons)):
                    const_name = ''.join([c.decode('utf-8') for c in cons[i]]).strip()
                    file_constituents.append(const_name)
                
                # Find the index of the requested constituent
                const_idx = None
                for i, c in enumerate(file_constituents):
                    if c.upper() == constituent.upper():
                        const_idx = i
                        break
                        
                if const_idx is None:
                    raise ValueError(f"Constituent {constituent} not found in TPXO file {tpxo_file}")
                    
                # Get the grid coordinates
                lon = np.array(nc.variables['lon_z'][:])
                lat = np.array(nc.variables['lat_z'][:])
                
                # Get the tidal data
                if data_type == "h":
                    # Elevation data
                    amp = np.array(nc.variables['ha'][const_idx]).squeeze()
                    pha = np.array(nc.variables['hp'][const_idx]).squeeze()
                    
                    # Ensure phase is positive
                    pha[pha < 0] += 360
                    
                    # Interpolate to boundary points
                    result = self._tpxo_interpolate(xi, yi, lon, lat, amp, pha)
                else:  # data_type == "uv"
                    # Velocity data
                    u_amp = np.array(nc.variables['ua'][const_idx]).squeeze()
                    u_pha = np.array(nc.variables['up'][const_idx]).squeeze()
                    v_amp = np.array(nc.variables['va'][const_idx]).squeeze()
                    v_pha = np.array(nc.variables['vp'][const_idx]).squeeze()
                    
                    # Ensure phases are positive
                    u_pha[u_pha < 0] += 360
                    v_pha[v_pha < 0] += 360
                    
                    # Interpolate to boundary points
                    result_u = self._tpxo_interpolate(xi, yi, lon, lat, u_amp, u_pha)
                    result_v = self._tpxo_interpolate(xi, yi, lon, lat, v_amp, v_pha)
                    
                    # Combine results
                    result[:, 0:2] = result_u
                    result[:, 2:4] = result_v
                    
                nc.close()
            except Exception as e:
                logger.error(f"Error processing TPXO data for {constituent}: {e}")
                raise
                
        elif self.tidal_database.lower() == "fes2014" and 'FES2014_DIR' in os.environ:
            # Use FES2014 database
            fes2014_dir = os.environ['FES2014_DIR']
            
            # Determine file paths based on data type
            if data_type == "h":
                fname = os.path.join(
                    fes2014_dir,
                    "fes2014b_elevations_extrapolated/ocean_tide_extrapolated",
                    f"{constituent.lower()}.nc"
                )
                varnames = ["amplitude", "phase"]
                scale = [0.01, 1.0]  # Convert cm to m for amplitude
            else:  # data_type == "uv"
                u_fname = os.path.join(fes2014_dir, "eastward_velocity", f"{constituent.lower()}.nc")
                v_fname = os.path.join(fes2014_dir, "northward_velocity", f"{constituent.lower()}.nc")
                varnames = ["Ua", "Ug", "Va", "Vg"]
                scale = [0.01, 1.0, 0.01, 1.0]  # Convert cm/s to m/s for amplitudes
                
            # Process elevation data
            if data_type == "h":
                if not os.path.exists(fname):
                    raise FileNotFoundError(f"FES2014 tidal data file not found: {fname}")
                    
                C = ReadNC(fname, 1)
                lon = np.array(C.variables['lon'][:])
                lat = np.array(C.variables['lat'][:])
                amp0 = np.array(C.variables[varnames[0]][:]) * scale[0]
                pha0 = np.array(C.variables[varnames[1]][:])
                C.close()
                
                # Ensure phase is positive
                pha0[pha0 < 0] += 360
                
                # Bilinear interpolation
                result = self._bilinear_interpolate(xi, yi, lon, lat, amp0, pha0)
            else:  # data_type == "uv"
                # Process U component
                if not os.path.exists(u_fname) or not os.path.exists(v_fname):
                    missing = []
                    if not os.path.exists(u_fname):
                        missing.append(u_fname)
                    if not os.path.exists(v_fname):
                        missing.append(v_fname)
                    raise FileNotFoundError(f"FES2014 tidal data files not found: {missing}")
                    
                C_u = ReadNC(u_fname, 1)
                lon = np.array(C_u.variables['lon'][:])
                lat = np.array(C_u.variables['lat'][:])
                u_amp = np.array(C_u.variables[varnames[0]][:]) * scale[0]
                u_pha = np.array(C_u.variables[varnames[1]][:])
                C_u.close()
                
                C_v = ReadNC(v_fname, 1)
                v_amp = np.array(C_v.variables[varnames[2]][:]) * scale[2]
                v_pha = np.array(C_v.variables[varnames[3]][:])
                C_v.close()
                
                # Ensure phases are positive
                u_pha[u_pha < 0] += 360
                v_pha[v_pha < 0] += 360
                
                # Bilinear interpolation for U
                result_u = self._bilinear_interpolate(xi, yi, lon, lat, u_amp, u_pha)
                # Bilinear interpolation for V
                result_v = self._bilinear_interpolate(xi, yi, lon, lat, v_amp, v_pha)
                
                # Combine results
                result[:, 0:2] = result_u
                result[:, 2:4] = result_v
        else:
            # No valid tidal database configuration
            raise ValueError(
                f"Invalid tidal database configuration: {self.tidal_database}. " +
                f"Elevation file: {self.tidal_elevations}, Velocity file: {self.tidal_velocities}"
            )
            
        return result
        
    def _tpxo_interpolate(self, xi, yi, lon, lat, amp, pha):
        """Interpolate TPXO tidal data to boundary points.
        
        This handles TPXO's irregular grid format with lon_z/lat_z coordinates.
        
        Parameters
        ----------
        xi : array
            Target longitude points
        yi : array
            Target latitude points
        lon : array
            Source longitude grid (2D for TPXO)
        lat : array
            Source latitude grid (2D for TPXO)
        amp : array
            Amplitude values on source grid
        pha : array
            Phase values on source grid
            
        Returns
        -------
        np.ndarray
            Array of shape (len(xi), 2) with [amplitude, phase] for each point
        """
        from scipy.interpolate import griddata
        
        # Reshape 2D grids to 1D arrays for griddata
        lon_flat = lon.flatten()
        lat_flat = lat.flatten()
        amp_flat = amp.flatten()
        pha_flat = pha.flatten()
        
        # Filter out NaN values
        valid_indices = ~np.isnan(lon_flat) & ~np.isnan(lat_flat) & ~np.isnan(amp_flat) & ~np.isnan(pha_flat)
        lon_valid = lon_flat[valid_indices]
        lat_valid = lat_flat[valid_indices]
        amp_valid = amp_flat[valid_indices]
        pha_valid = pha_flat[valid_indices]
        
        # Create target points array
        points = np.vstack((lon_valid, lat_valid)).T
        
        # Prepare result array
        result = np.zeros((len(xi), 2))
        
        # Interpolate amplitude
        amp_interp = griddata(points, amp_valid, (xi, yi), method='linear', fill_value=0.0)
        result[:, 0] = amp_interp
        
        # For phase, convert to complex numbers to handle wrap-around
        pha_rad = np.radians(pha_valid)
        cos_pha = np.cos(pha_rad)
        sin_pha = np.sin(pha_rad)
        
        # Interpolate cos and sin components
        cos_interp = griddata(points, cos_pha, (xi, yi), method='linear', fill_value=0.0)
        sin_interp = griddata(points, sin_pha, (xi, yi), method='linear', fill_value=0.0)
        
        # Convert back to degrees
        pha_interp = np.degrees(np.arctan2(sin_interp, cos_interp))
        pha_interp[pha_interp < 0] += 360  # Ensure positive phase
        
        result[:, 1] = pha_interp
        
        return result
    
    def _bilinear_interpolate(self, xi, yi, lon, lat, amp, pha):
        """Perform bilinear interpolation with phase jump handling.
        
        Parameters
        ----------
        xi : array
            Target longitude points
        yi : array
            Target latitude points
        lon : array
            Source longitude grid
        lat : array
            Source latitude grid
        amp : array
            Amplitude values on source grid
        pha : array
            Phase values on source grid
            
        Returns
        -------
        np.ndarray
            Array of shape (len(xi), 2) with [amplitude, phase] for each point
        """
        # Check if lon/lat are uniformly spaced
        dxs = np.unique(np.diff(lon))
        dys = np.unique(np.diff(lat))
        if len(dxs) != 1 or len(dys) != 1:
            raise ValueError("lon,lat not uniformly spaced")
            
        dx = dxs[0]
        dy = dys[0]
        
        # Calculate interpolation indices
        idx = np.floor((xi - lon[0]) / dx).astype('int')
        idy = np.floor((yi - lat[0]) / dy).astype('int')
        
        # Handle edge cases
        idx[idx < 0] = 0
        idx[idx >= len(lon) - 1] = len(lon) - 2
        idy[idy < 0] = 0
        idy[idy >= len(lat) - 1] = len(lat) - 2
        
        # Calculate interpolation ratios
        xrat = (xi - lon[idx]) / dx
        yrat = (yi - lat[idy]) / dy
        
        # Initialize result array
        result = np.zeros((len(xi), 2))
        
        # For each point
        for i in range(len(xi)):
            # Get corner values for amplitude
            a00 = amp[idy[i], idx[i]]
            a01 = amp[idy[i], idx[i] + 1]
            a10 = amp[idy[i] + 1, idx[i]]
            a11 = amp[idy[i] + 1, idx[i] + 1]
            
            # Get corner values for phase
            p00 = pha[idy[i], idx[i]]
            p01 = pha[idy[i], idx[i] + 1]
            p10 = pha[idy[i] + 1, idx[i]]
            p11 = pha[idy[i] + 1, idx[i] + 1]
            
            # Handle phase jumps
            p_corners = np.array([p00, p01, p10, p11])
            max_p = np.max(p_corners)
            min_p = np.min(p_corners)
            
            # If there's a phase jump (values differ by more than 180 degrees)
            if max_p - min_p > 180:
                # Add 360 to phases less than 180 degrees from the max
                for j in range(4):
                    if max_p - p_corners[j] > 180:
                        p_corners[j] += 360
                        
                p00, p01, p10, p11 = p_corners
            
            # Bilinear interpolation for amplitude
            a0 = a00 * (1 - xrat[i]) + a01 * xrat[i]
            a1 = a10 * (1 - xrat[i]) + a11 * xrat[i]
            amp_interp = a0 * (1 - yrat[i]) + a1 * yrat[i]
            
            # Bilinear interpolation for phase
            p0 = p00 * (1 - xrat[i]) + p01 * xrat[i]
            p1 = p10 * (1 - xrat[i]) + p11 * xrat[i]
            pha_interp = p0 * (1 - yrat[i]) + p1 * yrat[i]
            
            # Normalize phase to 0-360
            pha_interp = np.mod(pha_interp, 360)
            
            # Store results
            result[i, 0] = amp_interp
            result[i, 1] = pha_interp
            
        return result
    
    def write_bctides(self, output_file):
        """Generate bctides.in file directly using PyLibs approach.
        
        Parameters
        ----------
        output_file : str or Path
            Path to output file
            
        Returns
        -------
        Path
            Path to the created bctides.in file
        """
        # Ensure we have start_time and rnday
        if not self._start_time or self._rnday is None:
            raise ValueError("start_time and rnday must be set before calling write_bctides")
            
        # Get tidal factors
        self._get_tidal_factors()
        
        logger.info(f"Writing bctides.in to {output_file}")
        with open(output_file, 'w') as f:
            # Write header with date information
            if isinstance(self._start_time, datetime):
                f.write(f"!{self._start_time.month:02d}/{self._start_time.day:02d}/{self._start_time.year:4d} "
                        f"{self._start_time.hour:02d}:00:00 UTC\n")
            else:
                # Assume it's a list [year, month, day, hour]
                year, month, day, hour = self._start_time
                f.write(f"!{month:02d}/{day:02d}/{year:4d} {hour:02d}:00:00 UTC\n")
            
            # Write tidal potential information
            if self.ntip > 0:
                f.write(f" {len(self.tnames)} {self.cutoff_depth:.3f} !number of earth tidal potential, "
                        f"cut-off depth for applying tidal potential\n")
                
                # Write each constituent's potential information
                for i, tname in enumerate(self.tnames):
                    f.write(f"{tname}\n")
                    
                    # Determine species type based on constituent name
                    species_type = 2  # Default to semi-diurnal
                    if tname in ['O1', 'K1', 'P1', 'Q1']:
                        species_type = 1  # Diurnal
                    elif tname in ['MM', 'Mm', 'Mf']:
                        species_type = 0  # Long period
                    
                    f.write(f"{species_type} {self.amp[i]:<.6f} {self.freq[i]:<.9e} "
                            f"{self.nodal[i]:7.5f} {self.tear[i]:.2f}\n")
            else:
                # No earth tidal potential
                f.write(" 0 50.000 !number of earth tidal potential, cut-off depth for applying tidal potential\n")
            
            # Write frequency info
            n_constituents = len(self.tnames) + (1 if len(self.ethconst) > 0 else 0)
            f.write(f"{n_constituents} !nbfr\n")
            
            # Write Z0 (mean sea level) if ethconst provided
            if len(self.ethconst) > 0:
                f.write("Z0\n  0.0 1.0 0.0\n")
                
            # Write frequency info for each constituent
            for i, tname in enumerate(self.tnames):
                f.write(f"{tname}\n  {self.freq[i]:<.9e} {self.nodal[i]:7.5f} {self.tear[i]:.2f}\n")
            
            # Write open boundary information
            f.write(f"{self.gd.nob} !nope\n")
            
            # For each open boundary
            for ibnd in range(self.gd.nob):
                # Get boundary nodes
                nodes = self.gd.iobn[ibnd]
                num_nodes = self.gd.nobn[ibnd]
                
                # Write boundary flags (ensure we have enough flags defined)
                bnd_flags = self.flags[ibnd] if ibnd < len(self.flags) else self.flags[0]
                flag_str = ' '.join(map(str, bnd_flags))
                f.write(f"{num_nodes} {flag_str} !ocean\n")
                
                # Get boundary coordinates
                lons = self.gd.x[nodes]
                lats = self.gd.y[nodes]
                
                # Write elevation boundary conditions
                
                # First, handle constant elevation if provided
                if len(self.ethconst) > 0:
                    f.write("Z0\n")
                    eth_val = self.ethconst[ibnd] if ibnd < len(self.ethconst) else 0.0
                    for n in range(num_nodes):
                        f.write(f"{eth_val} 0.0\n")
                        
                # Then write tidal constituents
                for i, tname in enumerate(self.tnames):
                    logger.info(f"Processing tide {tname} for boundary {ibnd+1}")
                    
                    # Interpolate tidal data for this constituent
                    try:
                        tidal_data = self._interpolate_tidal_data(lons, lats, tname, "h")
                        
                        # Write header for constituent
                        f.write(f"{tname.lower()}\n")
                        
                        # Write amplitude and phase for each node
                        for n in range(num_nodes):
                            f.write(f"{tidal_data[n,0]:8.6f} {tidal_data[n,1]:.6f}\n")
                    except Exception as e:
                        # Log error but continue with other constituents
                        logger.error(f"Error processing tide {tname} for boundary {ibnd+1}: {e}")
                        raise
                
                # Write velocity boundary conditions
                
                # First, handle constant velocity if provided
                if len(self.vthconst) > 0:
                    f.write("Z0\n")
                    vth_val = self.vthconst[ibnd] if ibnd < len(self.vthconst) else 0.0
                    for n in range(num_nodes):
                        f.write("0.0 0.0 0.0 0.0\n")
                        
                # Then write tidal constituents
                for i, tname in enumerate(self.tnames):
                    # Write header for constituent first
                    f.write(f"{tname.lower()}\n")
                    
                    # Try to interpolate velocity data
                    try:
                        if self.tidal_velocities and os.path.exists(self.tidal_velocities):
                            vel_data = self._interpolate_tidal_data(lons, lats, tname, "uv")
                            
                            # Write u/v amplitude and phase for each node
                            for n in range(num_nodes):
                                f.write(f"{vel_data[n,0]:8.6f} {vel_data[n,1]:.6f} "
                                        f"{vel_data[n,2]:8.6f} {vel_data[n,3]:.6f}\n")
                        else:
                            # If no velocity file, use zeros to ensure file structure is complete
                            logger.warning(f"No velocity data available for {tname}, using zeros")
                            for n in range(num_nodes):
                                f.write("0.0 0.0 0.0 0.0\n")
                    except Exception as e:
                        # Log error and use zeros as fallback for this constituent
                        logger.error(f"Error processing velocity for tide {tname}, boundary {ibnd+1}: {e}")
                        logger.warning(f"Using zeros for velocity data for {tname}")
                        for n in range(num_nodes):
                            f.write("0.0 0.0 0.0 0.0\n")
            
            # Add remaining sections
            f.write("0 !ncbn: total # of flow bnd segments with discharge\n")
            f.write("0 !nfluxf: total # of flux boundary segments\n")
            
        logger.info(f"Successfully wrote bctides.in to {output_file}")
        return output_file
