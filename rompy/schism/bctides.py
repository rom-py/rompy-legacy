"""
Tidal boundary conditions for SCHISM.

A direct implementation based on PyLibs scripts/gen_bctides.py with no fallbacks.
"""

import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pyTMD
import timescale
import pandas as pd
from pylib import ReadNC
import xarray as xr
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


class Bctides:
    """Direct implementation of SCHISM tidal boundary conditions using PyLibs.

    Based on scripts/gen_bctides.py
    """

    def __init__(
        self,
        hgrid,
        flags=None,
        constituents="major",
        tidal_database=None,
        tidal_model="FES2014",
        tidal_potential=True,
        cutoff_depth=50.0,
        nodal_corrections=True,
        tide_interpolation_method="bilinear",
        extrapolate_tides=False,
        extrapolation_distance=100.0,
        extra_databases=[],
        mdt=None,
        ethconst=None,
        vthconst=None,
        tthconst=None,
        sthconst=None,
        tobc=None,
        sobc=None,
        relax=None,  # For backward compatibility
        inflow_relax=None,
        outflow_relax=None,
        ncbn=0,
        nfluxf=0,
        elev_th_path=None,
        elev_st_path=None,
        flow_th_path=None,
        vel_st_path=None,
        temp_th_path=None,
        temp_3d_path=None,
        salt_th_path=None,
        salt_3d_path=None,
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
        tidal_database : path, optional
            Path to pyTMD tidal database to use, by default None which uses the default
        tidal_model : str, optional
            Tidal model name (e.g., 'FES2014'), by default 'FES2014'
        tidal_potential : bool, optional
            Whether to apply tidal potential, by default True
        cutoff_depth : float, optional
            Cutoff depth for tidal potential, by default 50.0
        nodal_corrections : bool, optional
            Whether to apply nodal corrections, by default True
        tide_interpolation_method : str, optional
            Method for tidal interpolation, by default 'bilinear'
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
        tidal_elevations : str or Path, optional
            Path to tidal elevations file
        tidal_velocities : str or Path, optional
            Path to tidal velocities file
        ncbn : int, optional
            Number of flow boundary segments, by default 0
        nfluxf : int, optional
            Number of flux boundary segments, by default 0
        """
        # Set default values for any None parameters
        flags = flags or [[5, 5, 4, 4]]
        ethconst = ethconst or []
        vthconst = vthconst or []
        tthconst = tthconst or []
        sthconst = sthconst or []
        tobc = tobc or [1]
        sobc = sobc or [1]
        relax = relax or []  # Keep for backward compatibility
        inflow_relax = inflow_relax or [0.5]
        outflow_relax = outflow_relax or [0.1]

        # Assign to instance variables
        self.flags = flags

        # Store tidal file paths
        self.tidal_database = tidal_database
        self.tidal_model = tidal_model
        self.tidal_potential = tidal_potential
        self.cutoff_depth = cutoff_depth
        self.nodal_corrections = nodal_corrections
        self.tide_interpolation_method = tide_interpolation_method
        self.extrapolate_tides = extrapolate_tides
        self.extrapolation_distance = extrapolation_distance
        self.extra_databases = extra_databases
        self.mdt = mdt

        self.ethconst = ethconst
        self.vthconst = vthconst
        self.tthconst = tthconst
        self.sthconst = sthconst
        self.tobc = tobc
        self.sobc = sobc
        self.relax = relax
        self.inflow_relax = inflow_relax
        self.outflow_relax = outflow_relax
        self.ncbn = ncbn
        self.nfluxf = nfluxf

        # Store boundary condition file paths
        self.elev_th_path = elev_th_path  # Time history of elevation
        self.elev_st_path = elev_st_path  # Space-time elevation
        self.flow_th_path = flow_th_path  # Time history of flow
        self.vel_st_path = vel_st_path  # Space-time velocity
        self.temp_th_path = temp_th_path  # Temperature time history
        self.temp_3d_path = temp_3d_path  # 3D temperature
        self.salt_th_path = salt_th_path  # Salinity time history
        self.salt_3d_path = salt_3d_path  # 3D salinity

        # Store start time and run duration (will be set by SCHISMDataTides.get())
        self._start_time = None
        self._rnday = None

        # Load grid from file or object
        # Assume it's already a grid object
        self.gd = hgrid

        # Define constituent sets (using lowercase for pyTMD compatibility)
        self.major_constituents = ["o1", "k1", "q1", "p1", "m2", "s2", "k2", "n2"]
        self.minor_constituents = ["mm", "mf", "m4", "mn4", "ms4", "2n2", "s1"]

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
        # Ensure tnames are unique and lowercase (for pyTMD compatibility)
        self.tnames = list(set(t.lower() for t in self.tnames))

        # For storing tidal factors
        self.amp = []
        self.freq = []
        self.nodal = []
        self.tear = []
        self.species = []

    @property
    def start_date(self):
        """Get start date for tidal calculations."""
        return self._start_time or datetime.now()

    def _get_tidal_factors(self):
        """Get tidal amplitude, frequency, and species for constituents using pyTMD."""
        if hasattr(self, "amp") and len(self.amp) > 0:
            return
        logger.info("Computing tidal factors using pyTMD")
        # Use pyTMD for all calculations
        ts = timescale.time.Timescale().from_datetime(self._start_time)
        MJD = ts.MJD
        # Astronomical longitudes
        if self.tidal_model.startswith("FES"):
            # FES models use ASTRO5 method
            s, h, p, n, pp = pyTMD.astro.mean_longitudes(MJD, method="ASTRO5")
            u, f = pyTMD.arguments.nodal_modulation(
                n, p, self.tnames, corrections="FES"
            )
            freq = pyTMD.arguments.frequency(self.tnames, corrections="FES")
        else:
            # Other models use ASTRO2 method
            s, h, p, n, pp = pyTMD.astro.mean_longitudes(MJD, method="Cartwright")
            u, f = pyTMD.arguments.nodal_modulation(
                n, p, self.tnames, corrections="OTIS"
            )
            freq = pyTMD.arguments.frequency(self.tnames, corrections="OTIS")

        # Nodal corrections (u: phase, f: factor)
        u = u.squeeze()
        f = f.squeeze()
        u_deg = np.rad2deg(u)

        # Earth equilibrium argument
        hour = 24.0 * np.mod(MJD, 1)
        tau = 15.0 * hour - s + h
        k = 90.0 + np.zeros_like(MJD)
        fargs = np.c_[tau, s, h, p, n, pp, k]
        coef = pyTMD.arguments.coefficients_table(self.tnames)
        G = np.mod(np.dot(fargs, coef), 360.0)

        # Compose info
        self.amp = []
        self.freq = []
        self.nodal_factor = []
        self.nodal_phase_correction = []
        self.species = []
        for c, constituent in enumerate(self.tnames):
            params = pyTMD.arguments._constituent_parameters(constituent)
            self.amp.append(params[0])
            self.freq.append(freq[c])
            self.nodal_factor.append(f[c])
            self.nodal_phase_correction.append(u_deg[c])
            self.species.append(params[4])
        # Store earth equilibrium argument for each constituent
        self.earth_equil_arg = G[0, :]

    def _interpolate_tidal_data(self, lons, lats, constituent, data_type="h"):
        """
        Interpolate tidal data for a constituent to boundary points using pyTMD extract_constants.

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
            For elevation: [amp, pha] (shape: n_points, 2)
            For velocity: [u_amp, u_pha, v_amp, v_pha] (shape: n_points, 4)
        """
        tmd_model = pyTMD.io.model(
            self.tidal_database, extra_databases=self.extra_databases
        )
        if data_type == "h":
            amp, pha, _ = tmd_model.elevation(self.tidal_model).extract_constants(
                lons,
                lats,
                constituents=[constituent],
                method="bilinear",
                crop=True,
                extrapolate=self.extrapolate_tides,
                cutoff=self.extrapolation_distance,
            )
            amp = amp.squeeze()
            pha = pha.squeeze()
            # Return shape (n_points, 2)
            return np.column_stack((amp, pha))
        elif data_type == "uv":
            amp_u, pha_u, _ = tmd_model.current(self.tidal_model).extract_constants(
                lons,
                lats,
                type="u",
                constituents=[constituent],
                method="bilinear",
                crop=True,
                extrapolate=self.extrapolate_tides,
                cutoff=self.extrapolation_distance,
            )
            amp_v, pha_v, _ = tmd_model.current(self.tidal_model).extract_constants(
                lons,
                lats,
                type="v",
                constituents=[constituent],
                method="bilinear",
                crop=True,
                extrapolate=self.extrapolate_tides,
                cutoff=self.extrapolation_distance,
            )
            amp_u = (
                amp_u.squeeze() / 100
            )  # Convert cm/s to m/s - pyTMD always returns in cm/s
            pha_u = pha_u.squeeze()
            amp_v = (
                amp_v.squeeze() / 100
            )  # Convert cm/s to m/s - pyTMD always returns in cm/s
            pha_v = pha_v.squeeze()
            # Return shape (n_points, 4)
            return np.column_stack((amp_u, pha_u, amp_v, pha_v))
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

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
            raise ValueError(
                "start_time and rnday must be set before calling write_bctides"
            )

        # Ensure boundary information is computed before accessing boundary attributes
        if hasattr(self.gd, "compute_bnd") and not hasattr(self.gd, "nob"):
            logger.info("Computing boundary information for grid")
            self.gd.compute_bnd()
        elif not hasattr(self.gd, "nob"):
            logger.warning("Grid has no boundary information and no compute_bnd method")

        # Get tidal factors
        self._get_tidal_factors()

        if self.nodal_corrections:
            logger.info(
                "Applying nodal phase corrections to earth equilibrium argument"
            )
            self.earth_equil_arg = np.mod(
                self.earth_equil_arg + self.nodal_phase_correction, 360.0
            )
        else:
            logger.info("Setting nodal corrections to 1.0 (no corrections applied)")
            self.nodal_factor = [1.0] * len(self.tnames)

        logger.info(f"Writing bctides.in to {output_file}")
        with open(output_file, "w") as f:
            # Write header with date information
            if isinstance(self._start_time, datetime):
                f.write(
                    f"!{self._start_time.month:02d}/{self._start_time.day:02d}/{self._start_time.year:4d} "
                    f"{self._start_time.hour:02d}:00:00 UTC\n"
                )
            else:
                # Assume it's a list [year, month, day, hour]
                year, month, day, hour = self._start_time
                f.write(f"!{month:02d}/{day:02d}/{year:4d} {hour:02d}:00:00 UTC\n")

            # Write tidal potential information
            # Use only constituents with species 0, 1, or 2 (long period, diurnal, semi-diurnal)
            tidal_potential_indices = [
                i for i, s in enumerate(self.species) if s in (0, 1, 2)
            ]
            n_tidal_potential = len(tidal_potential_indices)
            if self.tidal_potential and n_tidal_potential > 0:
                f.write(
                    f" {n_tidal_potential} {self.cutoff_depth:.3f} !number of earth tidal potential, "
                    f"cut-off depth for applying tidal potential\n"
                )
                # Write each constituent's potential information
                for i in tidal_potential_indices:
                    tname = self.tnames[i]
                    species_type = self.species[i]
                    f.write(f"{tname}\n")
                    f.write(
                        f"{species_type} {self.amp[i]:<.6f} {self.freq[i]:<.6e} "
                        f"{self.nodal_factor[i]:.6f} {self.earth_equil_arg[i]:.6f}\n"
                    )
            else:
                # No earth tidal potential
                f.write(
                    " 0 50.000 !number of earth tidal potential, cut-off depth for applying tidal potential\n"
                )

            n_constituents = len(self.tnames)
            if not self.mdt is None:
                # If mdt is provided, we have a constant elevation for all constituents
                n_constituents += 1
            f.write(f"{n_constituents} !nbfr\n")
            if not self.mdt is None:
                # Write mdt as a special constant elevation
                f.write("z0\n")
                f.write(f"0.0 1.0 0.0\n")

            # Write frequency info for each constituent
            for i, tname in enumerate(self.tnames):
                f.write(
                    f"{tname}\n  {self.freq[i]:<.9e} {self.nodal_factor[i]:7.5f} {self.earth_equil_arg[i]:.5f}\n"
                )

            # Write open boundary information
            # Use the number of boundaries from self.flags or fallback to grid boundaries
            if hasattr(self, "flags") and self.flags:
                nope = len(self.flags)
                logger.info(f"Using {nope} user-defined boundaries from flags")
            elif hasattr(self.gd, "nob") and self.gd.nob > 0:
                nope = self.gd.nob
                logger.info(f"Using {nope} boundaries from grid")
            else:
                # No boundaries in grid and no user-defined flags
                logger.warning(
                    "Grid has no open boundaries and no user-defined boundary flags"
                )
                nope = 0

            f.write(f"{nope} !nope\n")

            # For each open boundary
            for ibnd in range(nope):
                # Get boundary nodes - prioritize grid boundaries if available
                if hasattr(self.gd, "nob") and self.gd.nob > 0 and ibnd < self.gd.nob:
                    # Use actual grid boundary
                    nodes = self.gd.iobn[ibnd]
                    num_nodes = self.gd.nobn[ibnd]
                    logger.debug(f"Using grid boundary {ibnd} with {num_nodes} nodes")
                elif (
                    hasattr(self.gd, "nob") and self.gd.nob > 0 and ibnd >= self.gd.nob
                ):
                    # User has defined more boundaries than grid has - reuse last grid boundary
                    last_bnd_idx = self.gd.nob - 1
                    nodes = self.gd.iobn[last_bnd_idx]
                    num_nodes = self.gd.nobn[last_bnd_idx]
                    raise ValueError(
                        f"Boundary {ibnd} exceeds grid boundaries, reusing boundary {last_bnd_idx}"
                    )
                else:
                    # Grid has no boundaries, but user has defined flags
                    # This is an inconsistent state - create a minimal dummy boundary
                    raise ValueError(
                        f"Grid has no open boundaries but user defined boundary {ibnd}, creating dummy boundary"
                    )

                # Write boundary flags (ensure we have enough flags defined)
                bnd_flags = (
                    self.flags[ibnd]
                    if hasattr(self, "flags") and ibnd < len(self.flags)
                    else [0, 0, 0, 0]
                )
                flag_str = " ".join(map(str, bnd_flags))
                f.write(f"{num_nodes} {flag_str} !ocean\n")

                # Get boundary coordinates
                lons = self.gd.x[nodes]
                lats = self.gd.y[nodes]

                # Write elevation boundary conditions

                # Handle elevation boundary conditions based on flags
                elev_type = bnd_flags[0] if len(bnd_flags) > 0 else 0

                # Type 1: Time history of elevation
                if elev_type == 1:
                    f.write("! Time history of elevation will be read from elev.th\n")
                # Type 2: Constant elevation
                elif elev_type == 2 and len(self.ethconst) > 0:
                    if not self.mdt is None:
                        logger.warning(
                            "Using mdt value for constant elevation, ignoring ethconst"
                        )
                        pass
                    else:
                        f.write("Z0\n")
                        eth_val = (
                            self.ethconst[ibnd] if ibnd < len(self.ethconst) else 0.0
                        )
                        for n in range(num_nodes):
                            f.write(f"{eth_val} 0.0\n")
                # Type 4: Space-time varying elevation
                elif elev_type == 4:
                    f.write(
                        "! Space-time varying elevation will be read from elev2D.th.nc\n"
                    )

                # Then write tidal constituents for elevation
                # Only write tidal constituents for tidal elevation types (3 or 5)
                if bnd_flags[0] == 3 or bnd_flags[0] == 5:
                    # If mdt is provided, write the Z0
                    if self.mdt is not None:
                        f.write("z0\n")
                        if isinstance(self.mdt, float):
                            # If mdt is a single float, write it for all nodes
                            for n in range(num_nodes):
                                f.write(f"{self.mdt:.6f} 0.0\n")
                        elif isinstance(self.mdt, (xr.Dataset, xr.DataArray)):
                            # Use a KDTree to efficiently find the closest mdt point for each boundary node
                            mdt_lons = self.mdt.x.values
                            mdt_lats = self.mdt.y.values
                            mdt_values = self.mdt.values
                            # Filter any NaN values in mdt
                            valid_mask = ~np.isnan(mdt_values)
                            mdt_lons = mdt_lons[valid_mask]
                            mdt_lats = mdt_lats[valid_mask]
                            mdt_values = mdt_values[valid_mask]
                            # Create KDTree for mdt points
                            mdt_points = np.column_stack((mdt_lons, mdt_lats))
                            bnd_points = np.column_stack((lons, lats))
                            tree = KDTree(mdt_points)
                            distances, indices = tree.query(bnd_points)
                            tolerance = 0.1
                            if np.any(distances > tolerance):
                                n_pts = np.sum(distances > tolerance)
                                logger.warning(
                                    f"Found {n_pts} boundary points with mdt distance > {tolerance} degrees"
                                )
                            # Extract the mdt values for these points
                            mdt_values = mdt_values[indices]
                            for n in range(num_nodes):
                                mdt_val = float(mdt_values[n])
                                f.write(f"{mdt_val:.6f} 0.0\n")
                        else:
                            # If mdt is not a float or xr.Dataset, raise an error
                            logger.error(
                                f"Invalid mdt type: {type(self.mdt)}. Expected float or xr.Dataset."
                            )

                    for i, tname in enumerate(self.tnames):
                        logger.info(f"Processing tide {tname} for boundary {ibnd+1}")

                        # Interpolate tidal data for this constituent
                        try:
                            tidal_data = self._interpolate_tidal_data(
                                lons, lats, tname, "h"
                            )

                            if self.nodal_corrections:
                                # Apply nodal correction to phase - amplitude is applied within the code?
                                tidal_data[:, 1] = (
                                    tidal_data[:, 1] + self.nodal_phase_correction[i]
                                ) % 360.0
                            else:
                                # If no nodal corrections, just use the phase as is
                                tidal_data[:, 1] = tidal_data[:, 1] % 360.0

                            # Write header for constituent
                            f.write(f"{tname}\n")

                            # Write amplitude and phase for each node
                            for n in range(num_nodes):
                                f.write(
                                    f"{tidal_data[n,0]:8.6f} {tidal_data[n,1]:.6f}\n"
                                )
                        except Exception as e:
                            # Log error but continue with other constituents
                            logger.error(
                                f"Error processing tide {tname} for boundary {ibnd+1}: {e}"
                            )
                            raise

                # Write velocity boundary conditions

                # Handle velocity boundary conditions based on flags
                vel_type = bnd_flags[1] if len(bnd_flags) > 1 else 0

                # Type -1: Flather type radiation boundary
                if vel_type == -1:
                    # Write mean elevation marker
                    f.write("eta_mean\n")

                    # Write mean elevation for each node (use 0 as default)
                    for n in range(num_nodes):
                        f.write("0.0\n")  # Default mean elevation

                    # Write mean normal velocity marker
                    f.write("vn_mean\n")

                    # Write mean normal velocity for each node
                    for n in range(num_nodes):
                        f.write("0.0\n")  # Default mean normal velocity
                # Type 1: Time history of discharge
                elif vel_type == 1:
                    f.write("! Time history of discharge will be read from flux.th\n")
                # Type 2: Constant discharge
                elif vel_type == 2 and len(self.vthconst) > 0:
                    vth_val = self.vthconst[ibnd] if ibnd < len(self.vthconst) else 0.0
                    for n in range(num_nodes):
                        # Write as integer if it's a whole number, otherwise as float
                        if vth_val == int(vth_val):
                            f.write(f"{int(vth_val)}\n")
                        else:
                            f.write(f"{vth_val}\n")
                # Type -4: Relaxed velocity with 3D input
                elif vel_type == -4:
                    f.write("! 3D velocity will be read from uv3D.th.nc\n")
                    if len(self.inflow_relax) > 0 and len(self.outflow_relax) > 0:
                        inflow = (
                            self.inflow_relax[ibnd]
                            if ibnd < len(self.inflow_relax)
                            else 0.5
                        )
                        outflow = (
                            self.outflow_relax[ibnd]
                            if ibnd < len(self.outflow_relax)
                            else 0.1
                        )
                        f.write(
                            f"{inflow:.4f} {outflow:.4f} ! Relaxation constants for inflow and outflow\n"
                        )

                # Then write tidal constituents for velocity
                # Only write tidal constituents for tidal velocity types (3 or 5)
                if vel_type == 3 or vel_type == 5:
                    if self.mdt is not None:
                        f.write("z0\n")
                        for n in range(num_nodes):
                            f.write(f"0.0 0.0 0.0 0.0\n")
                    for i, tname in enumerate(self.tnames):
                        # Write header for constituent first
                        f.write(f"{tname}\n")

                        # # Try to interpolate velocity data
                        # if self.tidal_velocities and os.path.exists(
                        #     self.tidal_velocities
                        # ):
                        vel_data = self._interpolate_tidal_data(lons, lats, tname, "uv")

                        if self.nodal_corrections:
                            # Apply nodal correction to phase for u and v components
                            vel_data[:, 1] = (
                                vel_data[:, 1] + self.nodal_phase_correction[i]
                            ) % 360.0
                            vel_data[:, 3] = (
                                vel_data[:, 3] + self.nodal_phase_correction[i]
                            ) % 360.0
                        else:
                            # If no nodal corrections, just use the phases as is
                            vel_data[:, 1] = vel_data[:, 1] % 360.0
                            vel_data[:, 3] = vel_data[:, 3] % 360.0

                        # Write u/v amplitude and phase for each node
                        for n in range(num_nodes):
                            f.write(
                                f"{vel_data[n,0]:8.6f} {vel_data[n,1]:.6f} "
                                f"{vel_data[n,2]:8.6f} {vel_data[n,3]:.6f}\n"
                            )
                        # else:
                        #     # If no velocity file, use zeros to ensure file structure is complete
                        #     logger.warning(
                        #         f"No velocity data available for {tname}, using zeros"
                        #     )
                        #     for n in range(num_nodes):
                        #         f.write("0.0 0.0 0.0 0.0\n")

                # Write temperature boundary conditions if specified
                if len(bnd_flags) > 2 and bnd_flags[2] > 0:
                    temp_type = bnd_flags[2]

                    # Handle different temperature boundary types
                    if temp_type == 1:  # Time history
                        # Write nudging factor for inflow
                        temp_nudge = (
                            self.tobc[ibnd]
                            if self.tobc and ibnd < len(self.tobc)
                            else 1.0
                        )
                        f.write(f"{temp_nudge:.6f} !temperature nudging factor\n")
                        if self.temp_th_path:
                            f.write(
                                f"! Temperature time history will be read from {self.temp_th_path}\n"
                            )
                    elif temp_type == 2:  # Constant value
                        # Write constant temperature and nudging factor
                        const_temp = (
                            self.tthconst[ibnd]
                            if self.tthconst and ibnd < len(self.tthconst)
                            else 20.0
                        )
                        temp_nudge = (
                            self.tobc[ibnd]
                            if self.tobc and ibnd < len(self.tobc)
                            else 1.0
                        )
                        f.write(f"{const_temp:.6f} !constant temperature\n")
                        f.write(f"{temp_nudge:.6f} !temperature nudging factor\n")
                    elif temp_type == 3:  # Initial profile
                        # Write nudging factor only
                        temp_nudge = (
                            self.tobc[ibnd]
                            if self.tobc and ibnd < len(self.tobc)
                            else 1.0
                        )
                        f.write(f"{temp_nudge:.6f} !temperature nudging factor\n")
                    elif temp_type == 4:  # 3D input
                        # Write nudging factor only
                        temp_nudge = (
                            self.tobc[ibnd]
                            if self.tobc and ibnd < len(self.tobc)
                            else 1.0
                        )
                        f.write(f"{temp_nudge:.6f} !temperature nudging factor\n")
                        if self.temp_3d_path:
                            f.write(
                                f"! 3D temperature will be read from {self.temp_3d_path}\n"
                            )

                # Write salinity boundary conditions if specified
                if len(bnd_flags) > 3 and bnd_flags[3] > 0:
                    salt_type = bnd_flags[3]

                    # Handle different salinity boundary types
                    if salt_type == 1:  # Time history
                        # Write nudging factor for inflow
                        salt_nudge = (
                            self.sobc[ibnd]
                            if self.sobc and ibnd < len(self.sobc)
                            else 1.0
                        )
                        f.write(f"{salt_nudge:.6f} !salinity nudging factor\n")
                        if self.salt_th_path:
                            f.write(
                                f"! Salinity time history will be read from {self.salt_th_path}\n"
                            )
                    elif salt_type == 2:  # Constant value
                        # Write constant salinity and nudging factor
                        const_salt = (
                            self.sthconst[ibnd]
                            if self.sthconst and ibnd < len(self.sthconst)
                            else 35.0
                        )
                        salt_nudge = (
                            self.sobc[ibnd]
                            if self.sobc and ibnd < len(self.sobc)
                            else 1.0
                        )
                        f.write(f"{const_salt:.6f} !constant salinity\n")
                        f.write(f"{salt_nudge:.6f} !salinity nudging factor\n")
                    elif salt_type == 3:  # Initial profile
                        # Write nudging factor only
                        salt_nudge = (
                            self.sobc[ibnd]
                            if self.sobc and ibnd < len(self.sobc)
                            else 1.0
                        )
                        f.write(f"{salt_nudge:.6f} !salinity nudging factor\n")
                    elif salt_type == 4:  # 3D input
                        # Write nudging factor only
                        salt_nudge = (
                            self.sobc[ibnd]
                            if self.sobc and ibnd < len(self.sobc)
                            else 1.0
                        )
                        f.write(f"{salt_nudge:.6f} !salinity nudging factor\n")
                        if self.salt_3d_path:
                            f.write(
                                f"! 3D salinity will be read from {self.salt_3d_path}\n"
                            )

            # Add flow and flux boundary information
            # Use instance attributes if available, otherwise default to 0
            ncbn = getattr(self, "ncbn", 0)
            nfluxf = getattr(self, "nfluxf", 0)

            f.write(f"{ncbn} !ncbn: total # of flow bnd segments with discharge\n")

            # If ncbn > 0, we need to write flow boundary information
            # For now, we're just writing placeholder values as this would require additional data
            for i in range(ncbn):
                f.write(f"1 1 !flow boundary {i+1}: number of nodes, boundary flag\n")
                f.write("1 !node number on the boundary\n")
                f.write("1 !number of vertical layers\n")
                f.write("0.0 !flow rate for each layer\n")

            f.write(f"{nfluxf} !nfluxf: total # of flux boundary segments\n")

            # If nfluxf > 0, we need to write flux boundary information
            # For now, we're just writing placeholder values
            for i in range(nfluxf):
                f.write(f"1 !flux boundary {i+1}: number of nodes\n")
                f.write("1 !node number on the boundary\n")

        logger.info(f"Successfully wrote bctides.in to {output_file}")
        return output_file
