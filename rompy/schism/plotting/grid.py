"""
Grid plotting functionality for SCHISM models.

This module provides comprehensive grid visualization capabilities including
bathymetry, boundaries, and grid structure plotting.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .core import BasePlotter, PlotConfig
from .utils import (add_boundary_overlay, add_grid_overlay, add_scale_bar,
                    get_geographic_extent, prepare_triangulation_data,
                    setup_colormap)

logger = logging.getLogger(__name__)


class GridPlotter(BasePlotter):
    """
    SCHISM grid plotting functionality.

    This class provides methods for visualizing SCHISM computational grids,
    including bathymetry, boundaries, and grid structure.

    Parameters
    ----------
    config : Optional[Any]
        SCHISM configuration object
    plot_config : Optional[PlotConfig]
        Plotting configuration parameters
    """

    def __init__(
        self, config: Optional[Any] = None, plot_config: Optional[PlotConfig] = None
    ):
        """Initialize GridPlotter."""
        super().__init__(config, plot_config)

    def plot(self, plot_type: str = "bathymetry", **kwargs) -> Tuple[Figure, Axes]:
        """
        Main plotting method with different plot types.

        Parameters
        ----------
        plot_type : str, optional
            Type of plot: 'grid', 'bathymetry', 'boundaries', 'contours'.
            Default is 'bathymetry'.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        if plot_type == "grid":
            return self.plot_grid(**kwargs)
        elif plot_type == "bathymetry":
            return self.plot_bathymetry(**kwargs)
        elif plot_type == "boundaries":
            return self.plot_boundaries(**kwargs)
        elif plot_type == "contours":
            return self.plot_contours(**kwargs)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

    def plot_grid(
        self,
        ax: Optional[Axes] = None,
        show_triangulation: bool = True,
        show_nodes: bool = False,
        node_size: float = 1.0,
        line_alpha: float = 0.5,
        line_color: str = "gray",
        line_width: float = 0.5,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot SCHISM computational grid structure.

        Parameters
        ----------
        ax : Optional[Axes]
            Existing axes to plot on
        show_triangulation : bool, optional
            Whether to show triangular mesh. Default is True.
        show_nodes : bool, optional
            Whether to show node points. Default is False.
        node_size : float, optional
            Size of node markers. Default is 1.0.
        line_alpha : float, optional
            Transparency of grid lines. Default is 0.5.
        line_color : str, optional
            Color of grid lines. Default is "gray".
        line_width : float, optional
            Width of grid lines. Default is 0.5.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, **kwargs)

        grid = self.config.grid
        hgrid = grid.pylibs_hgrid

        # Plot triangulation if requested
        if show_triangulation:
            x, y, triangles = prepare_triangulation_data(hgrid)
            ax.triplot(
                x,
                y,
                triangles,
                alpha=line_alpha,
                color=line_color,
                linewidth=line_width,
            )

        # Plot nodes if requested
        if show_nodes:
            ax.scatter(hgrid.x, hgrid.y, s=node_size, c="red", alpha=0.6, zorder=5)

        # Set extent based on grid coordinates
        extent = get_geographic_extent(hgrid.x, hgrid.y)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Add boundaries
        if self.plot_config.show_boundaries:
            add_boundary_overlay(
                ax,
                grid,
                boundary_colors=self.plot_config.boundary_colors,
                linewidth=self.plot_config.boundary_linewidth,
            )

        return self.finalize_plot(fig, ax, title="SCHISM Grid")

    def plot_bathymetry(
        self,
        ax: Optional[Axes] = None,
        cmap: Optional[str] = None,
        levels: Optional[Union[int, List[float]]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_contours: bool = False,
        contour_levels: Optional[Union[int, List[float]]] = None,
        contour_colors: str = "black",
        contour_alpha: float = 0.5,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot SCHISM bathymetry as filled contours.

        Parameters
        ----------
        ax : Optional[Axes]
            Existing axes to plot on
        cmap : Optional[str]
            Colormap name. If None, uses plot_config default.
        levels : Optional[Union[int, List[float]]]
            Contour levels for filled contours
        vmin : Optional[float]
            Minimum depth value for colormap
        vmax : Optional[float]
            Maximum depth value for colormap
        show_contours : bool, optional
            Whether to add contour lines. Default is False.
        contour_levels : Optional[Union[int, List[float]]]
            Levels for contour lines
        contour_colors : str, optional
            Color for contour lines. Default is "black".
        contour_alpha : float, optional
            Transparency for contour lines. Default is 0.5.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, **kwargs)

        grid = self.config.grid
        hgrid = grid.pylibs_hgrid

        # Get bathymetry data (depths are negative in SCHISM)
        depths = hgrid.dp
        x, y, triangles = prepare_triangulation_data(hgrid)

        # Set up colormap
        cmap = self.plot_config.cmap
        cmap_name, norm, levels_array = setup_colormap(
            depths, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels
        )

        # Create filled contour plot
        if levels_array is not None:
            cs = ax.tricontourf(
                x,
                y,
                triangles,
                depths,
                levels=levels_array,
                cmap=cmap_name,
                extend="both",
            )
        else:
            cs = ax.tricontourf(
                x,
                y,
                triangles,
                depths,
                levels=51,  # Default number of levels
                cmap=cmap_name,
                extend="both",
            )

        # Add colorbar
        cbar = self.add_colorbar(
            fig, ax, cs, label="Depth (m)" if depths.min() < 0 else "Elevation (m)"
        )

        # Add contour lines if requested
        if show_contours:
            contour_levels = contour_levels or levels_array
            if contour_levels is not None:
                cs_lines = ax.tricontour(
                    x,
                    y,
                    triangles,
                    depths,
                    levels=contour_levels,
                    colors=contour_colors,
                    alpha=contour_alpha,
                    linewidths=0.5,
                )

        # Set extent
        extent = get_geographic_extent(x, y)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        # Add boundaries
        if self.plot_config.show_boundaries:
            add_boundary_overlay(
                ax,
                grid,
                boundary_colors=self.plot_config.boundary_colors,
                linewidth=self.plot_config.boundary_linewidth,
            )

        return self.finalize_plot(fig, ax, title="SCHISM Bathymetry")

    def plot_contours(
        self,
        ax: Optional[Axes] = None,
        levels: Optional[Union[int, List[float]]] = None,
        colors: str = "black",
        linewidths: float = 1.0,
        alpha: float = 0.8,
        add_labels: bool = True,
        label_fontsize: int = 8,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot bathymetry contour lines.

        Parameters
        ----------
        ax : Optional[Axes]
            Existing axes to plot on
        levels : Optional[Union[int, List[float]]]
            Contour levels to plot
        colors : str, optional
            Color for contour lines. Default is "black".
        linewidths : float, optional
            Width of contour lines. Default is 1.0.
        alpha : float, optional
            Transparency of contour lines. Default is 0.8.
        add_labels : bool, optional
            Whether to add contour labels. Default is True.
        label_fontsize : int, optional
            Font size for contour labels. Default is 8.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, **kwargs)

        try:
            grid = self.config.grid
            hgrid = grid.pylibs_hgrid

            # Get bathymetry data
            depths = hgrid.dp
            x, y = hgrid.x, hgrid.y
            triangles = hgrid.elnode

            # Set default levels if not provided
            if levels is None:
                levels = np.arange(
                    np.floor(depths.min()),
                    np.ceil(depths.max()) + 1,
                    max(1, int((depths.max() - depths.min()) / 20)),
                )

            # Create contour plot
            cs = ax.tricontour(
                x,
                y,
                triangles,
                depths,
                levels=levels,
                colors=colors,
                linewidths=linewidths,
                alpha=alpha,
            )

            # Add labels if requested
            if add_labels:
                ax.clabel(cs, inline=True, fontsize=label_fontsize, fmt="%.0f")

            # Set extent
            extent = get_geographic_extent(x, y)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            # Add boundaries
            if self.plot_config.show_boundaries:
                add_boundary_overlay(
                    ax,
                    grid,
                    boundary_colors=self.plot_config.boundary_colors,
                    linewidth=self.plot_config.boundary_linewidth,
                )

        except Exception as e:
            logger.error(f"Error plotting contours: {e}")
            raise

        return self.finalize_plot(fig, ax, title="SCHISM Depth Contours")

    def plot_boundaries(
        self,
        ax: Optional[Axes] = None,
        boundary_types: List[str] = None,
        colors: Dict[str, str] = None,
        linewidth: float = 2.0,
        alpha: float = 0.8,
        add_labels: bool = True,
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot SCHISM model boundaries.

        Parameters
        ----------
        ax : Optional[Axes]
            Existing axes to plot on
        boundary_types : List[str], optional
            Types of boundaries to plot. Options: 'ocean', 'land', 'tidal'.
            If None, all available boundaries are plotted.
        colors : Dict[str, str], optional
            Colors for different boundary types
        linewidth : float, optional
            Width of boundary lines. Default is 2.0.
        alpha : float, optional
            Transparency of boundary lines. Default is 0.8.
        add_labels : bool, optional
            Whether to add boundary labels. Default is True.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, **kwargs)

        # Set default colors
        if colors is None:
            colors = self.plot_config.boundary_colors

        # Set default boundary types
        if boundary_types is None:
            boundary_types = ["ocean", "land"]

        try:
            grid = self.config.grid
            hgrid = grid.pylibs_hgrid

            # Ensure boundaries are computed
            if hasattr(hgrid, "compute_bnd") and not hasattr(hgrid, "nob"):
                hgrid.compute_bnd()

            legend_elements = []

            # Plot ocean boundaries
            if "ocean" in boundary_types and hasattr(grid, "ocean_boundary"):
                try:
                    x_ocean, y_ocean = grid.ocean_boundary()
                    # Safe array length check to avoid truth value ambiguity
                    if (
                        x_ocean is not None
                        and y_ocean is not None
                        and x_ocean.size > 0
                        and y_ocean.size > 0
                    ):
                        line = ax.plot(
                            x_ocean,
                            y_ocean,
                            color=colors.get("ocean", "red"),
                            linewidth=linewidth,
                            alpha=alpha,
                            label="Ocean Boundary",
                        )[0]
                        legend_elements.append(line)
                except Exception as e:
                    logger.warning(f"Could not plot ocean boundaries: {e}")

            # Plot land boundaries
            if "land" in boundary_types and hasattr(grid, "land_boundary"):
                try:
                    x_land, y_land = grid.land_boundary()
                    # Safe array length check to avoid truth value ambiguity
                    if (
                        x_land is not None
                        and y_land is not None
                        and x_land.size > 0
                        and y_land.size > 0
                    ):
                        line = ax.plot(
                            x_land,
                            y_land,
                            color=colors.get("land", "green"),
                            linewidth=linewidth,
                            alpha=alpha,
                            label="Land Boundary",
                        )[0]
                        legend_elements.append(line)
                except Exception as e:
                    logger.warning(f"Could not plot land boundaries: {e}")

            # Plot tidal boundaries if available
            if "tidal" in boundary_types:
                try:
                    # This would require tidal configuration information
                    # Implementation depends on how tidal boundaries are stored
                    logger.info("Tidal boundary plotting not yet implemented")
                except Exception as e:
                    logger.warning(f"Could not plot tidal boundaries: {e}")

            # Set extent based on grid
            extent = get_geographic_extent(hgrid.x, hgrid.y)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            # Add legend if we have boundary elements and labels are requested
            if legend_elements and add_labels:
                ax.legend(handles=legend_elements, loc="best")

        except Exception as e:
            logger.error(f"Error plotting boundaries: {e}")
            raise

        return self.finalize_plot(fig, ax, title="SCHISM Boundaries")

    def plot_grid_quality(
        self,
        ax: Optional[Axes] = None,
        metric: str = "skewness",
        cmap: str = "RdYlBu_r",
        **kwargs,
    ) -> Tuple[Figure, Axes]:
        """
        Plot grid quality metrics.

        Parameters
        ----------
        ax : Optional[Axes]
            Existing axes to plot on
        metric : str, optional
            Quality metric to plot. Options: 'skewness', 'aspect_ratio'.
            Default is 'skewness'.
        cmap : str, optional
            Colormap for quality visualization. Default is 'RdYlBu_r'.
        **kwargs : dict
            Additional plotting parameters

        Returns
        -------
        fig : Figure
            Figure object
        ax : Axes
            Axes object
        """
        fig, ax = self.create_figure(ax=ax, **kwargs)

        grid = self.config.grid
        hgrid = grid.pylibs_hgrid

        # Calculate quality metric
        if metric == "skewness":
            quality_values = self._calculate_skewness(hgrid)
            title = "Grid Skewness"
            cbar_label = "Skewness"
        elif metric == "aspect_ratio":
            quality_values = self._calculate_aspect_ratio(hgrid)
            title = "Grid Aspect Ratio"
            cbar_label = "Aspect Ratio"
        else:
            raise ValueError(f"Unknown quality metric: {metric}")

        # Plot quality values using proper triangulation
        import matplotlib.tri as tri

        from .utils import prepare_triangulation_data

        # Get proper triangulation with special node handling
        x_coords, y_coords, triangles = prepare_triangulation_data(hgrid)
        triangulation = tri.Triangulation(x_coords, y_coords, triangles)

        cs = ax.tripcolor(triangulation, quality_values, cmap=cmap, shading="flat")

        # Add colorbar
        cbar = self.add_colorbar(fig, ax, cs, label=cbar_label)

        # Set extent
        extent = get_geographic_extent(hgrid.x, hgrid.y)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        return self.finalize_plot(fig, ax, title=title)

    def _calculate_skewness(self, hgrid) -> np.ndarray:
        """
        Calculate grid element skewness.

        Parameters
        ----------
        hgrid : Any
            SCHISM horizontal grid object

        Returns
        -------
        skewness : np.ndarray
            Skewness values for each element
        """
        # Reason: Remove dummy data, require real skewness values
        if hasattr(hgrid, "skewness"):
            return hgrid.skewness
        raise RuntimeError(
            "Grid skewness values missing from grid object. ACTION: Ensure your grid preprocessing computes skewness and the grid object exposes a 'skewness' attribute. See /rompy/schism/plotting/__init__.py for grid requirements."
        )

    def _calculate_aspect_ratio(self, hgrid) -> np.ndarray:
        """
        Calculate grid element aspect ratios.

        Parameters
        ----------
        hgrid : Any
            SCHISM horizontal grid object

        Returns
        -------
        aspect_ratios : np.ndarray
            Aspect ratio values for each element
        """
        # Simplified aspect ratio calculation
        # Reason: Remove dummy data, require real aspect ratio values
        if hasattr(hgrid, "aspect_ratios"):
            return hgrid.aspect_ratios
        raise RuntimeError(
            "Grid aspect ratio values missing from grid object. ACTION: Ensure your grid preprocessing computes aspect ratios and the grid object exposes an 'aspect_ratios' attribute. See /rompy/schism/plotting/__init__.py for grid requirements."
        )
