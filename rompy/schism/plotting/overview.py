"""
Enhanced overview plotting functionality for SCHISM model visualization.

This module provides comprehensive multi-panel overview plots that combine
grid visualization, data analysis, and model setup validation in a single view.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .core import BasePlotter, PlotConfig, PlotValidator
from .data import DataPlotter
from .grid import GridPlotter
from .utils import (format_scientific_notation, get_geographic_extent,
                    setup_cartopy_axis)

logger = logging.getLogger(__name__)


class OverviewPlotter(BasePlotter):
    """
    Enhanced overview plotting for comprehensive SCHISM model visualization.

    This class creates multi-panel overview plots that provide a complete
    picture of the SCHISM model setup, including grid quality, boundary
    conditions, forcing data, and validation metrics.

    Parameters
    ----------
    config : Optional[Any]
        SCHISM configuration object
    plot_config : Optional[PlotConfig]
        Plot configuration parameters
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        plot_config: Optional[PlotConfig] = None,
    ):
        """Initialize OverviewPlotter."""
        super().__init__(config, plot_config)

        # Initialize sub-plotters
        self.grid_plotter = GridPlotter(config, plot_config)
        self.data_plotter = DataPlotter(config, plot_config)

    def plot_comprehensive_overview(
        self,
        figsize: Tuple[float, float] = (20, 16),
        include_validation: bool = True,
        include_quality_metrics: bool = True,
        include_data_summary: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create comprehensive multi-panel overview plot.

        This creates a detailed overview with the following panels:
        - Grid and bathymetry visualization
        - Boundary conditions and forcing locations
        - Data quality assessment plots
        - Model setup validation summary
        - Time series overview of forcing data

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (20, 16).
        include_validation : bool, optional
            Include model validation panel. Default is True.
        include_quality_metrics : bool, optional
            Include grid quality metrics panel. Default is True.
        include_data_summary : bool, optional
            Include data summary panel. Default is True.
        save_path : Optional[Union[str, Path]], optional
            Path to save the plot. If None, plot is not saved.
        **kwargs : dict
            Additional keyword arguments passed to plotting functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : Dict[str, matplotlib.axes.Axes]
            Dictionary of axes objects keyed by panel name.

        Examples
        --------
        >>> plotter = OverviewPlotter(config)
        >>> fig, axes = plotter.plot_comprehensive_overview()
        >>> fig.show()
        """
        # Create complex subplot layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Dictionary to store all axes
        axes = {}

        # Main grid panel (top-left, 2x2)
        axes["grid"] = fig.add_subplot(gs[0:2, 0:2], projection=setup_cartopy_axis())
        self._plot_main_grid_panel(axes["grid"], **kwargs)

        # Boundary conditions panel (top-right, 2x1)
        axes["boundaries"] = fig.add_subplot(
            gs[0:2, 2], projection=setup_cartopy_axis()
        )
        self._plot_boundary_panel(axes["boundaries"], **kwargs)

        # Data locations panel (top-right, 2x1)
        axes["data_locations"] = fig.add_subplot(
            gs[0:2, 3], projection=setup_cartopy_axis()
        )
        self._plot_data_locations_panel(axes["data_locations"], **kwargs)

        # Grid quality metrics (bottom-left)
        if include_quality_metrics:
            axes["quality"] = fig.add_subplot(gs[2, 0])
            self._plot_quality_metrics_panel(axes["quality"], **kwargs)

        # Model validation (bottom-center-left)
        if include_validation:
            axes["validation"] = fig.add_subplot(gs[2, 1])
            self._plot_validation_panel(axes["validation"], **kwargs)

        # Data summary (bottom-center-right)
        if include_data_summary:
            axes["data_summary"] = fig.add_subplot(gs[2, 2])
            self._plot_data_summary_panel(axes["data_summary"], **kwargs)

        # Time series overview (bottom-right)
        axes["timeseries"] = fig.add_subplot(gs[2, 3])
        self._plot_timeseries_overview_panel(axes["timeseries"], **kwargs)

        # Model info panel (bottom row, spanning all columns)
        axes["info"] = fig.add_subplot(gs[3, :])
        self._plot_model_info_panel(axes["info"], **kwargs)

        # Add overall title
        fig.suptitle("SCHISM Model Overview", fontsize=16, fontweight="bold", y=0.98)

        # Save if requested
        if save_path:
            self._save_plot(fig, save_path)

        return fig, axes

    def plot_grid_analysis_overview(
        self,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create grid-focused analysis overview.

        This creates a detailed grid analysis with:
        - Grid structure and bathymetry
        - Element quality metrics
        - Boundary condition locations
        - Grid statistics and histograms

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (16, 12).
        save_path : Optional[Union[str, Path]], optional
            Path to save the plot. If None, plot is not saved.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : Dict[str, matplotlib.axes.Axes]
            Dictionary of axes objects keyed by panel name.
        """
        # Create subplot layout
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        axes = {}

        # Main grid visualization (top row, spans 2 columns)
        axes["grid"] = fig.add_subplot(gs[0, 0:2], projection=setup_cartopy_axis())
        self.grid_plotter.plot_bathymetry(ax=axes["grid"], **kwargs)
        axes["grid"].set_title("Grid and Bathymetry", fontweight="bold")

        # Grid quality (top-right)
        axes["quality"] = fig.add_subplot(gs[0, 2], projection=setup_cartopy_axis())
        self.grid_plotter.plot_grid_quality(ax=axes["quality"], **kwargs)
        axes["quality"].set_title("Element Quality", fontweight="bold")

        # Boundaries (middle-left)
        axes["boundaries"] = fig.add_subplot(gs[1, 0], projection=setup_cartopy_axis())
        self.grid_plotter.plot_boundaries(ax=axes["boundaries"], **kwargs)
        axes["boundaries"].set_title("Boundaries", fontweight="bold")

        # Depth histogram (middle-center)
        axes["depth_hist"] = fig.add_subplot(gs[1, 1])
        self._plot_depth_histogram(axes["depth_hist"], **kwargs)

        # Element size histogram (middle-right)
        axes["size_hist"] = fig.add_subplot(gs[1, 2])
        self._plot_element_size_histogram(axes["size_hist"], **kwargs)

        # Grid statistics table (bottom row)
        axes["stats"] = fig.add_subplot(gs[2, :])
        self._plot_grid_statistics_table(axes["stats"], **kwargs)

        fig.suptitle("SCHISM Grid Analysis Overview", fontsize=16, fontweight="bold")

        if save_path:
            self._save_plot(fig, save_path)

        return fig, axes

    def plot_data_analysis_overview(
        self,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create data-focused analysis overview.

        This creates a comprehensive data analysis with:
        - Atmospheric forcing overview
        - Boundary data time series
        - Data quality metrics
        - Temporal coverage analysis

        Parameters
        ----------
        figsize : Tuple[float, float], optional
            Figure size in inches. Default is (16, 12).
        save_path : Optional[Union[str, Path]], optional
            Path to save the plot. If None, plot is not saved.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : Dict[str, matplotlib.axes.Axes]
            Dictionary of axes objects keyed by panel name.
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        axes = {}

        # Atmospheric data spatial (top-left)
        axes["atmospheric"] = fig.add_subplot(gs[0, 0], projection=setup_cartopy_axis())
        self._plot_atmospheric_overview(axes["atmospheric"], **kwargs)

        # Boundary data locations (top-center)
        axes["boundary_locations"] = fig.add_subplot(
            gs[0, 1], projection=setup_cartopy_axis()
        )
        self._plot_boundary_data_locations(axes["boundary_locations"], **kwargs)

        # Data coverage timeline (top-right)
        axes["coverage"] = fig.add_subplot(gs[0, 2])
        self._plot_data_coverage_timeline(axes["coverage"], **kwargs)

        # Atmospheric time series (middle row)
        axes["atm_timeseries"] = fig.add_subplot(gs[1, :])
        self._plot_atmospheric_timeseries_overview(axes["atm_timeseries"], **kwargs)

        # Boundary time series (bottom-left)
        axes["boundary_ts"] = fig.add_subplot(gs[2, 0])
        self._plot_boundary_timeseries_overview(axes["boundary_ts"], **kwargs)

        # Data quality metrics (bottom-center)
        axes["data_quality"] = fig.add_subplot(gs[2, 1])
        self._plot_data_quality_metrics(axes["data_quality"], **kwargs)

        # Data statistics (bottom-right)
        axes["data_stats"] = fig.add_subplot(gs[2, 2])
        self._plot_data_statistics_table(axes["data_stats"], **kwargs)

        fig.suptitle("SCHISM Data Analysis Overview", fontsize=16, fontweight="bold")

        if save_path:
            self._save_plot(fig, save_path)

        return fig, axes

    def _plot_main_grid_panel(self, ax: Axes, **kwargs) -> None:
        """Plot main grid panel with bathymetry and key features."""
        try:
            self.grid_plotter.plot_bathymetry(ax=ax, **kwargs)
            ax.set_title("Grid and Bathymetry", fontweight="bold")

            # Add grid statistics as text
            if hasattr(self, "grid") and self.grid is not None:
                grid_info = self._get_grid_info()
                ax.text(
                    0.02,
                    0.98,
                    grid_info,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=8,
                )
        except Exception as e:
            logger.error(f"Could not plot main grid panel: {e}")
            raise RuntimeError("Grid visualization unavailable")

    def _plot_boundary_panel(self, ax: Axes, **kwargs) -> None:
        """Plot boundary conditions panel."""
        try:
            self.grid_plotter.plot_boundaries(ax=ax, show_colorbar=False, **kwargs)
            ax.set_title("Boundary Conditions", fontweight="bold")
        except Exception as e:
            logger.error(f"Could not plot boundary panel: {e}")
            raise RuntimeError("Boundary data unavailable")

    def _plot_data_locations_panel(self, ax: Axes, **kwargs) -> None:
        """Plot data locations panel."""
        try:
            # Plot grid outline
            if hasattr(self, "grid") and self.grid is not None:
                self.grid_plotter.plot_grid(
                    ax=ax, show_elements=False, show_colorbar=False, **kwargs
                )

            # Add data source locations if available
            self._add_data_source_markers(ax)
            ax.set_title("Data Locations", fontweight="bold")
        except Exception as e:
            logger.error(f"Could not plot data locations panel: {e}")
            raise RuntimeError("Data locations unavailable")

    def _plot_quality_metrics_panel(self, ax: Axes, **kwargs) -> None:
        """Plot grid quality metrics."""
        try:
            if hasattr(self, "grid") and self.grid is not None:
                stats = self._calculate_grid_statistics()
                self._create_statistics_table(ax, stats, "Grid Statistics")
            else:
                raise RuntimeError("Grid statistics unavailable")
        except Exception as e:
            logger.error(f"Could not plot grid statistics: {e}")
            raise RuntimeError("Grid statistics unavailable: " + str(e))

            return

        names = list(metrics.keys())
        values = list(metrics.values())

        bars = ax.barh(
            names,
            values,
            color=[
                "green" if v > 0.8 else "orange" if v > 0.6 else "red" for v in values
            ],
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("Quality Score")

        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
            )

    def _run_validation_checks(self) -> Dict[str, str]:
        """Run model setup validation checks using real model/config data only."""
        # TODO: Implement real validation logic using model/config data
        raise NotImplementedError("Validation checks must use real model/config data.")

    def _plot_validation_results(self, ax: Axes, results: Dict[str, str]) -> None:
        """Plot validation results."""
        if not results:
            ax.text(
                0.5,
                0.5,
                "No validation results",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        y_pos = np.arange(len(results))
        labels = list(results.keys())
        statuses = list(results.values())

        colors = {"PASS": "green", "WARNING": "orange", "FAIL": "red"}
        bar_colors = [colors.get(status, "gray") for status in statuses]

        ax.barh(y_pos, [1] * len(results), color=bar_colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 1)
        ax.set_xticks([])

        # Add status labels
        for i, status in enumerate(statuses):
            ax.text(
                0.5,
                i,
                status,
                ha="center",
                va="center",
                fontweight="bold",
                color="white",
            )

    def _get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data from real model/config/data files only."""
        # TODO: Implement real data summary extraction from config/data files
        raise NotImplementedError("Data summary must use real model/config/data files.")

    def _plot_data_summary_chart(self, ax: Axes, summary: Dict[str, Any]) -> None:
        """Plot data summary as a simple chart."""
        if not summary:
            raise RuntimeError("No data summary available")
            return

        # Create a simple text summary
        summary_text = "\n".join(
            [f"{k.replace('_', ' ').title()}: {v}" for k, v in summary.items()]
        )
        ax.text(
            0.05,
            0.95,
            summary_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    def _plot_forcing_timeseries_overview(self, ax: Axes) -> None:
        """Plot overview of forcing time series using real data only."""
        # TODO: Implement real forcing time series plotting using model/config data
        raise NotImplementedError(
            "Forcing time series overview must use real model/config data."
        )

    def _get_model_info(self) -> Dict[str, str]:
        """Get model configuration information."""
        try:
            # Placeholder implementation
            return {
                "SCHISM Version": "5.11.0",
                "Grid Type": "Unstructured",
                "Vertical Layers": "10",
                "Physics": "Hydrostatic",
                "Modules": "WWM, ICE, SED",
            }
        except Exception as e:
            logger.warning(f"Could not get model info: {e}")
            return {}

    def _plot_model_info_table(self, ax: Axes, info: Dict[str, str]) -> None:
        """Plot model information as a table."""
        if not info:
            raise RuntimeError("No model info available")
            return

        # Create table data
        table_data = [[k, v] for k, v in info.items()]

        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=["Parameter", "Value"],
            cellLoc="left",
            loc="center",
            colWidths=[0.4, 0.6],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(table_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header row
                    cell.set_facecolor("#40466e")
                    cell.set_text_props(weight="bold", color="white")
                else:
                    cell.set_facecolor("#f1f1f2" if i % 2 == 0 else "white")

        ax.axis("off")

    def _add_data_source_markers(self, ax: Axes) -> None:
        """Add markers for data source locations."""
        try:
            # Placeholder implementation - would add actual data source locations
            # This would typically mark locations of boundary condition files,
            # atmospheric forcing grid points, etc.
            pass
        except Exception as e:
            logger.warning(f"Could not add data source markers: {e}")

    def _plot_depth_histogram(self, ax: Axes, **kwargs) -> None:
        """Plot depth histogram."""
        try:
            if hasattr(self, "grid") and self.grid is not None:
                # Get depth from pylibs_hgrid
                hgrid = self.grid.pylibs_hgrid
                if hasattr(hgrid, "dp"):
                    depths = hgrid.dp
                    depths = depths[~np.isnan(depths)]

                    ax.hist(depths, bins=50, alpha=0.7, edgecolor="black")
                    ax.set_xlabel("Depth (m)")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Depth Distribution", fontweight="bold")
                    ax.grid(True, alpha=0.3)
                else:
                    raise RuntimeError("Depth data unavailable")
            else:
                raise RuntimeError("Grid data unavailable")
        except Exception as e:
            logger.warning(f"Could not plot depth histogram: {e}")
            ax.text(
                0.5,
                0.5,
                "Depth histogram unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _plot_element_size_histogram(self, ax: Axes, **kwargs) -> None:
        """Plot element size histogram using real grid data only."""
        # TODO: Implement real element size histogram using grid data
        raise RuntimeError(
            "Element size histogram unavailable: must use real grid data."
        )

    def _plot_grid_statistics_table(self, ax: Axes, **kwargs) -> None:
        """Plot grid statistics as a table."""
        try:
            if hasattr(self, "grid") and self.grid is not None:
                hgrid = self.grid.pylibs_hgrid
                if hasattr(hgrid, "dp"):
                    depths = hgrid.dp
                    depths = depths[~np.isnan(depths)]

                    ax.hist(depths, bins=50, alpha=0.7, edgecolor="black")
                    ax.set_xlabel("Depth (m)")
                    ax.set_ylabel("Frequency")
                    ax.set_title("Depth Distribution", fontweight="bold")
                    ax.grid(True, alpha=0.3)
                else:
                    raise RuntimeError("Depth data unavailable")
            else:
                raise RuntimeError("Grid data unavailable")
        except Exception as e:
            logger.error(f"Could not plot depth histogram: {e}")
            raise RuntimeError("Depth histogram unavailable: " + str(e))

        except Exception as e:
            logger.warning(f"Could not plot grid statistics: {e}")
            raise RuntimeError("Grid statistics unavailable: " + str(e))

    def _calculate_grid_statistics(self) -> Dict[str, str]:
        """Calculate comprehensive grid statistics."""
        try:
            if hasattr(self, "grid") and self.grid is not None:
                # Use SCHISMGrid properties instead of coords
                n_nodes = self.grid.np  # number of points/nodes
                n_elements = self.grid.ne  # number of elements

                # Get depth from pylibs_hgrid if available
                hgrid = self.grid.pylibs_hgrid
                if hasattr(hgrid, "dp"):
                    depths = hgrid.dp
                    depths = depths[~np.isnan(depths)]

                    return {
                        "Total Nodes": f"{n_nodes:,}",
                        "Total Elements": f"{n_elements:,}",
                        "Min Depth": f"{depths.min():.2f} m",
                        "Max Depth": f"{depths.max():.2f} m",
                        "Mean Depth": f"{depths.mean():.2f} m",
                        "Depth Std": f"{depths.std():.2f} m",
                    }
                else:
                    return {
                        "Total Nodes": f"{n_nodes:,}",
                        "Total Elements": f"{n_elements:,}",
                        "Depth Stats": "unavailable",
                    }
            return {}
        except Exception as e:
            logger.warning(f"Could not calculate grid statistics: {e}")
            return {}

    def _create_statistics_table(
        self, ax: Axes, stats: Dict[str, str], title: str
    ) -> None:
        """Create a formatted statistics table."""
        if not stats:
            raise RuntimeError(f"No {title.lower()} available")
            return

        # Arrange statistics in columns
        items = list(stats.items())
        n_cols = 3
        n_rows = (len(items) + n_cols - 1) // n_cols

        table_data = []
        for row in range(n_rows):
            row_data = []
            for col in range(n_cols):
                idx = row * n_cols + col
                if idx < len(items):
                    key, value = items[idx]
                    row_data.extend([key, value])
                else:
                    row_data.extend(["", ""])
            table_data.append(row_data)

        # Create table
        table = ax.table(cellText=table_data, cellLoc="left", loc="center")

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Style the table
        for i in range(n_rows):
            for j in range(n_cols * 2):
                cell = table[(i, j)]
                if j % 2 == 0:  # Parameter names
                    cell.set_facecolor("#f8f9fa")
                    cell.set_text_props(weight="bold")
                else:  # Values
                    cell.set_facecolor("white")

        ax.set_title(title, fontweight="bold", pad=20)
        ax.axis("off")

    # Additional helper methods for data analysis overview
    def _plot_atmospheric_overview(self, ax: Axes, **kwargs) -> None:
        """Plot atmospheric forcing overview."""
        try:
            # Try to get atmospheric data from data plotter
            if hasattr(self.data_plotter, "plot_atmospheric_spatial"):
                # Remove show_colorbar parameter that's not accepted by plot_atmospheric_spatial
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() if k != "show_colorbar"
                }
                fig, _ = self.data_plotter.plot_atmospheric_spatial(
                    ax=ax, **filtered_kwargs
                )
                # Don't add colorbar for overview plots - keep it simple
                if hasattr(fig, "axes"):
                    for a in fig.axes:
                        if hasattr(a, "collections"):
                            for coll in a.collections:
                                if hasattr(coll, "colorbar") and coll.colorbar:
                                    coll.colorbar.remove()
            else:
                raise RuntimeError("Atmospheric data unavailable")
        except Exception as e:
            logger.warning(f"Could not plot atmospheric overview: {e}")
            raise RuntimeError("No atmospheric data available for analysis: " + str(e))
        ax.set_title("Atmospheric Forcing", fontweight="bold")

    def _plot_boundary_data_locations(self, ax: Axes, **kwargs) -> None:
        """Plot boundary data locations."""
        try:
            # Plot grid outline first
            if hasattr(self, "grid") and self.grid is not None:
                self.grid_plotter.plot_grid(
                    ax=ax, show_elements=False, show_colorbar=False, alpha=0.3, **kwargs
                )

            # Add boundary locations
            self.grid_plotter.plot_boundaries(ax=ax, show_colorbar=False, **kwargs)

        except Exception as e:
            logger.warning(f"Could not plot boundary data locations: {e}")
            raise RuntimeError("Boundary locations unavailable: " + str(e))
        ax.set_title("Boundary Data Locations", fontweight="bold")

    def _plot_data_coverage_timeline(self, ax: Axes, **kwargs) -> None:
        """Plot data coverage timeline using real data only."""
        # TODO: Implement real data coverage timeline using model/config/data
        raise NotImplementedError(
            "Data coverage timeline must use real model/config/data."
        )

    def _plot_atmospheric_timeseries_overview(self, ax: Axes, **kwargs) -> None:
        """Plot atmospheric time series overview using real data only."""
        # TODO: Implement real atmospheric time series plotting using model/config data
        raise NotImplementedError(
            "Atmospheric time series overview must use real model/config data."
        )

    def _plot_boundary_timeseries_overview(self, ax: Axes, **kwargs) -> None:
        """Plot boundary time series overview using real data only."""
        # TODO: Implement real boundary time series plotting using model/config data
        raise NotImplementedError(
            "Boundary time series overview must use real model/config data."
        )

    def _plot_data_quality_metrics(self, ax: Axes, **kwargs) -> None:
        """Plot data quality metrics using real data only."""
        # TODO: Implement real data quality metrics plotting using model/config/data
        raise NotImplementedError(
            "Data quality metrics must use real model/config/data."
        )

    def _plot_data_statistics_table(self, ax: Axes, **kwargs) -> None:
        """Plot data statistics table using real data only."""
        # TODO: Implement real data statistics table using model/config/data
        raise NotImplementedError(
            "Data statistics table must use real model/config/data."
        )

    def plot(self, **kwargs) -> Tuple[Figure, Dict[str, Axes]]:
        """
        Create default overview plot (alias for plot_comprehensive_overview).

        This implements the abstract plot method from BasePlotter.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to plot_comprehensive_overview.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        axes : Dict[str, matplotlib.axes.Axes]
            Dictionary of axes objects keyed by panel name.
        """
        return self.plot_comprehensive_overview(**kwargs)

    def _save_plot(self, fig: Figure, save_path: Union[str, Path]) -> bool:
        """Save plot to file."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=self.plot_config.dpi,
                bbox_inches="tight",
                facecolor="white",
            )
            logger.info(f"Overview plot saved to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Could not save plot to {save_path}: {e}")
            return False
