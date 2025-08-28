import logging
import re
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.tri import Triangulation
from pyproj import Proj
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()

logging.basicConfig(level=logging.INFO)


def schism_load(schfile):
    """
    Ron: Load schism output file and parse the element/node values to create a (matplotlib) trimesh object for plotting

    Returns xarray dataset and trimesh object
    """
    schout = xr.open_dataset(schfile)
    elems = np.int32(schout.SCHISM_hgrid_face_nodes[:, :-1] - 1)
    # Ron: get lat/lon coordinates of nodes - weird it appears x,y are switched
    lons = schout.SCHISM_hgrid_node_y.values
    lats = schout.SCHISM_hgrid_node_x.values
    # create trimesh object
    meshtri = Triangulation(lats, lons, triangles=elems)
    return schout, meshtri


def schism_plot(
    schout,
    meshtri,
    varname,
    varscale=[],
    bbox=[],
    time=-1,
    mask=True,
    vectors=False,
    plotmesh=False,
    project=False,
    contours=[10, 30, 50],
    pscale=20,
    cmap=plt.cm.jet,
    add_coastlines=True,
):
    """
    plot output variable in xarray dataset (schout) using mesh information meshtri.
    input:
        schout: xarray dataset returned by def schism_load
        meshtri: (matplotlib) trimesh object returned by def schism_load
        varname: name of data variable in schout
        OPTIONAL:
            varscale: min/max plotting colour values (defalts to min/max)
            bbox: bounding box for plotting [minlon,minlat,maxlon,maxlat] (defalts to data bounds)
            time: time to plot (if a variable dimension), can be int (index) or datetime64-like object
            plotmesh: plot the grid mesh (elemment boundaries)
            project: use a map projection (cartopy, so that e.g. gis data can be added - this is slower
            mask: mask out parts of the SCHISM output based on a minumum depth threshold
    Returns xarray dataset and
    We should modify this to load multiple files ... probably need assistance from DASK
    """
    if "time" in list(schout.dims):
        if type(time) == int:  # input ts is index
            schout = schout.isel(time=time)
        else:  # assume is datetime64-like object
            schout = schout.sel(time=time)
    # By default, plot depth contours
    # ... I like depth to be z (negative)
    z = schout.depth * -1
    if np.mean(contours) > 0:
        contours = np.flip(-1 * np.asarray(contours))
    else:
        contours = np.asarray(contours)
    if varname == "depth" or varname == "z":
        var = z
    elif varname == "wind_speed":
        var = np.sqrt(schout.wind_speed[:, 0] ** 2 + schout.wind_speed[:, 1] ** 2)
    elif varname == "dahv":
        var = np.sqrt(schout.dahv[:, 0] ** 2 + schout.dahv[:, 1] ** 2)
    else:
        var = schout[varname]

    if len(varscale) == 0:
        vmin = var.values.min()
        vmax = var.values.max()
    else:
        vmin, vmax = varscale
    if project:
        x, y = meshtri.x, meshtri.y
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(pscale, pscale),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        if len(bbox) == 4:
            ax.set_extent([bbox[0], bbox[2], bbox[1], bbox[3]], ccrs.PlateCarree())
        else:
            bbox = [x.min(), y.min(), x.max(), y.max()]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(30, 30))
    if plotmesh:
        ax.triplot(meshtri, color="k", alpha=0.3)
    ### MASKING ** doesnt work with tripcolor, must use tricontouf ###############################
    # mask all places in the grid where depth is greater than elev (i.e. are "dry") by threshold below
    if mask:
        # temp=var.values
        # threshold of + 0.05 seems pretty good   *** But do we want to use the minimum depth
        # defined in the SCHISM input (H0) and in output schout.minimum_depth
        # bad_idx= schout.elev.values+schout.depth.values<0.05
        bad_idx = (
            schout.elevation.values + schout.depth.values < schout.minimum_depth.values
        )
        # new way
        mask = np.all(np.where(bad_idx[meshtri.triangles], True, False), axis=1)
        meshtri.set_mask(mask)

        extend = "neither"
        if (var.min() < vmin) & (var.max() > vmax):
            extend = "both"
        elif var.min() < vmin:
            extend = "min"
        elif var.max() > vmax:
            extend = "max"

        cax = ax.tricontourf(
            meshtri, var, cmap=cmap, levels=np.linspace(vmin, vmax, 50), extend=extend
        )
    # no mask#############################################################
    else:
        cax = ax.tripcolor(meshtri, var, cmap=cmap, vmin=vmin, vmax=vmax)
        # quiver variables if asked
    if vectors:
        if re.search("sigWaveHeight", varname):
            vtype = "waves"
        if re.search("wind", varname):
            vtype = "wind"
        else:
            vtype = "currents"
        LonI, LatI, UI, VI = schism_calculate_vectors(ax, schout, vtype=vtype)
        ax.quiver(LonI, LatI, UI, VI, color="k")

    con = ax.tricontour(meshtri, z, contours, colors="k")
    # ax.clabel(con, con.levels, inline=True, fmt='%i', fontsize=12)
    if not (project):
        ax.set_aspect("equal")
    else:
        # draw lat/lon grid lines every n degrees.
        #  n = (max(lat)-min(lat))/8
        n = (bbox[2] - bbox[0]) / 5
        for fac in [1, 10, 100]:
            nr = np.round(n * fac) / fac
            if nr > 0:
                n = nr
                xticks = np.arange(
                    np.round(bbox[0] * fac) / fac, np.round(bbox[2] * fac) / fac + n, n
                )
                yticks = np.arange(
                    np.round(bbox[1] * fac) / fac, np.round(bbox[3] * fac) / fac + n, n
                )
                break
        #        ax.set_xticks(xticks, crs=ccrs.PlateCarree()
        ax.set_xticks(xticks)
        #         ax.set_xticklabels(np.arange(np.round(min(x)),np.round(max(x)),n))
        #        ax.set_yticks(yticks, crs=ccrs.PlateCarree()
        ax.set_yticks(yticks)
        #         ax.set_yticklabels(np.arange(np.round(min(y)),np.round(max(y)),n))
        ax.yaxis.tick_left()
        # lon_formatter = cticker.LongitudeFormatter()
        # lat_formatter = cticker.LatitudeFormatter()
        # New versions of marplotlib throw warnings on this - does it matter
        # ax.xaxis.set_major_formatter(lon_formatter)
        # ax.yaxis.set_major_formatter(lat_formatter)
        # ax.set_xticks(lon_formatter)
        # ax.set_yticks(lat_formatter)
        ax.grid(linewidth=1, color="black", alpha=0.5, linestyle="--")
        ax.add_feature(cfeature.BORDERS, linewidth=2)
    if len(bbox) == 4:
        ax.set_ylim(bbox[1], bbox[3])
        ax.set_xlim(bbox[0], bbox[2])
    cbar = plt.colorbar(mappable=cax, shrink=0.5)
    cbar.set_ticks(np.round(np.linspace(vmin, vmax, 5) * 100) / 100)
    cbar.set_label(varname)
    if add_coastlines:
        ax.coastlines()

    return fig, ax


# Nautical convention
def pol2cart2(rho, deg):
    x, y = pol2cart(rho, deg / 180.0 * np.pi)
    return y, x


# Cartesian convention
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def schism_calculate_vectors(ax, schout, vtype="waves", dX="auto", mask=True):
    pUTM55 = Proj("epsg:32755")
    # pWGS84 = Proj('epsg:4326')
    if vtype == "waves":
        idx = (schout.sigWaveHeight > 0.05) & (schout.elevation - schout.depth < 0.1)
        dp = schout.discretePeakDirection[idx]
        # hs=schout.WWM_1[idx]
        hs = np.ones(dp.shape)
        [u, v] = pol2cart2(hs, np.mod(dp + 180, 360))
    elif vtype == "elev" or re.search("curr", vtype):
        idx = np.sqrt(schout.depthAverageVelX**2 + schout.depthAverageVelX**2) > 0.0
        u = schout.depthAverageVelX[idx]
        v = schout.depthAverageVelY[idx]
    elif vtype == "wind":
        idx = np.ones_like(schout.windSpeedX, dtype=bool)
        u = schout.windSpeedX[idx]
        v = schout.windSpeedY[idx]
    else:
        raise ValueError("*** Warning input vecter data not understood")
    x, y = pUTM55(
        schout.SCHISM_hgrid_node_x.values[idx], schout.SCHISM_hgrid_node_y.values[idx]
    )
    xlim, ylim = pUTM55(ax.get_xlim(), ax.get_ylim())
    # might have to play with this - we assume a total of 100 arrows a side will be pleasing
    if dX == "auto":
        n = 30
        dX = np.ceil((ylim[1] - ylim[0]) / n)
    xi = np.arange(xlim[0], xlim[1] + dX, dX)
    yi = np.arange(ylim[0], ylim[1] + dX, dX)
    XI, YI = np.meshgrid(xi, yi)

    UI = griddata((x, y), u, (XI, YI), method="linear")
    VI = griddata((x, y), v, (XI, YI), method="linear")
    # Create a mask so that place with very little data are removed
    if mask:
        xbins = np.arange(xlim[0], xlim[1] + 2 * dX, dX)
        ybins = np.arange(ylim[0], ylim[1] + 2 * dX, dX)
        densityH, _, _ = np.histogram2d(x, y, bins=[xbins, ybins])
        densityH = densityH.T
        # might want to adjust this...
        idx = densityH < 1
        UI[idx] = np.nan
        VI[idx] = np.nan
    LonI, LatI = pUTM55(XI, YI, inverse=True)

    return LonI, LatI, UI, VI


if __name__ == "__main__":
    # load schism files
    # schfile = "../../notebooks/schism/schism_procedural/test_schism/outputs/schout_1.nc"
    # take schfile off the command line
    schfile = sys.argv[1]
    schout, meshtri = schism_load(schfile)
    lons = schout.SCHISM_hgrid_node_y.values
    lats = schout.SCHISM_hgrid_node_x.values
    # plot gridded fields - elevation
    for variable in [
        "elevation",
        "windSpeedX",
        "sigWaveHeight",
        "depthAverageVelX",
        "airPressure",
    ]:
        # for variable in ["airPressure"]:
        for ix, time in enumerate(schout.time.values):
            fig, ax = schism_plot(
                schout,
                meshtri,
                variable,
                bbox=[145, -25, 155, -16],
                project=True,
                plotmesh=True,
                time=time,
                mask=False,
                vectors=True,
                # varscale=(-9, 9),
                contours=[0],
            )
            plt.tight_layout()
            plt.savefig(
                f"schism_plot_{variable}_{ix}.png", dpi=300, bbox_inches="tight"
            )
    # plt.show()
    # plt.close()
