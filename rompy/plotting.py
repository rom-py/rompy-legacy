# -----------------------------------------------------------------------------
# Copyright (c) 2020 - 2021, CSIRO
#
# All rights reserved.
#
# The full license is in the LICENSE file, distributed with this software.
# -----------------------------------------------------------------------------


def scatter(ds, color, minLon=None, minLat=None, maxLon=None, maxLat=None, fscale=10):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from datetime import datetime
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # First set some plot parameters:
    if not minLon:
        minLon = ds.LONGITUDE.min().values.item()
    if not minLat:
        minLat = ds.LATITUDE.min().values.item()
    if not maxLon:
        maxLon = ds.LONGITUDE.max().values.item()
    if not maxLat:
        maxLat = ds.LATITUDE.max().values.item()
    extents = [minLon, maxLon, minLat, maxLat]

    # create figure and plot/map
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(fscale, fscale * (maxLon - minLon) / (maxLat - minLat)),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.set_extent(extents, crs=ccrs.PlateCarree())

    coastline = cfeature.GSHHSFeature(
        scale="auto", edgecolor="black", facecolor=cfeature.COLORS["land"]
    )
    ax.add_feature(coastline, zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=2)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    if "PARTITION" in ds:
        sc = ax.scatter(
            [ds.sel(PARTITION=part).LONGITUDE for part in ds.PARTITION],
            [ds.sel(PARTITION=part).LATITUDE for part in ds.PARTITION],
            c=[ds.sel(PARTITION=part)[color] for part in ds.PARTITION],
        )
    else:
        sc = ax.scatter(ds.LONGITUDE, ds.LATITUDE, c=ds[color])

    axins0 = inset_axes(
        ax,
        width="5%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.07, 0, 1, 1),
        bbox_transform=ax.transAxes,
    )
    cb = fig.colorbar(sc, orientation="vertical", cax=axins0, ticklocation="right")
    if color == "TIME":
        cb.ax.set_yticklabels(
            [
                (datetime.fromtimestamp(i // 10**9)).strftime("%b %d %Y")
                for i in cb.get_ticks()
            ]
        )
    if "units" in ds[color].attrs.keys():
        cb.set_label(f"{ds[color].standard_name}\n [{ds[color].units}]")

    return fig, ax
