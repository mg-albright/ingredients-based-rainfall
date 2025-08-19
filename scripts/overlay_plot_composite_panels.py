# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
overlay_plot_composite_panels.py

Author: Mary Grace Albright

Usage:
    python overlay_plot_composite_panels.py

This script reads all monthly netCDF files for a given year and ARI duration,
computes composites (mean over events) either annually or by season,
and plots composites for specified lead times and regions as 2x2 panel plots with a
single, shared vertical colorbar. Only the outer axes have tick labels and axis labels
(delta Latitude on left panels; delta Longitude on bottom panels), and the wind-speed quiver
key appears on the top-right panel. The colorbar is extended on both ends.
Modified: MGA 2025 July 24
"""

print("Loading packages...")
import os
import glob
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import csv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colormaps
import warnings
from plotting_functions import truncate_colormap, save_composite_files, instantaneous_moisture_flux, get_ncl_colormap

warnings.filterwarnings(
    "ignore",
    message=".*multi-part geometries is deprecated.*"
)


centroid_file = '/export/hpc-lw-director/malbright/scripts/centroid_region_averages.csv'
centroids = {}
with open(centroid_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # note: store as (lon, lat) for Cartopy
        centroids[row['Region']] = (
            float(row['Avg_Lon']), float(row['Avg_Lat'])
        )
        
        
def make_composites(
    data_dir,
    var,
    ari_dur,
    year,
    period,
    lead_times,
    regions,
    seasons,
    user_cmap,
    ncl=False,
    save_vars=False,
    user_vmin=None,
    user_vmax=None,
    user_vstep=None,
    out_dir=None,
    color_list=None,
    u_var=None,
    v_var=None,
    wind_skip=5,
    wind_scale=None
):
    
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # find main files
    main_paths = sorted(glob.glob(os.path.join(data_dir, var, f"{ari_dur}yrARI/hrrr_3h{ari_dur}yrARI_{var}_{year}*.nc")))
    
    if save_vars:
        save_composite_files(
            data_dir, var, ari_dur, year, period,
            lead_times, regions, seasons,
            out_dir, u_var, v_var
        )
        return
    
    # read grid spacing
    ds0 = xr.open_dataset(main_paths[0])
    dlon = ds0.attrs.get('dlon', 1.0)
    dlat = ds0.attrs.get('dlat', 1.0)
    ds0.close()
    tick_degs = np.arange(-2.0, 2.5, 1.0)
        
    for region_name, bbox in regions.items():
        # prepare save directory
        save_dir = os.path.join(out_dir, f"{ari_dur}yrARI/{region_name}/{period}")
        os.makedirs(save_dir, exist_ok=True)
        read_dir = os.path.join(out_dir, f"{ari_dur}yrARI/{region_name}/{period}/ncfiles")
        
        # build list of expected filenames
        expected = [
            os.path.join(read_dir,
                         f"{var}_composite_{region_name}_{period}_lead{lt:+d}h.nc")
            for lt in lead_times
        ]

        # if any one is missing, skip this region/season
        if not all(os.path.isfile(fp) for fp in expected):
            print(f"   Missing composites for {region_name}/{period}, skipping")
            continue

        # rebuild the main composites dict
        comps_main = {}
        for lt in lead_times:
            fn = f"{var}_composite_{region_name}_{period}_lead{lt:+d}h.nc"
            print(f"Reading in {fn}...")
            ds_comp = xr.open_dataset(os.path.join(read_dir, fn))
            comps_main[lt] = ds_comp[var]        # now a DataArray
            ds_comp.close()
        
        # if you have u/v to plot as well, do the same
        if u_var:
            comps_u = {}
            for lt in lead_times:
                fn_u = f"{u_var}_composite_{region_name}_{period}_lead{lt:+d}h.nc"
                
                print(f"Reading in {fn_u}...")
                
                du = xr.open_dataset(os.path.join(read_dir, fn_u))
                
                comps_u[lt] = du[u_var]
                
                du.close()
                
        if v_var:
            comps_v = {}
            for lt in lead_times:
                fn_v = f"{v_var}_composite_{region_name}_{period}_lead{lt:+d}h.nc"
                
                print(f"Reading in {fn_v}...")
                
                dv = xr.open_dataset(os.path.join(read_dir, fn_v))
                
                comps_v[lt] = dv[v_var]
                
                dv.close()
                
        if var == "absv500":
            comps_hgt = {}
            for lt in lead_times:
                fn_hgt = f"hgt500_composite_{region_name}_{period}_lead{lt:+d}h.nc"
                
                print(f"Reading in {fn_hgt}...")
                
                dhgt = xr.open_dataset(os.path.join(read_dir, fn_hgt))
                
                comps_hgt[lt] = dhgt['hgt500']
                
                dhgt.close()
                
        if var in ["q850", "q925"]:
            for lt in lead_times:
                print("  Calculating moisture transport...")
                comps_main[lt] = instantaneous_moisture_flux(
                                    comps_main[lt],
                                    comps_u[lt],
                                    comps_v[lt]
                                 )

        # determine colorbar range
        if user_vmin is not None and user_vmax is not None:
            vmin, vmax, vstep = user_vmin, user_vmax, user_vstep
        else:
            all_vals = np.concatenate([c.values.ravel() for c in comps_main.values()])
            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
            vstep = (vmax - vmin) / 10.0

        # choose colormap
        if ncl:
            cmap = get_ncl_colormap(user_cmap, n_levels=256)
        elif color_list:
            cmap = LinearSegmentedColormap.from_list("Custom", color_list, N=256)
        elif user_cmap in ["GnBu", "PuBuGn"]:
            cmap = truncate_colormap(user_cmap, 0.2, 1.0)
        else:
            #cmap = get_cmap(user_cmap) if isinstance(user_cmap, str) else user_cmap
            cmap = colormaps.get(user_cmap) if isinstance(user_cmap, str) else user_cmap


        if var == "pwat":
            title = "precipitable water"
            var_label = "in"
            pres_lev = "850 mb"
        elif var == "cape":
            title = "Most Unstable CAPE"
            var_label = "J/kg"
        elif var == "q850":
            title = "Instantaneous Moisture Flux"
            var_label= r"g/kg m s$^{-1}$"
            pres_lev = "850 mb"
        elif var == "q925":
            title = "Instantaneous Moisture Flux"
            var_label= r"g/kg m s$^{-1}$"
            pres_lev = "925 mb"
        elif var == "u300":
            title = "Upper Level Winds"
            var_label = "mph"
            pres_lev = "300 mb"
        elif var == 'hgt500':
            title = "500 mb Geopotential Height"
            var_label = "height"
        elif var == 'absv500':
            title = 'Geopotential Height and Absolute Vorticity'
            pres_lev = "500 mb"
            var_label = r'10$^{-5}$s$^{-1}$'

        # plot panels
        fig, axes = plt.subplots(
            2, 2, figsize=(12, 10), sharex=True, sharey=True,
            subplot_kw={'projection': ccrs.PlateCarree()}
        )

        # compute map extents and coordinate arrays
        center_lon, center_lat = centroids[region_name]
        ny, nx = comps_main[lead_times[0]].sizes['y'], comps_main[lead_times[0]].sizes['x']
        lat_coords = center_lat + (np.arange(ny) - (ny//2)) * dlat
        lon_coords = center_lon + (np.arange(nx) - (nx//2)) * dlon

        for ax in axes.flatten():
            ax.coastlines(resolution='50m')
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            ax.add_feature(cfeature.STATES, linewidth=0.5)
            ax._autoscaleXon = False; ax._autoscaleYon = False
            
            ax.set_extent(
                [lon_coords.min(), lon_coords.max(),
                 lat_coords.min(), lat_coords.max()],
                crs=ccrs.PlateCarree()
            )

        fig.suptitle(fr"{ari_dur} year {title} {period} composite — {region_name}", fontsize=16, y=1.02)
        axes_flat = axes.flatten()
        levels = np.arange(vmin, vmax + vstep, vstep)

        for i, lt in enumerate(lead_times):
            ax = axes_flat[i]
            if var == "u300":
                comp = np.sqrt(comps_main[lt].values**2 + comps_v[lt].values**2) * 2.23694
            elif var == 'absv500':
                comp = comps_main[lt].values * 1e5
            else:
                comp = comps_main[lt].values
            X, Y = np.meshgrid(lon_coords, lat_coords)
            cf = ax.contourf(
                X, Y, comp,
                levels=levels,
                cmap=cmap,
                extend='both',
                transform=ccrs.PlateCarree()
            )

            # overlay wind in lon/lat space
            if u_var and v_var:
                uarr = comps_u[lt].values * 2.23694
                varr = comps_v[lt].values * 2.23694
                X_ll, Y_ll = np.meshgrid(lon_coords, lat_coords)
                Xs = X_ll[::wind_skip, ::wind_skip]; Ys = Y_ll[::wind_skip, ::wind_skip]
                Us = uarr[::wind_skip, ::wind_skip]; Vs = varr[::wind_skip, ::wind_skip]
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                sample_vals = comp[::wind_skip, ::wind_skip]
                rgba = cmap(norm(sample_vals))
                lum = (0.299*rgba[...,0] + 0.587*rgba[...,1] + 0.114*rgba[...,2]).mean()
                arrow_color = 'white' if lum < 0.3 else 'black'
                Q = ax.quiver(Xs, Ys, Us, Vs, color=arrow_color, scale=wind_scale, headwidth=5, width=0.003)
                if i == 1:
                    ax.quiverkey(Q, X=0.8, Y=1.05, U=20, label='20 mph', labelpos='E', fontproperties={'size': 12}, color='black')
            elif v_var and not u_var:
                uarr = comps_main[lt].values * 2.23694
                varr = comps_v[lt].values * 2.23694
                strm = ax.streamplot(
                    lon_coords, lat_coords,
                    uarr, varr,
                    transform=ccrs.PlateCarree(),
                    density=0.6,
                    linewidth=1,
                    arrowsize=1,
                    color="black",
                    #broken_streamlines=False
                )
            elif var == "absv500":
                hgt = comps_hgt[lt].values / 10
                hmin, hmax = np.nanmin(hgt), np.nanmax(hgt)

                all_levels_hgt = np.arange(500, 600, 6)
                levels_hgt     = all_levels_hgt[(all_levels_hgt >= hmin) & (all_levels_hgt <= hmax)]
                comp_hgt = comps_hgt[lt].values / 10
                cs = ax.contour(
                    X, Y, comp_hgt,
                    levels=levels_hgt,
                    colors='k',
                    linewidths=1,
                    transform=ccrs.PlateCarree()
                )
                ax.clabel(cs, fmt='%d', inline=True, fontsize=10)
                
            # center dot in lon/lat space
            ax.plot(center_lon, center_lat, 'o', color='red', mec='black', markersize=8, transform=ccrs.PlateCarree())

            # ticks and labels on outer axes only
            row, col = divmod(i, 2)
            x_off = tick_degs / dlon; y_off = tick_degs / dlat
            if col == 0:
                ax.set_yticks(lat_coords[(ny//2 + y_off).astype(int)])
                ax.set_yticklabels([f"{d:.1f}°" for d in tick_degs])
                ax.set_ylabel(r'$\Delta$ Latitude')
            else:
                ax.set_ylabel('')
            if row == 1:
                ax.set_xticks(lon_coords[(nx//2 + x_off).astype(int)])
                ax.set_xticklabels([f"{d:.1f}°" for d in tick_degs])
                ax.set_xlabel(r'$\Delta$ Longitude')
            else:
                ax.set_xticks([])

            if i == 0:
                ax.set_title("Event start")
            elif i == 1:
                ax.set_title("3 hours prior")
            elif i == 2:
                ax.set_title("6 hours prior")
            elif i == 3:
                ax.set_title("12 hours prior")
                #ax.set_title(f"Lead {lt:+d} h")

            if i == 0:
                if v_var:
                    ax.text(x=0, y=1.065, s=pres_lev, ha='left', va='top', fontsize=12, transform=ax.transAxes)

        # shared colorbar
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        #sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
        #sm.set_array([])
        fig.colorbar(cf, cax=cax, label=var_label, extend='both')

        # adjust layout
        fig.subplots_adjust(left=0.1, right=0.88, top=0.95, bottom=0.05, hspace=0.15, wspace=0.0)
        if out_dir:
            if var == "q850":
                fname = f"moist_trans850_2x2_panel_{ari_dur}yrARI_{region_name}_{period}.pdf"
            elif var == "q925":
                fname = f"moist_trans925_2x2_panel_{ari_dur}yrARI_{region_name}_{period}.pdf"
            else:
                fname = f"{var}_2x2_panel_{ari_dur}yrARI_{region_name}_{period}.pdf"
            fig.savefig(os.path.join(save_dir, fname), dpi=200, bbox_inches='tight')
        plt.close(fig)
        print("Plotted and saved panel!\n")
        #sys.exit()


if __name__ == '__main__':
    data_dir   = '/export/hpc-lw-director/malbright/working/HRRR'
    var        = 'absv500' 
    ari_dur    = 2 
    year       = 2024
    period     = 'ANN'
    lead_times = [0, -3, -6, -12]

    if var == "pwat":
        user_vmin  = 0.2
        user_vmax  = 2.2
        user_vstep = 0.1
    
        user_cmap  = 'precip4_11lev'
        ncl        = True
        color_list = None

        save_vars  = False 
        u_var      = 'u850'
        v_var      = 'v850'
        wind_skip  = 20
        wind_scale = 200
        
    if var == "cape":
        user_vmin  = 0.0
        user_vmax  = 2700
        user_vstep = 100
    
        user_cmap  = 'gist_ncar'
        ncl        = False
        color_list = None

        save_vars  = False 
        u_var      = None
        v_var      = None
        wind_skip  = 5
        wind_scale = None
       
    if var == "q850":
        user_vmin  = 0 
        user_vmax  = 350 
        user_vstep = 10 
    
        user_cmap  = 'jet'
        ncl        = False
        color_list = None

        save_vars  = False 
        u_var      = 'u850'
        v_var      = 'v850'
        wind_skip  = 20
        wind_scale = 200

    if var == "q925":
        user_vmin  = 0
        user_vmax  = 350
        user_vstep = 10

        user_cmap  = 'jet'
        ncl        = False
        color_list = None

        save_vars  = False
        u_var      = 'u925'
        v_var      = 'v925'
        wind_skip  = 20
        wind_scale = 200
        
    if var == "u300":
        user_vmin  = 5 
        user_vmax  = 120
        user_vstep = 5

        user_cmap  = 'bwr'
        ncl        = False
        color_list = None

        save_vars  = False
        u_var      = None 
        v_var      = 'v300'
        wind_skip  = 20
        wind_scale = 200

    if var == "hgt500":
        user_vmin  = 5000
        user_vmax  = 6000
        user_vstep = 10

        user_cmap  = 'bwr'
        ncl        = False
        color_list = None

        save_vars  = False
        u_var      = None
        v_var      = None
        wind_skip  = 20
        wind_scale = 200

    if var == "absv500":
        user_vmin  = 12
        user_vmax  = 30 
        user_vstep = 2 

        user_cmap  = 'WhiteYellowOrangeRed'
        ncl        = True 
        color_list = None

        save_vars  = False
        u_var      = None
        v_var      = None
        wind_skip  = 20
        wind_scale = 200
        
    out_dir    = '/export/hpc-lw-director/malbright/working/composites'

    regions = {
        'West': (-125,-117, 32,49),
        'Southwest': (-117,-104, 28,42),
        'NorthernPlains': (-104,-85, 38,49),
        'SouthernPlains': (-104,-90, 24,38),
        'Southeast': (-90,-75, 24,38),
        'Northeast': (-85,-66, 38,49),
    }
    regions = {k:{'lons':(v[0],v[1]), 'lats':(v[2],v[3])} for k,v in regions.items()}

    seasons = {
        'ANN': list(range(1,13)), 'DJF':[12,1,2],
        'MAM':[3,4,5], 'JJA':[6,7,8], 'SON':[9,10,11]
    }
    

    for season in seasons.keys():
        make_composites(
            data_dir, var, ari_dur, year, season, lead_times,
            regions, seasons,
            user_cmap, ncl, save_vars, user_vmin, user_vmax, user_vstep,
            out_dir, color_list,
            u_var, v_var, wind_skip, wind_scale
        )
