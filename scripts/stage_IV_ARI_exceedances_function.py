#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#############################################################################################
#
# Script: ari_pipeline.py
# Author: Erica Bower and Mary Grace Albright
# Purpose: Find 3-hr, 2-year ARI exceedances in Stage IV data, generate binary masks,
#          and run MET gen_vx_mask + MODE for each timestep.
# Modified: 2025-07 by MGA — added triangulation caching and month-by-month driver.
#############################################################################################
print("Loading packages...\n")
import os
import sys
import pickle
import subprocess
import datetime
import numpy as np
import pandas as pd
import pygrib
from netCDF4 import Dataset
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
from dateutil.relativedelta import relativedelta
import functions_Lapenta_ingredients

# --- CONFIGURATION -----------------------------------------------------------
# Paths (adjust as needed)
WORKING        = '/export/hpc-lw-director/malbright/working/'
ST4_PATH       = '/export/hpc-lw-dtbdev5/wpc_cpgffh/gribs/ST4/'
#ARI_PATH       = '/export/hpc-lw-hmtdev3/ebower/Atlas14_Oct_2024/'
ARI_PATH       = '/backup/ebower/Atlas14_Oct_2024/'
MET_PATH       = '/opt/MET/METPlus4_0/bin/'
MET_FILES_PATH = '/export/hpc-lw-director/malbright/met_exec_files/'

# ARI settings
ARI_year       = 2                       # 2-year ARI
ARI_DUR        = 3                       # 3-hour accumulation
ARI_DURs       = ['60m','02h','03h','06h','12h']

# Caching triangulation
TRI_PICKLE     = os.path.join(WORKING, 'triangulation.pkl')

# Load CONUS land mask and target grid
conus_file = os.path.join(WORKING, 'CONUS_9km.nc')
with Dataset(conus_file, 'r') as df:
    CONUSmask = df.variables['CONUSmask'][:]
    lats_c     = df.variables['lat'][:]
    lons_c     = df.variables['lon'][:]

# Ensure working dirs exist
for p in (WORKING, ST4_PATH, ARI_PATH):
    assert os.path.isdir(p), f"Path not found: {p}"

# --- MAIN PIPELINE FUNCTION -------------------------------------------------
def run_ari_pipeline(start_date, end_date, month, year):
    """
    Process Stage IV and ARI data from start_date → end_date (inclusive),
    generate exceedance txt/nc files, and run gen_vx_mask + MODE.
    """
    # Build date/hour lists
    dates = pd.date_range(start=start_date, end=end_date, freq='1D')
    hours = np.arange(0, 24)
    nt     = len(dates) * len(hours)

    # Pre-allocate Stage IV array
    stage_iv = np.zeros((lats_c.shape[0], lats_c.shape[1], nt))
    tri      = None
    missing_dates = []

    # Read Stage IV, interpolate to target grid
    for di, dt in enumerate(dates):
        date_str = dt.strftime('%Y%m%d')
        print(f"--- Processing Stage IV for {date_str}")
        # copy & unzip raw ST4 files
        day_dir = os.path.join(WORKING, date_str)
        os.system(f"cp -r {ST4_PATH}{date_str} {WORKING}")
        os.system(f"gunzip {day_dir}/*.gz")

        # build or load triangulation
        if tri is None:
            if os.path.exists(TRI_PICKLE):
                with open(TRI_PICKLE, 'rb') as f:
                    tri = pickle.load(f)
                print("Loaded cached triangulation.")
            else:
                sample = os.path.join(day_dir, f"ST4.{date_str}00.01h")
                with pygrib.open(sample) as f:
                    msg = f.select()[0]
                    lats_src, lons_src = msg.latlons()
                pts_src = np.stack((lons_src.ravel(), lats_src.ravel()), axis=1)
                tri = Delaunay(pts_src)
                with open(TRI_PICKLE, 'wb') as f:
                    pickle.dump(tri, f)
                print("Built and cached triangulation.")

        # hour-by-hour interpolation
        for hi, hr in enumerate(hours):
            h_str = f"{hr:02d}"
            fname = os.path.join(day_dir, f"ST4.{date_str}{h_str}.01h")
            if not os.path.exists(fname):
                print(f"  Missing '{fname}'. Skipping...")
                missing_dates.append(date_str + h_str)
            else:
                print(f"  Reading {fname}")
                with pygrib.open(fname) as f:
                    msg = f.select()[0]
                    st4 = msg.values
                st4_flat = np.where(st4 > 9000, np.nan, st4).ravel()

                interp = LinearNDInterpolator(tri, st4_flat, fill_value=np.nan)
                regr   = interp(np.stack((lons_c.ravel(), lats_c.ravel()), axis=1))
                regr   = regr.reshape(lats_c.shape)

                # store
                ti = di * len(hours) + hi
                stage_iv[:, :, ti] = regr

        # cleanup
        os.system(f"rm -rf {day_dir}")

    print("Completed Stage IV load & interpolation.")
    
    missing_txt = f"missing_ST4_files_{year}{month:02d}.txt"
    missing_fp = os.path.join(WORKING, missing_txt)
    if missing_dates:
        with open(missing_fp, 'w') as f:
            f.write('\n'.join(missing_dates))

    # Read ARI climatology into array
    print("Reading ARI climatology...")
    for idx, durstr in enumerate(ARI_DURs):
        fn = os.path.join(ARI_PATH, f"nationalAtlas14_{ARI_year}yr{durstr}a_REGRID_mpd.nc")
        with Dataset(fn, 'r') as df:
            var = df.variables['ARI'][:]
        var = np.where(var == -9999, np.nan, var)
        if idx == 0:
            ARI = np.empty((var.shape[0], var.shape[1], len(ARI_DURs)))
        ARI[:, :, idx] = var
    ARI = ARI / 0.039  # convert to mm

    # Compute exceedances
    print("Calculating ARI exceedances...")
    ST4gARI = functions_Lapenta_ingredients.calc_ST4gARI_duration(stage_iv, ARI, ARI_DUR)
    print(f"Unique values in ST4gARI: {np.unique(ST4gARI)}")

    # Write outputs and call external tools
    txt_dir  = os.path.join(WORKING, f"ST4g_ARI_txt/{ARI_year}yr_ARI/{ARI_DUR}h/{year}/{month:02d}")
    nc_dir   = os.path.join(WORKING, f"gen_vx_mask_out/{ARI_year}yr_ARI/{ARI_DUR}h/{year}/{month:02d}")
    mode_dir = os.path.join(WORKING, f"mode_out/{ARI_year}yr_ARI/{ARI_DUR}h/{year}/{month:02d}")
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(nc_dir, exist_ok=True)
    os.makedirs(mode_dir, exist_ok=True)

    for t in range(ST4gARI.shape[2]):
        i_pts, j_pts = np.where(ST4gARI[:, :, t])
        if i_pts.size == 0:
            continue
        day_i = t // len(hours)
        hr_i  = t % len(hours)
        dt    = dates[day_i]
        dt_str= dt.strftime('%Y%m%d')
        h_str = f"{hours[hr_i]:02d}"
        dtstamp = f"{dt_str}{h_str}"

        # txt
        txt_fn = f"ST4g_ARI_{dtstamp}_{ARI_DUR}h{ARI_year}yrARI.txt"
        txt_fp = os.path.join(txt_dir, txt_fn)
        with open(txt_fp, 'w') as f:
            f.write("ST4gARI\n")
            for i,j in zip(i_pts, j_pts):
                lat = lats_c[i,j]
                lon = lons_c[i,j]
                f.write(f"{lat}-{abs(lon)}\n")

        # nc via gen_vx_mask
        nc_fn = f"ST4g_ARI_{dtstamp}_{ARI_DUR}h{ARI_year}yrARI.nc"
        nc_fp = os.path.join(nc_dir, nc_fn)
        cmd1 = (
            f"{MET_PATH}gen_vx_mask "
            f"-type circle -thresh '<=20' "
            f"{MET_FILES_PATH}dummy_usgs.nc "
            f"{txt_fp} "
            f"-name ST4gARI "
            f"{nc_fp}"
        )
        print(f"Running: {cmd1}")
        subprocess.call(cmd1, shell=True)

        # MODE
        cmd2 = (
            f"{MET_PATH}mode {nc_fp} {nc_fp} {MET_FILES_PATH}MODEconfig_nc"
        )
        print(f"Running: {cmd2}")
        subprocess.call(cmd2, shell=True)

        # move MODE outputs
        for mf in os.listdir('.'):
            if mf.startswith('mode_ST4gARI_'):
                base, ext = os.path.splitext(mf)
                new = f"{base}_{ARI_DUR}h{ARI_year}yrARI_{dtstamp}{ext}"
                os.rename(mf, os.path.join(mode_dir, new))

    print(f"Finished pipeline for {start_date.date()} → {end_date.date()}")


# --- DRIVER: Loop over months -----------------------------------------------
if __name__ == '__main__':
    # process all months -- start with 2024
    #months = [(2024, m) for m in [1, 2, 10, 11, 12]]
    months = [(2023, m) for m in range(8, 13)]
    #months = [(2024, m) for m in [9]]
    for year, month in months:
        sd = datetime.datetime(year, month, 1)
        ed = sd + relativedelta(months=1) - relativedelta(days=1)
        print(f"\n=== Running ARI pipeline for {year}-{month:02d} ===")
        run_ari_pipeline(sd, ed, month, year)
