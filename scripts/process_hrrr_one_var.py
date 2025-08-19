#!/usr/bin/env python3
"""
process_hrrr.py

Author: Mary Grace Albright

New workflow:
 - For each MODE file, read CO0* centroids.
 - For each centroid, extract a fixed-size box at five lead times before the event.
 - Label the time dimension in the output as [0, -3, -6, -9, -12] hours to indicate hours before the exceedance start.
 - Accumulate these boxes per calendar month into one netCDF per month with dimensions (event, time, y, x).
 - Events are uniquely named "YYYYMMDDHH_objID".
 - Store centroid lat/lon per event; store grid spacing metadata for reconstructed coordinates.

Dependencies:
  - numpy, pygrib, pandas, xarray
  - urllib.request.urlretrieve
  - datetime
  - subprocess, ThreadPoolExecutor

Modified: MGA 2025 July 18
"""
print("Loading packages...")
import os
import glob
import subprocess
import sys
from datetime import datetime, timedelta
from urllib.request import urlretrieve
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pygrib
import xarray as xr

# ---------------- USER PARAMETERS ---------------- #
ARI_YEAR = 2 
ARI_DUR = 3
MIN_AREA = 23

MODEL           = 'hrrr'
VAR_SHORT_NAME  = 'ustm'
MODE_DIR        = f'/export/hpc-lw-director/malbright/working/mode_out/{ARI_YEAR}yr_ARI/{ARI_DUR}h'
OUTDIR          = '/export/hpc-lw-director/malbright/working/HRRR'

os.makedirs(OUTDIR, exist_ok=True)

MAX_OFFSETS = [3, 6, 9, 15]
TIME_LABELS = [0, -3, -6, -12]
DELTA_DEG = 2.5

IDX_LABELS = {
    'pwat'  : 'PWAT:entire atmosphere',
    'cape'  : 'CAPE:255-0 mb above ground',
    'u300'  : 'UGRD:300 mb', 'v300': 'VGRD:300 mb',
    'u850'  : 'UGRD:850 mb', 'v850': 'VGRD:850 mb',
    'u925'  : 'UGRD:925 mb', 'v925': 'VGRD:925 mb',
    'q850'  : 'SPFH:850 mb', 'q925': 'SPFH:925 mb',
    'ustm'  : 'USTM:0-6000 m above ground', 'vstm': 'VSTM:0-6000 m above ground',
    'hgt500': 'HGT:500 mb', 'absv500': 'ABSV:500 mb' 
}

# Cache for downloaded arrays and lat/lon grids
CACHE = {}
LATS, LONS = None, None

# ---------------- Helper Functions ---------------- #
def select_grib_msg_full_file(grbs, short_name):
    if short_name == 'pwat':
        msgs = grbs.select(shortName=short_name)
    elif short_name == 'cape':
        msgs = grbs.select(shortName=short_name, typeOfLevel='pressureFromGroundLayer')
    else:
        msgs = grbs.select(shortName=short_name)
    if not msgs:
        raise ValueError(f"No GRIB messages with shortName={short_name}")
    if len(msgs) > 1:
        print(f"Warning: {len(msgs)} msgs for '{short_name}', using first")
    return msgs[0]


def select_grib_msg(grbs, short_name):
    msgs = grbs.select()[:]
    if not msgs:
        raise ValueError(f"File empty: no GRIB messages")
    if len(msgs) > 1:
        print(f"Warning: {len(msgs)} msgs for '{short_name}', using first")
    return msgs[0]


def find_byte_range(date_dt, grid, var_short):
    key = date_dt.strftime('%Y%m%d%H')
    idx_url = (
        f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
        f"{MODEL}.{date_dt:%Y%m%d}/conus/"
        f"{MODEL}.t{date_dt:%H}z.wrf{grid}f00.grib2.idx"
    )
    idx_local = os.path.join(OUTDIR, f"{key}.idx")
    urlretrieve(idx_url, idx_local)
    label = IDX_LABELS.get(var_short)
    if label is None:
        os.remove(idx_local)
        raise KeyError(f"No IDX label defined for variable '{var_short}'")

    start = end = None
    with open(idx_local, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if label in line:
            parts = line.strip().split(':')
            start = int(parts[1])
            next_parts = lines[i+1].strip().split(':')
            end = int(next_parts[1])
            print(f"  Byte-range for {label} @ {date_dt}: start={start}, end={end}")
            break

    os.remove(idx_local)
    if start is None or end is None:
        raise RuntimeError(f"Could not find byte range for {label}")
    return start, end


def download_and_cache(dt):
    global CACHE, LATS, LONS

    key = dt.strftime('%Y%m%d%H')
    if key in CACHE:
        return

    grid = 'nat' if VAR_SHORT_NAME in ('pwat','cape','ustm','vstm') else 'prs'
    start, end = find_byte_range(dt, grid, VAR_SHORT_NAME)

    slice_name = f"{MODEL}.t{dt:%H}z.wrf{grid}f00.{VAR_SHORT_NAME}.{key}.grib2"
    slice_path = os.path.join(OUTDIR, slice_name)
    full_url = (
        f"https://noaa-hrrr-bdp-pds.s3.amazonaws.com/"
        f"{MODEL}.{dt:%Y%m%d}/conus/"
        f"{MODEL}.t{dt:%H}z.wrf{grid}f00.grib2"
    )

    cmd = [
        'curl', full_url,
        '--range', f'{start}-{end}',
        '--silent',
        '--output', slice_path
    ]
    subprocess.run(cmd, check=True)

    with pygrib.open(slice_path) as grbs:
        msg = select_grib_msg(grbs, VAR_SHORT_NAME)
        arr = msg.values
        if LATS is None:
            LATS, LONS = msg.latlons()

    os.remove(slice_path)
    CACHE[key] = arr


def compute_five_hr_dataset(valid_dt):
    """
    Download the five required offsets in parallel, then build an xarray Dataset
    with full 2D lat/lon coords.
    """
    dts = [valid_dt - timedelta(hours=o) for o in MAX_OFFSETS]
    with ThreadPoolExecutor(max_workers=len(dts)) as executor:
        list(executor.map(download_and_cache, dts))  # force exceptions

    arrays = [CACHE[dt.strftime('%Y%m%d%H')] for dt in dts]
    arr = np.stack(arrays, axis=0)

    ds = xr.Dataset(
        {VAR_SHORT_NAME: (('time','y','x'), arr)},
        coords={
            'time': ('time', TIME_LABELS),
            'y': np.arange(arr.shape[1]),
            'x': np.arange(arr.shape[2]),
            'lat': (('y','x'), LATS),
            'lon': (('y','x'), LONS)
        }
    )
    ds.attrs['delta_deg'] = DELTA_DEG
    return ds


def compute_grid_params():
    lat1d = LATS[:,0]
    lon1d = LONS[0,:]
    dlat = float(np.mean(np.diff(lat1d)))
    dlon = float(np.mean(np.diff(lon1d)))
    ni = int(round(DELTA_DEG / dlat))
    nj = int(round(DELTA_DEG / dlon))
    return ni, nj, dlat, dlon


def extract_box_array(ds, cent_lat, cent_lon, ni, nj):
    """
    Extract a fixed-size box around the closest grid cell on the 2D lat/lon grid.
    """
    data = ds[VAR_SHORT_NAME].values  # shape (time, y, x)
    # 2D nearest neighbor
    dist2 = (ds['lat'].values - cent_lat)**2 + (ds['lon'].values - cent_lon)**2
    i0, j0 = np.unravel_index(dist2.argmin(), dist2.shape)

    i1, i2 = i0 - ni, i0 + ni
    j1, j2 = j0 - nj, j0 + nj
    out = np.full((data.shape[0], 2*ni+1, 2*nj+1), np.nan, dtype=data.dtype)

    si1, si2 = max(0, i1), min(data.shape[1], i2+1)
    sj1, sj2 = max(0, j1), min(data.shape[2], j2+1)
    ti1 = si1 - i1
    tj1 = sj1 - j1
    ti2 = ti1 + (si2 - si1)
    tj2 = tj1 + (sj2 - sj1)

    out[:, ti1:ti2, tj1:tj2] = data[:, si1:si2, sj1:sj2]
    return out


def accumulate_monthly(monthly, event_id, arr5, clat, clon):
    monthly['arr'].append(arr5)
    monthly['id'].append(event_id)
    monthly['clat'].append(clat)
    monthly['clon'].append(clon)


def save_monthly(year, month, data, ni, nj, dlat, dlon):
    arrs = np.stack(data['arr'], axis=0)
    events = np.array(data['id'], dtype='S')
    ds = xr.Dataset(
        {VAR_SHORT_NAME: (('event','time','y','x'), arrs),
         'centroid_lat': (('event',), data['clat']),
         'centroid_lon': (('event',), data['clon'])},
        coords={
            'event': events,
            'time': TIME_LABELS,
            'y': np.arange(arrs.shape[2]),
            'x': np.arange(arrs.shape[3])
        }
    )
    ds.attrs.update({'dlat': dlat, 'dlon': dlon, 'ni': ni, 'nj': nj, 'delta_deg': DELTA_DEG})
    ds[VAR_SHORT_NAME] = ds[VAR_SHORT_NAME].astype('float32')
    encoding = {var: {'zlib': True, 'complevel': 4, 'shuffle': True} for var in ds.data_vars}
    outdir_var = os.path.join(OUTDIR, f"{VAR_SHORT_NAME}/{ARI_YEAR}yrARI")
    os.makedirs(outdir_var, exist_ok=True)
    out = os.path.join(
        outdir_var,
        f"hrrr_{ARI_DUR}h{ARI_YEAR}yrARI_{VAR_SHORT_NAME}_{year}{month:02d}.nc"
    )
    ds.to_netcdf(out, engine='netcdf4', encoding=encoding)


def find_mode_files(year, month):
    pattern = os.path.join(MODE_DIR, f"{year}/{month:02d}/*_obj_*.txt")
    return sorted(glob.glob(pattern))


def parse_centroids(mode_file):
    df = pd.read_csv(mode_file, delim_whitespace=True)
    df = df[df['OBJECT_ID'].str.startswith('CO0')]
    if ARI_YEAR == 2:
        df = df[df['AREA'] >= MIN_AREA]
    return df[['OBJECT_ID','CENTROID_LAT','CENTROID_LON']]

# ---------------- Main ---------------- #
def main(year, month, test_slice=None):
    files = find_mode_files(year, month)
    if test_slice:
        files = files[test_slice]
    monthly = {'arr':[], 'id':[], 'clat':[], 'clon':[]}
    ni = nj = dlat = dlon = None
    for mode_file in files:
        print(f"\nProcessing {mode_file}...")
        ts = os.path.basename(mode_file).split('_')[-1].replace('.txt','')
        valid_dt = datetime.strptime(ts, '%Y%m%d%H')
        creds = parse_centroids(mode_file)
        if creds.empty:
            print("No centroids found, skipping...")
            continue
        print(f"Parsed {len(creds)} centroids.")
        ds = compute_five_hr_dataset(valid_dt)
        if ni is None:
            ni, nj, dlat, dlon = compute_grid_params()
        for _, row in creds.iterrows():
            arr5    = extract_box_array(ds, row['CENTROID_LAT'], row['CENTROID_LON'], ni, nj)
            event_id = f"{ts}_{row['OBJECT_ID']}"
            accumulate_monthly(monthly, event_id, arr5, row['CENTROID_LAT'], row['CENTROID_LON'])
    save_monthly(year, month, monthly, ni, nj, dlat, dlon)
    print(f"Saved month {month}!\n")


if __name__ == "__main__":
    year = 2024
    #run_tests()
    for month in range(8,13):
    #for month in range(1,8):
        main(year, month)
