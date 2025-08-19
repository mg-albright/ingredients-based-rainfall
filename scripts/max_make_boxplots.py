# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
boxplot_hrrr_centroid_max.py

Generate boxplots of a given HRRR variable at the centroid point,
using the maximum grid value instead of the center point,
for ARI durations 2 and 100, across specified hours, regions, and seasons.

Produces three sets of figures saved as PDF:
 1) One file per season: panels = regions (2x3 grid), boxplots of value vs. hour, ARI=2 vs. 100.
 2) One file per region: panels = seasons (2x3 grid but customized layout), boxplots of value vs. hour, ARI=2 vs. 100.
 3) One file per season: panels = hours (2x2 grid), boxplots of value vs. region, ARI=2 vs. 100.

Usage:
    python boxplot_hrrr_centroid_max.py --variable pwat

Outputs:
    /export/hpc-lw-director/malbright/working/boxplots/*.pdf
"""

import os
import argparse
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plotting_functions import instantaneous_moisture_flux

# Parameters
ARI_DURS = [2, 100]
HOURS = [0, -3, -6, -12]
REGIONS = {
    'West': (-125, -117, 32, 49),
    'Southwest': (-117, -104, 28, 42),
    'NorthernPlains': (-104, -85, 38, 49),
    'SouthernPlains': (-104, -90, 24, 38),
    'Southeast': (-90, -75, 24, 38),
    'Northeast': (-85, -66, 38, 49),
}
SEASONS = {
    'ANN': list(range(1, 13)),
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}

INPUT_BASE = '/export/hpc-lw-director/malbright/working/HRRR'
OUTPUT_DIR = '/export/hpc-lw-director/malbright/working/boxplots'
MM_TO_INCH = 0.0393701

# Map months only to non-annual seasons
def month_to_season_map(seasons):
    m2s = {}
    for s, months in seasons.items():
        if s == 'ANN':
            continue
        for m in months:
            m2s[m] = s
    return m2s


def assign_region(lat, lon, regions):
    for name, (lo_min, lo_max, la_min, la_max) in regions.items():
        if lo_min <= lon <= lo_max and la_min <= lat <= la_max:
            return name
    return None

# Pretty names for variables
PRETTY = {'pwat':    'Precipitable Water',
          'cape':    'Most Unstable CAPE',
          'wspd850': '850 mb Wind Speed',
          'wspd925': '925 mb Wind Speed',
          'wspd300': '300 mb Wind Speed',
          'mt850':   '850 mb Moisture Flux',
          'mt925':   '925 mb Moisture Flux'
        }

# Collect data event-by-event, using the maximum grid value

def collect_data(variable):
    records = []
    m2s = month_to_season_map(SEASONS)

    for ari in ARI_DURS:
        is_wspd = variable.startswith("wspd")
        is_mt = variable.startswith("mt")
        # Identify component variable names
        if is_wspd:
            level = variable.replace("wspd", "")
            u_name, v_name = f"u{level}", f"v{level}"
        elif is_mt:
            level = variable.replace("mt", "")
            q_name, u_name, v_name = f"q{level}", f"u{level}", f"v{level}"
        subdir = os.path.join(INPUT_BASE, variable, f'{ari}yrARI')

        for month in range(1, 13):
            season = m2s.get(month)
            if season is None:
                continue

            # Open datasets
            if is_wspd or is_mt:
                ds_u = xr.open_dataset(os.path.join(INPUT_BASE, u_name, f"{ari}yrARI",
                                   f"hrrr_3h{ari}yrARI_{u_name}_2024{month:02d}.nc"))
                ds_v = xr.open_dataset(os.path.join(INPUT_BASE, v_name, f"{ari}yrARI",
                                   f"hrrr_3h{ari}yrARI_{v_name}_2024{month:02d}.nc"))
                if is_mt:
                    ds_q = xr.open_dataset(os.path.join(INPUT_BASE, q_name, f"{ari}yrARI",
                                       f"hrrr_3h{ari}yrARI_{q_name}_2024{month:02d}.nc"))
            else:
                path = os.path.join(subdir, f'hrrr_3h{ari}yrARI_{variable}_2024{month:02d}.nc')
                if not os.path.isfile(path):
                    print(f"Warning: file not found {path}")
                    continue
                ds = xr.open_dataset(path)

            times = ds_u['time'].values if (is_wspd or is_mt) else ds['time'].values

            for evt in range((ds_u if (is_wspd or is_mt) else ds).dims['event']):
                lat = float((ds_u if is_wspd else ds_q if is_mt else ds)['centroid_lat'].isel(event=evt).values)
                lon = float((ds_u if is_wspd else ds_q if is_mt else ds)['centroid_lon'].isel(event=evt).values)
                region = assign_region(lat, lon, REGIONS)
                if region is None:
                    continue

                for hr in HOURS:
                    if hr not in times:
                        continue
                    t_index = int(np.where(times == hr)[0])

                    if is_wspd:
                        u_slice = ds_u[u_name].isel(event=evt, time=t_index)
                        v_slice = ds_v[v_name].isel(event=evt, time=t_index)
                        speed = np.sqrt(u_slice**2 + v_slice**2) * 2.23694
                        val = float(speed.max().values)

                    elif is_mt:
                        q_slice = ds_q[q_name].isel(event=evt, time=t_index)
                        u_slice = ds_u[u_name].isel(event=evt, time=t_index)
                        v_slice = ds_v[v_name].isel(event=evt, time=t_index)
                        mf = instantaneous_moisture_flux(q_slice, u_slice, v_slice)
                        val = float(mf.max().values)

                    else:
                        da = ds[variable].isel(event=evt, time=t_index)
                        val = float(da.max().values)

                    if np.isnan(val):
                        continue
                    if variable == 'pwat':
                        val *= MM_TO_INCH

                    records.append({
                        'ari_dur': ari,
                        'region': region,
                        'season': season,
                        'hour': hr,
                        'value': val
                    })

            # Close datasets
            if is_wspd or is_mt:
                ds_u.close()
                ds_v.close()
                if is_mt:
                    ds_q.close()
            else:
                ds.close()

    return pd.DataFrame.from_records(records)

# Styling
BOX_WIDTH = 0.2
GAP = 0.1
MEDIANPROPS = {'color': 'black', 'linewidth': 1.5}
FLIERPROPS = {'marker': 'o', 'markerfacecolor': 'black', 'markeredgecolor': 'black', 'markersize': 3, 'linestyle': 'none'}

# Helpers

def get_ylabel(variable):
    labels = {
        'pwat':    'inches',
        'cape':    'J/kg',
        'wspd850': 'mph',
        'wspd925': 'mph',
        'wspd300': 'mph',
        'mt850':   r'g/kg m s$^{-1}$',
        'mt925':   r'g/kg m s$^{-1}$'
    }
    return labels.get(variable, variable)

def pretty_var(variable):
    return PRETTY.get(variable, variable.replace('_', ' ').title())

# Plot 1: by region
def plot_by_region(df, variable):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    seasons = list(SEASONS.keys())
    regions = list(REGIONS.keys())
    x = np.arange(len(HOURS))

    for season in seasons:
        df_s = df[df['season'] == season]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        axes_flat = axes.flatten()
        
        for ax, region in zip(axes_flat, regions):
            df_sr = df_s[df_s['region'] == region]
            for k, ari in enumerate(ARI_DURS):
                data_k = [df_sr[(df_sr.hour == hr) & (df_sr.ari_dur == ari)]['value'].values
                          for hr in HOURS]
                pos = x + (k - 0.5) * BOX_WIDTH + GAP
                bplot = ax.boxplot(data_k, positions=pos, widths=BOX_WIDTH,
                                   patch_artist=True, medianprops=MEDIANPROPS,
                                   flierprops=FLIERPROPS)
                for patch in bplot['boxes']:
                    patch.set_facecolor(f'C{k}')
            ax.set_title(region)
            ax.set_xticks(x)
            ax.set_xticklabels([str(h) for h in HOURS])
            ax.set_xlabel('Hour offset')
            ax.set_ylabel(get_ylabel(variable))
            if region == regions[0]:
                handles = [mpatches.Patch(facecolor=f'C{k}', label=f'{ari} year')
                           for k, ari in enumerate(ARI_DURS)]
                ax.legend(handles=handles, title='ARI duration')
        for ax in axes_flat[len(regions):]:
            fig.delaxes(ax)

        fig.suptitle(f"Maximum {pretty_var(variable)} by Region — Season {season}", fontsize=16)
        fig.savefig(os.path.join(OUTPUT_DIR, f'{variable}/max_{variable}_season_{season}_byRegion.pdf'))
        plt.close(fig)

# Plot 2: by season (custom layout)
def plot_by_season(df, variable):
    regions = list(REGIONS.keys())
    season_pos = {'ANN': 0, 'DJF': 1, 'MAM': 2, 'JJA': 4, 'SON': 5}

    for region in regions:
        df_r = df[df['region'] == region]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
        axes_flat = axes.flatten()
        x = np.arange(len(HOURS))

        for season, pos in season_pos.items():
            ax = axes_flat[pos]
            df_rs = df_r[df_r['season'] == season]
            for k, ari in enumerate(ARI_DURS):
                data_k = [df_rs[(df_rs.hour == hr) & (df_rs.ari_dur == ari)]['value'].values
                          for hr in HOURS]
                p = x + (k - 0.5) * BOX_WIDTH + GAP
                bplot = ax.boxplot(data_k, positions=p, widths=BOX_WIDTH,
                                   patch_artist=True, medianprops=MEDIANPROPS,
                                   flierprops=FLIERPROPS)
                for patch in bplot['boxes']:
                    patch.set_facecolor(f'C{k}')
            ax.set_title(season)
            ax.set_xticks(x)
            ax.set_xticklabels([str(h) for h in HOURS])
            ax.set_xlabel('Hour offset')
            ax.set_ylabel(get_ylabel(variable))
            if season == 'ANN':
                handles = [mpatches.Patch(facecolor=f'C{k}', label=f'{ari} year')
                           for k, ari in enumerate(ARI_DURS)]
                ax.legend(handles=handles, title='ARI duration')
        # remove unused subplot at index 3
        fig.delaxes(axes_flat[3])

        fig.suptitle(f"Maximum {pretty_var(variable)} by Season — Region {region}", fontsize=16)
        fig.savefig(os.path.join(OUTPUT_DIR, f'max_{variable}_region_{region}_bySeason.pdf'))
        plt.close(fig)

# Plot 3: by hour offset
def plot_by_hour(df, variable):
    seasons = list(SEASONS.keys())
    regions = list(REGIONS.keys())

    for season in seasons:
        df_s = df[df['season'] == season]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes_flat = axes.flatten()
        x = np.arange(len(regions))

        for ax, hr in zip(axes_flat, HOURS):
            df_sh = df_s[df_s['hour'] == hr]
            for k, ari in enumerate(ARI_DURS):
                data_k = [df_sh[(df_sh.region == reg) & (df_sh.ari_dur == ari)]['value'].values
                          for reg in regions]
                pos = x + (k - 0.5) * BOX_WIDTH + GAP
                bplot = ax.boxplot(data_k, positions=pos, widths=BOX_WIDTH,
                                   patch_artist=True, medianprops=MEDIANPROPS,
                                   flierprops=FLIERPROPS)
                for patch in bplot['boxes']:
                    patch.set_facecolor(f'C{k}')
            ax.set_title(f'Hour {hr}')
            ax.set_xticks(x)
            ax.set_xticklabels(regions, rotation=45, ha='right')
            ax.set_xlabel('Region')
            ax.set_ylabel(get_ylabel(variable))
            if hr == HOURS[0]:
                handles = [mpatches.Patch(facecolor=f'C{k}', label=f'{ari} year')
                           for k, ari in enumerate(ARI_DURS)]
                ax.legend(handles=handles, title='ARI duration')

        fig.suptitle(f"Maximum {pretty_var(variable)} by Hour offset — Season {season}", fontsize=16)
        fig.savefig(os.path.join(OUTPUT_DIR, f'max_{variable}_season_{season}_byHour.pdf'))
        plt.close(fig)


def main(variable):
    print(f"Collecting data for variable '{variable}'...")
    df = collect_data(variable)
    if df.empty:
        print("No data found. Exiting.")
        return

    # Add annual (ANN) aggregation: duplicate all records with season='ANN'
    df_ann = df.copy()
    df_ann['season'] = 'ANN'
    df = pd.concat([df, df_ann], ignore_index=True)

    print("Generating plots by region...")
    plot_by_region(df, variable)
    print("Generating plots by season...")
    plot_by_season(df, variable)
    print("Generating plots by hour offset...")
    plot_by_hour(df, variable)
    print("Done. PDFs saved in:", OUTPUT_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make boxplots of HRRR variable at centroids.')
    parser.add_argument('--variable', default='pwat', help='Variable name (e.g. pwat)')
    args = parser.parse_args()
    main(args.variable)
