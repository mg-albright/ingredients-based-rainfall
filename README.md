# ingredients-based-rainfall
Scripts from Lapenta NOAA Internship -- summer 2025

Workflow (all files in scripts/): 
1. stage_IV_ARI_exceedances_function.py
	- Stage IV exceedances here: /export/hpc-lw-director/malbright/working/ST4g_ARI_txt/
	- gen_vx_mask output here: /export/hpc-lw-director/malbright/working/gen_vx_mask_out/
	- MODE output here: /export/hpc-lw-director/malbright/working/mode_out/
2. process_hrrr_one_var.py
	- HRRR associated with MODE output here: /export/hpc-lw-director/malbright/working/HRRR/
3. overlay_plot_composite_panels.py
	- selecting save_vars=True will calculate composite and save netcdf files but not plot
	- save_vars=False assumes you already have composites calculated and will try to read in the saved composites
	- outputted figures here: /export/hpc-lw-director/malbright/working/composites/ 
4. make_boxplots.py OR max_make_boxplots.py
   	- No need to change anything inside the script, unless you want to change the plotting aesthetics 
   		- Run as: python make_boxplots.py --variable variable 
   			- possible variables: pwat, cape, wspd300 (wind speed at 300 mb), wspd925, wspd850, mt925 (moisture transport at 925 mb), or mt850
	- outputted figures here: /export/hpc-lw-director/malbright/working/boxplots
