#!/usr/bin/python

###################################################################################
# Functions needed for calculating Stage IV exceedances of ARIs or FFG
###################################################################################

import pygrib
import subprocess
import time
import os
import numpy as np
import datetime
import time
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from netCDF4 import Dataset
from scipy import interpolate
import cartopy.feature as cf
import cartopy.crs as ccrs
import cartopy_maps

#########################################################################################
#########################################################################################
#########################################################################################

#########################################################################################
# New function for calculating the St4 exceedances of ARI. 
# A straightforward calculation with documentation and comments to accompany.
# Just read in the hourly data, step through hour by hour to find all the exceedances,
# and assign exceedances to the hour in which they occurred. NOT the start hour.
# Output the flattened obs object the same way Mike did. EB. 20250226.
# 
# This is a modified version to calculate ARI exceedances over an extended period of time
# so you always include the 12-hour.
#####################INPUT FILES FOR calc_ST4gARI########################################
#vhr_ST4       = Valid hours to be considered; dimension (vhr)
#ST4_01hr      = ST4 1-hour accumulated precipitation; dimension (lat,lon,vhr)
#ARI           = ARI precipitation values (lat,lon,thresholds for 1/3/6/12/24 hour windows)
####################OUTPUT FILES FOR calc_ST4gARI#############################
#flood_obs     = Daily flood obs grid representing all possible combinations of ST4 > ARI
#########################################################################################

### Just load in the hourly Stage IV

def calc_ST4gARI_duration(ST4_01hr,ARI,DUR):
    
    #print('COMPLETELY REWRITTEN FUNCTION')
    
    #Split ARI into 1,3,6,12, and 24 hour variables
    ARI_01hr  = ARI[:,:,0]
    ARI_03hr  = ARI[:,:,1]
    ARI_06hr  = ARI[:,:,2]
    ARI_12hr  = ARI[:,:,3]
    
    ### number of hours we have to cycle through
    ### keep in mind, this array might always be 24 hours long, but only some hours
    ### will be filled in. It's okay to cycle through the whole thing, but we'll still
    ### add a check to see if there's data in that hour.
    num_hrs=ST4_01hr.shape[2]
    ### create these arrays so we can find exceedances at each hour. Not hard coded.
    flood_01hr  = np.zeros((ST4_01hr.shape))
    flood_03hr  = np.zeros((ST4_01hr.shape))
    flood_06hr  = np.zeros((ST4_01hr.shape))
    flood_12hr  = np.zeros((ST4_01hr.shape))
    
    
    ### now cycle through the hours for the other durations.
    ### start at index 2 (hour 3), and then build on that
    for n in range(num_hrs):
      
      ### check for data in this hour- should only need to skip at the end of the array
      ### np.unique will show only [0.] if there's no data (all zeros), otherwise, there
      ### will be numbers other than zero, making the length of np.unique greater than 1
      if len(np.unique(ST4_01hr[:,:,n]))==1:
         print('no data at hour index '+str(n))
         continue
      
      ### start with 1-hour exceedances
      ### 1-hour exceedances are straightforward. Just compare the hourly Stage IV to the 1-hr ARI.
      flood_01hr[:,:,n] = ST4_01hr[:,:,n] > ARI_01hr

      ### if we haven't had 3 hours yet, don't try to calculate anything.
      if n<2:
        continue
      
      if n>=2:
        ### let's go- start by finding the 3-hour rainfall accumulation
        rainfall=np.sum(ST4_01hr[:,:,n-2:n+1],axis=2)
        ### check if n-1:n is enough or if python does that weird thing where it makes you go to index+1
        ### just to get the array you SHOULD have gotten by going to index n.
        ### answer: it must be n+1 to get shape lat,lon,3
        ### now compare to the ARI
        flood_03hr[:,:,n]= rainfall > ARI_03hr
        #print(np.unique(flood_03hr[:,:,n]))
        #print('3-hour')
        #exit()
      
      ### if this is the 6th hour of data or more, we can look for 6-hour exceedances
      if n>=5:
        rainfall = np.sum(ST4_01hr[:,:,n-5:n+1],axis=2)
        ### compare to ARI
        flood_06hr[:,:,n]= rainfall > ARI_06hr
        #print(np.unique(flood_06hr[:,:,n]))
        #print('6-hour')
        #exit()
    
      ### if this is the 12th hour of data or more, we can look for 6-hour exceedances
      if n>=11:
        rainfall = np.sum(ST4_01hr[:,:,n-11:n+1],axis=2)
        ### compare to ARI
        flood_12hr[:,:,n]= rainfall > ARI_12hr
        #print(np.unique(flood_12hr[:,:,n]))
        #print('12-hour')
        #exit()

    
    
    #print((flood_01hr>0).sum())
    #print((flood_03hr>0).sum())
    #print((flood_06hr>0).sum())
    #print((flood_12hr>0).sum())
    
    
    ### add them all up to make one large object that has ARI exceedances all together
     
    ### always include the 12-hour
######    flood_obs=((flood_01hr + flood_03hr + flood_06hr + flood_12hr) > 0)
    if DUR==1:
        flood_obs = flood_01hr > 0
    elif DUR==3:
        flood_obs = flood_03hr > 0
    elif DUR==6:
        flood_obs = flood_06hr > 0
    elif DUR==12:
        flood_obs = flood_12hr > 0
    
    #print((flood_01hr[:,:,0:2]>0).sum())
    #print((flood_03hr[:,:,0:2]>0).sum())
    #print((flood_06hr[:,:,0:2]>0).sum())
    #
    #print((flood_01hr[:,:,3::]>0).sum())
    #print((flood_03hr[:,:,3::]>0).sum())
    #print((flood_06hr[:,:,3::]>0).sum())
    #print((flood_12hr>0).sum())
    #exit()
    
    ### RETURN THE FINAL, TOTAL OBSERVATION OBJECT/GRID.
    return flood_obs



#########################################################################################
#########################################################################################
#########################################################################################




#########################################################################################
# New function for calculating the St4 exceedances of ARI. 
# A straightforward calculation with documentation and comments to accompany.
# Just read in the hourly data, step through hour by hour to find all the exceedances,
# and assign exceedances to the hour in which they occurred. NOT the start hour.
# Output the flattened obs object the same way Mike did. EB. 20250226.
#####################INPUT FILES FOR calc_ST4gARI########################################
#vhr_ST4       = Valid hours to be considered; dimension (vhr)
#ST4_01hr      = ST4 1-hour accumulated precipitation; dimension (lat,lon,vhr)
#ARI           = ARI precipitation values (lat,lon,thresholds for 1/3/6/12/24 hour windows)
#val_hours     = valid hours of the MPD- to determine if we use the 12-hour exceedances or not
####################OUTPUT FILES FOR calc_ST4gARI#############################
#flood_obs     = Daily flood obs grid representing all possible combinations of ST4 > ARI
#########################################################################################

### Just load in the hourly Stage IV

def calc_ST4gARI_MPD(vhr_ST4,ST4_01hr,ARI,val_hours):
    
    print('COMPLETELY REWRITTEN FUNCTION')
    
    #Split ARI into 1,3,6,12, and 24 hour variables
    ARI_01hr  = ARI[:,:,0]
    ARI_03hr  = ARI[:,:,1]
    ARI_06hr  = ARI[:,:,2]
    ARI_12hr  = ARI[:,:,3]
    
    ### number of hours we have to cycle through
    ### keep in mind, this array might always be 24 hours long, but only some hours
    ### will be filled in. It's okay to cycle through the whole thing, but we'll still
    ### add a check to see if there's data in that hour.
    num_hrs=ST4_01hr.shape[2]
    ### create these arrays so we can find exceedances at each hour. Not hard coded.
    flood_01hr  = np.zeros((ST4_01hr.shape))
    flood_03hr  = np.zeros((ST4_01hr.shape))
    flood_06hr  = np.zeros((ST4_01hr.shape))
    flood_12hr  = np.zeros((ST4_01hr.shape))
    
    
    ### now cycle through the hours for the other durations.
    ### start at index 2 (hour 3), and then build on that
    for n in range(num_hrs):
      
      ### check for data in this hour- should only need to skip at the end of the array
      ### np.unique will show only [0.] if there's no data (all zeros), otherwise, there
      ### will be numbers other than zero, making the length of np.unique greater than 1
      if len(np.unique(ST4_01hr[:,:,n]))==1:
         print('no data at hour index '+str(n))
         continue
      
      ### start with 1-hour exceedances
      ### 1-hour exceedances are straightforward. Just compare the hourly Stage IV to the 1-hr ARI.
      flood_01hr[:,:,n] = ST4_01hr[:,:,n] > ARI_01hr

      ### if we haven't had 3 hours yet, don't try to calculate anything.
      if n<2:
        continue
      
      if n>=2:
        ### let's go- start by finding the 3-hour rainfall accumulation
        rainfall=np.sum(ST4_01hr[:,:,n-2:n+1],axis=2)
        ### check if n-1:n is enough or if python does that weird thing where it makes you go to index+1
        ### just to get the array you SHOULD have gotten by going to index n.
        ### answer: it must be n+1 to get shape lat,lon,3
        ### now compare to the ARI
        flood_03hr[:,:,n]= rainfall > ARI_03hr
        #print(np.unique(flood_03hr[:,:,n]))
        #print('3-hour')
        #exit()
      
      ### if this is the 6th hour of data or more, we can look for 6-hour exceedances
      if n>=5:
        rainfall = np.sum(ST4_01hr[:,:,n-5:n+1],axis=2)
        ### compare to ARI
        flood_06hr[:,:,n]= rainfall > ARI_06hr
        #print(np.unique(flood_06hr[:,:,n]))
        #print('6-hour')
        #exit()
    
      ### if this is the 12th hour of data or more, we can look for 6-hour exceedances
      if n>=11:
        rainfall = np.sum(ST4_01hr[:,:,n-11:n+1],axis=2)
        ### compare to ARI
        flood_12hr[:,:,n]= rainfall > ARI_12hr
        #print(np.unique(flood_12hr[:,:,n]))
        #print('12-hour')
        #exit()

    
    
    #print((flood_01hr>0).sum())
    #print((flood_03hr>0).sum())
    #print((flood_06hr>0).sum())
    #print((flood_12hr>0).sum())
    
    
    ### add them all up to make one large object that has ARI exceedances all together
     
    ### FOR MPD: CHECK VALID HOURS. IF THE VALID TIME OF THE MPD IS GREATER THAN 10 HOURS
    ### THAT MEANS IT'S AN AR MPD. USED 10 BECAUSE IF WE DO LEAD TIME, WE'RE LOOKING AT THE 
    ### MPD PLUS 3 HOURS BEFORE, WHICH IS 9 HOURS. AND WE DON'T WANT THE 12-HOUR EXCEEDANCES
    ### INCLUDED FOR THAT. ONLY FOR THE VALID TIMES LONGER THAN 10 HOURS BECAUSE THOSE ARE AR
    ### MPDS AND ARE USUALLY VALID FOR 12 FULL HOURS.
    if val_hours>=10:
      flood_obs=((flood_01hr + flood_03hr + flood_06hr + flood_12hr) > 0)
    else:
      flood_obs=((flood_01hr + flood_03hr + flood_06hr) > 0)
    
    
    
    
    #print((flood_01hr[:,:,0:2]>0).sum())
    #print((flood_03hr[:,:,0:2]>0).sum())
    #print((flood_06hr[:,:,0:2]>0).sum())
    #
    #print((flood_01hr[:,:,3::]>0).sum())
    #print((flood_03hr[:,:,3::]>0).sum())
    #print((flood_06hr[:,:,3::]>0).sum())
    #print((flood_12hr>0).sum())
    #exit()
    
    ### RETURN THE FINAL, TOTAL OBSERVATION OBJECT/GRID.
    return flood_obs


#########################################################################################
# totally rewritten version of calc_ST4gFFG. Indexing now makes sense. Simple, straightforward.
# you still find every possible exceedance, but we don't have the double counting and missing
# indices problems. Just step through the array, find each 3- and 6-hour window, and find 
# any/all exceedances. Hourly is simple. Also creating the 6-hour array from the hourly
# because the MPDs are not issued on the synoptic hours. So the Stage IV 6-hourly data isn't
# working for us. SAME PROCESS as ARI calculations. Just different thresholds.
#####################INPUT FILES FOR calc_ST4gFFG###########################################
#vhr_ST4       = Valid hours to be considered; dimension (vhr)
#ST4_01hr      = ST4 1-hour accumulated precipitation; dimension (lat,lon,vhr)
#FFG_01hr      = FFG 1-hour accumulated precipitation; dimension (lat,lon,vhr)
#FFG_03hr      = FFG 3-hour accumulated precipitation; dimension (lat,lon,vhr/3)
#FFG_06hr      = FFG 6-hour accumulated precipitation; dimension (lat,lon,vhr/6)
####################OUTPUT FILES FOR calc_ST4gFFG###########################################
#flood_obs     = Daily flood obs grid representing all possible combinations of ST4 > FFG
############################################################################################

def calc_ST4gFFG_MPD(vhr_ST4,ST4_01hr,FFG_01hr,FFG_03hr,FFG_06hr):
    
    ############COMMENT OUT WHEN NOT IN FUNCTION MODE#######################
    #vhr_ST4 = vhr1_ahead[1::]
    ########################################################################
    
    ### number of hours we have to cycle through
    ### keep in mind, this array might always be 24 hours long, but only some hours
    ### will be filled in. It's okay to cycle through the whole thing, but we'll still
    ### add a check to see if there's data in that hour.
    num_hrs=ST4_01hr.shape[2]
    
    ### generate the flood arrays. Same shape/size as the hourly Stage IV.
    flood_01hr = np.zeros((ST4_01hr.shape))
    flood_03hr = np.zeros((ST4_01hr.shape))
    flood_06hr = np.zeros((ST4_01hr.shape))
    
    
    ### now cycle through the hours for the other durations.
    ### start at index 2 (hour 3), and then build on that
    for n in range(num_hrs):
      
      ### check for data in this hour- should only need to skip at the end of the array
      ### np.unique will show only [0.] if there's no data (all zeros), otherwise, there
      ### will be numbers other than zero, making the length of np.unique greater than 1
      if len(np.unique(ST4_01hr[:,:,n]))==1:
         print('no data at hour index '+str(n))
         continue
      
      ### start with 1-hour exceedances
      ### 1-hour exceedances are straightforward. Just compare the hourly Stage IV to the 1-hr ARI.
      flood_01hr[:,:,n] = ST4_01hr[:,:,n] > FFG_01hr[:,:,n]

      ### if we haven't had 3 hours yet, don't try to calculate anything.
      if n<2:
        continue
      
      if n>=2:
        ### let's go- start by finding the 3-hour rainfall accumulation
        rainfall=np.sum(ST4_01hr[:,:,n-2:n+1],axis=2)
        ### check if n-1:n is enough or if python does that weird thing where it makes you go to index+1
        ### just to get the array you SHOULD have gotten by going to index n.
        #print(ST4_01hr[:,:,n-2:n+1].shape)
        ### now compare to the FFG. Note that 3-hour FFG is only issued on the synoptic hours. 0, 6, 12, 18.
        ### so we'll have to choose an index closest. Choose the most recent one basically.
        if n>=0 and n<6:
           n2=0
        elif n>=6 and n<12:
           n2=1
        elif n>=12 and n<18:
           n2=2
        elif n>=18 and n<24:
           n2=3
        flood_03hr[:,:,n]= rainfall > FFG_03hr[:,:,n2]
        #if n==6:
        #  print(n2)
        #  print(np.nanmax(FFG_03hr[:,:,n2]))
        #  print(np.nanmax(FFG_03hr[0:25,500:550,n2]))
        #  print(np.nanmax(rainfall[0:25,500:550]))
        #  ### FL keys
        #  print(np.unique(flood_03hr[0:25,500:550,n]))
          
      
      ### if this is the 6th hour of data or more, we can look for 6-hour exceedances
      if n>=5:
        rainfall = np.sum(ST4_01hr[:,:,n-5:n+1],axis=2)
        ### compare to FFG- on the synoptic hours- choose the index for the FFG array
        if n>=0 and n<6:
           n2=0
        elif n>=6 and n<12:
           n2=1
        elif n>=12 and n<18:
           n2=2
        elif n>=18 and n<24:
           n2=3
        flood_06hr[:,:,n]= rainfall > FFG_06hr[:,:,n2]
    

    
    #print(np.unique(flood_01hr))
    #print(np.unique(flood_03hr))
    #print(np.unique(flood_06hr))
    
    #Combine all flooding info
    flood_obs= flood_01hr + flood_03hr+ flood_06hr
    #print(np.unique(flood_obs))

    return flood_obs


#########################################################################################
#########################################################################################
#########################################################################################
