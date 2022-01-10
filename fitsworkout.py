from astropy.io import fits
import pandas as pd
from astropy.table import Table
import numpy as np
import math
# scp vlefranc@lfs2.mpi-hd.mpg.de:/lfs/l7/hess/fits/hap-hd/fits_prod05/hess1/std_ImPACT_fullEnclosure/*.fits.gz  .
table = Table.read('/Users/vl238644/Documents/GitHub/GammaPyGui/Main/static/FITS/obs-index.fits.gz', format='fits')
df = table.to_pandas()

df.columns
df['OBS_ID']
df['OBJECT'] = df['OBJECT'].str.decode("utf-8")

df_crab = df[df['OBJECT'] == 'Crab Nebula']
run_list = list(df_crab['OBS_ID'])

import os


run_list
e = run_list[0]
for e in run_list:
    try:
        runmin = math.floor(e/100)*100
        if int(str(runmin)[2]) not in [0,2,4,6,8]:
            to_change = int(str(runmin)[2])-1
            runmin_str = str(runmin)
            runmin = int(runmin_str[:2] + str(to_change)+'00')
        runmax = runmin +199
        file = 'run0'+str(runmin)+'-0'+str(runmax)
        os.system("scp -r  vlefranc@lfs2.mpi-hd.mpg.de:/lfs/l7/hess/fits/hap-hd/fits_prod05/hess1/std_ImPACT_fullEnclosure/"+file+"/run0"+ str(e)+" /Users/vl238644/Documents/GitHub/Analysis/GammaPy/fits/CrabNebulaHESS")
    except Exception:
        print("ERROR WITH "+str(e))
