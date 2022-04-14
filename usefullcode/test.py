import pandas as pd
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go
import datetime
from datetime import timedelta
from werkzeug.utils import secure_filename
import os
import math
import numpy as np
import sshtunnel
import sys
import paramiko
import re
import pymongo
from pymongo import MongoClient
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
from astropy.table import Table

from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom, Map
from gammapy.makers import MapDatasetMaker
from gammapy.makers.utils import make_theta_squared_table
from gammapy.visualization import plot_theta_squared_table


def Get_MongoDB():
    '''
        Connect to mongo machine on port 27017 and get LeLardon
    '''
    # Connection Parameters
    Params = "mongodb+srv://lardon:140889@lelardon-gpgol.mongodb.net/test?retryWrites=true&w=majority"
    #Params = "mongodb+srv://lardon:Lardon56!@lelardon-gpgol.mongodb.net/test?retryWrites=true"
    # GET DB via pymongo
    client = pymongo.MongoClient(Params)
    db = client['Lardon']
    return db


path = "/Users/vl238644/Documents/GitHub/HESS/GammaPyGui"

allruns = np.unique(df['OBS_ID'])

for run in allruns:
    print(run)
    dat = Table.read('/Users/vl238644/Documents/GitHub/Analysis/GammaPy/hess_dl3_dr1/data/hess_dl3_dr1_obs_id_020136.fits.gz', format='fits')
    df = dat.to_pandas()
    df['run']=run
    client = pymongo.MongoClient("mongodb://hess:****@51.15.204.85/HESS")
    db = client['HESS']
    my_dict = df.to_dict('record')
    db['run_'+str(run)].insert_many(my_dict)


client = pymongo.MongoClient("mongodb://hess:****@51.15.204.85/HESS")
db = client['HESS']
cursor = db['PKS2155304_steady'].find()
df =  pd.DataFrame(list(cursor))
df

dat = Table.read(path + '/Main/static/hess_dl3_dr1/obs-index.fits.gz', format='fits')
hdudat = Table.read(path + '/Main/static/hess_dl3_dr1/hdu-index.fits.gz', format='fits')
df = dat.to_pandas()
for e in df.columns:
    print(e)
    print(type(list(df[e])[0]))
    if type(list(df[e])[0]) == bytes :
        df[e] = [item.decode("utf-8") for item in df[e]]

df_hdu = hdudat.to_pandas()
for e in df_hdu.columns:
    print(e)
    print(type(list(df_hdu[e])[0]))
    if type(list(df_hdu[e])[0]) == bytes :
        df_hdu[e] = [item.decode("utf-8") for item in df_hdu[e]]

my_dict = df.to_dict('record')
db['obs-index'].insert_many(my_dict)

my_dict = df_hdu.to_dict('record')
db['hdu-index'].insert_many(my_dict)

db['test'].drop()



dat = Table.read('/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure/run037400-037599/run037496/hess_events_037496.fits.gz', format='fits')
df = dat.to_pandas()

import plotly.express as px
import plotly
import plotly.figure_factory as ff
import plotly.graph_objects as go

fig = px.density_heatmap(df, x="RA", y="DEC",nbinsx=500, nbinsy=500)
fig.show()
