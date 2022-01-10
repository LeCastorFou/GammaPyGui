import astropy.io.fits as pyfits
import pandas as pd
import numpy as np
from astropy.table import Table, join, vstack, setdiff
import os
from astropy.io import ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import re
import urllib.parse
from astroquery.simbad import Simbad
import pandas as pd
import pymongo
from pymongo import MongoClient
#from sshtunnel import SSHTunnelForwarder
import pandas as pd
hdul = pyfits.open('/Users/vl238644/Downloads/myfits05.fits')
datas = hdul[1].data
# print(hdul[1].header)
hdul[0].header
hdul[1].header
len(hdul[1].header)

hdul.info()
datas[0]
obs = []

def Get_MongoDB():
    '''
        Connect to mongo machine on port 27017 and get Colibri
    '''
    # Connection Parameters
    Params = "mongodb+srv://Colibri:Colibri2020!@cluster0.qqawu.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
    # GET DB via pymongo
    client = pymongo.MongoClient(Params)
    db = client['Colibri']
    return db

db_mongo = Get_MongoDB()


for data in datas:
    to_register = {'dec':[data[0]],'energy':[data[1]],'err':[data[2]],'far':[data[3]],'ivorn':[data[4]],'ivorn_time':[data[5]],'obervatory':[data[6]],'pkt_ser_num':[data[7]],
                    'ra':[data[8]],'reference':[data[9]],'signalness':[data[10]],'source_name':[data[11]],'timestamp':[data[12]],'trigger_id':[data[13]],'type':[data[14]],
                    'comment':[data[15]],'dm':[data[16]],'assoc':[data[17]],'energy_time_url':[data[18]],'lc_url':[data[19]],'sed_url':[data[20]],'simbad_link':[data[21]]}
    to_register = pd.DataFrame.from_dict(to_register)
    my_dict = to_register.to_dict('record')
    db_mongo['VoEvents'].insert_many(my_dict)


to_register
len(data[1])
data[1][0]
type(data[1])

np.unique(obs)

for e in data:
    for i in range(len(e)):
        if e[i] == 'swift':
            print(e)


def read_fits_as_dataframe(filename, index_columns):
    # This is the way to read the FITS data into a numpy structured array
    # (using astropy.io.fits.getdata didn't work out of the box
    # because it gives a FITSRec)
    table = Table.read(filename)
    data = table._data
    # Fix byte order.
    # See https://github.com/astropy/astropy/issues/1156
    data = data.byteswap().newbyteorder()
    df = pd.DataFrame.from_records(data)
    # Strip whitespace for string columns that will become indices
    for index_column in index_columns:
        df[index_column] = df[index_column].map(str.strip)
    df = df.set_index(index_columns)
    return df

df = read_fits_as_dataframe('/Users/vl238644/Downloads/myfits.py', ['ROI', 'Solution'])
print(df.head())
print(df.xs('HESS_J1023m575'))
print(df[['nfev', 'statname']])
pd.DataFrame(data)
