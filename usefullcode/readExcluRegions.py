import pandas as pd
import astropy
from astropy.coordinates import SkyCoord, ICRS, Galactic, FK4, FK5, Angle, Latitude, Longitude  #
import astropy.units as u
from astropy.table import Table

df = pd.read_csv("/Users/vl238644/Documents/HESS/analysis/All_Regions_GAL.csv")

ra = []
dec = []
for i in range(len(df)):
    gc = SkyCoord(l=list(df['L'])[i]*u.degree, b=list(df['B'])[i]*u.degree, frame='galactic')
    res = gc.fk5
    ra  = ra + [res.ra.value]
    dec = dec + [res.dec.value]

df['ra'] = ra
df['dec'] = dec

df

df.to_csv("/Users/vl238644/Documents/HESS/analysis/All_Regions_GAL_radec.csv")
