import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles

from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom, Map
from gammapy.makers import MapDatasetMaker
from gammapy.makers.utils import make_theta_squared_table
from gammapy.visualization import plot_theta_squared_table
import pandas as pd
import os
import plotly
import plotly.express as px
import plotly.graph_objects as go # High-level coordinates

data_store = DataStore.from_dir("/Users/vl238644/Documents/GitHub/Analysis/GammaPy/hess_dl3_dr1")
data_store

data_store.info()
data_store.obs_table[:][["OBS_ID", "DATE-OBS", "RA_PNT", "DEC_PNT", "OBJECT"]]
data_store.obs_table[:]
obs = data_store.obs(23523)
events = obs.events.select_offset([0, 2.5] * u.deg)

eEnergy =[ e.value for e in  list(events.energy) ]
eOffset = [ e.value for e in list(events.offset) ]


eCoord = [e.transform_to('galactic') for e in list(events.radec)]
eLat = [e.l.value for e in eCoord]
eLong = [e.b.value for e in eCoord]



df = pd.DataFrame(list(zip(eEnergy, eOffset,eLat,eLong)),columns =['energy', 'Offset','lat','long'])

fig = px.density_heatmap(df, x="lat", y="long",nbinsx=50, nbinsy=50)
fig.show()


obs.aeff.peek()
obs.edisp.peek()
obs.psf.peek()
obs.bkg.to_2d().plot()
position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
theta2_axis = MapAxis.from_bounds(0, 0.2, nbin=20, interp="lin", unit="deg2")

observations = data_store.get_observations([23523, 23526])
theta2_table = make_theta_squared_table(
    observations=observations,
    position=position,
    theta_squared_axis=theta2_axis,
)

plt.figure(figsize=(10, 5))
plot_theta_squared_table(theta2_table)

# Get the observations
obs_id = data_store.obs_table["OBS_ID"][data_store.obs_table["OBJECT"] == "MSH 15-5-02"]
observations = data_store.get_observations(obs_id)
print(len(observations))
