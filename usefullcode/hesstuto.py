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
import pandas as pd
import os
import plotly
import plotly.express as px
import plotly.graph_objects as go # High-level coordinates
from gammapy.maps import MapAxis
from gammapy.maps.utils import edges_from_lo_hi
import numpy as np
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

table = Table.read('/Users/vl238644/Documents/GitHub/Analysis/GammaPy/hess_dl3_dr1/obs-index.fits.gz', format='fits')
obsindex = table.to_pandas()
obsindex["OBJECT"] =  [ e.decode("utf-8")  for e in obsindex["OBJECT"] ]
obsindex[obsindex['OBJECT']=="PKS 2155-304"]

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

def plot_theta_squared_table_custom(table):
    import matplotlib.pyplot as plt

    theta2_edges = edges_from_lo_hi(
        table["theta2_min"].quantity, table["theta2_max"].quantity
    )
    theta2_axis = MapAxis.from_edges(theta2_edges, interp="lin", name="theta_squared")

    ax0 = plt.subplot(2, 1, 1)

    x = theta2_axis.center.value
    x_edges = theta2_axis.edges.value
    xerr = (x - x_edges[:-1], x_edges[1:] - x)

    ax0.errorbar(
        x,
        table["counts"],
        xerr=xerr,
        yerr=np.sqrt(table["counts"]),
        linestyle="None",
        label="Counts",
    )

    ax0.errorbar(
        x,
        table["counts_off"],
        xerr=xerr,
        yerr=np.sqrt(table["counts_off"]),
        linestyle="None",
        label="Counts Off",
    )

    ax0.errorbar(
        x,
        table["excess"],
        xerr=xerr,
        yerr=(-table["excess_errn"], table["excess_errp"]),
        fmt="+",
        linestyle="None",
        label="Excess",
    )

    ax0.set_ylabel("Counts")
    ax0.set_xticks([])
    ax0.set_xlabel("")
    ax0.legend()

    ax1 = plt.subplot(2, 1, 2)
    ax1.errorbar(x, table["sqrt_ts"], xerr=xerr, linestyle="None")
    ax1.set_xlabel(f"Theta  [{theta2_axis.unit}]")
    ax1.set_ylabel("Significance")
    return ax0


plt.figure(figsize=(10, 5))
plot_theta_squared_table_custom(theta2_table).get_figure().savefig('/Users/vl238644/Documents/GitHub/Analysis/GammaPy/eventmap.png')

# Get the observations
obs_id = data_store.obs_table["OBS_ID"][data_store.obs_table["OBJECT"] == "MSH 15-5-02"]
observations = data_store.get_observations(obs_id)
print(len(observations))
