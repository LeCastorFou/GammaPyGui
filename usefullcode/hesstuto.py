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

from gammapy.data import EventList

table = Table.read('/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure/obs-index.fits.gz', format='fits')
df = table.to_pandas()
df.columns
data_store = DataStore.from_dir("/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure")
data_store
data_store.info()
data_store.obs_table[:][["OBS_ID", "DATE-OBS", "RA_PNT", "DEC_PNT", "OBJECT"]]
data_store.obs_table[:]
obs = data_store.get_observations([18361,18410,18415])
print(obs)

datasets = Datasets()
obs_ids = [18361,18410,18415]
for obs_id, observation in zip(obs_ids, obs):
    dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)
    dataset_on_off = bkg_maker.run(dataset, observation)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
    datasets.append(dataset_on_off)


obs_list_events = [e.events for e in obs]
combined_events = EventList.from_stack(obs_list_events)
combined_events.peek()


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


import gammapy

import regions


from regions import CircleSkyRegion
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset,
)
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    create_crab_spectral_model,
    SkyModel,
)
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.visualization import plot_spectrum_datasets_off_regions

data_store = DataStore.from_dir("/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure")
obs_ids = [18361,18410,18415]
observations = data_store.get_observations(obs_ids)
target_position = SkyCoord(ra=83.63, dec=22.01, unit="deg", frame="icrs")
on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)
exclusion_region = CircleSkyRegion(
center=SkyCoord(183.604, -8.708, unit="deg", frame="galactic"),radius=0.5 * u.deg)

skydir = target_position.galactic
geom = WcsGeom.create(npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs")

exclusion_mask = ~geom.region_mask([exclusion_region])
exclusion_mask.plot()

energy_axis = MapAxis.from_energy_bounds(  0.1, 40, nbin=10, per_decade=True, unit="TeV", name="energy")
energy_axis_true = MapAxis.from_energy_bounds(0.05, 100, nbin=20, per_decade=True, unit="TeV", name="energy_true")

geom = RegionGeom.create(region=on_region, axes=[energy_axis])
dataset_empty = SpectrumDataset.create(geom=geom, energy_axis_true=energy_axis_true)
dataset_maker = SpectrumDatasetMaker(containment_correction=True, selection=["counts", "exposure", "edisp"])
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

datasets = Datasets()

for obs_id, observation in zip(obs_ids, observations):
    dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)
    dataset_on_off = bkg_maker.run(dataset, observation)
    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
    datasets.append(dataset_on_off)


plt.figure(figsize=(8, 8))
ax = exclusion_mask.plot()
on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="k")
plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)

info_table = datasets.info_table(cumulative=True)
plt.plot(info_table["livetime"].to("h"), info_table["excess"], marker="o", ls="none")
plt.xlabel("Livetime [h]")
plt.ylabel("Excess");

plt.plot(
    info_table["livetime"].to("h"),
    info_table["sqrt_ts"],
    marker="o",
    ls="none",
)
plt.xlabel("Livetime [h]")
plt.ylabel("Sqrt(TS)");

spectral_model = ExpCutoffPowerLawSpectralModel(
    amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
    index=2,
    lambda_=0.1 * u.Unit("TeV-1"),
    reference=1 * u.TeV,
)
model = SkyModel(spectral_model=spectral_model, name="crab")

datasets.models = [model]
fit_joint = Fit(datasets)
result_joint = fit_joint.run(datasets=datasets)

# we make a copy here to compare it later
model_best_joint = model.copy()
