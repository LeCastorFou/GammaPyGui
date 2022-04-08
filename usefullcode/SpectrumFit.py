
import gammapy

import regions
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
import requests
import json
import time
import pandas as pd
from pathlib import Path
from PIL import Image
from wtforms.fields.html5 import DateField
from flask import jsonify
import requests
import astropy
from astropy.coordinates import SkyCoord, ICRS, Galactic, FK4, FK5, Angle, Latitude, Longitude  #
import astropy.units as u
from astropy.table import Table



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
    PowerLawSpectralModel,
    LogParabolaSpectralModel,
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



table = Table.read('/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure/obs-index.fits.gz', format='fits')
obsindex = table.to_pandas()
obsindex["OBJECT"] =  [ e.decode("utf-8")  for e in obsindex["OBJECT"] ]
object  = np.unique(list(obsindex["OBJECT"]))
object

datastore = DataStore.from_dir('/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure/')
obsindex = obsindex[obsindex['OBJECT'] == 'Crab Nebula']
listrun = list(obsindex['OBS_ID'])

ra_obj = list(obsindex['RA_OBJ'])[0]
dec_obj = list(obsindex['DEC_OBJ'])[0]

obs_ids = listrun
obs_ids = [23134,23155,23156,23304,23309,23310,23523]
print(obs_ids)
print("GETTING RUNS INFOS")
observations = datastore.get_observations(obs_ids)
target_position = SkyCoord(ra=ra_obj, dec=dec_obj, unit="deg", frame="icrs")
on_region_radius = Angle("0.11 deg")
on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


exclusion_region = CircleSkyRegion(
    center=SkyCoord(0, 0, unit="deg", frame="galactic"),
    radius=0.0 * u.deg,
)

skydir = target_position.galactic
geom = WcsGeom.create(
    npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
)

exclusion_mask = ~geom.region_mask([exclusion_region])
exclusion_mask.plot()

energy_axis = MapAxis.from_energy_bounds(
    0.1, 40, nbin=10, per_decade=True, unit="TeV", name="energy"
)
energy_axis_true = MapAxis.from_energy_bounds(
    0.05, 100, nbin=20, per_decade=True, unit="TeV", name="energy_true"
)

geom = RegionGeom.create(region=on_region, axes=[energy_axis])
dataset_empty = SpectrumDataset.create(
    geom=geom, energy_axis_true=energy_axis_true
)

dataset_maker = SpectrumDatasetMaker(
    containment_correction=True, selection=["counts", "exposure", "edisp"]
)
bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

datasets = Datasets()
print("MERGING OBS ...")
for obs_id, observation in zip(obs_ids, observations):
    try:
        print(obs_id)
        print(observation)
        dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        datasets.append(dataset_on_off)
    except Exception as inst:
        print(inst)
        print("ERROR WITH RUN " + str(obs_id))
print("DONE")

spectral_model = ExpCutoffPowerLawSpectralModel(
    amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
    index=2,
    lambda_=0.1 * u.Unit("TeV-1"),
    reference=1 * u.TeV,
)
model = SkyModel(spectral_model=spectral_model, name="crab")

datasets.models = [model]
fit_joint = Fit()
result_joint = fit_joint.run(datasets=datasets)

# we make a copy here to compare it later
model_best_joint = model.copy()
datasets.info_table()
print("PLOTTING FITS")
ax_spectrum, ax_residuals = datasets[0].plot_fit()
ax_spectrum.set_ylim(0.1, 40)

e_min, e_max = 0.7, 30
energy_edges = np.geomspace(e_min, e_max, 11) * u.TeV
print("STARTING FluxPointsEstimator ...")
fpe = FluxPointsEstimator(energy_edges=energy_edges, source="crab", selection_optional="all")
flux_points = fpe.run(datasets=datasets)
print(flux_points)
df = flux_points.to_table(sed_type="dnde", formatted=True)
names = [name for name in df.colnames if len(df[name].shape) <= 1]
df = df[names].to_pandas()
print("PLOT RESULTS ...")
plt.figure(figsize=(8, 5))
ax = flux_points.plot(sed_type="e2dnde", color="darkorange")
flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde");


def tableToPandas(table):
    allCol = table.colnames
    coldict = {}
    for e in allCol:
        coldict[e] = list(flux_points.to_table(sed_type="e2dnde", formatted=True)[e])
    return pd.DataFrame(coldict)

dataset_gammacat = FluxPointsDataset(
    data=flux_points, name="gammacat"
)

flux_points_df = tableToPandas(flux_points.to_table(sed_type="e2dnde", formatted=True))
np.max(flux_points_df['e2dnde'])

dataset_gammacat.data.to_table(sed_type="dnde", formatted=True)

pwl = PowerLawSpectralModel(
    index=2, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV"
)
model = SkyModel(spectral_model=pwl, name="crab")

print(model)
datasets = Datasets([dataset_gammacat])
datasets.models = model

fitter = Fit()
result_pwl = fitter.run(datasets=datasets)

print(datasets.models)
print(result_pwl)
ax = plt.subplot()

kwargs = {"ax": ax, "sed_type": "e2dnde"}

for d in datasets:
    d.data.plot(label=d.name, **kwargs)

energy_bounds = [1e-1, 1e2] * u.TeV
pwl.plot(energy_bounds=energy_bounds, color="k", **kwargs)
pwl.plot_error(energy_bounds=energy_bounds, **kwargs)
ax.set_ylim(np.min(flux_points_df['e2dnde'])/5, np.max(flux_points_df['e2dnde'])*5)
ax.set_xlim(energy_bounds)
ax.legend()
ax.get_figure().savefig(resPath+analysisName+'/spectrumfit.jpg')
