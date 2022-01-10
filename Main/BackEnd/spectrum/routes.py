from flask import Blueprint
from flask import render_template, url_for,flash, redirect, request, abort, send_from_directory, make_response, jsonify
from Main import db, bcrypt, mail
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
from flask_restful import Api, Resource, reqparse
import pandas as pd
from Main.BackEnd.spectrum.forms import  LaunchSpectrum
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Check package versions
import gammapy
import numpy as np
import astropy
import regions

from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
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

spectrumbp = Blueprint('spectrumbp',__name__)


@spectrumbp.route("/spectrum1D", methods=['GET', 'POST'])
def spectrum1D():
    form = LaunchSpectrum()
    if form.validate_on_submit():
        datastore = DataStore.from_dir("/Users/vl238644/Documents/GitHub/Analysis/GammaPy/hess_dl3_dr1")
        obs_ids = [23523, 23526, 23559, 23592]
        observations = datastore.get_observations(obs_ids)
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

        resname= form.picname.data+'.png'
        plt.figure(figsize=(8, 8))
        plt.savefig("/Users/vl238644/Documents/GitHub/GammaPyGui/Main/static/plot_results/"+ resname)
        ax = exclusion_mask.plot()
        on_region.to_pixel(ax.wcs).plot(ax=ax, edgecolor="k")
        plot_spectrum_datasets_off_regions(ax=ax, datasets=datasets)

        return render_template('spectrum/index_spectrum.html', resname=resname , form=form)


    return render_template('spectrum/index_spectrum.html', resname='def.png' , form=form)
