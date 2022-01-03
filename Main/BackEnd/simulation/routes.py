from flask import Blueprint
from flask import render_template, url_for,flash, redirect, request, abort, send_from_directory, make_response, jsonify
from Main import db, bcrypt, mail
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
from flask_restful import Api, Resource, reqparse
import pandas as pd
from flask_simple_geoip import SimpleGeoIP
from Main.BackEnd.simulation.forms import  LaunchSimu
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection
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
import pathlib

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import astropy.units as u
import gammapy
from astropy.coordinates import SkyCoord
from gammapy.irf import load_cta_irfs
from gammapy.maps import WcsGeom, MapAxis
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    GaussianSpatialModel,
    SkyModel,
    Models,
    FoVBackgroundModel,
)
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.modeling import Fit
from gammapy.data import Observation
from gammapy.datasets import MapDataset
import plotly.graph_objects as go
import pandas as pd


simubp = Blueprint('simubp',__name__)

@simubp.route("/simu", methods=['GET', 'POST'])
def simu():
    form = LaunchSimu()
    if form.validate_on_submit():
        plong = form.plong.data
        plat = form.plat.data
        slong = form.slong.data
        slat = form.slat.data
        print("PLONG")
        print(plong)
        path  = pathlib.Path().resolve()
        irfs = load_cta_irfs(str(path)+"/Main/static/irfs/North_z20_50h/irf_file.fits")

        livetime = 0.1 * u.hr
        pointing = SkyCoord(plong, plat, unit="deg", frame="galactic")

        # Define map geometry for binned simulation
        energy_reco = MapAxis.from_edges(np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log")
        geom = WcsGeom.create(skydir=(plong, plat),binsz=0.02,width=(6, 6),frame="galactic",axes=[energy_reco])

        # It is usually useful to have a separate binning for the true energy axis
        energy_true = MapAxis.from_edges(np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy_true", interp="log")

        empty = MapDataset.create(geom, name="dataset-simu", energy_axis_true=energy_true)

        # Define sky model to used simulate the data.
        # Here we use a Gaussian spatial model and a Power Law spectral model.
        spatial_model = GaussianSpatialModel(lon_0=str(slong)+" deg", lat_0=str(slat)+" deg", sigma="0.3 deg", frame="galactic")
        spectral_model = PowerLawSpectralModel(index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV")
        model_simu = SkyModel(spatial_model=spatial_model,spectral_model=spectral_model,name="model-simu")
        bkg_model = FoVBackgroundModel(dataset_name="dataset-simu")
        models = Models([model_simu, bkg_model])
        #print(models)

        # Create an in-memory observation
        obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
        #print(obs)

        # Make the MapDataset
        maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])

        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)

        dataset = maker.run(empty, obs)
        dataset = maker_safe_mask.run(dataset, obs)
        #print(dataset)

        # Add the model on the dataset and Poission fluctuate
        dataset.models = models
        dataset.fake()
        # Do a print on the dataset - there is now a counts maps
        #print(dataset)
        print(dataset)

        # To plot, eg, counts:
        alldata = dataset.counts.smooth(0.05 * u.deg).data
        stacked_data =np.sum(alldata, axis = 0)
        fig = go.Figure(data =
            go.Contour(
                z=stacked_data
            ))
        graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('simu/indexSimu.html', form = form, graphJSON = graph)
    return render_template('simu/indexSimu.html', form = form, graphJSON = {})
