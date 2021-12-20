from flask import Blueprint
from flask import render_template, url_for,flash, redirect, request, abort, send_from_directory, make_response, jsonify
from Main import db, bcrypt, mail
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
from flask_restful import Api, Resource, reqparse
import pandas as pd
from flask_simple_geoip import SimpleGeoIP
from Main.BackEnd.main.forms import  StartAnalysis
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


main = Blueprint('main',__name__)

@main.route("/", methods=['GET', 'POST'])
@main.route("/home", methods=['GET', 'POST'])
def home():
    db_mongo = Get_MongoDB()
    df = load_DB_collection(db_mongo,"obs-index")
    targetList = np.unique(df['TARGET_NAME'])
    print(targetList)
    form = StartAnalysis()
    form.source.choices = [ (e,e) for e in np.sort(targetList)]
    if form.validate_on_submit():

        obs_id_list = list(df[df['TARGET_NAME']==form.source.data]['OBS_ID'])

        obs_id_list = [e for e in obs_id_list if e >= form.rmin.data]
        obs_id_list = [e for e in obs_id_list if e <= form.rmax.data]

        n_list = 0
        for obs in obs_id_list:
            if n_list == 0:
                df = load_DB_collection(db_mongo,"run_"+str(obs))
            else:
                df = pd.concat([df,load_DB_collection(db_mongo,"run_"+str(obs) )])
            n_list = n_list + 1

        fig = px.density_heatmap(df, x="RA", y="DEC",nbinsx=50, nbinsy=50)
        graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('main/index.html',form = form, graphJSON = graph)
    return render_template('main/index.html', form = form, graph ={})
