from flask import Blueprint
from flask import render_template, url_for,flash, redirect, request, abort, send_from_directory, make_response, jsonify
from Main import db, bcrypt, mail
from flask_login import login_user, current_user, logout_user, login_required
from flask_mail import Message
from flask_restful import Api, Resource, reqparse
import pandas as pd
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

######## to DELETE BEFORE PROD
import os
import pandas as pd
import numpy as np
import secrets
import os
from PIL import Image
from wtforms.fields.html5 import DateField
import datetime
from datetime import timedelta
from flask import current_app, url_for
from flask_mail import Message
from datetime import datetime
import pymongo
from pymongo import MongoClient
#from sshtunnel import SSHTunnelForwarder
import pandas as pd
import ast

def Get_MongoDB():
    # Connection Parameters
    client = pymongo.MongoClient("mongodb://hess:CT5io!@51.15.204.85/HESS")
    db = client['HESS']
    return db

def load_DB_collection(db_mongo,collection):
    cursor = db_mongo[collection].find()
    df =  pd.DataFrame(list(cursor))
    if len(df)>0:
        df = df.drop(['_id'], axis=1)
    return df
#########################################

@main.route("/", methods=['GET', 'POST'])
@main.route("/home", methods=['GET', 'POST'])
def home():
    db_mongo = Get_MongoDB()
    df = load_DB_collection(db_mongo,"obs-index")
    targetList = np.unique(df['TARGET_NAME'])

    form = StartAnalysis()
    form.source.choices = [ (e,e) for e in np.sort(targetList)]

    isRunList = True

    if form.validate_on_submit():
        if 'file' not in request.files:
            flash('No file part')
            isRunList = False
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            isRunList = False
        if isRunList:
            filename = secure_filename(file.filename)
            file.save("/tmp/"+filename)
            f = open("/tmp/"+file.filename, "r")
            runlist = []
            for x in f:
                runlist = runlist + [x]
            if os.path.exists("/tmp/"+filename):
                os.remove("/tmp/"+ filename)
            runlist = [int(i) for i in runlist if i != '\n']
            print(runlist)

        obs_id_list = list(df[df['TARGET_NAME']==form.source.data]['OBS_ID'])

        if isRunList:
            print(runlist)
            obs_id_list = [e  for e in obs_id_list if e in runlist]
        else:
            obs_id_list = [e for e in obs_id_list if e >= form.rmin.data]
            obs_id_list = [e for e in obs_id_list if e <= form.rmax.data]
        print('LIST of KEPT runs')
        print(obs_id_list)
        if len(obs_id_list) >0:
            n_list = 0
            for obs in obs_id_list:
                newdata = load_DB_collection(db_mongo,"run_"+str(obs))
                if len(newdata) > 0:
                    if n_list == 0:
                        df = newdata
                    else:
                        df = pd.concat([df,newdata])
                    n_list = n_list + 1

            fig = px.density_heatmap(df, x="RA", y="DEC",nbinsx=50, nbinsy=50)
            graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template('main/index.html',form = form, graphJSON = graph)
        else:
            flash('NO runs available','error')
            return render_template('main/index.html', form = form, graphJSON ={})
    return render_template('main/index.html', form = form, graphJSON ={})
