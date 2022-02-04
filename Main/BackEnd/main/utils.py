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
from gammapy.maps import WcsNDMap
from gammapy.maps.utils import edges_from_lo_hi
from gammapy.maps import MapAxis

#Lardon56!
#mongodb+srv://<username>:<password>@lelardon-gpgol.mongodb.net/test?retryWrites=true&w=majority

def Get_MongoDB():
    '''
        Connect to mongo machine on port 27017 and get LeLardon
    '''
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

def plot_map_image(eventlist,path,analysisName,source):
    if eventlist.is_pointed_observation:
        offset = eventlist.offset
    else:
        offset = eventlist

    plot_width = 2 * offset.max()

    if  eventlist.is_pointed_observation:
        plot_center = eventlist.pointing_radec
    else:
        plot_center = eventlist.galactic_median

    opts = {"width": plot_width,"binsz": 0.05,"proj": "TAN","frame": "galactic","skydir": plot_center,}

    m = WcsNDMap.create(**opts)
    m.fill_by_coord(eventlist.radec)
    m = m.smooth(width=0.5)
    os.makedirs(path+analysisName)
    m.plot(stretch="sqrt").get_figure().savefig(path+analysisName+'/'+source+'_eventmap.png')

def plot_theta_squared_table_custom(table,path,analysisName,source):
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
    ax0.get_figure().savefig(path+analysisName+'/'+source+'_eventmap.png')
    return ax0
