import os
import pandas as pd
import numpy as np
#import secrets
import os
from PIL import Image
from wtforms.fields.html5 import DateField
import datetime
from datetime import timedelta
from flask import current_app, url_for
from flask_mail import Message
from datetime import datetime
#from sshtunnel import SSHTunnelForwarder
import pandas as pd
import ast
import arrow
import requests
import urllib3
import certifi
import xml.etree.ElementTree as ET
import datetime
import re
import math
import sys

import random
import socket
import struct
import json

import sshtunnel
import sys
import paramiko




def getting_ip(row):
    """This function calls the api and return the response"""
    url = f"https://freegeoip.app/json/{row}"       # getting records from getting ip address
    headers = {
        'accept': "application/json",
        'content-type': "application/json"
        }
    response = requests.request("GET", url, headers=headers)
    respond = json.loads(response.text)
    return respond

http = urllib3.PoolManager(ca_certs=certifi.where())


def getCollForStat(col):
    allCols = getMdbCursor()
    allCols.execute('SHOW COLUMNS FROM '+ 'HD_test'+'.'+col)
    cols = allCols.fetchall()

    list_allCols = []
    for i in  range(len(cols)):
        list_allCols = list_allCols + [cols[i][0]]
    list_allCols_str =  ','.join(list_allCols[1:])
    cur = getMdbCursor()
    CMD = "SELECT " + list_allCols_str +" FROM HD_test."+col+" WHERE Run > 100 ORDER BY HD_test."+col+".WhenEntered asc"
    cur.execute(CMD)
    df = pd.DataFrame(cur.fetchall())
    df_atm =  df
    df_atm.columns = list_allCols[1:]
    #df_atm = df_atm.drop_duplicates(['Run','Telescope'],keep='last')

    df_atm =df_atm.merge(df_run, on=['Run'])

    df_atm['Year'] = [e.year  for e in list(df_atm['TimeOfStart'])]
    df_atm['Month'] = [e.month  for e in list(df_atm['TimeOfStart'])]
    df_atm['doy'] = [e.timetuple().tm_yday for e in list(df_atm['TimeOfStart'])]
    df_atm['Nrun']=1
    df_atm.groupby(['Month']).sum()
    return df_atm
