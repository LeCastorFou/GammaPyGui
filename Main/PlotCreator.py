
import pandas as pd
import plotly.express as px
import chart_studio.plotly as py
import json
import plotly
import datetime
from datetime import timedelta
import numpy as np
import math
import plotly.express as px
import plotly.graph_objects as go

class PlotCreator(object):

    def __init__(self):
        self.style = ""

    def stdScatterPlotTelCol(self,data=pd.DataFrame([]),xcol="",ycol="",colorcol="",hoverdata="",name='plot'):
        fig = px.scatter(data, x=xcol, y=ycol, color=colorcol,hover_data=[hoverdata],color_discrete_map = {'1': 'blue', '2': 'red', '3': 'green','4':'orange','5':'grey'})
        fig.update_traces(mode="markers", hovertemplate=None)
        fig.update_layout(hovermode='x unified')
        fig.write_image(name)
        return None
