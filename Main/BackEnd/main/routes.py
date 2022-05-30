from Main.BackEnd.imports.pythonSTD import *
from Main.BackEnd.imports.ploting import *
from Main.BackEnd.imports.flaskSTD import *
from Main.BackEnd.imports.astropySTD import *

from Main import db, bcrypt, mail
from Main.BackEnd.main.forms import  StartAnalysis, StartHessAna, SetUpConfig, StartHess2D, StartHessDataq
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection, plot_map_image, plot_theta_squared_table_custom, plotRingSigni, GetDistanceFromCoor

from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom, Map,WcsNDMap
from gammapy.makers import MapDatasetMaker, RingBackgroundMaker
from gammapy.makers.utils import make_theta_squared_table
from gammapy.visualization import plot_theta_squared_table
from gammapy.data import EventList
from gammapy.datasets import MapDataset
from gammapy.makers import SafeMaskMaker, FoVBackgroundMaker
from regions import CircleSkyRegion
from gammapy.estimators import ExcessMapEstimator

from Main.PlotCreator import PlotCreator

main = Blueprint('main',__name__)


@main.route("/home", methods=['GET', 'POST'])
def home():
    try:
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
    except Exception as inst:
        inst = str(inst)
        return render_template('main/error.html',  inst =inst)

@main.route("/account", methods=['GET', 'POST'])
def account():
    try:
        form = SetUpConfig()
        if not os.path.exists( os.getcwd() + "/Main/static/configFile/"):
            os.makedirs( os.getcwd() + "/Main/static/configFile/")
        configPath = os.getcwd() + "/Main/static/configFile/"
        confExist = os.path.isfile(configPath+"config.csv")
        fileConfig = configPath+"config.csv"
        hessDataPath = ""

        if form.validate_on_submit():
            hessDataPath_new = form.hessDataPath.data
            ctaIRFSpath_new = form.ctaIrfsPath.data
            excludedRegionHESS = form.excludedRegionHESS.data
            linkWebSummary = form.linkWebSummary.data

            if hessDataPath_new[-1] != '/':
                hessDataPath_new = hessDataPath_new + '/'
            if ctaIRFSpath_new != '' and ctaIRFSpath_new != None:
                if ctaIRFSpath_new[-1] != '/':
                    ctaIRFSpath_new = ctaIRFSpath_new + '/'
            else:
                ctaIRFSpath_new = ''

            if excludedRegionHESS != '' and excludedRegionHESS != None:
                excludedRegionHESS = excludedRegionHESS
            else:
                excludedRegionHESS = ''

            if linkWebSummary != '' and linkWebSummary != None:
                if linkWebSummary[-1] != '/':
                    linkWebSummary = linkWebSummary + '/'
            else:
                linkWebSummary = ''

            df_config = pd.DataFrame.from_dict({'hessDataPath':[hessDataPath_new],'ctaIRFSpath':[ctaIRFSpath_new]
            ,'excludedRegionHESS':[excludedRegionHESS] ,'linkWebSummary':[linkWebSummary] })
            df_config.to_csv(fileConfig)

        elif request.method ==  'GET':
            if not confExist:
                open(fileConfig, 'a').close()
            else:
                try:
                    df_config = pd.read_csv(fileConfig)
                except:
                    df_config = pd.DataFrame.from_dict({})
                    print("No config file or empty ")

                if len(df_config) == 0:
                    print("NO default config")
                else:
                    try:
                        hessDataPath = list(df_config['hessDataPath'])[0]
                        form.hessDataPath.data =  hessDataPath
                    except:
                        print("No HESS PATH defined")
                    try:
                        ctaIRFSpath = list(df_config['ctaIRFSpath'])[0]
                        form.ctaIrfsPath.data =  ctaIRFSpath
                    except:
                        print("No CTA IRFS PATH defined")
                    try:
                        excludedRegionHESS = list(df_config['excludedRegionHESS'])[0]
                        form.excludedRegionHESS.data =  excludedRegionHESS
                    except:
                        print("No CTA IRFS PATH defined")
                    try:
                        linkWebSummary = list(df_config['linkWebSummary'])[0]
                        form.linkWebSummary.data =  linkWebSummary
                    except:
                        print("No CTA IRFS PATH defined")


        flash('Welcome to your account')
        return render_template('main/accountConfig.html', form = form)
    except Exception as inst:
        inst = str(inst)
        return render_template('main/error.html',  inst =inst)

@main.route("/", methods=['GET', 'POST'])
@main.route("/hessana", methods=['GET', 'POST'])
def hessana():
    try:
        form = StartHessAna()
        configPath = os.getcwd() + "/Main/static/configFile/"
        resPath = os.getcwd() + "/Main/static/results/"
        confExist = os.path.isfile(configPath+"config.csv")
        fileConfig = configPath+"config.csv"
        hessDataPath = ""
        pathConf = False
        try:
            df_config = pd.read_csv(fileConfig)
            hessDataPath = list(df_config['hessDataPath'])[0]
            pathConf = True
        except:
            print("No HESS DATA PATH defined")

        try:
            df_config = pd.read_csv(fileConfig)
            excludedRegionHESS = list(df_config['excludedRegionHESS'])[0]
        except:
            excludedRegionHESS = []
            print("No HESS excluded regions")

        table = Table.read(hessDataPath+'/obs-index.fits.gz', format='fits')
        obsindex = table.to_pandas()
        obsindex["OBJECT"] =  [ e.decode("utf-8")  for e in obsindex["OBJECT"] ]

        object  = np.unique(list(obsindex["OBJECT"]))
        objects = [(e,e) for e in object]
        form.source.choices = objects

        isRunList = True

        if form.validate_on_submit():
            if pathConf:
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


                data_store = DataStore.from_dir(hessDataPath)
                #data_store.obs_table[:][["OBS_ID", "DATE-OBS", "RA_PNT", "DEC_PNT", "OBJECT"]]
                obsindex_all = obsindex
                #obsindex['Distance'] =  obsindex.apply(lambda x : round(GetDistanceFromCoor(x['RA_PNT'], x['DEC_PNT'],float(lat),float(long))), axis=1)
                #obsindex = obsindex[obsindex['OBJECT'] == form.source.data]
                #listrun = list(obsindex_all['OBS_ID'])
                ra_obj = form.ra_src.data
                dec_obj = form.dec_src.data
                print("### POSITION #####")
                print(ra_obj)
                print(dec_obj)
                obsindex_all = obsindex_all[( abs(obsindex_all['RA_PNT']) < abs(ra_obj) + 3) & (abs(obsindex_all['RA_PNT']) > abs(ra_obj) - 3 )]
                obsindex_all = obsindex_all[( abs(obsindex_all['DEC_PNT']) < abs(dec_obj) + 3) & (abs(obsindex_all['DEC_PNT']) > abs(dec_obj) - 3 )]
                obsindex_all['Distance'] = [ math.sqrt((ra_obj-list(obsindex_all['RA_PNT'])[i])**2+(dec_obj-list(obsindex_all['DEC_PNT'])[i])**2 ) for i in range(len(obsindex_all))]
                obsindex_all = obsindex_all[obsindex_all['Distance'] <= form.distance.data]

                listrun_select_circle = list(obsindex_all['OBS_ID'])

                if isRunList:
                    print(runlist)
                    listrun = [e  for e in listrun_select_circle if e in runlist]
                else:
                    listrun = [e for e in listrun_select_circle if e >= form.rmin.data]
                    listrun = [e for e in listrun_select_circle if e <= form.rmax.data]

                res_analysisName_base = form.analysisName.data
                res_analysisName = res_analysisName_base
                i = 0
                while  os.path.exists(resPath+'/'+res_analysisName):
                    i = i + 1
                    res_analysisName = res_analysisName_base +'_'+str(i)
                # CREATE RES DIRECTORY
                os.makedirs(resPath+'/'+res_analysisName)
                ## Check if file exists
                list_run_all = listrun
                runs_notexist = []
                for e in list_run_all:
                    print(e)
                    try:
                        obs = data_store.obs(e)
                        print("### RUN : "+str(e)+" = " + str(obs.observation_time_duration)+" DURATION  #####")
                    except Exception:
                        print("RUN : "+str(e)+" ERROR")
                        runs_notexist = runs_notexist + [e]
                listrun = [run for run in list_run_all if run not in runs_notexist]
                print(listrun)
                ra_src = ra_obj
                dec_src = dec_obj

                energy_axis = MapAxis.from_energy_bounds(form.ana_emin.data, form.ana_emax.data, 20, unit="TeV")

                geom = WcsGeom.create(
                    skydir=(ra_src, dec_src),
                    binsz=0.02,
                    width=(form.map_size_X.data, form.map_size_Y.data),
                    frame="icrs",
                    proj="CAR",
                    axes=[energy_axis],
                )

                # Reduced IRFs are defined in true energy (i.e. not measured energy).
                energy_axis_true = MapAxis.from_energy_bounds(form.ana_emin.data*0.85, form.ana_emax.data*1.15, 30, unit="TeV", name="energy_true")

                stacked = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true, name="analyis-stacked")
                offset_max = form.max_evt_offset.data * u.deg

                maker = MapDatasetMaker()
                if form.apply_aeff_mask.data:
                    maker_safe_mask = SafeMaskMaker(methods=["offset-max", "aeff-max"], offset_max=offset_max, aeff_percent=form.apply_aeff_mask_value.data) #Possible to apply the aeff mask later?
                else:
                    maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=offset_max)

                # TODO : IMPLEMENT REGIONS HANDLING
                excluded_regions_list = pd.read_csv(excludedRegionHESS)
                excluded_regions_list['Distance'] = [ math.sqrt((ra_obj-list(excluded_regions_list['ra'])[i])**2+(dec_obj-list(excluded_regions_list['dec'])[i])**2 ) for i in range(len(excluded_regions_list))]
                search_size = np.max ([form.map_size_X.data,form.map_size_Y.data])
                excluded_regions_list =  excluded_regions_list[excluded_regions_list['Distance'] < search_size]
                excluded_regions_list.to_csv(resPath+res_analysisName+'/excludedregions_'+res_analysisName+'.csv')

                print("### POSITION #####")
                print(ra_obj)
                print(dec_obj)
                print(search_size)
                print(excluded_regions_list)

                all_circle_regions = [CircleSkyRegion(center=SkyCoord(str(ra_obj)+" deg", str(dec_obj)+" deg"), radius=0.3 * u.deg)]
                for i in range(len(excluded_regions_list)):
                    all_circle_regions = all_circle_regions + [CircleSkyRegion(center=SkyCoord(str(list(excluded_regions_list['ra'])[0])+" deg", str(list(excluded_regions_list['dec'])[0])+" deg"), radius=list(excluded_regions_list['radius'])[0] * u.deg)]
                exclusion_mask = ~geom.region_mask(regions=all_circle_regions)
                maker_fov = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

                observations = data_store.get_observations(listrun)
                for obs in observations:
                    try:
                        # First a cutout of the target map is produced
                        cutout = stacked.cutout(obs.pointing_radec, width=2 * offset_max, name=f"obs-{obs.obs_id}")
                        # A MapDataset is filled in this cutout geometry
                        dataset = maker.run(cutout, obs)
                        # The data quality cut is applied
                        dataset = maker_safe_mask.run(dataset, obs)
                        # fit background model
                        dataset = maker_fov.run(dataset)
                        print(f"Background norm obs {obs.obs_id}: {dataset.background_model.spectral_model.norm.value:.2f}")
                        # The resulting dataset cutout is stacked onto the final one
                        stacked.stack(dataset)
                    except Exception:
                        print('ERROR with '+ str(obs))
                        print(str(obs))
                print(stacked)
                filename = resPath+'/'+res_analysisName + "/stacked-dataset.fits.gz"
                stacked.write(filename, overwrite=True)
                pd.DataFrame(listrun).to_csv(resPath+res_analysisName+'/run_list_'+res_analysisName+'.csv')

                f = open(resPath+res_analysisName+'/run_list_'+res_analysisName+'.txt', "a")
                for e in listrun:
                    f.write(str(e)+'\n')
                f.close()


            return render_template('main/hessana.html',form = form, graphJSON = {})

        return render_template('main/hessana.html', form = form, graphJSON ={})
    except Exception as inst:
        inst = str(inst)
        return render_template('main/error.html',  inst =inst)

@main.route("/hessdataq", methods=['GET', 'POST'])
def hessdataq():
    configPath = os.getcwd() + "/Main/static/configFile/"
    resPath = os.getcwd() + "/Main/static/results/"
    listresfiles = os.listdir(resPath)
    confExist = os.path.isfile(configPath+"config.csv")
    fileConfig = configPath+"config.csv"
    if not confExist:
        open(fileConfig, 'a').close()
        webSumAddress = ''
    else:
        df_config = pd.read_csv(fileConfig)
        print(df_config)
        try:
            df_config = pd.read_csv(fileConfig)
            webSumAddress = list(df_config['linkWebSummary'])[0]
        except:
            df_config = pd.DataFrame.from_dict({})
            print("No config file or empty ")
            webSumAddress = ''
    form = StartHessDataq()

    listresfilesdict = []
    for e in listresfiles:
        listresfilesdict = listresfilesdict + [(e,e)]
    form.analysis.choices = listresfilesdict

    isRunList = True
    print(webSumAddress)
    if form.validate_on_submit():
        res_analysisName = form.analysis.data
        try:
            listrun_csv = pd.read_csv(resPath+res_analysisName+'/run_list_'+res_analysisName+'.csv')
            listrun = list(listrun_csv[listrun_csv.columns[1]])

            response = requests.post(webSumAddress+"uploadrunlistSumAPI", data=json.dumps({"runlist": listrun}))

            res= response.json()
            df_final = pd.DataFrame({
            'Ntel':res['plotNTel']['Ntel'],
            'Value':res['plotNTel']['valTel'],
            'Duration':res['plotDuration']['runDuration'],
            'Distance':res['plotOffAxis']['offAxis']
            })


            fig1 = px.histogram(df_final, x="Ntel", color ='Value')
            fig1.update_traces(hovertemplate=None)
            fig1.update_layout(hovermode='x unified')
            fig1.write_image(resPath+res_analysisName+"/plotNTel.png")
            fig2 = px.histogram(df_final, x="Duration")
            fig2.update_traces(hovertemplate=None)
            fig2.update_layout(hovermode='x unified')
            fig2.write_image(resPath+res_analysisName+"/plotRunsDuration.png")
            fig3 = px.histogram(df_final, x="Distance")
            fig3.update_traces(hovertemplate=None)
            fig3.update_layout(hovermode='x unified')
            fig3.write_image(resPath+res_analysisName+"/plotOffAxis.png")

            df_atm = pd.DataFrame({
            'TransparencyCoefficient_CT1':res['transparency']['TransparencyCoefficient_CT1'],
            'TransparencyCoefficient_CT2':res['transparency']['TransparencyCoefficient_CT2'],
            'TransparencyCoefficient_CT3':res['transparency']['TransparencyCoefficient_CT3'],
            'TransparencyCoefficient_CT4':res['transparency']['TransparencyCoefficient_CT4'],
            'TransparencyCoefficient_CT5':res['transparency']['TransparencyCoefficient_CT5'],
            'TimeOfStart':res['transparency']['TimeOfStart'],
            'Run':res['transparency']['Run']
            })
            fig = go.Figure()
            for e in ['TransparencyCoefficient_CT1','TransparencyCoefficient_CT2','TransparencyCoefficient_CT3','TransparencyCoefficient_CT4','TransparencyCoefficient_CT5']:
                fig.add_trace(go.Scatter( x=df_atm.TimeOfStart,y=df_atm[e],name=e))
            fig.update_traces(mode="markers", hovertemplate=None)
            fig.update_layout(hovermode='x unified')
            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            fig = go.Figure()
            for e in ['TransparencyCoefficient_CT1','TransparencyCoefficient_CT2','TransparencyCoefficient_CT3','TransparencyCoefficient_CT4','TransparencyCoefficient_CT5']:
                fig.add_trace(go.Scatter( x=df_atm.Run,y=df_atm[e],name=e))
            fig.update_traces(mode="markers", hovertemplate=None)
            fig.update_layout(hovermode='x unified')
            fig.write_image(resPath+res_analysisName+"/transparencyCoeff.png")

            df_bp = pd.DataFrame({
            'Num_Broken':res['bp']['Num_Broken'],
            'Telescope':res['bp']['Telescope'],
            'TimeOfStart':res['bp']['TimeOfStart']
            })
            PlotCreator().stdScatterPlotTelCol(data=df_bp, xcol='TimeOfStart', ycol='Num_Broken', colorcol='Telescope', hoverdata='Num_Broken',name =resPath+res_analysisName+"/nBrokenPix.png")


            df_dead = pd.DataFrame({
            'Deadtime_mean':res['deadtime']['Deadtime_mean'],
            'Telescope':res['deadtime']['Telescope'],
            'TimeOfStart':res['deadtime']['TimeOfStart']
            })
            PlotCreator().stdScatterPlotTelCol(data=df_dead, xcol='TimeOfStart', ycol='Deadtime_mean', colorcol='Telescope', hoverdata='Deadtime_mean',name =resPath+res_analysisName+"/Deadtime_mean.png")


            df_trigger = pd.DataFrame({
            'Rate_mean':res['trigger']['Rate_mean'],
            'Telescope':res['trigger']['Telescope'],
            'TimeOfStart':res['trigger']['TimeOfStart']
            })
            PlotCreator().stdScatterPlotTelCol(data=df_trigger, xcol='TimeOfStart', ycol='Rate_mean', colorcol='Telescope', hoverdata='Rate_mean',name =resPath+res_analysisName+"/Rate_mean.png")

        except Exception:
            print("NO CONNECTION TO WEB SUMMARY !!!!")

    return render_template('main/hessdataq.html', form = form, graphJSON ={})

@main.route("/hess2d", methods=['GET', 'POST'])
def hess2d():
    configPath = os.getcwd() + "/Main/static/configFile/"
    resPath = os.getcwd() + "/Main/static/results/"
    listresfiles = os.listdir(resPath)

    confExist = os.path.isfile(configPath+"config.csv")
    fileConfig = configPath+"config.csv"
    hessDataPath = ""
    pathConf = False
    try:
        df_config = pd.read_csv(fileConfig)
        hessDataPath = list(df_config['hessDataPath'])[0]
        pathConf = True
    except:
        print("No HESS DATA PATH defined")

    data_store = DataStore.from_dir(hessDataPath)
    form = StartHess2D()

    listresfilesdict = []
    for e in listresfiles:
        listresfilesdict = listresfilesdict + [(e,e)]
    form.analysis.choices = listresfilesdict

    isRunList = True

    if form.validate_on_submit():
        res_analysisName = form.analysis.data

        listrun_csv = pd.read_csv(resPath+res_analysisName+'/run_list_'+res_analysisName+'.csv')
        listrun = list(listrun_csv[listrun_csv.columns[1]])

        stacked = MapDataset.read(resPath+res_analysisName+"/stacked-dataset.fits.gz")
        #stacked.counts.sum_over_axes().smooth(0.05 * u.deg).plot(stretch="sqrt", add_cbar=True)
        print(stacked.info_dict())
        print(stacked.geoms['geom'].center_skydir.ra.degree)
        ra_src = stacked.geoms['geom'].center_skydir.ra.degree
        dec_src = stacked.geoms['geom'].center_skydir.dec.degree
        print(stacked.to_dict())

        # Define map geometry
        #energy_reco = MapAxis.from_edges(np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log")
        #energy_true = MapAxis.from_edges(np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy_true", interp="log")
        energy = stacked.geoms['geom'].axes['energy']#     MapAxis.from_edges(np.logspace(-1.0, 0.0, 30), unit="TeV", name="energy", interp="log") #Is this actually used???
        geom = WcsGeom.create(skydir=(ra_src, dec_src),binsz=0.02,width=stacked.geoms['geom'].width,frame="icrs",axes=[energy])

        #geom = datasets[0].geoms['geom']
        energy_axis = energy
        geom_image = geom.to_image().to_cube([energy_axis.squash()])

        # Make the exclusion mask
        # LOAD EXCLUDED regions
        try:
            excluded_regions_list = pd.read_csv(resPath+res_analysisName+'/excludedregions_'+res_analysisName+'.csv')
        except:
            excluded_regions_list = pd.DataFrame.from_dict({})
        print("EXCLUDED REGIONS")
        print(excluded_regions_list)
        pointing = SkyCoord(ra_src, dec_src, unit="deg", frame="icrs")
        regions = [CircleSkyRegion(center=SkyCoord(ra_src, dec_src, unit="deg"), radius=0.3 *u.deg)]
        if len(excluded_regions_list) >0:
            for i in range(len(excluded_regions_list)):
                regions = regions + [CircleSkyRegion(center=SkyCoord(str(list(excluded_regions_list['ra'])[0])+" deg", str(list(excluded_regions_list['dec'])[0])+" deg"), radius=list(excluded_regions_list['radius'])[0] * u.deg)]
        ######
        exclusion_mask = ~geom_image.region_mask(regions)
        exclusion_mask.geom

        print("### Ring background ... #####")
        ring_maker = RingBackgroundMaker(r_in=form.ring_inner_radius.data * u.deg, width=form.ring_width.data *u.deg)

        #energy_axis_true = energy
        dataset_on_off = ring_maker.run(stacked.to_image())
        stacked.to_image().info_dict()
        print("=>",form.os_radius.data)
        estimator = ExcessMapEstimator(form.os_radius.data *u.deg, selection_optional=[])
        lima_maps = estimator.run(dataset_on_off)

        significance_map = lima_maps["sqrt_ts"]
        print(significance_map)
        excess_map = lima_maps["npred_excess"]
        masked_significance_map = exclusion_mask * significance_map
        print("### PLOTING ... #####")

        print("saving in " + resPath+res_analysisName)
        plotRingSigni(significance_map,excess_map,np.array([exclusion_mask.data[0]]),resPath+res_analysisName,res_analysisName)

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(221, projection=significance_map.geom.wcs)
        ax2 = plt.subplot(222, projection=excess_map.geom.wcs)
        ax3 = plt.subplot(223, projection=masked_significance_map.geom.wcs)
        ax1.set_title("Significance map")
        significance_map.plot(ax=ax1, add_cbar=True)
        ax2.set_title("Excess map")
        excess_map.plot(ax=ax2, add_cbar=True)
        ax3.set_title("Masked significance map")
        masked_significance_map.plot(ax=ax3, add_cbar=True)
        ax3 .get_figure().savefig(resPath+res_analysisName +'/2Dmaps.png')

        # Theta 2 Plot
        position = SkyCoord(ra=ra_src, dec=dec_src, unit="deg", frame="icrs")
        theta2_axis = MapAxis.from_bounds(0, 0.2, nbin=20, interp="lin", unit="deg2")

        observations = data_store.get_observations(listrun)
        theta2_table = make_theta_squared_table(
            observations=observations,
            position=position,
            theta_squared_axis=theta2_axis,
        )
        plot_theta_squared_table_custom(theta2_table,resPath,res_analysisName,'source')

    return render_template('main/hess2d.html', form = form, graphJSON ={})

@main.route("/tuto", methods=['GET'])
def tuto():
    return render_template('main/tutorial.html')

@main.route("/getCoordFromSource/<string:source>", methods=['GET','POST'])
def getCoordFromSource(source):
    print('IN getCoordFromSource')
    configPath = os.getcwd() + "/Main/static/configFile/"
    resPath = os.getcwd() + "/Main/static/results/"
    confExist = os.path.isfile(configPath+"config.csv")
    fileConfig = configPath+"config.csv"
    hessDataPath = ""
    pathConf = False

    sourcesearch=source
    try:
        df_config = pd.read_csv(fileConfig)
        hessDataPath = list(df_config['hessDataPath'])[0]
        pathConf = True
    except:
        print("No HESS DATA PATH defined")

    table = Table.read(hessDataPath+'/obs-index.fits.gz', format='fits')
    obsindex = table.to_pandas()
    obsindex["OBJECT"] =  [ e.decode("utf-8")  for e in obsindex["OBJECT"] ]
    res = obsindex[obsindex['OBJECT'] ==sourcesearch ][:1].to_dict('records')
    res = [ res[0]['RA_OBJ'],res[0]['DEC_OBJ']]
    return jsonify(matching_results=res)


@main.route("/results", methods=['GET'])
def results():
    try:
        resPath = os.getcwd() + "/Main/static/results/"
        listres = os.listdir(resPath)
        return render_template('main/results.html',listres=listres)
    except Exception as inst:
        inst = str(inst)
        return render_template('main/error.html',  inst =inst)


@main.route("/resultsplots/<string:folder>", methods=['GET'])
def resultsplots(folder):
    #try:
    resPath = os.getcwd() + "/Main/static/results/"+folder+'/'
    listres = os.listdir(resPath)
    listres = [e for e in listres if (e.endswith('.png') or e.endswith('.jpg')) ]
    isCTA = False
    isAnalysis= False
    if folder != 'CTA':
        # Check if spectrum
        isAnalysis = '2Dmaps.png' in listres
        isSpectrum = 'spectrum.jpg' in listres
        if not isSpectrum:
            listres = listres+['spectrum.jpg','spectrum2.jpg','spectrumfit.jpg']

        # check if data quelity runs here
        isDQ = 'plotNTel.png' in listres
        #ORDENING PICTURES
        listres_order = []
        for pic in listres :
            #print(pic)
            if pic.endswith('2Dmaps.png'):
                listres_order = listres_order + [0]
            if pic.endswith('distribution.png'):
                listres_order = listres_order + [1]
            if pic.endswith('theta2.png'):
                listres_order = listres_order + [2]
            if pic.endswith('trum.jpg'):
                listres_order = listres_order + [3]
            if pic.endswith('trum2.jpg'):
                listres_order = listres_order + [4]
            if pic.endswith('trumfit.jpg'):
                listres_order = listres_order + [5]
            if isDQ:
                if pic.endswith('plotNTel.png'):
                    listres_order = listres_order + [6]
                if pic.endswith('ration.png'):
                    listres_order = listres_order + [7]
                if pic.endswith('Axis.png'):
                    listres_order = listres_order + [8]
                if pic.endswith('Coeff.png'):
                    listres_order = listres_order + [9]
                if pic.endswith('te_mean.png'):
                    listres_order = listres_order + [10]
                if pic.endswith('time_mean.png'):
                    listres_order = listres_order + [11]
                if pic.endswith('Pix.png'):
                    listres_order = listres_order + [12]

        #print(listres)
        #print(len(listres))
        #print(listres_order)
        #print(len(listres_order))
        df = pd.DataFrame({'files' :listres,'order': listres_order})
        df = df.sort_values(['order'])
        listres = list(df['files'])
    else:
        isCTA = True
        isSpectrum = False
        isDQ = False

    return render_template('main/resultsplots.html',listres=listres, folder = folder, isSpectrum=isSpectrum, isDQ=isDQ, isCTA=isCTA,isAnalysis=isAnalysis)
    #except Exception as inst:
    #    inst = str(inst)
    #    return render_template('main/error.html',  inst =inst)
