from Main.BackEnd.imports.pythonSTD import *
from Main.BackEnd.imports.ploting import *
from Main.BackEnd.imports.flaskSTD import *
from Main.BackEnd.imports.astropySTD import *

from Main import db, bcrypt, mail
from Main.BackEnd.main.forms import  StartAnalysis, StartHessAna, SetUpConfig
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection, plot_map_image, plot_theta_squared_table_custom, plotRingSigni

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

main = Blueprint('main',__name__)



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

@main.route("/account", methods=['GET', 'POST'])
def account():
    form = SetUpConfig()
    if not os.path.exists( os.getcwd() + "/Main/static/configFile/"):
        os.makedirs( os.getcwd() + "/Main/static/configFile/")
    configPath = os.getcwd() + "/Main/static/configFile/"
    confExist = os.path.isfile(configPath+"config.csv")
    fileConfig = configPath+"config.csv"
    hessDataPath = ""

    if form.validate_on_submit():
        hessDataPath_new = form.hessDataPath.data
        if hessDataPath_new[-1] != '/':
            hessDataPath_new = hessDataPath_new + '/'
        df_config = pd.DataFrame.from_dict({'hessDataPath':[hessDataPath_new]})
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
                except:
                    print("No HESS PATH defined")

            form.hessDataPath.data =  hessDataPath
    flash('Welcome to your account')
    return render_template('main/accountConfig.html', form = form)

@main.route("/", methods=['GET', 'POST'])
@main.route("/hessana", methods=['GET', 'POST'])
def hessana():
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

            listrun = list(obsindex[obsindex['OBJECT'] == form.source.data]['OBS_ID'])

            if isRunList:
                print(runlist)
                listrun = [e  for e in listrun if e in runlist]
            else:
                listrun = [e for e in listrun if e >= form.rmin.data]
                listrun = [e for e in listrun if e <= form.rmax.data]

            ra_obj = list(obsindex['RA_OBJ'])[0]
            dec_obj = list(obsindex['DEC_OBJ'])[0]

            obs = data_store.obs(listrun[0])
            res_analysisName_base = form.analysisName.data
            res_analysisName = res_analysisName_base
            i = 0
            while  os.path.exists(resPath+'/'+res_analysisName):
                i = i + 1
                res_analysisName = res_analysisName_base +'_'+str(i)

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
            print(pd.DataFrame(listrun))
            print(resPath+'run_list_'+res_analysisName+'.csv')

            obs = data_store.get_observations(listrun)
            obs_list_events = [e.events for e in obs]
            combined_events = EventList.from_stack(obs_list_events)

            ### Creat events map
            plot_map_image(combined_events,resPath,res_analysisName,form.source.data)

            ### Create Theta2
            position = SkyCoord(ra=ra_obj, dec=dec_obj, unit="deg", frame="icrs")
            theta2_axis = MapAxis.from_bounds(0, 0.2, nbin=20, interp="lin", unit="deg2")

            observations = data_store.get_observations(listrun)
            theta2_table = make_theta_squared_table(
                observations=observations,
                position=position,
                theta_squared_axis=theta2_axis,
            )
            plot_theta_squared_table_custom(theta2_table,resPath,res_analysisName,form.source.data)
            pd.DataFrame(listrun).to_csv(resPath+res_analysisName+'/run_list_'+res_analysisName+'.csv')

            # Compute ring background significance
            print("### Computing ring background #####")
            ra_src = ra_obj
            dec_src = dec_obj
            print("### POSITION #####")
            print(ra_src)
            print(dec_src)
            observations = data_store.get_observations(listrun)

            energy_axis = MapAxis.from_energy_bounds(1.0, 10.0, 4, unit="TeV")

            geom = WcsGeom.create(
                skydir=(ra_src, dec_src),
                binsz=0.02,
                width=(2, 2),
                frame="icrs",
                proj="CAR",
                axes=[energy_axis],
            )

            # Reduced IRFs are defined in true energy (i.e. not measured energy).
            energy_axis_true = MapAxis.from_energy_bounds(0.5, 20, 10, unit="TeV", name="energy_true")

            stacked = MapDataset.create(geom=geom, energy_axis_true=energy_axis_true, name="crab-stacked")
            offset_max = 2.5 * u.deg
            maker = MapDatasetMaker()
            maker_safe_mask = SafeMaskMaker(methods=["offset-max", "aeff-max"], offset_max=offset_max)
            circle = CircleSkyRegion(center=SkyCoord(str(ra_src)+" deg", str(dec_src)+" deg"), radius=0.2 * u.deg)
            exclusion_mask = ~geom.region_mask(regions=[circle])
            maker_fov = FoVBackgroundMaker(method="fit", exclusion_mask=exclusion_mask)

            print("### Stack observations ... ")
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


            stacked.counts.sum_over_axes().smooth(0.05 * u.deg).plot(stretch="sqrt", add_cbar=True);

            # Define map geometry for binned simulation
            energy_reco = MapAxis.from_edges(np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log")
            geom = WcsGeom.create(skydir=(ra_src, dec_src),binsz=0.02,width=(2, 2),frame="icrs",axes=[energy_reco])

            # It is usually useful to have a separate binning for the true energy axis
            energy_true = MapAxis.from_edges(np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy_true", interp="log")
            energy = MapAxis.from_edges(np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy", interp="log")

            #geom = datasets[0].geoms['geom']
            energy_axis = energy
            geom_image = geom.to_image().to_cube([energy_axis.squash()])

            # Make the exclusion mask
            pointing = SkyCoord(ra_src, dec_src, unit="deg", frame="icrs")
            regions = CircleSkyRegion(center=SkyCoord(ra_src, dec_src, unit="deg"), radius=0.5 * u.deg)
            exclusion_mask = ~geom_image.region_mask([regions])
            exclusion_mask.geom

            print("### Ring background ... #####")
            ring_maker = RingBackgroundMaker(r_in="0.5 deg", width="0.3 deg")

            energy_axis_true = energy
            dataset_on_off = ring_maker.run(stacked.to_image())
            stacked.to_image().info_dict()

            estimator = ExcessMapEstimator(0.04 * u.deg, selection_optional=[])
            lima_maps = estimator.run(dataset_on_off)

            significance_map = lima_maps["sqrt_ts"]
            excess_map = lima_maps["npred_excess"]
            print("### PLOTING ... #####")
            print("saving in " + resPath+res_analysisName)
            plotRingSigni(significance_map,excess_map,np.array([exclusion_mask.data[0]]),resPath+res_analysisName,res_analysisName)

            plt.figure(figsize=(10, 10))
            ax1 = plt.subplot(221, projection=significance_map.geom.wcs)
            ax2 = plt.subplot(222, projection=excess_map.geom.wcs)

            ax1.set_title("Significance map")
            significance_map.plot(ax=ax1, add_cbar=True)

            ax2.set_title("Excess map")
            excess_map.plot(ax=ax2, add_cbar=True)

        return render_template('main/hessana.html',form = form, graphJSON = {})

    return render_template('main/hessana.html', form = form, graphJSON ={})


@main.route("/results", methods=['GET'])
def results():
    resPath = os.getcwd() + "/Main/static/results/"
    print('LIST DIR')
    listres = os.listdir(resPath)
    return render_template('main/results.html',listres=listres)

@main.route("/resultsplots/<string:folder>", methods=['GET'])
def resultsplots(folder):
    resPath = os.getcwd() + "/Main/static/results/"+folder+'/'
    print('LIST DIR')
    listres = os.listdir(resPath)
    return render_template('main/resultsplots.html',listres=listres,folder = folder)
