from Main.BackEnd.imports.pythonSTD import *
from Main.BackEnd.imports.ploting import *
from Main.BackEnd.imports.flaskSTD import *
from Main.BackEnd.imports.astropySTD import *

from Main import db, bcrypt, mail
from Main.BackEnd.main.forms import  StartAnalysis, StartHessAna, SetUpConfig
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection, plot_map_image, plot_theta_squared_table_custom

from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom, Map,WcsNDMap
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

            plot_map_image(obs.events,resPath,res_analysisName,form.source.data)
            position = SkyCoord(ra=ra_obj, dec=dec_obj, unit="deg", frame="icrs")
            theta2_axis = MapAxis.from_bounds(0, 0.2, nbin=20, interp="lin", unit="deg2")

            observations = data_store.get_observations(listrun)
            theta2_table = make_theta_squared_table(
                observations=observations,
                position=position,
                theta_squared_axis=theta2_axis,
            )

            plot_theta_squared_table_custom(theta2_table,resPath,res_analysisName,form.source.data)
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
