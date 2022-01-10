from Main.BackEnd.imports.pythonSTD import *
from Main.BackEnd.imports.ploting import *
from Main.BackEnd.imports.flaskSTD import *
from Main.BackEnd.imports.astropySTD import *

from Main import db, bcrypt, mail
from Main.BackEnd.main.forms import  StartAnalysis
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection

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
