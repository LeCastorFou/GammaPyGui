from Main.BackEnd.imports.pythonSTD import *
from Main.BackEnd.imports.ploting import *
from Main.BackEnd.imports.flaskSTD import *
from Main.BackEnd.imports.astropySTD import *

from Main import db, bcrypt, mail
from Main.BackEnd.spectrum.forms import  LaunchSpectrum
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection
from Main.BackEnd.spectrum.utils import tableToPandas

import gammapy

import regions


from regions import CircleSkyRegion
from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    SpectrumDataset,
    SpectrumDatasetOnOff,
    FluxPointsDataset,
)
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    LogParabolaSpectralModel,
    BrokenPowerLawSpectralModel,
    ExpCutoffPowerLawSpectralModel,
    create_crab_spectral_model,
    SkyModel,
)
from gammapy.makers import (
    SafeMaskMaker,
    SpectrumDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.visualization import plot_spectrum_datasets_off_regions

spectrumbp = Blueprint('spectrumbp',__name__)


@spectrumbp.route("/spectrum1D", methods=['GET', 'POST'])
def spectrum1D():
    #try:
    form = LaunchSpectrum()
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
    folder = ''
    plot =''
    print('Is path Config existing?')
    print(pathConf)
    if pathConf:
        table = Table.read(hessDataPath+'/obs-index.fits.gz', format='fits')
        obsindex = table.to_pandas()
        obsindex["OBJECT"] =  [ e.decode("utf-8")  for e in obsindex["OBJECT"] ]
        object  = np.unique(list(obsindex["OBJECT"]))
        objects = [(e,e) for e in object]
        form.source.choices = objects

        isRunList = True

        if form.validate_on_submit():
            analysisName = request.form['tags']
            if analysisName not in listresfiles:
                print('CREATING ANALYIS FOLDER')
                os.makedirs(resPath+analysisName)
            print('Name of the spectrum : ' +analysisName)
            resPath+analysisName

            if 'file' not in request.files:
                flash('No RUN LIST SELECTED')
                isRunList = False
            file = request.files['file']
            # If the user does not select a file, the browser submits an
            # empty file without a filename.
            if file.filename == '':
                flash('No RUN LIST SELECTED')
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

            datastore = DataStore.from_dir(hessDataPath)
            obsindex = obsindex[obsindex['OBJECT']== form.source.data]
            listrun = list(obsindex['OBS_ID'])

            if isRunList:
                listrun = [e  for e in listrun if e in runlist]
            else:
                listrun = [e for e in listrun if e >= form.rmin.data]
                listrun = [e for e in listrun if e <= form.rmax.data]

            ra_obj = list(obsindex['RA_OBJ'])[0]
            dec_obj = list(obsindex['DEC_OBJ'])[0]

            obs_ids = listrun
            #obs_ids = [23134,23155,23156,23304,23309,23310,23523]
            #print(obs_ids)
            print("GETTING RUNS INFOS")
            observations = datastore.get_observations(obs_ids)
            target_position = SkyCoord(ra=ra_obj, dec=dec_obj, unit="deg", frame="icrs")
            on_region_radius = Angle("0.11 deg")
            on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)


            exclusion_region = CircleSkyRegion(
                center=SkyCoord(0, 0, unit="deg", frame="galactic"),
                radius=0.0 * u.deg,
            )

            skydir = target_position.galactic
            geom = WcsGeom.create(
                npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
            )

            exclusion_mask = ~geom.region_mask([exclusion_region])
            exclusion_mask.plot()

            energy_axis = MapAxis.from_energy_bounds(
                0.1, 40, nbin=10, per_decade=True, unit="TeV", name="energy"
            )
            energy_axis_true = MapAxis.from_energy_bounds(
                0.05, 100, nbin=20, per_decade=True, unit="TeV", name="energy_true"
            )

            geom = RegionGeom.create(region=on_region, axes=[energy_axis])
            dataset_empty = SpectrumDataset.create(
                geom=geom, energy_axis_true=energy_axis_true
            )

            dataset_maker = SpectrumDatasetMaker(
                containment_correction=True, selection=["counts", "exposure", "edisp"]
            )
            bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
            safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=10)

            datasets = Datasets()
            print("MERGING OBS ...")
            for obs_id, observation in zip(obs_ids, observations):
                try:
                    dataset = dataset_maker.run(dataset_empty.copy(name=str(obs_id)), observation)
                    dataset_on_off = bkg_maker.run(dataset, observation)
                    dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
                    datasets.append(dataset_on_off)
                except Exception as inst:
                    print(inst)
                    print("ERROR WITH RUN " + str(obs_id))
            print("DONE")

            spectral_model = ExpCutoffPowerLawSpectralModel(
                amplitude=1e-12 * u.Unit("cm-2 s-1 TeV-1"),
                index=2,
                lambda_=0.1 * u.Unit("TeV-1"),
                reference=1 * u.TeV,
            )
            model = SkyModel(spectral_model=spectral_model, name="crab")

            datasets.models = [model]
            fit_joint = Fit()
            result_joint = fit_joint.run(datasets=datasets)

            # we make a copy here to compare it later
            model_best_joint = model.copy()
            datasets.info_table()
            print("PLOTTING FITS")
            ax_spectrum, ax_residuals = datasets[0].plot_fit()
            ax_spectrum.set_ylim(0.1, 40)
            ax_spectrum.get_figure().savefig(resPath+analysisName+'/spectrum.jpg')

            e_min, e_max = 0.7, 30
            energy_edges = np.geomspace(e_min, e_max, 11) * u.TeV
            print("STARTING FluxPointsEstimator ...")
            fpe = FluxPointsEstimator(energy_edges=energy_edges, source="crab", selection_optional="all")
            flux_points = fpe.run(datasets=datasets)
            df = flux_points.to_table(sed_type="dnde", formatted=True)
            names = [name for name in df.colnames if len(df[name].shape) <= 1]
            df = df[names].to_pandas()
            print("PLOT RESULTS ...")
            plt.figure(figsize=(8, 5))
            ax = flux_points.plot(sed_type="e2dnde", color="darkorange")
            flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde");
            ax.get_figure().savefig(resPath+analysisName+'/spectrum2.jpg')

            #### FITTING WITh flux point ##############
            dataset_gammacat = FluxPointsDataset(data=flux_points, name="gammacat")
            flux_points_df = tableToPandas(flux_points.to_table(sed_type="e2dnde", formatted=True))
            flux_points_df.to_csv(resPath+analysisName+'/fluxpoint_res.csv')
            dataset_gammacat.data.to_table(sed_type="dnde", formatted=True)

            if form.model.data == 'PowerLawSpectralModel':
                pwl = PowerLawSpectralModel(index=2, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV")
            if form.model.data == 'LogParabolaSpectralModel':
                pwl = LogParabolaSpectralModel(amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV",alpha=2.3,beta=0.2)
            if form.model.data == 'ExpCutoffPowerLawSpectralModel':
                pwl = ExpCutoffPowerLawSpectralModel(index=2, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV",lambda_="0.1 TeV-1",alpha=2.3)
            if form.model.data == 'BrokenPowerLawSpectralModel':
                pwl = BrokenPowerLawSpectralModel(index1=2, index2=3,  amplitude="1e-12 cm-2 s-1 TeV-1", ebreak=3)

            model = SkyModel(spectral_model=pwl, name="crab")


            datasets = Datasets([dataset_gammacat])
            datasets.models = model
            fitter = Fit()
            result_pwl = fitter.run(datasets=datasets)

            original_stdout = sys.stdout
            with open(resPath+analysisName+'/modelres.txt', 'w') as f:
                sys.stdout = f # Change the standard output to the file we created.
                print(datasets.models)
                sys.stdout = original_stdout

            ax = plt.subplot()
            kwargs = {"ax": ax, "sed_type": "e2dnde"}

            for d in datasets:
                d.data.plot(label=d.name, **kwargs)

            energy_bounds = [1e-1, 1e2] * u.TeV
            pwl.plot(energy_bounds=energy_bounds, color="k", **kwargs)
            pwl.plot_error(energy_bounds=energy_bounds, **kwargs)
            ax.set_ylim(np.min(flux_points_df['e2dnde'])/5, np.max(flux_points_df['e2dnde'])*5)
            ax.set_xlim(energy_bounds)
            ax.legend()
            ax.get_figure().savefig(resPath+analysisName+'/spectrumfit.jpg')



            return render_template('spectrum/index_spectrum.html', resname='' , form=form, folder = folder,plot=plot,listresfiles=listresfiles)

    return render_template('spectrum/index_spectrum.html', resname='def.png' , form=form, folder = folder,plot=plot,listresfiles=listresfiles)
    #except Exception as inst:
    #    inst = str(inst)
    #    return render_template('main/error.html',  inst =inst)
