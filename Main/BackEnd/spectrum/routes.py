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
    MapDataset,
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
    plot = ''
    print('Is path Config existing?')
    print(pathConf)

    if pathConf:
        table = Table.read(hessDataPath+'/obs-index.fits.gz', format='fits')
        obsindex = table.to_pandas()
        obsindex["OBJECT"] =  [ e.decode("utf-8")  for e in obsindex["OBJECT"] ]
        object  = np.unique(list(obsindex["OBJECT"]))
        objects = [(e,e) for e in object]

        listresfilesdict = []
        for e in listresfiles:
            listresfilesdict = listresfilesdict + [(e,e)]
        form.source.choices = listresfilesdict
        #form.source.choices = objects

        isRunList = True
        print("form.validate_on_submit()")
        print(form.errors)
        if form.validate_on_submit():
            res_analysisName = form.source.data
            analysisName = res_analysisName
            listrun_csv = pd.read_csv(resPath+res_analysisName+'/run_list_'+res_analysisName+'.csv')
            listrun = list(listrun_csv[listrun_csv.columns[1]])

            stacked = MapDataset.read(resPath+res_analysisName+"/stacked-dataset.fits.gz")
            stacked.counts.sum_over_axes().smooth(0.05 * u.deg).plot(stretch="sqrt", add_cbar=True)
            print(stacked.info_dict())
            print(stacked.geoms['geom'].center_skydir.ra.degree)
            ra_src = stacked.geoms['geom'].center_skydir.ra.degree
            dec_src = stacked.geoms['geom'].center_skydir.dec.degree
            datastore = DataStore.from_dir(hessDataPath)

            ##
            ra_obj = ra_src
            dec_obj = dec_src
            obs_ids = listrun

            print("GETTING RUNS INFOS")
            observations = datastore.get_observations(obs_ids)
            target_position = SkyCoord(ra=ra_obj, dec=dec_obj, unit="deg", frame="icrs")
            on_region_radius = Angle(form.on_size.data *u.deg)
            on_region = CircleSkyRegion(center=target_position, radius=on_region_radius)

            ## TODO : TAKE CARE OF EXCLUSION REGIONS
            exclusion_region = CircleSkyRegion(
                center=SkyCoord(0, 0, unit="deg", frame="galactic"),
                radius=0.0 * u.deg,
            )

            excluded_regions_list = pd.read_csv(resPath+res_analysisName+'/excludedregions_'+res_analysisName+'.csv')
            print("EXCLUDED REGIONS")
            print(excluded_regions_list)
            pointing = SkyCoord(ra_src, dec_src, unit="deg", frame="icrs")
            exclusion_region = [CircleSkyRegion(center=SkyCoord(ra_src, dec_src, unit="deg"), radius=0.3 *u.deg)]
            if len(excluded_regions_list) >0:
                for i in range(len(excluded_regions_list)):
                    exclusion_region = exclusion_region + [CircleSkyRegion(center=SkyCoord(str(list(excluded_regions_list['ra'])[0])+" deg", str(list(excluded_regions_list['dec'])[0])+" deg"), radius=list(excluded_regions_list['radius'])[0] * u.deg)]


            skydir = target_position.galactic
            geom = WcsGeom.create(
                npix=(150, 150), binsz=0.05, skydir=skydir, proj="TAN", frame="icrs"
            )

            exclusion_mask = ~geom.region_mask(exclusion_region)
            exclusion_mask.plot()

            energy = stacked.geoms['geom'].axes['energy']#     MapAxis.from_edges(np.logspace(-1.0, 0.0, 30), unit="TeV", name="energy", interp="log") #Is this actually used???
            geom = WcsGeom.create(skydir=(ra_src, dec_src),binsz=0.02,width=stacked.geoms['geom'].width,frame="icrs",axes=[energy])

            energy_axis = MapAxis.from_energy_bounds(
                form.spec_emin.data, form.spec_emax.data, nbin=form.spec_ebins.data, per_decade=True, unit="TeV", name="energy"
            )
            energy_axis_true = MapAxis.from_energy_bounds(
                form.spec_emin.data*0.85, form.spec_emax.data*1.15, nbin=form.spec_ebins.data*2, per_decade=True, unit="TeV", name="energy_true"
            )

            geom = RegionGeom.create(region=on_region, axes=[energy_axis])
            dataset_empty = SpectrumDataset.create(
                geom=geom, energy_axis_true=energy_axis_true
            )

            dataset_maker = SpectrumDatasetMaker(
                containment_correction=form.containment_bool.data, selection=["counts", "exposure", "edisp"]
            )
            bkg_maker = ReflectedRegionsBackgroundMaker(exclusion_mask=exclusion_mask)
            safe_mask_masker = SafeMaskMaker(methods=["aeff-max"], aeff_percent=form.aeff_mask_value.data)

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
            ## POSSIBLE TO DO THAT ONLY ONCE ? (to be able to refit with different parameters later without remerging)

            print("FIT SPECTRUM...")
            if request.form['modelfit'] == '0':
                spectral_model = PowerLawSpectralModel(index=form.index.data, amplitude=form.amplitude.data+" cm-2 s-1 TeV-1", reference=str(form.reference.data)+" TeV")
            elif request.form['modelfit'] == '1':
                spectral_model = BrokenPowerLawSpectralModel(index1=form.index.data, index2=form.index2.data,  amplitude=form.amplitude.data+" cm-2 s-1 TeV-1", ebreak=str(form.ebreak.data)+' 1 TeV')
            elif request.form['modelfit'] == '2':
                spectral_model = LogParabolaSpectralModel(amplitude=form.amplitude.data+" cm-2 s-1 TeV-1", reference=str(form.reference.data)+" TeV",alpha=form.alpha.data,beta=form.beta.data)
            elif request.form['modelfit'] == '3':
                spectral_model = ExpCutoffPowerLawSpectralModel(index=form.index.data, amplitude=form.amplitude.data+" cm-2 s-1 TeV-1", reference=str(form.reference.data)+" TeV",lambda_=str(form.lambdat.data)+" TeV-1",alpha=form.alpha.data)
            else:
                spectral_model = PowerLawSpectralModel(index=2, amplitude="1e-12 cm-2 s-1 TeV-1", reference="1 TeV")
            model = SkyModel(spectral_model=spectral_model, name="myskymod")

            datasets.models = [model]
            fit_joint = Fit()
            result_joint = fit_joint.run(datasets=datasets)

            model_best_joint = model.copy()
            print(datasets.models.to_parameters_table())
            print("DONE")


            print("PLOTTING FITS")
            ax_spectrum, ax_residuals = datasets[0].plot_fit()
            #ax_spectrum.set_ylim(0.01, 40)
            ax_spectrum.get_figure().savefig(resPath+analysisName+'/spectrum.jpg')

            if form.compute_points.data:
                print("STARTING FluxPointsEstimator ...")
                print(datasets)
                e_min, e_max = 0.3, 10
                energy_edges = np.geomspace(e_min, e_max, 15) * u.TeV

                fpe = FluxPointsEstimator(energy_edges=energy_edges, source="myskymod", selection_optional="all")
                flux_points = fpe.run(datasets=datasets)
                df = flux_points.to_table(sed_type="dnde", formatted=True)
                names = [name for name in df.colnames if len(df[name].shape) <= 1]
                df = df[names].to_pandas()

                print("PLOT RESULTS ...")
                plt.figure(figsize=(8, 5))
                ax = flux_points.plot(sed_type="e2dnde", color="darkorange")
                flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde");
                ax.get_figure().savefig(resPath+analysisName+'/spectrum2.jpg')

                flux_points_dataset = FluxPointsDataset(data=flux_points, models=model_best_joint)
                ax_spectrum, ax_residuals = flux_points_dataset.plot_fit()
                ax_spectrum.get_figure().savefig(resPath+analysisName+'/spectrumfit.jpg')

                original_stdout = sys.stdout
                with open(resPath+analysisName+'/Spectral_model_results.txt', 'w') as f:
                    sys.stdout = f # Change the standard output to the file we created.
                    print(flux_points_dataset)
                    sys.stdout = original_stdout

                dataset_gammacat = FluxPointsDataset(data=flux_points, name=form.source.data+" Flux points")
                flux_points_df = tableToPandas(flux_points.to_table(sed_type="e2dnde", formatted=True))
                flux_points_df.to_csv(resPath+analysisName+'/Flux_point_results.csv')

                return render_template('spectrum/index_spectrum.html', resname='' , form=form, folder = folder,plot=plot,listresfiles=listresfiles)
        else:
            print("form.validate_on_submit()")
            print(form.errors)
        return render_template('spectrum/index_spectrum.html', resname='def.png' , form=form, folder = folder,plot=plot,listresfiles=listresfiles)
    #except Exception as inst:
    #    inst = str(inst)
    #    return render_template('main/error.html',  inst =inst)
