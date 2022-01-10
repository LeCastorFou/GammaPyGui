
from Main.BackEnd.imports.pythonSTD import *
from Main.BackEnd.imports.ploting import *
from Main.BackEnd.imports.flaskSTD import *
from Main.BackEnd.imports.astropySTD import *

from Main import db, bcrypt, mail
from Main.BackEnd.simulation.forms import  LaunchSimu
from Main.BackEnd.main.utils import  Get_MongoDB, load_DB_collection

import gammapy
from gammapy.irf import load_cta_irfs
from gammapy.maps import WcsGeom, MapAxis
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    GaussianSpatialModel,
    SkyModel,
    Models,
    FoVBackgroundModel,
)
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.modeling import Fit
from gammapy.data import Observation
from gammapy.datasets import MapDataset


from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.makers import RingBackgroundMaker
from gammapy.estimators import ExcessMapEstimator
from gammapy.maps import Map
from gammapy.datasets import MapDatasetOnOff
from regions import CircleSkyRegion

simubp = Blueprint('simubp',__name__)

@simubp.route("/simu", methods=['GET', 'POST'])
def simu():
    form = LaunchSimu()
    if form.validate_on_submit():
        plong = form.plong.data
        plat = form.plat.data
        slong = form.slong.data
        slat = form.slat.data
        livetimeint = form.livetime.data
        sIndex = form.sIndex.data
        sFlux = form.sFlux.data
        path  = pathlib.Path().resolve()
        irfs = load_cta_irfs(str(path)+"/Main/static/irfs/North_z20_50h/irf_file.fits")

        livetime = livetimeint * u.hr
        pointing = SkyCoord(plong, plat, unit="deg", frame="galactic")

        # Define map geometry for binned simulation
        energy_reco = MapAxis.from_edges(np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log")
        geom = WcsGeom.create(skydir=(plong, plat),binsz=0.02,width=(6, 6),frame="galactic",axes=[energy_reco])

        # It is usually useful to have a separate binning for the true energy axis
        energy_true = MapAxis.from_edges(np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy_true", interp="log")
        energy = MapAxis.from_edges(np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy", interp="log")

        empty = MapDataset.create(geom, name="dataset-simu", energy_axis_true=energy_true)

        # Define sky model to used simulate the data.
        # Here we use a Gaussian spatial model and a Power Law spectral model.
        spatial_model = GaussianSpatialModel(lon_0=str(slong)+" deg", lat_0=str(slat)+" deg", sigma="0.15 deg", frame="galactic")
        spectral_model = PowerLawSpectralModel(index=sIndex, amplitude=sFlux +" cm-2 s-1 TeV-1", reference="1 TeV")
        model_simu = SkyModel(spatial_model=spatial_model,spectral_model=spectral_model,name="model-simu")
        bkg_model = FoVBackgroundModel(dataset_name="dataset-simu")
        models = Models([model_simu, bkg_model])
        #print(models)

        # Create an in-memory observation
        obs = Observation.create(pointing=pointing, livetime=livetime, irfs=irfs)
        #print(obs)

        # Make the MapDataset
        maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])

        maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)

        dataset = maker.run(empty, obs)
        dataset = maker_safe_mask.run(dataset, obs)
        #print(dataset)

        # Add the model on the dataset and Poission fluctuate
        dataset.models = models
        dataset.fake()
        # Do a print on the dataset - there is now a counts maps
        #print(dataset)

        # To plot, eg, counts:
        #alldata = dataset.counts.smooth(0.05 * u.deg).data
        #stacked_data =np.sum(alldata, axis = 0)


        #### RING BackGROUND MODEL ###
        geom = dataset.geoms['geom']
        energy_axis = energy
        geom_image = geom.to_image().to_cube([energy_axis.squash()])

        # Make the exclusion mask
        regions = CircleSkyRegion(center=pointing, radius=0.0 * u.deg)
        exclusion_mask = ~geom_image.region_mask([regions])

        ring_maker = RingBackgroundMaker(r_in="0.5 deg", width="0.3 deg", exclusion_mask=exclusion_mask)

        energy_axis_true = energy
        dataset_on_off = ring_maker.run(dataset.to_image())


        estimator = ExcessMapEstimator(0.04 * u.deg, selection_optional=[])
        lima_maps = estimator.run(dataset_on_off)
        significance_map = lima_maps["sqrt_ts"]
        excess_map = lima_maps["npred_excess"]

        maxsign = np.nan_to_num(significance_map.data).max()

        isSigni = True if np.nan_to_num(significance_map.data).max() > 5 else False

        if isSigni:
            data = np.nan_to_num(significance_map.data[0])
            posmax = np.unravel_index(np.argmax(data, axis=None), data.shape)
            latsource =  0
            longsource = 0
            pointingExcludedSource = SkyCoord(latsource-3+posmax[0]*0.02, longsource-3+posmax[1]*0.02, unit="deg", frame="galactic")
            df_data = pd.DataFrame(data)
            #Get the size of the source
            size_long = len(df_data[posmax[1]][df_data[posmax[1]] > 5])*0.02
            size_lat = len([e for e in df_data.iloc[[posmax[0]]].values.tolist()[0] if e >5])*0.02
            maxsize = np.max([size_long,size_lat])
            if maxsize > 0.6:
                maxsize = 0.6
            # Re Run exclude regions
            geom = dataset.geoms['geom']
            energy_axis = energy
            geom_image = geom.to_image().to_cube([energy_axis.squash()])

            # Make the exclusion mask
            regions = CircleSkyRegion(center=pointingExcludedSource, radius=maxsize * u.deg)
            exclusion_mask = ~geom_image.region_mask([regions])

            ring_maker = RingBackgroundMaker(r_in="0.5 deg", width="0.3 deg", exclusion_mask=exclusion_mask)

            energy_axis_true = energy
            dataset_on_off = ring_maker.run(dataset.to_image())


            estimator = ExcessMapEstimator(0.04 * u.deg, selection_optional=[])
            lima_maps = estimator.run(dataset_on_off)
            significance_map = lima_maps["sqrt_ts"]
            excess_map = lima_maps["npred_excess"]

        df = pd.DataFrame(np.nan_to_num(significance_map.data)[0])
        df = df.where(df >5, 0)
        df = df.where(df <=5, 1)
        NpixOver = df.values.sum()

        fig = go.Figure(data =
            go.Contour(
                z=significance_map.data[0],

             colorscale='Electric',
                contours=dict(
                    start=0,
                    end=8,
                    size=2,
                )
                ))
        graph = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('simu/indexSimu.html', form = form, graphJSON = graph)
    return render_template('simu/indexSimu.html', form = form, graphJSON = {})
