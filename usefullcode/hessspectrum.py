%matplotlib inline
import matplotlib.pyplot as plt
# Check package versions
import gammapy
import numpy as np
import astropy
import regions

print("gammapy:", gammapy.__version__)
print("numpy:", np.__version__)
print("astropy", astropy.__version__)
print("regions", regions.__version__)

from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
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
from astropy.table import Table
from gammapy.makers import RingBackgroundMaker

table = Table.read('/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure/obs-index.fits.gz', format='fits')
df = table.to_pandas()

df['OBJECT'] = [e.decode("utf-8") for e in  df['OBJECT'] ]
run_list =  list(df[df['OBJECT'] =='Crab Nebula']['OBS_ID'])[50:70]
plong = list(df[df['OBJECT'] =='Crab Nebula']['GLON_PNT'])[0]
plat= list(df[df['OBJECT'] =='Crab Nebula']['GLAT_PNT'])[0]
datastore = DataStore.from_dir("/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure")
obs_ids = run_list
observations = datastore.get_observations(obs_ids)
target_position = SkyCoord(ra=list(df[df['OBJECT'] =='Crab Nebula']['RA_OBJ'])[0], dec=list(df[df['OBJECT'] =='Crab Nebula']['DEC_OBJ'])[0], unit="deg", frame="icrs")
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

for obs_id, observation in zip(obs_ids, observations):
    try:
        print(obs_id)
        dataset = dataset_maker.run(
            dataset_empty.copy(name=str(obs_id)), observation
        )
        dataset_on_off = bkg_maker.run(dataset, observation)
        dataset_on_off = safe_mask_masker.run(dataset_on_off, observation)
        datasets.append(dataset_on_off)
    except Exception as inst:
        print(inst)
        print("ERROR WITH RUN " + str(obs_id))

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

datasets[0].plot_excess()
ax_spectrum, ax_residuals = datasets[0].plot_fit()
ax_spectrum.set_ylim(0.1, 40)
ax_spectrum.get_figure().savefig('/Users/vl238644/Documents/GitHub/GammaPyGui/Main/static/results/'+'spectum.jpg')

e_min, e_max = 0.7, 30
energy_edges = np.geomspace(e_min, e_max, 11) * u.TeV

fpe = FluxPointsEstimator(energy_edges=energy_edges, source="crab", selection_optional="all")
flux_points = fpe.run(datasets=datasets)
df = flux_points.to_table(sed_type="dnde", formatted=True)
names = [name for name in df.colnames if len(df[name].shape) <= 1]
df[names].to_pandas()

plt.figure(figsize=(8, 5))
ax = flux_points.plot(sed_type="e2dnde", color="darkorange")
flux_points.plot_ts_profiles(ax=ax, sed_type="e2dnde");
ax.get_figure().savefig('/Users/vl238644/Documents/GitHub/GammaPyGui/Main/static/results/'+'spectum2.jpg')
df[df['OBJECT'] =='Crab Nebula'].columns

# Define map geometry for binned simulation
energy_reco = MapAxis.from_edges(np.logspace(-1.0, 1.0, 10), unit="TeV", name="energy", interp="log")
geom = WcsGeom.create(skydir=(plong, plat),binsz=0.02,width=(6, 6),frame="galactic",axes=[energy_reco])
datasets[0]

# It is usually useful to have a separate binning for the true energy axis
energy_true = MapAxis.from_edges(np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy_true", interp="log")
energy = MapAxis.from_edges(np.logspace(-1.5, 1.5, 30), unit="TeV", name="energy", interp="log")

#TEsting ring BackGROUND
#geom = datasets[0].geoms['geom']
energy_axis = energy
geom_image = geom.to_image().to_cube([energy_axis.squash()])

# Make the exclusion mask
pointing = SkyCoord(plong, plat, unit="deg", frame="galactic")
regions = CircleSkyRegion(center=pointing, radius=0.0 * u.deg)
exclusion_mask = ~geom_image.region_mask([regions])

ring_maker = RingBackgroundMaker(r_in="0.5 deg", width="0.3 deg", exclusion_mask=exclusion_mask)

energy_axis_true = energy
datasets[0].to_image()
dataset_on_off = ring_maker.run(dataset.to_image())


estimator = ExcessMapEstimator(0.04 * u.deg, selection_optional=[])
lima_maps = estimator.run(dataset_on_off)
significance_map = lima_maps["sqrt_ts"]
excess_map = lima_maps["npred_excess"]
