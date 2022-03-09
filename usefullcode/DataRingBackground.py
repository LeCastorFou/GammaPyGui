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


from gammapy.maps import MapAxis, RegionGeom, WcsGeom
from gammapy.modeling import Fit
from gammapy.data import DataStore
from gammapy.datasets import (
    Datasets,
    MapDataset,
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
    FoVBackgroundMaker,
    SpectrumDatasetMaker,
    MapDatasetMaker,
    ReflectedRegionsBackgroundMaker,
)
from gammapy.estimators import FluxPointsEstimator
from gammapy.visualization import plot_spectrum_datasets_off_regions
from astropy.table import Table
from gammapy.makers import RingBackgroundMaker
from gammapy.analysis import Analysis, AnalysisConfig
from gammapy.makers import RingBackgroundMaker
from gammapy.estimators import ExcessMapEstimator
from gammapy.maps import Map
from gammapy.datasets import MapDatasetOnOff
from scipy.stats import norm
table = Table.read('/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure/obs-index.fits.gz', format='fits')
df = table.to_pandas()

df['OBJECT'] = [e.decode("utf-8") for e in  df['OBJECT'] ]
run_list =  list(df[df['OBJECT'] =='Crab Nebula']['OBS_ID'])[50:70]
datastore = DataStore.from_dir("/Users/vl238644/Documents/HESS/GammaPy/fits_files/std_ImPACT_fullEnclosure")


def plotRingSigni(significance_map,excess_map,mask,path_res,filename):
    offdata = significance_map.data[mask]
    offdata = offdata[np.isfinite(offdata)]

    significance_all = significance_map.data[np.isfinite(significance_map.data)]
    significance_off = offdata

    plt.hist(
        significance_all,
        density=True,
        alpha=0.5,
        color="red",
        label="all bins",
        bins=21,
    )

    plt.hist(
        significance_off,
        density=True,
        alpha=0.5,
        color="blue",
        label="off bins",
        bins=21,
    )

    # Now, fit the off distribution with a Gaussian
    mu, std = norm.fit(significance_off)
    x = np.linspace(-8, 8, 50)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, lw=2, color="black")
    plt.legend()
    plt.xlabel("Significance")
    plt.yscale("log")
    plt.ylim(1e-5, 10)
    xmin, xmax = np.min([np.min(significance_all),-10]), np.max( [np.max(significance_all),10])
    plt.xlim(xmin, xmax)
    plt.text(-8,1,f"Fit results: mu = {mu:.2f}, std = {std:.2f}")
    plt.savefig(path_res+'/'+filename+'_ringBackground_significance_distribution.png')
    plt.clf()
    print(f"Fit results: mu = {mu:.2f}, std = {std:.2f}")
