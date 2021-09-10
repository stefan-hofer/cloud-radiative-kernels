import glob
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.gridspec as gridspec
import xesmf as xe

import cartopy.feature as feat

folder = '/trd-project1/NS9252K/noresm/cases/WP4_shofer/n.n202.NSSP585frc2.f09_tn14.ssp585.001_global/atm/hist/COSP/cloud_feedbacks/'


list_files = sorted(glob.glob(folder + 'SW_cloud_feedbacks_ALL*.nc'))

ds = xr.open_mfdataset(
    list_files)
