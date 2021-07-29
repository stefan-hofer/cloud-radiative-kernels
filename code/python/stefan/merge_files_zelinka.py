import xarray as xr
import glob

main_dir = '/projects/NS9252K/noresm/cases/WP4_shofer/n.n202.NSSP585frc2.f09_tn14.ssp585.001_global/atm/hist/COSP/'
subdirs = ['HIST_0/',  'HIST_1/',  'HIST_2/',  'PI/',  'SSP_0/',  'SSP_1/']
all_dirs = [main_dir + s for s in subdirs]

vars = ["FISCCP1_COSP", "TS", "FSDSC", "FSNSC"]


for dir in all_dirs:
    print('Going to directory: {}'.format(dir))
    ds = xr.open_mfdataset(dir + '*_COSP.nc')
    ds = ds.load()
    for var in vars:
        if var == 'FISCCP1_COSP':
            var_new = 'FISCCP1'
            save_str = dir + sorted(glob.glob(dir + '*.cam.h0*.nc')
                                    )[-1].split('/')[-1][:-11] + '_' + var_new + '.nc'
        else:
            save_str = dir + sorted(glob.glob(dir + '*.cam.h0*.nc')
                                    )[-1].split('/')[-1][:-11] + '_' + var + '.nc'

        print('Saving the file: {}'.format(save_str))
        ds[var].to_netcdf(save_str)
