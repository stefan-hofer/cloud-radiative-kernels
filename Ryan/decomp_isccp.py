#!/Users/ryan/anaconda2/envs/cdat_lite/bin/python

'''
Zelinka et al 2013 decomposition modified using oroginal code from Mark Zelinka

files required
cloud kernels from Zelinka et al 2012
control cosp run with ISCCP simulator
future cosp run with ISCCP simulator

clisccp variable dims: (240, 7, 7, 96, 192)
rsuscs/rsdscs variable dims: (240, 96, 192)
tas variable dims: (240, 96, 192)
output LW and SW feedback variable dims: (10, 90, 144)

RL - Sep 2017

'''

import cdms2 as cdms
import cdutil
import MV2 as MV
import numpy as np
import pylab as pl

# files
run = 'otb'
ker = 'cloud_kernels2.nc' # MZ kernel file
fctrl = run+'_ctrl_isccp.nc' # control ISCCP simulator output
fdble = run+'_dble_isccp.nc' # 2xCO2 ISCCP simulator output
direc="/Users/ryan/Desktop/2nd/cesm/data/cosp_data/"

ecs = 3.73949874
tau = [0.,0.3,1.3,3.6,9.4,23.,60.,380.]
CTP = [1000,800,680,560,440,310,180,50]
dlow = 5 # low cloud index, 3 for up to (including) 680hPa, 5 for 440hPa

# variables to load from files, last 2 are for surface clear sky albedo
vars = ["FISCCP1_COSP","TS","FSDSC","FSNSC"] 


# helpful functions provided by MZ
###########################################################################
def add_cyclic(data):
    # Add Cyclic point around 360 degrees longitude:
    lons=data.getLongitude()[:]
    dx=np.gradient(lons)[-1]
    data2 = data(longitude=(0, dx+np.max(lons)), squeeze=True)    
    return data2

###########################################################################
def nanarray(vector):

    # this generates a masked array with the size given by vector
    # example: vector = (90,144,28)

    # similar to this=NaN*ones(x,y,z) in matlab

    this=MV.zeros(vector)
    this=MV.masked_where(this==0,this)

    return this

###########################################################################
def map_SWkern_to_lon(Ksw,albcsmap):

    from scipy.interpolate import interp1d
    ## Map each location's clear-sky surface albedo to the correct albedo bin
    # Ksw is size 12,7,7,lats,3
    # albcsmap is size A,lats,lons
    albcs=np.arange(0.0,1.5,0.5) 
    A=albcsmap.shape[0]
    TT=Ksw.shape[1]
    PP=Ksw.shape[2]
    lenlat=Ksw.shape[3]
    lenlon=albcsmap.shape[2]
    SWkernel_map=nanarray((A,TT,PP,lenlat,lenlon))
    for M in range(A):
        MM=M
        while MM>11:
            MM=MM-12
        for LA in range(lenlat):
            alon=albcsmap[M,LA,:] 
            # interp1d can't handle mask but it can deal with NaN (?)
            try:
                alon2=MV.where(alon.mask,np.nan,alon)   
            except:
                alon2=alon
            if np.ma.count(alon2)>1: # at least 1 unmasked value
                if len(pl.find(Ksw[MM,:,:,LA,:]>0))==0:
                    SWkernel_map[M,:,:,LA,:] = 0
                else:
                    f = interp1d(albcs,Ksw[MM,:,:,LA,:],axis=2,fill_value="extrapolate")
                    ynew = f(alon2.data)
                    ynew=MV.masked_where(alon2.mask,ynew)
                    SWkernel_map[M,:,:,LA,:] = ynew
            else:
                continue

    return SWkernel_map

def KT_decomposition_4D(c1,c2,Klw,Ksw):
 
    # this function takes in a (tau,CTP,lat,lon) matrix and performs the
    # decomposition of Zelinka et al 2013 doi:10.1175/JCLI-D-12-00555.1
 
    # reshape to be (CTP,tau,lat,lon)
    c1 = MV.transpose(c1,(1,0,2,3)) # control cloud fraction histogram
    c2 = MV.transpose(c2,(1,0,2,3)) # perturbed cloud fraction histogram
    Klw = MV.transpose(Klw,(1,0,2,3)) # LW Kernel histogram
    Ksw = MV.transpose(Ksw,(1,0,2,3)) # SW Kernel histogram
 
    P=c1.shape[0]
    T=c1.shape[1]
 
    c=c1
    sum_c=np.tile(MV.sum(MV.sum(c,1),0),(P,T,1,1))                                  # Eq. B2
    dc = c2-c1
    sum_dc=np.tile(MV.sum(MV.sum(dc,1),0),(P,T,1,1))
    dc_prop = c*(sum_dc/sum_c)
    dc_star = dc - dc_prop                                                          # Eq. B1
 
    # LW components
    Klw0 = np.tile(MV.sum(MV.sum(Klw*c/sum_c,1),0),(P,T,1,1))                       # Eq. B4
    Klw_prime = Klw - Klw0                                                          # Eq. B3
    this=MV.sum(Klw_prime*np.tile(MV.sum(c/sum_c,0),(P,1,1,1)),1)                   # Eq. B7a
    Klw_p_prime=np.tile(np.tile(this,(1,1,1,1)),(T,1,1,1))(order=[1,0,2,3])         # Eq. B7b
    that=np.tile(np.tile(MV.sum(c/sum_c,1),(1,1,1,1)),(T,1,1,1))(order=[1,0,2,3])   # Eq. B8a
    Klw_t_prime = np.tile(MV.sum(Klw_prime*that,0),(P,1,1,1))                       # Eq. B8b
    Klw_resid_prime = Klw_prime - Klw_p_prime - Klw_t_prime                         # Eq. B9
    dRlw_true = MV.sum(MV.sum(Klw*dc,1),0)                                          # LW total
    dRlw_prop = Klw0[0,0,:,:]*sum_dc[0,0,:,:]                                       # LW amount component
    dRlw_dctp = MV.sum(MV.sum(Klw_p_prime*dc_star,1),0)                             # LW altitude component
    dRlw_dtau = MV.sum(MV.sum(Klw_t_prime*dc_star,1),0)                             # LW optical depth component
    dRlw_resid = MV.sum(MV.sum(Klw_resid_prime*dc_star,1),0)                        # LW residual
    dRlw_sum = dRlw_prop + dRlw_dctp + dRlw_dtau + dRlw_resid                       # sum of LW components -- should equal LW total
 
    # SW components
    Ksw0 = np.tile(MV.sum(MV.sum(Ksw*c/sum_c,1),0),(P,T,1,1))                       # Eq. B4
    Ksw_prime = Ksw - Ksw0                                                          # Eq. B3
    this=MV.sum(Ksw_prime*np.tile(MV.sum(c/sum_c,0),(P,1,1,1)),1)                   # Eq. B7a
    Ksw_p_prime=np.tile(np.tile(this,(1,1,1,1)),(T,1,1,1))(order=[1,0,2,3])         # Eq. B7b 
    that=np.tile(np.tile(MV.sum(c/sum_c,1),(1,1,1,1)),(T,1,1,1))(order=[1,0,2,3])   # Eq. B8a
    Ksw_t_prime = np.tile(MV.sum(Ksw_prime*that,0),(P,1,1,1))                       # Eq. B8b
    Ksw_resid_prime = Ksw_prime - Ksw_p_prime - Ksw_t_prime                         # Eq. B9
    dRsw_true = MV.sum(MV.sum(Ksw*dc,1),0)                                          # SW total
    dRsw_prop = Ksw0[0,0,:,:]*sum_dc[0,0,:,:]                                       # SW amount component
    dRsw_dctp = MV.sum(MV.sum(Ksw_p_prime*dc_star,1),0)                             # SW altitude component
    dRsw_dtau = MV.sum(MV.sum(Ksw_t_prime*dc_star,1),0)                             # SW optical depth component
    dRsw_resid = MV.sum(MV.sum(Ksw_resid_prime*dc_star,1),0)                        # SW residual
    dRsw_sum = dRsw_prop + dRsw_dctp + dRsw_dtau + dRsw_resid                       # sum of SW components -- should equal SW total
 
    dc_star = MV.transpose(dc_star,(1,0,2,3))
    dc_prop = MV.transpose(dc_prop,(1,0,2,3))
 
    return dRlw_true,dRlw_prop,dRlw_dctp,dRlw_dtau,dRlw_resid,dRsw_true,dRsw_prop,dRsw_dctp,dRsw_dtau,dRsw_resid

# Load in the Zelinka et al 2012 kernels:
f=cdms.open(direc+ker)
LWkernel=f('LWkernel')
SWkernel=f('SWkernel')
f.close()

# Load in variables for control simulation
fc = cdms.open(direc+fctrl,'r')

# Load in clisccp: ISCCP cloud area fraction (in atmosphere layer), surface air temperature (K)
clisccp1=fc(vars[0])
tas1 = fc(vars[1])

# Load in rsuscs, rsdscs
# rsuscs: surface upwelling shortwave flux in air assuming clear sky
# rsdscs: surface downwelling shortwave flux in air assuming clear sky
rsuscs1 = fc(vars[2])-fc(vars[3]) # net - down
rsdscs1 = fc(vars[3])
# "rsuscs/rsdscs variable dims:" (240, 96, 144)
albcs=np.arange(0.0,1.5,0.5) # the clear-sky albedos over which the kernel is computed

fc.close()

# Do the same for future
fd = cdms.open(direc+fdble,'r')
clisccp2=fd(vars[0])
tas2 = fd(vars[1])
fd.close()

# Define the cloud kernel axis attributes
lats=LWkernel.getLatitude()[:]
lons=np.arange(1.25,360,2.5)
grid = cdms.createGenericGrid(lats,lons)

# Make sure clisccp is in percent  
sumclisccp1=MV.sum(MV.sum(clisccp1,2),1)
sumclisccp2=MV.sum(MV.sum(clisccp2,2),1)   
if np.max(sumclisccp1) <= 1.:
    clisccp1 = clisccp1*100.        
if np.max(sumclisccp2) <= 1.:
    clisccp2 = clisccp2*100.

# Compute climatological annual cycle:
avgclisccp1=cdutil.ANNUALCYCLE.climatology(clisccp1) #(12, TAU, CTP, LAT, LON)
avgclisccp2=cdutil.ANNUALCYCLE.climatology(clisccp2) #(12, TAU, CTP, LAT, LON)
del(clisccp1,clisccp2)

# Compute climatological annual cycle:
avgtas1=cdutil.ANNUALCYCLE.climatology(tas1) #(12, 90, 144)
avgtas2=cdutil.ANNUALCYCLE.climatology(tas2) #(12, 90, 144)
# Compute global annual mean tas anomalies
avgdtas = cdutil.averager(MV.average(avgtas2 - avgtas1,axis=0), axis='xy', weights='weighted') # (scalar)
print "globally averaged temperature anomalies: "+str(avgdtas)+" K"

# Compute clear-sky surface albedo
albcs1=rsuscs1/rsdscs1
avgalbcs1=cdutil.ANNUALCYCLE.climatology(albcs1) #(12, 90, 144)
avgalbcs1=MV.where(avgalbcs1>1.,1,avgalbcs1) # where(condition, x, y) is x where condition is true, y otherwise
avgalbcs1=MV.where(avgalbcs1<0.,0,avgalbcs1)
del(rsuscs1,rsdscs1,albcs1)

# Regrid everything to the kernel grid:
avgalbcs1 = add_cyclic(avgalbcs1)
avgclisccp1 = add_cyclic(avgclisccp1)
avgclisccp2 = add_cyclic(avgclisccp2)
avgalbcs1_grd = avgalbcs1.regrid(grid,regridTool="esmf",regridMethod = "linear")
clisccp1_r = avgclisccp1.regrid(grid,regridTool="esmf",regridMethod = "linear")
clisccp2_r = avgclisccp2.regrid(grid,regridTool="esmf",regridMethod = "linear")

# CESM cosp output default is ctp,tau,lat,lon - swap these axes to tau,ctp,lat,lon
clisccp1 = cdms.asVariable(np.transpose(clisccp1_r,(0,2,1,3,4)))
clisccp2 = cdms.asVariable(np.transpose(clisccp2_r,(0,2,1,3,4)))
clisccp1.setAxisList(clisccp1_r.getAxisList())

# LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
LWK=np.tile(np.tile(LWkernel[:,:,:,:,0],(1,1,1,1,1)),(144,1,1,1,1))(order=[1,2,3,4,0])

# Use control albcs to map SW kernel to appropriate longitudes
SWK = map_SWkern_to_lon(SWkernel,avgalbcs1_grd)

print "Decomposing kernels..."

# decompose the kernels per month
all_fbks = np.zeros((12,10,clisccp1.shape[-2],144))
all_fbks_low = np.zeros((12,10,clisccp1.shape[-2],144)) #low cloud feedback
all_fbks_nonlow = np.zeros_like(all_fbks_low) # non low cloud feedback
for i in range(12):
    # decompose low clouds by high cloud top pressures
    decomp_fbks = KT_decomposition_4D(clisccp1[i,:,:dlow,:,:],clisccp2[i,:,:dlow,:,:],
                                              LWK[i,:,:dlow,:,:],SWK[i,:,:dlow,:,:])
    var_out = np.array(decomp_fbks)
    all_fbks_low[i,:,:,:] = var_out[:,:,:]
    # decompose the non low clouds
    decomp_fbks = KT_decomposition_4D(clisccp1[i,:,dlow:,:,:],clisccp2[i,:,dlow:,:,:],
                                              LWK[i,:,dlow:,:,:],SWK[i,:,dlow:,:,:])
    var_out = np.array(decomp_fbks)
    all_fbks_nonlow[i,:,:,:] = var_out[:,:,:]
    decomp_fbks = KT_decomposition_4D(clisccp1[i,:,:,:,:],clisccp2[i,:,:,:,:],
                                              LWK[i,:,:,:,:],SWK[i,:,:,:,:])
    var_out = np.array(decomp_fbks)
    all_fbks[i,:,:,:] = var_out[:,:,:]
    
    
print "done"
var_out1 = np.average(all_fbks_low,axis=0)
var_out1 /= ecs
var_out2 = np.average(all_fbks_nonlow,axis=0)
var_out2 /= ecs
var_out3 = np.average(all_fbks,axis=0)
var_out3 /= ecs

print "Saving to netCDF file"
from netCDF4 import Dataset

# Write to a netcdf file
dataset = Dataset(run+'_cloud_fbk_decomp.nc','w')

# Create dimensions
i = dataset.createDimension('i',var_out.shape[0])
lat = dataset.createDimension('lat',len(lats))
lon = dataset.createDimension('lon',len(lons))

# Create and Write dimensional variables
LAT = dataset.createVariable('LAT', np.float32,('lat',))
LON = dataset.createVariable('LON', np.float32,('lon',))
LAT[:] = lats
LON[:] = lons

FB_LOW = dataset.createVariable('FB_LOW',np.float32,('i','lat','lon'))
FB_LOW.units = 'W/m2/K'
FB_LOW.description = "Low cloud feedback decomposition from KT_decomposition_4D function.\nDimensions are [feedback type,lat,lon]"
FB_LOW[:,:,:] = var_out1

FB_NLOW = dataset.createVariable('FB_NLOW',np.float32,('i','lat','lon'))
FB_NLOW.units = 'W/m2/K'
FB_NLOW.description = "Non-low cloud feedback decomposition from KT_decomposition_4D function.\nDimensions are [feedback type,lat,lon]"
FB_NLOW[:,:,:] = var_out2

FB_ALL = dataset.createVariable('FB_ALL',np.float32,('i','lat','lon'))
FB_ALL.units = 'W/m2/K'
FB_ALL.description = "Total cloud feedback decomposition from KT_decomposition_4D function.\nDimensions are [feedback type,lat,lon]"
FB_ALL[:,:,:] = var_out3


dataset.close()
