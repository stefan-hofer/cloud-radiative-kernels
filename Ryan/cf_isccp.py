#!/Users/ryan/anaconda2/envs/cdat_lite/bin/python

'''
% Cloud Property Histograms. Part I: Cloud Radiative Kernels. J. Climate, 25, 3715?3735. doi:10.1175/JCLI-D-11-00248.1.
% Modified from original script provided by Mark Zelinka (zelinka1@llnl.gov) on 14 July 2017

files required:
cloud kernels from Zelinka et al 2012
control cosp run with ISCCP simulator
future cosp run with ISCCP simulator

clisccp variable dims: (240, 7, 7, 96, 192)
rsuscs/rsdscs variable dims: (240, 96, 192)
tas variable dims: (240, 96, 192)
output LW and SW feedback variable dims: (12, 7, 7, 90, 144)

RL - Sep 2017

'''

import cdms2 as cdms
import cdutil
import MV2 as MV
import numpy as np
import pylab as pl
from mpl_toolkits.basemap import Basemap,shiftgrid,addcyclic

run = 'otb'
ker = 'cloud_kernels2.nc' # MZ kernel file
fctrl = run+'_ctrl_isccp.nc' # control ISCCP simulator output
fdble = run+'_dble_isccp.nc' # 2xCO2 ISCCP simulator output
direc="/Users/ryan/Desktop/2nd/cesm/data/cosp_data/"

# Define helpful functions
###########################################################################
def add_cyclic(data):
    # Add Cyclic point around 360 degrees longitude:
    lons=data.getLongitude()[:]
    dx=np.gradient(lons)[-1]
    data2 = data(longitude=(0, dx+np.max(lons)), squeeze=True)    
    return data2

def averager(x,latitudes,lat_dim,lon_dim):
    if np.max(latitudes) > 3.4:
        ls = np.deg2rad(latitudes)
    else:
        ls = latitudes
    weights = np.cos(ls)
    zonal = np.average(x,axis=lon_dim)
    return np.average(x,axis=lat_dim,weights=weights)

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
                    # print albcs
                    # fill_value to get rid of above range error
                    f = interp1d(albcs,Ksw[MM,:,:,LA,:],axis=2,fill_value="extrapolate")
                    ynew = f(alon2.data)
                    ynew=MV.masked_where(alon2.mask,ynew)
                    SWkernel_map[M,:,:,LA,:] = ynew
            else:
                continue

    return SWkernel_map

# Main Routine follows
###########################################################################

# variables to load from files, last 2 are for surface clear sky albedo
vars = ["FISCCP1_COSP","TS","FSDSC","FSNSC"] 

# Load in the Zelinka et al 2012 kernels:
f=cdms.open(direc+ker)
LWkernel=f('LWkernel')
SWkernel=f('SWkernel')
f.close()

# Load in variables for control simulation
fc = cdms.open(direc+fctrl,'r')

# Load in clisccp: ISCCP cloud area fraction (in atmosphere layer) 
clisccp1=fc(vars[0])

# Load in rsuscs, rsdscs
# rsuscs: surface upwelling shortwave flux in air assuming clear sky
# rsdscs: surface downwelling shortwave flux in air assuming clear sky
rsuscs1 = fc(vars[2])-fc(vars[3]) # net - down
rsdscs1 = fc(vars[3])

# Load surface air temperature (K)
tas1 = fc(vars[1])
fc.close()

# Do the same for future
fd = cdms.open(direc+fdble,'r')
clisccp2=fd(vars[0])
tas2 = fd(vars[1])
fd.close()

# "clisccp variable dims:" (240, 7, 7, 96, 144)
# "rsuscs/rsdscs variable dims:" (240, 96, 144)

albcs=np.arange(0.0,1.5,0.5) # the clear-sky albedos over which the kernel is computed

# LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
LWkernel_map=np.tile(np.tile(LWkernel[:,:,:,:,0],(1,1,1,1,1)),(144,1,1,1,1))(order=[1,2,3,4,0])

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

# Compute clisccp anomalies
# CESM cosp output default is ctp,tau - swap these axes
anomclisccp = cdms.asVariable(np.transpose(avgclisccp2 - avgclisccp1,(0,2,1,3,4)))
anomclisccp.setAxisList(avgclisccp1.getAxisList())

# Compute clear-sky surface albedo
albcs1=rsuscs1/rsdscs1
avgalbcs1=cdutil.ANNUALCYCLE.climatology(albcs1) #(12, 90, 144)
avgalbcs1=MV.where(avgalbcs1>1.,1,avgalbcs1) # where(condition, x, y) is x where condition is true, y otherwise
avgalbcs1=MV.where(avgalbcs1<0.,0,avgalbcs1)
del(rsuscs1,rsdscs1,albcs1)

# Compute climatological annual cycle:
avgtas1=cdutil.ANNUALCYCLE.climatology(tas1) #(12, 90, 144)
avgtas2=cdutil.ANNUALCYCLE.climatology(tas2) #(12, 90, 144)
del(tas1,tas2)

# Compute global annual mean tas anomalies
anomtas = avgtas2 - avgtas1
avgdtas = cdutil.averager(MV.average(anomtas,axis=0), axis='xy', weights='weighted') # (scalar)
print "globally averaged temperature anomalies: "+str(avgdtas)+" K"
#print "test: "+str(averager(anomtas,lats,0,1))+" K"

# Regrid everything to the kernel grid:
avgalbcs1 = add_cyclic(avgalbcs1)
avgclisccp1 = add_cyclic(avgclisccp1)
avgclisccp2 = add_cyclic(avgclisccp2)
avganomclisccp = add_cyclic(anomclisccp)
avgalbcs1_grd = avgalbcs1.regrid(grid,regridTool="esmf",regridMethod = "linear")
avgclisccp1_grd = avgclisccp1.regrid(grid,regridTool="esmf",regridMethod = "linear")
avgclisccp2_grd = avgclisccp2.regrid(grid,regridTool="esmf",regridMethod = "linear")
avganomclisccp_grd = avganomclisccp.regrid(grid,regridTool="esmf",regridMethod = "linear")

# Use control albcs to map SW kernel to appropriate longitudes
SWkernel_map = map_SWkern_to_lon(SWkernel,avgalbcs1_grd)

# Compute clisccp anomalies normalized by global mean delta tas
anomclisccp = avganomclisccp_grd/avgdtas

# CESM ISCCP fields are P,tau,lat,lon --> change to tau,P,lat,lon
#anomclisccp = np.transpose(anomclisccp,(0,2,1,3,4))
#cdms.axis.latitude_aliases.append("Y")
#cdms.axis.longitude_aliases.append("X")

# Compute feedbacks: Multiply clisccp anomalies by kernels
SW0 = SWkernel_map*anomclisccp
LW_cld_fbk = LWkernel_map*anomclisccp
LW_cld_fbk.setAxisList(anomclisccp.getAxisList())

SW_cld_fbk = SW0
SW_cld_fbk.setAxisList(anomclisccp.getAxisList())
'''
# Set the SW cloud feedbacks to zero in the polar night
# The sun is down if every bin of the SW kernel is zero:
sundown=MV.sum(MV.sum(SWkernel_map,axis=2),axis=1)  #12,90,144
repsundown=np.tile(np.tile(sundown,(1,1,1,1,1)),(7,7,1,1,1))(order=[2,1,0,3,4])
SW1 = MV.where(repsundown==0, 0, SW0) # where(condition, x, y) is x where condition is true, y otherwise
SW_cld_fbk = MV.where(repsundown.mask, 0, SW1) # where(condition, x, y) is x where condition is true, y otherwise
SW_cld_fbk.setAxisList(anomclisccp.getAxisList())
'''
# SW_cld_fbk and LW_cld_fbk contain the contributions to the feedback from cloud anomalies in each bin of the histogram
'''
# Quick sanity check:
# print the global, annual mean LW and SW cloud feedbacks:
sumLW = MV.average(MV.sum(MV.sum(LW_cld_fbk,axis=2),axis=1),axis=0)
print "feedback mean over all tau, CTPs, months:", sumLW.shape
avgLW_cld_fbk = cdutil.averager(sumLW, axis='xy', weights='weighted')
print 'avg LW cloud feedback = '+str(avgLW_cld_fbk)+' W/m2/K'
sumSW = MV.average(MV.sum(MV.sum(SW_cld_fbk,axis=2),axis=1),axis=0)
avgSW_cld_fbk = cdutil.averager(sumSW, axis='xy', weights='weighted')
print 'avg SW cloud feedback = '+str(avgSW_cld_fbk)+' W/m2/K'
sumCLD = MV.average(MV.sum(MV.sum(avganomclisccp_grd,axis=2),axis=1),axis=0)
avgCLD_change = cdutil.averager(sumCLD, axis='xy', weights='weighted')
print 'avg cloud change = '+str(avgCLD_change)+'%'

print "LW and SW feedback variable dims:",LW_cld_fbk.shape
print "map dims:",sumLW.shape
'''

# Some sample global mean figures
tau=[0.,0.3,1.3,3.6,9.4,23.,60.,380.]
ctp=[1000,800,680,560,440,310,180,50]


print "Writing to a netCDF file"
from netCDF4 import Dataset

# Write to a netcdf file
dataset = Dataset(run+'_cloud_isccp_output.nc','w')

# Create dimensions
mon = dataset.createDimension('mon', 12)
tau_nc = dataset.createDimension('tau', len(tau)-1)
ctp_nc = dataset.createDimension('ctp', len(ctp)-1)
lat = dataset.createDimension('lat',len(lats))
lon = dataset.createDimension('lon',len(lons))

# Create and Write dimensional variables
taus = dataset.createVariable('taus', np.float32, ('tau',))
ctps = dataset.createVariable('ctps', np.float32, ('ctp',))
LAT = dataset.createVariable('LAT', np.float32,('lat',))
LON = dataset.createVariable('LON', np.float32,('lon',))

ctps.units = "hPa"
taus[:] = tau[1:]
ctps[:] = ctp[1:]
LAT[:] = lats
LON[:] = lons

SWCF = dataset.createVariable('SWCF',np.float32,('mon','tau','ctp','lat','lon'))
LWCF = dataset.createVariable('LWCF',np.float32,('mon','tau','ctp','lat','lon'))

SWCF[:,:,:,:,:] = SW_cld_fbk
LWCF[:,:,:,:,:] = LW_cld_fbk

dataset.close()


print "Plotting..."

# amip cloud fraction histogram:
pl.subplots()
data = cdutil.averager(MV.average(avgclisccp1_grd,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='Reds')#,vmin=0, vmax=10)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean cloud fraction')
pl.xlabel('TAU')
pl.ylabel('CTP')
pl.colorbar()
pl.savefig(run+"_cf.png")
# amipFuture cloud fraction histogram:
pl.subplots()
data = cdutil.averager(MV.average(avgclisccp2_grd,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='Reds')#,vmin=0, vmax=10)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean Future cloud fraction')
pl.xlabel('TAU')
pl.ylabel('CTP')
pl.colorbar()
pl.savefig(run+"_2xCO2_cf.png")
# difference of cloud fraction histograms:
pl.subplots()
data = cdutil.averager(MV.average(anomclisccp,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='RdBu_r')#,vmin=-0.2, vmax=0.2)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean normalized change in cloud fraction')
pl.colorbar()
pl.savefig(run+"_dcf.png")
# LW cloud feedback contributions:
pl.subplots()
data = cdutil.averager(MV.average(LW_cld_fbk,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='RdBu_r')#,vmin=-0.2, vmax=0.2)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean LW cloud feedback contributions')
pl.colorbar()
pl.savefig(run+"_lwf.png")
# SW cloud feedback contributions:
pl.subplots()
data = cdutil.averager(MV.average(SW_cld_fbk,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='RdBu_r')#,vmin=-0.75, vmax=0.75)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean SW cloud feedback contributions')
pl.colorbar()
pl.savefig(run+"_swf.png")

# Feedback distributions

sumSW,LON = shiftgrid(180.,sumSW,lons,start=False)
newSW,LON = addcyclic(sumSW,LON)
pl.figure()
x,y = np.meshgrid(LON,lats)
m = Basemap(projection='cyl',resolution='c')
cm = m.pcolormesh(x,y,newSW,cmap=pl.cm.RdBu_r,alpha=0.6)
cbar = pl.colorbar(cm,shrink=0.6)
m.drawcoastlines()
m.drawparallels([-90,-30,30,90],labels=[1,0,0,0],linewidth=0.001)
m.drawmeridians([-120,0,120],labels=[1,0,0,1],linewidth=0.001)
pl.title("SW cloud feedback distribution")
pl.savefig(run+"_swfmap.png")

sumLW,LON = shiftgrid(180.,sumLW,lons,start=False)
newLW,LON = addcyclic(sumLW,LON)
pl.figure()
x,y = np.meshgrid(LON,lats)
m = Basemap(projection='cyl',resolution='c')
cm = m.pcolormesh(x,y,newLW,cmap=pl.cm.RdBu_r,alpha=0.6)
cbar = pl.colorbar(cm,shrink=0.6)
m.drawcoastlines()
m.drawparallels([-90,-30,30,90],labels=[1,0,0,0],linewidth=0.001)
m.drawmeridians([-120,0,120],labels=[1,0,0,1],linewidth=0.001)
pl.title("LW cloud feedback distribution")
pl.savefig(run+"_lwfmap.png")

