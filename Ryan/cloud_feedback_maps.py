#!/usr/bin/env python

"""
Plot the cloud feedback decomposition using output from decomp_isccp.py.
Cloud feedbacks are computed using the Zelinka et. al (2012) kernels, in
the format: (feedback type, latitude, longitude). 
The first index, feedback type, includes:
Longwave: "Total","Amount","Altitude","Optical Depth","Residual"
Repeated for shortwave.

RL - Nov 2017
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.basemap import Basemap, shiftgrid, addcyclic
from netCDF4 import Dataset
plt.rcParams["font.family"] = "sans-serif"

# for plotting
runs = ['otb','iris3','iris2','iris1']
#direc = "/Users/ryan/Desktop/2nd/cesm/data/cosp_data/"
run = runs[0]
filename = run+"_cloud_fbk_decomp.nc"

types = ["Total","Amount","Altitude","Optical Depth"]
calcs = ['LW','SW','Net']
start,end = -3.0,3.0 # max and min values on the colorbar
nc = 20 # number of contours for each colormap
step = (end-start)/float(nc)
levels = np.arange(start,end+1e-5,step) # levels of contours set by nc
alpha = 0.7 # transparency of contours

###### Load netCDF file #######
nfile = Dataset(filename)
LAT = nfile.variables['LAT'][:]
LON = nfile.variables['LON'][:]
data = nfile.variables["FB_ALL"][:,:,:] #data(feedback,lat,lon)
nfile.close()

###### Separate LW and SW components ######
# cosine weighting
wgt = np.cos(np.deg2rad(LAT))

# compute global averages for title.
nan_index = 28 # at 33S latitude band, interpolate using nearby values
LW = data[:5,:,:]
for i in range(5): # linear interpolation for latitude band
    LW[i,nan_index,:] = (LW[i,nan_index-1,:] + LW[i,nan_index+1,:])/2.0
LW_globe = np.ma.average(np.mean(LW,axis=2),axis=1,weights=wgt)
SW = data[5:,:,:]
SW_globe = np.average(np.mean(SW,axis=2),axis=1,weights=wgt)
net = LW+SW
net_globe = LW_globe + SW_globe

# put variables into list for looping ease. 
all_feedbacks = np.array([LW,SW,net])
#all_feedbacks = np.clip(all_feedbacks,start,end)
globes = np.array([LW_globe,SW_globe,net_globe])

###### Plotting using Basemap and contourf ######
x,y = np.meshgrid(LON,LAT)

# Create figure
# axes = ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9),(ax10,ax11,ax12)) ! column, row
fig,axes = plt.subplots(4,3,sharex='col',sharey='row',figsize=(14.5,9.92))
#fig.subplots_adjust(hspace=0.2,wspace=0.1)
plt.tight_layout(pad=3.0, w_pad=1.0, h_pad=1.0)

def plot_ax(ax,z,title):
    """ Basemap Plotting function for each individual axis. Returns colormap instance.
    ax: Matplotlib.pyplot.ax instance
    z: variable to plot z[lat,long]
    """
    #z[z>vmax] = vmax
    #z[z<vmin] = vmin
    ax.set_title(title)
    m = Basemap(projection='cyl',resolution='c',llcrnrlat=-90,urcrnrlat=90,
                    llcrnrlon=0,urcrnrlon=360,ax=ax)
    m.drawcoastlines()
    m.drawmapboundary(fill_color='white')
    #newz = np.clip(z,start,end)
    m.contour(x,y,z,levels,colors='k',linewidths=0.1,alpha=0.3)
    cm = m.contourf(x,y,z,levels,cmap=plt.cm.RdBu_r,alpha=alpha,extend="both")
    return cm

chars = [['a','b','c'],['d','e','f'],['g','h','i'],['j','k','l']]
# loop through
for i in range(len(types)):
    for j in range(len(calcs)):
        cm = plot_ax(axes[i,j],all_feedbacks[j][i,:,:],
                    "("+chars[i][j]+") "+calcs[j]+" "+types[i]+" (%.2f)"%globes[j][i])
        pass
# plot 1 to save time, see if code works
#i,j = -1,-1
#cm = plot_ax(axes[i,j],all_feedbacks[j][i,:,:],calcs[j]+" "+types[i]+" [%.2f]"%globes[j][i])

#cm.cmap.set_under('k')
#cm.set_clim(-3.0, 3.0)
# put the colorbar to the right of the figure. 
fig.colorbar(cm, ax=axes.ravel().tolist(), shrink=0.5,pad=0.025,ticks=np.arange(start,end+1e-5,1.),fraction=0.025)
plt.savefig(run+"_cloud_feedback.pdf")
