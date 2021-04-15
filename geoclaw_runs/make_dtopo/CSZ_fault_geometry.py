#!/usr/bin/env python
# coding: utf-8

#====== Converted from the notebook CSZ_fault_geometry.ipynb ==========

# # Generating dtopo files for CSZ fakequakes
# 
# ### Modified to make plots for the ML paper
# 
# 
# This notebooks demonstrates how the GeoClaw `dtopotools` module can be used to generate the dtopo file for a kinematic rupture specified on a set of triangular subfaults (available starting in Clawpack Version 5.5.0).  
# 
# This uses one of the 1300 "fakequake" realizations from the paper
# 
# - *Kinematic rupture scenarios and synthetic displacement data: An example application to the Cascadia subduction zone* by Diego Melgar, R. J. LeVeque, Douglas S. Dreger, Richard M. Allen,  J. Geophys. Res. -- Solid Earth 121 (2016), p. 6658. [doi:10.1002/2016JB013314](http://dx.doi.org/10.1002/2016JB013314). 
# 
# This requires `cascadia30.mshout` containing the geometry of the triangulated fault surface, from
#   https://github.com/dmelgarm/MudPy/blob/master/examples/fakequakes/3D/cascadia30.mshout
# 
# It also requires a rupture scenario in the form of a `.rupt` file from the collection of fakequakes archived at  <https://zenodo.org/record/59943#.WgHuahNSxE4>.
# 
# This sample uses one rupture scenario extracted from `data/cascadia.001012`.

# ### Version
# 
# Runs with Clawpack v5.7.1.



# In[ ]:


from clawpack.geoclaw import dtopotools, topotools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from clawpack.visclaw.plottools import plotbox
import os
from clawpack.visclaw import animation_tools
from IPython.display import HTML


# ## Load etopo1 topo and extract coastline for plotting

# In[ ]:


extent = [-130,-122,39,52]
topo = topotools.read_netcdf('etopo1', extent=extent)

# generate coast_xy from etopo1 data:
coast_xy = topo.make_shoreline_xy()


# ### Set up CSZ geometry

# In[ ]:


fault_geometry_file = './cascadia30.mshout'
print('Reading fault geometry from %s' % fault_geometry_file)
print('\nHeader:\n')
print(open(fault_geometry_file).readline())


# In[ ]:


# read in .mshout (CSZ geoemetry)

cascadia = np.loadtxt(fault_geometry_file,skiprows=1)
cascadia[:,[3,6,9,12]] = 1e3*abs(cascadia[:,[3,6,9,12]])

print('Loaded geometry for %i triangular subfaults' % cascadia.shape[0])


# For example, the first triangular fault in the given geometry of CSZ has the nodes

# In[ ]:


print(cascadia[0,4:7])
print(cascadia[0,7:10])
print(cascadia[0,10:13])


# ### Plot the subfaults:

# Set up a fault model with these subfaults, without yet specifying a particular earthquake scenario. 

# In[ ]:


fault0 = dtopotools.Fault()
fault0.subfaults = []

nsubfaults = cascadia.shape[0]

for j in range(nsubfaults):
    subfault0 = dtopotools.SubFault()
    node1 = cascadia[j,4:7].tolist()
    node2 = cascadia[j,7:10].tolist()
    node3 = cascadia[j,10:13].tolist()
    node_list = [node1,node2,node3]
    subfault0.set_corners(node_list,projection_zone='10T')
    fault0.subfaults.append(subfault0)


# Now we can plot the triangular subplots:

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = plt.axes()
plt.contourf(topo.X, topo.Y, topo.Z, [0,10000], colors=[[.6,1,.6]])
for s in fault0.subfaults:
    c = s.corners
    c.append(c[0])
    c = np.array(c)
    ax.plot(c[:,0],c[:,1], 'b',linewidth=0.3)
plt.plot(coast_xy[:,0], coast_xy[:,1], 'g', linewidth=0.7)
plotbox([-125,-122.1,47.88,48.75], {'color':'k','linewidth':0.7})
ax.set_xlim(-130,-122)
ax.set_ylim(39,52)
ax.set_aspect(1./np.cos(45*np.pi/180.))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude');
#ax.set_title('Triangular subfaults');
if 1:
    fname = 'CSZ_subfaults.png'
    plt.savefig(fname, bbox_inches='tight')
    print('Created ',fname)


# ### Rupture scenario

# We now read in rupture scenario, using data from [https://zenodo.org/record/59943#.WgHuahNSxE4].

# In[ ]:


rnum = 1127
rupt_fname = '_cascadia.%s.rupt' % str(rnum).zfill(6)
print("Reading earthquake data from %s" % rupt_fname)
rupture_parameters = np.loadtxt(rupt_fname,skiprows=1)


# This data is used to set the slip and rake on each of the subfaults loaded above.  Since this is a dynamic rupture, we also set the `rupture_time` and `rise_time` of each subfault.

# In[ ]:


fault0 = dtopotools.Fault()
fault0.subfaults = []
fault0.rupture_type = 'kinematic'
rake = 90. # assume same rake for all subfaults

J = int(np.floor(cascadia.shape[0]))

for j in range(J):
    subfault0 = dtopotools.SubFault()
    node1 = cascadia[j,4:7].tolist()
    node2 = cascadia[j,7:10].tolist()
    node3 = cascadia[j,10:13].tolist()
    node_list = [node1,node2,node3]
    
    ss_slip = rupture_parameters[j,8]
    ds_slip = rupture_parameters[j,9]
    
    rake = np.rad2deg(np.arctan2(ds_slip, ss_slip))
    
    subfault0.set_corners(node_list,projection_zone='10T')
    subfault0.rupture_time = rupture_parameters[j,12]
    subfault0.rise_time = rupture_parameters[j,7]
    subfault0.rake = rake

    slip = np.sqrt(ds_slip ** 2 + ss_slip ** 2)
    subfault0.slip = slip
    fault0.subfaults.append(subfault0)


# ### Compute seafloor deformations with GeoClaw 
# 
# We now run the ``create_dtopography`` routine to generate dynamic seafloor deformations at a given set of times.  This applies the Okada model to each of the subfaults and evaluates the surface displacement on the grid given by `x,y`, at each time. These are summed up over all subfaults to compute the total deformation.

# In[ ]:


x,y = fault0.create_dtopo_xy(dx = 4/60.)
print('Will create dtopo on arrays of shape %i by %i' % (len(x),len(y)))
tfinal = max([subfault1.rupture_time + subfault1.rise_time for subfault1 in fault0.subfaults])
times0 = np.linspace(0.,tfinal,100)
dtopo0 = fault0.create_dtopography(x,y,times=times0,verbose=True);


# In[ ]:


x.shape,y.shape, dtopo0.dZ.shape


# In[ ]:


fig,(ax0,ax1,ax2, ax3) = plt.subplots(ncols=4,nrows=1,figsize=(16,6))
fault0.plot_subfaults(axes=ax0,slip_color=True,plot_box=False);
ax0.set_title('Slip on Fault');

X = dtopo0.X; Y = dtopo0.Y; dZ_at_t = dtopo0.dZ_at_t
dz_max = dtopo0.dZ.max()

t0 = 0.25*tfinal    # time to plot deformation
dtopotools.plot_dZ_colors(X,Y,dZ_at_t(t0),axes=ax1, 
                          cmax_dZ = dz_max, add_colorbar=False);
ax1.set_title('Seafloor at time t=' + str(t0));

t0 = 0.5*tfinal    # time to plot deformation
dtopotools.plot_dZ_colors(X,Y,dZ_at_t(t0),axes=ax2,
                          cmax_dZ = dz_max, add_colorbar=False);

ax2.set_title('Seafloor at time t=' + str(t0));

t0 = tfinal    # time to plot deformation
dtopotools.plot_dZ_colors(X,Y,dZ_at_t(t0),axes=ax3,
                          cmax_dZ = dz_max, add_colorbar=True);
ax3.set_title('Seafloor at time t=' + str(t0));

#fig.savefig('CSZ_triangular.png');


# ### Plot the rupture time and rise time of each subfault
# 
# This shows where the rupture originates and how it propagates outward. Each vertical bar shows the rupture time and duration of one subfault.

# In[ ]:


plt.figure(figsize=(14,8))
plt.axes()
latitudes = [s.latitude for s in fault0.subfaults]
rise_times = [s.rise_time for s in fault0.subfaults]
rupture_times = [s.rupture_time for s in fault0.subfaults]
for j,lat in enumerate(latitudes):
    plt.plot([lat,lat],[rupture_times[j],rupture_times[j]+rise_times[j]],'b')
plt.xlabel('latitude')
plt.ylabel('seconds')
plt.title('rupture time + rise time of each triangle vs. latitude')


# ## Plots for paper

# In[ ]:


cmap = mpl.cm.jet
cmap.set_under(color='w', alpha=0)  # transparent for zero slip


# In[ ]:


cmap_slip = mpl.cm.jet

def plot_triangular_slip(subfault,ax,cmin_slip,cmax_slip):
    x_corners = [subfault.corners[2][0],
                 subfault.corners[0][0],
                 subfault.corners[1][0],
                 subfault.corners[2][0]]

    y_corners = [subfault.corners[2][1],
                 subfault.corners[0][1],
                 subfault.corners[1][1],
                 subfault.corners[2][1]]
    
    slip = subfault.slip
    s = min(1, max(0, (slip-cmin_slip)/(cmax_slip-cmin_slip)))
    c = np.array(cmap_slip(s*.99))  # since 1 does not map properly with jet
    if slip <= cmin_slip:
        c[-1] = 0  # make transparent

    ax.fill(x_corners,y_corners,color=c,edgecolor='none')


# In[ ]:


cmin_slip = 0.01  # smaller values will be transparent
cmax_slip = np.array([s.slip for s in fault0.subfaults]).max()


# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = plt.axes()
#plt.contourf(topo.X, topo.Y, topo.Z, [0,10000], colors=[[.6,1,.6]])
for s in fault0.subfaults:
    c = s.corners
    c.append(c[0])
    c = np.array(c)
    ax.plot(c[:,0],c[:,1], 'k', linewidth=0.2)
    plot_triangular_slip(s,ax,cmin_slip,cmax_slip)
    
plt.plot(coast_xy[:,0], coast_xy[:,1], 'g', linewidth=0.7)
plotbox([-125,-122.1,47.88,48.75], {'color':'k','linewidth':0.7})

if 0:
    fault0.plot_subfaults(axes=ax,slip_color=True,plot_box=False,
                      colorbar_ticksize=12, colorbar_labelsize=12);

ax.set_xlim(-130,-122)
ax.set_ylim(39,52)
ax.set_aspect(1./np.cos(45*np.pi/180.))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude');
#ax.set_title('Triangular subfaults');

if 1:
    # add colorbar
    cax,kw = mpl.colorbar.make_axes(ax, shrink=1)
    norm = mpl.colors.Normalize(vmin=cmin_slip,vmax=cmax_slip)
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap_slip, norm=norm)
    cb1.set_label("Slip (m)",fontsize=12)
    cb1.ax.tick_params(labelsize=12)

if 1:
    fname = 'cascadia_%s.png' % rnum
    plt.savefig(fname, bbox_inches='tight')
    print('Created ',fname)


# ### plot dtopo

# In[ ]:


fig = plt.figure(figsize=(15,10))
ax = plt.axes()
#plt.contourf(topo.X, topo.Y, topo.Z, [0,10000], colors=[[.6,1,.6]])
for s in fault0.subfaults:
    c = s.corners
    c.append(c[0])
    c = np.array(c)
    ax.plot(c[:,0],c[:,1], 'k', linewidth=0.2)
    #plot_triangular_slip(s,ax,cmin_slip,cmax_slip)
    
plt.plot(coast_xy[:,0], coast_xy[:,1], 'g', linewidth=0.7)
plotbox([-125,-122.1,47.88,48.75], {'color':'k','linewidth':0.7})

dtopo0.plot_dZ_colors(t=3600., axes=ax)

ax.set_xlim(-130,-122)
ax.set_ylim(39,52)
ax.set_aspect(1./np.cos(45*np.pi/180.))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#ax.set_xlabel('Longitude')
#ax.set_ylabel('Latitude');
ax.set_title('');

if 1:
    fname = 'cascadia_%s_dtopo.png' % rnum
    plt.savefig(fname, bbox_inches='tight')
    print('Created ',fname)


# ### Create dtopo file for running GeoClaw

# In[ ]:


ruptno = rupt_fname.split('.')[1]
fname = 'cascadia' + ruptno + '.dtt3'
dtopo0.write(fname, dtopo_type=3)
print('Created %s, with dynamic rupture of a Mw %.2f event' % (fname, fault0.Mw()))


# In[ ]:




