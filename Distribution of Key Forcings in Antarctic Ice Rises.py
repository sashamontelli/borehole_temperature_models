# ----------------------------------------------------------------------
# This code is used to produce diagrams included in the paper "Montelli, A. and Kingslake, J.:
# Geothermal heat flux is the dominant source of uncertainty in englacial-temperature-based dating of
# ice-rise formation, 2022, EGUsphere [preprint]", https://doi.org/10.5194/egusphere-2022-236.
#
# Originally, it was a part of the Jupyter notebook in this directory. Unfortunately, the code doesn't
# currently produce the same results as those produced in the paper. This different in results may
# be explained by code changes introduced between when the original diagrams were produced and now.
#
# Because this code isn't critical to the core algorithms themselves, we have decided to remove this
# code from the notebook and preserve it here. This way, the notebook is completely reproducible and
# the code itself is not lost.
# ----------------------------------------------------------------------

import warnings

import geopandas as gpd
import numpy as np
import pyproj
import scipy
import xarray as xr

from matplotlib import pyplot as plt
from tqdm import tqdm


# Distribution of key forcings in Antarctic ice rises

# Ultimately, the primary purpose of the paper is to examine what information about climate and
# glacial history can be inferred from englacial temperatures measured within ice rises. Therefore,
# to provide some context for the models presented above, we load the Antarctic ice rises inventory
# by [Matsouka et al 2015](https://www.sciencedirect.com/science/article/pii/S0012825215300416), and
# produce a map that shows the locations of present-day ice rises with respect to distribution of
# geothermal heat flux and surface mass balance of the Antarctic Ice Sheet.

# ----------------------------------------------------------------------
# Importing ice rise inventory and extracting coordinates of ice rises and converting from projected
# to WGS84 coordinate system:

icerises = gpd.read_file('Antarctic ice rises inventory from Matsouka et al 2015/IceRisesInventory_v1/icerises_inventory_v1.shp')
ir = icerises[icerises['type']<3]

proj = pyproj.Transformer.from_crs(4326, 3031, always_xy=True)

longi = np.array(ir.longi)
lati = np.array(ir.lati)

xirlong, yirlat = (longi, lati)
x2, y2 = proj.transform(xirlong, yirlat)

# ----------------------------------------------------------------------
# Importing RACMO climate model and converting from projected to WGS84 coordinate system:

ds = xr.open_dataset('RACMO smb model 1979-2014/RACMO2.3p1_ANT27_SMB_yearly_1979_2014.nc')
smb = ds.smb

rad2deg = 180./np.pi

p = pyproj.Proj('+ellps=WGS84 +proj=ob_tran +o_proj=latlon +o_lat_p=-180.0 +o_lon_p=-170.0 +lon_0=180.0')

rlon = ds.rlon.values
rlat = ds.rlat.values

xmesh,ymesh = np.meshgrid(rlon, rlat)

lon, lat = p(xmesh, ymesh)
lon, lat = lon*rad2deg, lat*rad2deg    # radians --> degrees

# ----------------------------------------------------------------------
# Extracting surface mass balance data at ice rise locations based on the RACMO model:

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    smb_data = smb.mean(dim = 'time')

    smb_data = np.array(smb_data)
    smb_data_rshp = smb_data.reshape(len(rlat), len(rlon))

    tck = scipy.interpolate.bisplrep(lon, lat, smb_data_rshp*24*365*3600, s = 0)

    smb_ice_rises = []

    for i in tqdm(range(len(xirlong))):
        smb_1 = scipy.interpolate.bisplev(xirlong[i],yirlat[i],tck)
        smb_ice_rises.append(smb_1)

    smb_ice_rises = np.array(smb_ice_rises)
    smb_ice_rises[smb_ice_rises<0] = 0

# ----------------------------------------------------------------------
# Importing heat flux data and converting from projected to WGS84 coordinate system:

f = open(('Antarctic GHF model from Martos et al 2017/Antarctic_GHF_Martos.xyz'), 'r')

xval = np.zeros(0)
yval = np.zeros(0)
bhflx = np.zeros(0)

for i,line in enumerate(tqdm(f)):
        line = line.strip()
        columns = line.split()
        xval=np.append(xval, float(columns[0]))
        yval=np.append(yval, float(columns[1]))
        bhflx=np.append(bhflx, float(columns[2]))
f.close()

# ----------------------------------------------------------------------
x_new = np.sort(np.unique(xval)) # are sorted already
y_new = np.sort(np.unique(yval))

fillvalue = -9.e+33
bheatflx = np.zeros((len(y_new), len(x_new))) + fillvalue

for i in tqdm(range(len(bhflx))):
    # go through all vals and fill them into the right place
    ix = np.in1d(x_new.ravel(), xval[i]).reshape(x_new.shape)
    iy = np.in1d(y_new.ravel(), yval[i]).reshape(y_new.shape)
    bheatflx[iy,ix] = bhflx[i]*1.0e-3


hf = np.array(bheatflx)
hf = hf.reshape((len(y_new), len(x_new)))

# ----------------------------------------------------------------------
# Extracting heat flux data at ice rise locations:

X,Y = np.meshgrid(x_new, y_new)

xs = X.reshape((350*291, 1))
ys = Y.reshape((350*291, 1))

points = np.array([x2, y2])

hf_ice_rises = []

for i in tqdm(range(len(x2))):
    hf_1 = zs2 = scipy.interpolate.griddata(np.hstack((xs, ys)), hf.reshape(-1), points[:,i])
    hf_ice_rises.append(hf_1)

hf_ice_rises = np.array(hf_ice_rises)
hf_ice_rises[hf_ice_rises>1e-01] = np.nan
hf_ice_rises[hf_ice_rises<0] = np.nan

# ----------------------------------------------------------------------
# Plotting the distribution of key ice-rise parameters across the Antarctic Ice Sheet

fig, ax = plt.subplots(1, 4, figsize=(10, 4), constrained_layout=True)

irthick = np.array(ir.thick)
irthick[irthick == -9999.0] = 0

vel_mean = np.array(ir.vel_mean)
vel_mean[vel_mean == -9999.0] = 0
vel_mean[vel_mean == 608.136446] = 0

for index, (title, values, color, legend, x_label) in enumerate(
    [
        ("Ice rise thickness", irthick, "azure", "\u03bc = {:0.1f}, \u03C3 = {:0.1f}".format(np.mean(irthick),np.std(irthick)), "Thickness, m"),
        ("Flow velocity", vel_mean, "tomato", "\u03bc = {:0.1f}, \u03C3 = {:0.1f}".format(np.mean(vel_mean),np.std(vel_mean)), "Velocity, m/s"),
        ("Ice rise accumulation rates", smb_ice_rises / 918, "aquamarine", "\u03bc = {:0.1f}, \u03C3 = {:0.1f}".format(np.nanmean(smb_ice_rises/918),np.nanstd(smb_ice_rises // 918)), "Accumulation rate, m/y"),
        ("Heat flux", hf_ice_rises, "orange", "\u03bc = {:0.2f}, \u03C3 = {:0.2f}".format(np.nanmean(hf_ice_rises),np.nanstd(hf_ice_rises)), "Heat flux, W/m^2"),
    ],
):
    ax[index].hist(values, 16, density=True, facecolor=color, edgecolor='black', alpha=1, linewidth=0.2, label=legend)
    ax[index].grid()
    ax[index].legend(borderpad=0.5,prop={'size':7})
    ax[index].set_xlabel(x_label)
    ax[index].set_ylabel('Probability density')
    ax[index].set_title(title)

fig.savefig('Ice rise statistics.pdf')

# ----------------------------------------------------------------------
fig, axes = plt.subplots(figsize=(10,6))
ax1 = axes

pcm = ax1.contourf(x_new, y_new, hf, alpha= 1, levels=np.linspace(0.03,0.15, num=7), cmap="binary")
cbar1 = fig.colorbar(pcm, ax=ax1)
cbar1.set_label('Heat flux, W/m^2')
ax1.contour(x_new, y_new, hf, levels=0, colors ='k',linewidths=0.5)
sc = ax1.scatter(x2, y2, alpha = 1, s=smb_ice_rises/5, c=irthick, edgecolors = 'k', linewidths = 1, cmap="gnuplot2")
cbar2 = fig.colorbar(sc, ax=ax1)
cbar2.set_label('Ice thickness, m')
fig.legend(*sc.legend_elements("sizes", num=8), loc = 'center right', prop={'size': 10})
ax1.axis('equal')
ax1.grid()
ax1.set_xlabel('x, m')
ax1.set_ylabel('y, m')


fig.savefig('Ice rise dating locations.pdf')
