
from networkx.algorithms.bipartite.basic import color
from scipy.sparse.data import _minmax_mixin
import get_polygon
import matplotlib.pyplot as plt
import SWconnect
import pandas as pd
import numpy as np

proj = SWconnect.sw_connect("CanacolFaults")

cult = get_polygon.getLayer(proj,"Icotea Top CDO Faults_poly - Copy")

grid = SWconnect.get_grid(proj,"Icotea TopCDO Depth-July2021 Grid")

bin_size = 50

# Replace null grid values with NaN

grid[grid > 100000000] = np.NaN

layers = get_polygon.get_polyobjects(cult)

faultDFlist = []
count = 0

for layer in layers:

    count += 1 

    faultDF = get_polygon.get_strikes(layer,count)

    faultDFlist.append(faultDF)


faultDF = pd.concat(faultDFlist).dropna()
grid.dropna(inplace=True)
print(faultDF)

from scipy.interpolate import griddata

# NOTE set a max distance for interpolation

faultDF["Z1x"] = griddata((grid.X,grid.Y),grid.X,(faultDF.X1,faultDF.Y1),method="nearest")
faultDF["Z1y"] = griddata((grid.X,grid.Y),grid.Y,(faultDF.X1,faultDF.Y1),method="nearest")

faultDF["Z2x"] = griddata((grid.X,grid.Y),grid.X,(faultDF.X2,faultDF.Y2),method="nearest")
faultDF["Z2y"] = griddata((grid.X,grid.Y),grid.Y,(faultDF.X2,faultDF.Y2),method="nearest")

faultDF["Z1"] = griddata((grid.X,grid.Y),grid.Z,(faultDF.X1,faultDF.Y1),method="nearest")
faultDF["Z2"] = griddata((grid.X,grid.Y),grid.Z,(faultDF.X2,faultDF.Y2),method="nearest")

# Calculate the difference between the point on polygon and the nearest grid data value
faultDF["Interp_distance1"] = ((faultDF.Z1x-faultDF.X1)**2+(faultDF.Z1y-faultDF.Y1)**2)**(0.5)

faultDF["Interp_distance2"] = ((faultDF.Z2x-faultDF.X2)**2+(faultDF.Z2y-faultDF.Y2)**2)**(0.5)

faultDF["Zdiff"] = faultDF.Z1-faultDF.Z2

for i in faultDF.Object.unique():
    plotDF = faultDF.loc[faultDF['Object'] == i]

    # Drop values if the distance is greater than the bin size
    ZplotDF = plotDF.drop(plotDF[(plotDF["Interp_distance1"] > bin_size) | (plotDF["Interp_distance2"] > bin_size)].index)

    ZplotDF['Zdiffsmooth'] = ZplotDF.loc[:,('Zdiff')].rolling(13).mean()

    #plotDF['Zdiffsmooth'] = plotDF['Zdiff']

    zero_crossings = np.where(np.diff(np.sign(ZplotDF['Zdiffsmooth'])))[0]
    #zero_crossings = np.where(np.logical_and(plotDF['Zdiffsmooth']>=-0.2, plotDF['Zdiffsmooth']<=0.2))[0]
    
    if len(zero_crossings) > 0:
        #zero_crossings = [x + 7 for x in zero_crossings]

        zc_DF = ZplotDF.iloc[zero_crossings]
        
        zc_DF["UniqueID"] = zc_DF['Object'].map(str)+'_'+zc_DF['Length'].map(str)
    else:
        zc_DF = pd.DataFrame(columns = ["Length","Zdiffsmooth"])
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(ZplotDF.Length,ZplotDF['Zdiffsmooth'])
    fig.suptitle(f"Fault #{i}")
    #ax1.xlabel("Distance Along Fault (m)")
    #ax1.ylabel("Grid Difference (m)")
    ax1.plot(zc_DF.Length,zc_DF['Zdiffsmooth'],'x')
    ax1.axhline(y=0,color='r')

    ax2.plot(plotDF.X1,plotDF.Y1)
    ax2.plot(plotDF.X2,plotDF.Y2)
    ax2.plot(plotDF.MidX,plotDF.MidY)
    zc_DF= zc_DF[zc_DF['Zdiffsmooth'].notna()]
    ax2.plot(zc_DF.MidX,zc_DF.MidY,'x')
    ax2.set_aspect('equal',adjustable='box')
    
    fig.savefig(f".\Images\Fault{i}.png")
    zc_DF.to_csv(f".\CSV Files\zc_DF{i}.csv")

    plt.show()

#print(faultDF.reset_index())

faultDF.reset_index().to_csv("result.csv")

