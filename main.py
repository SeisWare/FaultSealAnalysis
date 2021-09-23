
from networkx.algorithms.bipartite.basic import color
from scipy.sparse.data import _minmax_mixin
import get_polygon
import matplotlib.pyplot as plt
import SWconnect
import pandas as pd
import numpy as np

proj = SWconnect.sw_connect("CanacolFaults")

cult = get_polygon.getLayer(proj,"Icotea Top CDO Faults_poly - Copy2")

grid = SWconnect.get_grid(proj,"icotea_resample Grid")

# Replace null grid values with NaN

grid[grid > 100000000] = np.NaN

print(grid)

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
print(len(layers))


from scipy.interpolate import griddata

# NOTE set a max distance for interpolation



faultDF["Z1"] = griddata((grid.X,grid.Y),grid.Z,(faultDF.X1,faultDF.Y1),method="nearest")

faultDF["Z2"] = griddata((grid.X,grid.Y),grid.Z,(faultDF.X2,faultDF.Y2),method="nearest")

faultDF["Zdiff"] = faultDF.Z1-faultDF.Z2

print(faultDF.reset_index())

faultDF.reset_index().to_csv("result.csv")

plt.show()

