
from networkx.algorithms.bipartite.basic import color
from scipy.sparse.data import _minmax_mixin
import get_polygon
import matplotlib.pyplot as plt
import get_centerline
import SWconnect
import pandas as pd
import numpy as np

proj = SWconnect.sw_connect("CanacolFaults")

cult = get_polygon.getLayer(proj,"Icotea Top CDO Faults_poly - Copy")

grid = SWconnect.get_grid(proj,"Icotea TopCDO Depth-July2021 Grid")

# Replace bad grid values with NaN

grid[grid > 100000000] = np.nan

print(grid)

layers = get_polygon.get_polyobjects(cult)


from shapely.geometry import LineString

from shapely.geometry.polygon import LinearRing

faultDFlist = []
count = 0
for layer in layers:

    x, y = layer.exterior.xy

    count += 1 

    # Convert to a ring for intersections
    lring = LinearRing(list(layer.exterior.coords))
    

    plt.plot(x,y)

    x3, y3 = get_polygon.center_extract(layer)

    plt.plot(x3,y3,'-')

    strikelistx1 = []
    strikelisty1 = []
    strikelistx2 = []
    strikelisty2 = []
    midlistx = []
    midlisty = []

    # Scroll through elements in x and y lists to calculate perpendicular lines
    # Perpendicular lines are calculated by connecting parallel offsets to line strings
    for k,v in enumerate((zip(x3,y3))):

        if (k+1 < len(list(zip(x3,y3))) and k - 1 >= 0):
            
            cd_length = 250

            ab = LineString(([(list(zip(x3,y3))[k-1][0],list(zip(x3,y3))[k-1][1]),(v[0],v[1])]))

            left = ab.parallel_offset(cd_length / 2, 'left')
            right = ab.parallel_offset(cd_length / 2, 'right')

            c = left.boundary[1]
            d = right.boundary[0]  # note the different orientation for right offset
            cd = LineString([c, d])
            
            cd = cd.intersection(lring)

            try:
                midlistx.append(v[0])
                midlisty.append(v[1])
                strikelistx1.append(cd[0].x)
                strikelistx2.append(cd[1].x)
                strikelisty1.append(cd[0].y)
                strikelisty2.append(cd[1].y)
            except TypeError:
                strikelistx1.append(np.NaN)
                strikelistx2.append(np.NaN)
                strikelisty1.append(np.NaN)
                strikelisty2.append(np.NaN)
            

    faultpolydict = {"Object":count,"X1":strikelistx1,"Y1":strikelisty1,"X2":strikelistx2,"Y2":strikelisty2,"MidX":midlistx,"MidY":midlisty}
        
    faultDF = pd.DataFrame(faultpolydict)

    faultDFlist.append(faultDF)
    #print(cd.intersection(layer))


faultDF = pd.concat(faultDFlist).dropna()
grid.dropna(inplace=True)
print(faultDF)
print(len(layers))


from scipy.interpolate import griddata

faultDF["Z1"] = griddata((grid.X,grid.Y),grid.Z,(faultDF.X1,faultDF.Y1),method="nearest")

faultDF["Z2"] = griddata((grid.X,grid.Y),grid.Z,(faultDF.X2,faultDF.Y2),method="nearest")

faultDF["Zdiff"] = faultDF.Z1-faultDF.Z2

print(faultDF.reset_index())

### Plotting endpoints

#plt.plot(faultDF.X1,faultDF.Y1,'o')

#plt.plot(faultDF.X2,faultDF.Y2,'o')

faultDF.reset_index().to_csv("result.csv")
plt.show()

