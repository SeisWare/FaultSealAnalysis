
from networkx.algorithms.bipartite.basic import color
from scipy.sparse.data import _minmax_mixin
import get_polygon
import matplotlib.pyplot as plt
import SWconnect
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

#proj = SWconnect.sw_connect("CanacolFaults")

#cult = get_polygon.getLayer(proj,"Icotea Top CDO Faults_poly - Copy")

#grid = SWconnect.get_grid(proj,"Icotea TopCDO Depth-July2021 Grid")


def fault_throw_viz(proj,cult_name,grid_name,folder_name):
    # Replace null grid values with NaN
    
    cult = get_polygon.getLayer(proj,cult_name)
    grid = SWconnect.get_grid(proj,grid_name)

    bin_size = 50

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

        get_polygon.build_plots(plotDF,bin_size,i,folder_name)

    faultDF.reset_index().to_csv(f"{folder_name}\\result.csv")

