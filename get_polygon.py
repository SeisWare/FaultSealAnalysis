
import SeisWare
import networkx as nx
import geometry_custom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing
from shapely import geometry
from shapely.geometry import LineString
import os

# Collection of functions designed to interact with SeisWare SDK

def getLayer(login_instance,name):
    #Here we get a single culture layer based on a login instance and a given layer name
    #We will also populate the layer because why not
    culture_list = SeisWare.CultureList()

    login_instance.CultureManager().GetAll(culture_list)

    cult = [i for i in culture_list if i.Name()==name]
    
    login_instance.CultureManager().PopulateObjects(cult[0])

    objects = SeisWare.CultureObjects()
    
    cult[0].Objects(objects)

    polyObjects = SeisWare.PolygonObjectList()

    objects.PolygonObjects(polyObjects)

    return polyObjects

def get_polyobjects(cult):
    #Return a list of all areas in a layer, uses a populated layer as input such as from getLayer function
    polylist = []
    #Go through objects in layer
    for val in cult:
        
        # Get all the x,y points from the culture layer
        if val.closed == True:
            polylist.append(geometry.Polygon([[p.x.Value(SeisWare.Unit.Meter), p.y.Value(SeisWare.Unit.Meter)] for p in val.points]))
    
    return polylist

def set_polygon_object(points, closed = True):

    # Set the polygon inside a SeisWare project using a list of xy values

    polygon_points = SeisWare.WorldPosList()

    xylist = []

    for i in points:
        #print(i[0][0],i[1])
        wp2 = SeisWare.WorldPos2(SeisWare.Measurement(i[0][0],SeisWare.Unit.Meter),SeisWare.Measurement(i[1][0],SeisWare.Unit.Meter))
        polygon_points.append(wp2)
        xylist.append((i[0][0],i[1][0]))


    polyobjectlist = SeisWare.PolygonObjectList()
    
    #login_instance.CultureManager().SetPolygonObjects(polyobjectlist)

    return xylist

def set_polygon(login_instance, name, objects):

    return None


def center_extract(culture_layer):

    # Function to extract the centerline from seisware polygon object and return two lists. Corresponding x,y points for centerline

    # Extract centerline of polygon
    center = geometry_custom.Centerline(culture_layer,interpolation_distance=3)

    centerlist = [list(x.coords) for x in list(center)]

    # Sort list of xy points from centerline

    # Build graph of points

    G_center = nx.Graph()

    G_center.add_edges_from(centerlist)

    end_nodes = [x for x in G_center.nodes() if G_center.degree(x)==1]

    current_max_path = []
    
    from itertools import combinations

    for combo in combinations(end_nodes,2):
        dij_path = nx.dijkstra_path(G_center,combo[0],combo[1])
        if len(dij_path) > len(current_max_path):
                current_max_path = dij_path
                
    print(len(current_max_path))

    x_c = np.array(list(zip(*current_max_path))[0])
    y_c = np.array(list(zip(*current_max_path))[1])

    return x_c,y_c

def get_strikes(layer, count):

    # Get the x,y points of each polygon
    x, y = layer.exterior.xy

    # Convert to a ring for intersections
    lring = LinearRing(list(layer.exterior.coords))
    
    #plt.plot(x,y)

    x3, y3 = center_extract(layer)

    #plt.plot(x3,y3,'-')
    
    strikelistx1 = []
    strikelisty1 = []
    strikelistx2 = []
    strikelisty2 = []
    midlistx = []
    midlisty = []
    abdistance = []
    
    # Scroll through elements in x and y lists to calculate perpendicular lines
    # Perpendicular lines are calculated by connecting parallel offsets to line strings
    for k,v in enumerate((zip(x3,y3))):

        if (k+1 < len(list(zip(x3,y3))) and k - 1 >= 0):
            
            cd_length = 550

            ab = LineString(([(list(zip(x3,y3))[k-1][0],list(zip(x3,y3))[k-1][1]),(v[0],v[1])]))

            left = ab.parallel_offset(cd_length / 2, 'left')
            right = ab.parallel_offset(cd_length / 2, 'right')

            c = left.boundary[1]
            d = right.boundary[0]  # note the different orientation for right offset

            cd = LineString([c, d])
            
            cd = cd.intersection(lring)
            '''
            # NOTE this is for plotting to QC values
            if ab.xy[1][0] < ab.xy[1][1]: # check if the line is going up or down
                plt.plot(cd[0].x,cd[0].y, 'o')
                plt.plot(cd[1].x,cd[1].y, 'x')
            elif ab.xy[1][0] > ab.xy[1][1]:
                plt.plot(cd[1].x,cd[1].y, 'o')
                plt.plot(cd[0].x,cd[0].y, 'x')
            '''
            
            if len(abdistance) == 0:
                abdistance = [ab.length]
            else:
                abdistance.append(ab.length+abdistance[k-2])

            try:
                midlistx.append(v[0])
                midlisty.append(v[1])
                if ab.xy[1][0] < ab.xy[1][1]:
                    strikelistx1.append(cd[0].x)
                    strikelistx2.append(cd[1].x)
                    strikelisty1.append(cd[0].y)
                    strikelisty2.append(cd[1].y)
                elif ab.xy[1][0] > ab.xy[1][1]: # when the line is going down, change the direction of the strike intersection points
                    strikelistx1.append(cd[1].x)
                    strikelistx2.append(cd[0].x)
                    strikelisty1.append(cd[1].y)
                    strikelisty2.append(cd[0].y)
            except TypeError:
                strikelistx1.append(np.NaN)
                strikelistx2.append(np.NaN)
                strikelisty1.append(np.NaN)
                strikelisty2.append(np.NaN)
    
    faultpolydict = {"Object":count,"X1":strikelistx1,"Y1":strikelisty1,"X2":strikelistx2,"Y2":strikelisty2,"MidX":midlistx,"MidY":midlisty,"Length":abdistance}
          
    faultDF = pd.DataFrame(faultpolydict)

    return faultDF

def build_plots(plotDF,bin_size,plot_number,folder_path = ".",smoothing_window = 13):
    # Drop values if the distance is greater than the bin size
    ZplotDF = plotDF.drop(plotDF[(plotDF["Interp_distance1"] > bin_size) | (plotDF["Interp_distance2"] > bin_size)].index)

    ZplotDF['Zdiffsmooth'] = ZplotDF.loc[:,('Zdiff')].rolling(13).mean()

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
    fig.suptitle(f"Fault #{plot_number}")
    #ax1.xlabel("Distance Along Fault (m)")
    #ax1.ylabel("Grid Difference (m)")
    ax1.plot(zc_DF.Length,zc_DF['Zdiffsmooth'],'x')
    try:    
        ax1.text(ZplotDF.Length.iloc[2],0,"A")

        ax1.text(ZplotDF.Length.iloc[-1],0,"B")
    except IndexError:
        None
    ax1.axhline(y=0,color='r')

    ax2.plot(plotDF.X1,plotDF.Y1,color = "r")
    ax2.plot(plotDF.X2,plotDF.Y2,color = "r")
    

    try:
        ax2.plot(plotDF.MidX,plotDF.MidY)
    except AttributeError:
        None
    
    zc_DF= zc_DF[zc_DF['Zdiffsmooth'].notna()]
    try:
        ax2.plot(plotDF.MidX,plotDF.MidY)

        ax2.text(plotDF.MidX.iloc[0],plotDF.MidY.iloc[0],"A")
        ax2.text(plotDF.MidX.iloc[-1],plotDF.MidY.iloc[-1],"B")

        ax2.plot(zc_DF.MidX,zc_DF.MidY,'x')
    except AttributeError:
        None
    
    
    ax2.set_aspect('equal',adjustable='box')
    
    if not os.path.isdir(f"{folder_path}/Images/"):
        os.makedirs(f"{folder_path}/Images/")
        os.makedirs(f"{folder_path}/CSV Files/")
    
    fig.savefig(f"{folder_path}/Images/Fault{plot_number}.png")
    zc_DF.to_csv(f"{folder_path}/CSV Files/zc_DF{plot_number}.csv")

    #plt.show()


