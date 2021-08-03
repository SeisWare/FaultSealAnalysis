import SeisWare

import SWconnect

from shapely import geometry

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
            xpoints = []    
            ypoints = []
            '''
            for j in val.points:
                xpoints.append(j.x.Value(SeisWare.Unit.Meter))
                ypoints.append(j.y.Value(SeisWare.Unit.Meter))
            '''   
        polylist.append(geometry.Polygon([[p.x.Value(SeisWare.Unit.Meter), p.y.Value(SeisWare.Unit.Meter)] for p in val.points]))
    
    return polylist

proj = SWconnect.sw_connect("CanacolFaults")

cult = getLayer(proj,"Icotea Top CDO Faults_poly")

layers = get_polyobjects(cult)

print(len(layers))
