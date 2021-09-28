# -*- coding: utf-8 -*-
"""
This is a helper function for connecting to a SeisWare Project
"""
import SeisWare
import sys 
import pandas as pd
import numpy as np

def handle_error(message, error):
    """
    Helper function to print out the error message to stderr and exit the program.
    """
    print(message, file=sys.stderr)
    print("Error: %s" % (error), file=sys.stderr)
    sys.exit(1)

def sw_project_list():
    '''
    Return a list of SeisWare projects

    Values will be SeisWare Project objects

    '''
    connection = SeisWare.Connection()
    
    try:
        serverInfo = SeisWare.Connection.CreateServer()
        connection.Connect(serverInfo.Endpoint(), 10000)
    except RuntimeError as err:
        handle_error("Failed to connect to the server", err)

    project_list = SeisWare.ProjectList()
    
    connection.ProjectManager().GetAll(project_list)
    
    return project_list

def sw_connect(project_name):
    '''
    project_name : String containing project name

    return : login_instance object used to access data within a project

    Connect to a SeisWare project

    '''
    connection = SeisWare.Connection()
    
    try:
        serverInfo = SeisWare.Connection.CreateServer()
        connection.Connect(serverInfo.Endpoint(), 5000)
    except RuntimeError as err:
        handle_error("Failed to connect to the server", err)

    project_list = SeisWare.ProjectList()

    try:
        connection.ProjectManager().GetAll(project_list)
    except RuntimeError as err:
        handle_error("Failed to get the project list from the server", err)

    projects = [project for project in project_list if project.Name() == project_name]
    if not projects:
        print("No project was found", file=sys.stderr)
        sys.exit(1)
        
    login_instance = SeisWare.LoginInstance()
    
    attempts = 0

    while attempts < 3:
        try:
            login_instance.Open(connection, projects[0])
            break
        except RuntimeError as err:
            attempts += 1
            handle_error("Failed to connect to the project", err)
            
            
    return login_instance

def get_well(login_instance,uwi):
    
    '''
    uwi : String containing well UWI

    returns : SeisWare well object

    '''

    well_list = SeisWare.WellList()
    
    try:
        login_instance.WellManager().GetAll(well_list)
    except RuntimeError as err:
        handle_error("Failed to get all the wells from the project", err)
        
    wells = [well for well in well_list if well.UWI() == uwi]
    
    return wells[0]


def get_all_wells(login_instance):
    
    '''

    Returns all wells in a SeisWare project as a list of well objects

    '''

    well_list = SeisWare.WellList()
    
    try:
        login_instance.WellManager().GetAll(well_list)
    except RuntimeError as err:
        handle_error("Failed to get all the wells from the project", err)
        
    wells = [well for well in well_list]
    
    return wells

def get_filter_wells(login_instance,filter_name):
    
    '''
    filter_name: String containing filter name

    Returns filtered wells in a SeisWare project as a list of well objects

    '''
    well_filter = SeisWare.FilterList()

    login_instance.FilterManager().GetAll(well_filter)

    well_filter = [i for i in well_filter if i.Name() == filter_name]

    keys = SeisWare.IDSet()

    failed_keys = SeisWare.IDSet()

    well_list = SeisWare.WellList()
    
    try:
        login_instance.WellManager().GetKeysByFilter(well_filter[0],keys)
        login_instance.WellManager().GetByKeys(keys,well_list,failed_keys)
    except RuntimeError as err:
        handle_error("Failed to get all the wells from the project", err)
        
    wells = [well for well in well_list]
    
    return wells


def get_grid(login_instance, grid_name):

    '''
    grid_name: String with grid name

    returns: Dataframe containing X Y Z, where the header for Z is the Grid Name
    
    Get a grid and return a dataframe with X Y Z values
    '''

    # Get the grids from the project
    grid_list = SeisWare.GridList()
    try:
        login_instance.GridManager().GetAll(grid_list)
    except RuntimeError as err:
        handle_error("Failed to get the grids from the project", err)

    # Get the grid we want
    grids = [grid for grid in grid_list if grid.Name() == grid_name]
    
    if not grids:
        print("No grids were found", file=sys.stderr)
        sys.exit(1)

    # Populate the grid with it's values
    try:
        login_instance.GridManager().PopulateValues(grids[0])
    except RuntimeError as err:
        handle_error("Failed to populate the values of grid %s from the project" % (grid_name), err)
    
    grid = grids[0]

    # Get the values from the grid
    grid_values = SeisWare.GridValues()
    grid.Values(grid_values)
    #Fill a DF with X,Y,Z values
    #Make a list of tuples
    xyzcoords = []
    grid_values_list = list(grid_values.Data())
    counter = 0
    grid_df = pd.DataFrame()
    for i in range(grid_values.Height()):
        for j in range(grid_values.Width()):
            xyzcoords.append((grid.Definition().RangeY().start+i*grid.Definition().RangeY().delta,
                            grid.Definition().RangeX().start+j*grid.Definition().RangeX().delta,
                            grid_values_list[counter]))
            counter = counter + 1
            #print(counter)
            
    grid_df = pd.DataFrame(xyzcoords,columns=["Y","X","Z"])

    return grid_df


def get_srvy(login_instance, well, depth_unit = SeisWare.Unit.Meter):
    '''
    well: SeisWare well object
    depth_unit: Defaults to meter, alternatively can be SeisWare.Unit.Foot

    Get the directional survey based on well, login_instance. 
    Will fail for well without Directional Survey.
    Return directional survey as dataframe with 
    columns 
    UWI X Y TVDSS MD
    
    '''

    surfaceX = well.TopHole().x.Value(depth_unit)
    surfaceY = well.TopHole().y.Value(depth_unit)
    surfaceDatum = well.DatumElevation().Value(depth_unit)

    dirsrvylist = SeisWare.DirectionalSurveyList()

    login_instance.DirectionalSurveyManager().GetAllForWell(well.ID(),dirsrvylist)

    #Select the directional survey if it exists
    dirsrvy = [i for i in dirsrvylist if i.OffsetNorthType()>0]
    
    login_instance.DirectionalSurveyManager().PopulateValues(dirsrvy[0])
    
    srvypoints = SeisWare.DirectionalSurveyPointList()
    
    dirsrvy[0].Values(srvypoints)
    
    srvytable = []
    
    for i in srvypoints:
        srvytable.append((
            well.Name(),
            surfaceX + i.xOffset.Value(depth_unit),
            surfaceY + i.yOffset.Value(depth_unit),
            surfaceDatum - i.tvd.Value(depth_unit),
            i.md.Value(depth_unit)))

    return pd.DataFrame(srvytable, columns = ['UWI','X','Y','TVDSS','MD'])


def get_log_curve(login_instance,well,log_curve_name,depth_unit = SeisWare.Unit.Meter):
    '''

    Takes well object, log curve name, and login instance to return a dataframe containing
    
    columns
    MD curvevalue

    '''
    log_curve_list = SeisWare.LogCurveList()
    
    try:
        login_instance.LogCurveManager().GetAllForWell(well.ID(), log_curve_list)
    except RuntimeError as err:
        handle_error("Failed to get the log curves of well %s from the project" % (well.UWI()), err)
    
    log_curves = [log_curve for log_curve in log_curve_list if log_curve.Name() == log_curve_name]
    
    if not log_curves:
        
        return pd.DataFrame(data=None,columns = ['MD',log_curve_name])
        #print("No log curve was found", file=sys.stderr)
        #sys.exit(1) 
    
    try:
        login_instance.LogCurveManager().PopulateValues(log_curves[0])
    except RuntimeError as err:
        handle_error("Failed to populate the values of log curve %s of well %s from the project" % (log_curve_name, well.UWI()), err)

    log_curve_values = SeisWare.DoublesList()
    log_curves[0].Values(log_curve_values)
    
    log_table = []

    for i in range(len(log_curve_values)):
        log_table.append(((log_curves[0].TopDepth()+log_curves[0].DepthInc()*i).Value(depth_unit),log_curve_values[i]))


    return pd.DataFrame(log_table,columns = ['MD',log_curve_name])

def get_horizon(login_instance,seismic_name,horizon_name,proj_units = SeisWare.Unit.Meter):

    '''
    seismic_name : string containing seismic line name
    horizon_name : string containing horizon name
    proj_units : Coordinate system XY units. Defaults to meters. Can be changed to SeisWare.Unit.Foot

    '''
    
    surveys = SeisWare.SeismicSurveyList()

    login_instance.SeismicSurveyManager().GetAll(surveys)

    survey = [i for i in surveys if i.Name() == seismic_name]

    horizons = SeisWare.HorizonList()

    login_instance.HorizonManager().GetAll(horizons)

    horizon = [i for i in horizons if i.Name() == horizon_name]

    picks = SeisWare.HorizonPicksList()

    # Horizon and Survey keys must be passed in as an IDPair list. The IDPair is created and then made into a single element list

    login_instance.HorizonPicksManager().GetByHorizonAndSeismicSurveyKeys([SeisWare.IDPair(horizon[0].ID(),survey[0].ID())],picks)

    # Populate the pick values
    login_instance.HorizonPicksManager().PopulateValues(picks[0])

    # Create an empty constructor
    values = SeisWare.HorizonPickValues()

    # Get the values from the picks and put them into the constructor
    picks[0].Values(values)

    # Get the values from the constructor and put them into a list

    horizon_points = []

    #inline_range = (survey[0].Survey().InlineRange().start,survey[0].Survey().InlineRange().start+survey[0].Survey().InlineRange().count*survey[0].Survey().InlineRange().delta-1)
    #crossline_range = (survey[0].Survey().CrosslineRange().start,survey[0].Survey().CrosslineRange().start+survey[0].Survey().CrosslineRange().count*survey[0].Survey().CrosslineRange().delta-1)

    fifc = (survey[0].Survey().CornerFiFc().x.Value(proj_units),survey[0].Survey().CornerFiFc().y.Value(proj_units))
    filc = (survey[0].Survey().CornerFiLc().x.Value(proj_units),survey[0].Survey().CornerFiLc().y.Value(proj_units))
    lifc = (survey[0].Survey().CornerLiFc().x.Value(proj_units),survey[0].Survey().CornerLiFc().y.Value(proj_units))

    delta_x_il = (lifc[0] - fifc[0])/(survey[0].Survey().CrosslineRange().count-1)
    delta_y_il = (lifc[1] - fifc[1])/(survey[0].Survey().CrosslineRange().count-1)

    delta_x_xl = (filc[0] - fifc[0])/(survey[0].Survey().InlineRange().count-1)
    delta_y_xl = (filc[1] - fifc[1])/(survey[0].Survey().InlineRange().count-1)


    for i in range(values.InlineCount()):
        for j in range(values.CrosslineCount()):
            #print(i,j)
            if values.IsPicked(SeisWare.GridIndex2(j,i)):
                pick_ij = values.Pick(SeisWare.GridIndex2(j,i)).structure.Value(SeisWare.Unit.Millisecond)
                pick_ij_amp = values.Pick(SeisWare.GridIndex2(j,i)).amplitude
            else:
                pick_ij = np.nan
            x = fifc[0] + i*delta_x_il + j*delta_x_xl
            y = fifc[1] + i*delta_y_il + j*delta_y_xl
            
            horizon_points.append((x,y,pick_ij,pick_ij_amp,i,j))
            
    return pd.DataFrame(horizon_points)