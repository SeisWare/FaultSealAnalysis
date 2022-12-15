# FaultSealAnalysis
Fault Seal Analysis is a utility to generate fault throw graphs across a fault using fault polygon layers and a depth grid, created for use with SeisWare projects. (www.seisware.com)

The required libraries are:
- pandas
- numpy
- matplotlib
- scipy
- shapely==1.7.1
- networkx
- PyQt5 (used only for UI)

You will also need the SeisWare SDK installed and available for your python environment. 

### To use this project:

1. Run UI.py to get the interface for selection the project, grid and polygons.
2. Select the project, grid and associated fault polygons to be used. Specify the Midpoint Sample Interval and Output location for the files. Click Output Data.

> The program will generate the following files in the specified output location, and you will get a message in your terminal that the File was output when it’s done. NOTE: this can take a few minutes, and there is no progress bar so just be patient – if there are any errors, the terminal will alert you. 

### Results generated:
|Location|Path|File|Description|
|---|---|---|---|
|\OutputLocation ||\Map.png | Image of the fault polygons with the associated numbers used for the output files.|		
|\OutputLocation ||\results.csv | All results from ZplotDF for all faults in one file.|	
|\OutputLocation |\CSV Files|\zc_DF#.csv | Location of the points where the fault throw is zero. 1 file per fault, numbered based on the fault numbering in Map.png |
|\OutputLocation |\CSV Files|\ZplotDF#.csv | All of the information used to create the plots (the XY of each side of the polygon, the intersection values with the grid and the difference used to create the throw graph). 1 file per fault, numbered based on the fault numbering in Map.png| 
|\OutputLocation |\Images|\Fault#.png | Graph of fault throw along mid line with fault polygon used. 1 file per fault, numbered based on the fault numbering in Map.png|


### Known issues
- If the operation times out, just run the operation again. This is often caused by the SeisWare SDK server taking a bit too long to respond.
- Using a different version of the shapely library will generate incorrect results.
