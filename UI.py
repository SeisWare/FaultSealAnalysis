
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QFileDialog
import sys

import SWconnect
import SeisWare
from fault_throw import fault_throw_viz


def projectlist():

    projlist = []
    for i in SWconnect.sw_project_list():
        projlist.append(i.Name())
    projlist = sorted(projlist)
    proj = SWconnect.sw_connect(projlist[0])

    return [proj, projlist]

def gridlist(proj):

    grid_list = SeisWare.GridList()

    proj.GridManager().GetAll(grid_list)

    grid_list = [i.Name() for i in grid_list]

    return grid_list

def polylist(proj):

    poly_list = SeisWare.CultureList()

    proj.CultureManager().GetAll(poly_list)

    poly_list = [i.Name() for i in poly_list]

    return poly_list

class ui(QDialog):

    def __init__(self):
        super(ui, self).__init__()
        uic.loadUi("dialog.ui",self)
        self.show()
        self.projectName.addItems(['Select a project']+(projectlist()[1]))
        
        self.projectName.currentIndexChanged.connect(self.selectionchange)

        self.outputFile.clicked.connect(self.on_click)

        self.browseFile.clicked.connect(self.browseClick)

        self.setWindowTitle("Fault Throw Imager")
        self.setWindowIcon(QtGui.QIcon('SeisWareLogo.png'))
 
    def selectionchange(self):

        proj = SWconnect.sw_connect(self.projectName.currentText()) #get the proj based on selection
        self.gridName.clear()
        self.polygonName.clear()
        
        self.gridName.addItems(sorted(gridlist(proj)))
        
        self.polygonName.addItems(sorted(polylist(proj)))

    def on_click(self):

        proj = SWconnect.sw_connect(self.projectName.currentText())
        grid = self.gridName.currentText()
        cult = self.polygonName.currentText()
        fault_throw_viz(proj,cult,grid,self.folderLocation.text())

        print(f"File output to {self.folderLocation.text()}")
        #define what happens on button click


    def browseClick(self):
        filename = QFileDialog.getExistingDirectory(self, "Select Folder")
        
        if filename:
            self.folderLocation.setText(filename)

app = QApplication(sys.argv)
window = ui()
app.exec_()