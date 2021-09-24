
from PyQt5 import QtCore
from PyQt5.QtWidgets import *

import SWconnect
import SeisWare

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args,**kwargs)
        
        self.widget = QWidget(self)
        self.label = QLabel('Note: Length and Area measurements are estimates due to the Map Projection')
        self.setCentralWidget(self.widget)
        
        layout = QHBoxLayout(self.widget)
        
    
        layout.addItem(QSpacerItem(139, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        vlay = QVBoxLayout()
        vlay.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        layout.addLayout(vlay)
        layout.addItem(QSpacerItem(139, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        

        self.cb = QComboBox(self.widget)
        self.cb.addItems(projectlist()[1])
        self.cb.currentIndexChanged.connect(self.selectionchange)
        
        self.cb2 = QComboBox(self.widget)
        self.cb2.addItems(cultureDF(projectlist()[0]))

        self.b1 = QPushButton("Output  File",self.widget)
        self.b1.clicked.connect(self.on_click)
        
        
        vlay.addWidget(QLabel("Project Name"))
        vlay.addWidget(self.cb)
        vlay.addWidget(QLabel("Layer Name"))
        vlay.addWidget(self.cb2)
        vlay.addWidget(self.b1)
        vlay.addWidget(self.label)
        vlay.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        self.setWindowTitle("Culture Statistics")
        

	

    def selectionchange(self,i):

        proj = SWconnect.sw_connect(self.cb.currentText()) #get the proj based on selection
            
        self.cb2.clear()
        self.cb2.addItems(cultureDF(proj))
        
    def on_click(self):

        proj = SWconnect.sw_connect(self.cb.currentText())
        culturestats = {}
        culturelist = []    
        for i in getAllLayers(proj):
            
            cult = getLayer(proj,i.Name())
            culturestats[i.Name()] = areaList(cult) #Create the dictionary based on {Name:{"Count":value,"Area":[Values]}}
            culturelist.append(i.Name()) #Create the list

            
        contourDF = pd.DataFrame(culturestats[self.cb2.currentText()])
        contourDF.index = contourDF.index + 1
        current_time = date.today().strftime('%d%m%Y')
        contourDF.to_csv(f"C:\\temp\\{self.cb2.currentText()} stats {current_time}.csv")
        print(r"File output to C:\temp")
        #define what happens on button click
        


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()