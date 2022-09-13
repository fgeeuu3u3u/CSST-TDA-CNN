
from multiprocessing.managers import BaseManager
from posixpath import basename
import sys
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QFileDialog, QApplication, QPushButton, QLineEdit, QLabel, QComboBox, QCheckBox, QGridLayout
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
#class Example


from SpecGen.SpecGenerator import SpecGenerator
from SpecGen.Config import Config

from astropy.io import fits
from astropy.table import Table

import galsim


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib

import numpy as np

from pylab import *




class Figure_Canvas(FigureCanvas):
    """
    创建画板类
    """

    def __init__(self, width=3.2, height=2.7):
        self.fig = Figure(figsize=(width, height), dpi=70)
        super(Figure_Canvas, self).__init__(self.fig)

    def add_line(self, x_data, y_data, x2_data, y2_data, specImg, grating):
        self.fig.clear()
        self.ax = self.fig.add_subplot(211)
        self.line = Line2D(x_data, y_data)



        self.ax.grid(True)  
        self.ax.set_title('1-d Spec')  
        

        self.ax.set_xlabel('Wavelength($\AA$)') 
        self.ax.set_ylabel('F$\lambda$(erg/s/cm$^2$/$\AA$)')
        self.line.set_color('blue')


        # ------------------------------------------------------#
        self.ax.add_line(self.line)

        self.line2 = Line2D(x2_data, y2_data)
        self.ax.add_line(self.line2)
        self.line2.set_color('red')
        self.ax.legend([self.line, self.line2], ['Sim Spec', 'Orig Spec']) 

        if grating == 'GU':
            self.ax.set_xlim(2400, 4200)
        elif grating == 'GV':
            self.ax.set_xlim(3800, 6400)
        elif grating == 'GI':
            self.ax.set_xlim(6000, 10200)
        # self.ax.set_xlim(np.min(x_data), np.max(x_data))
        self.ax.set_ylim(np.min([np.min(y_data),np.min(y2_data)]), np.max([np.max(y_data), np.max(y2_data)]))



        self.ax1 = self.fig.add_subplot(212)
        import matplotlib.colors as colors
        self.ax1.imshow(specImg, norm = colors.LogNorm())
        self.ax1.set_title('Sim Slitless Spec Img')




class Example(QMainWindow):
    def __init__(self):
        super().__init__()
        self.InitUI()
 
    def InitUI(self):
        # self.textEdit = QTextEdit()
        # self.setCentralWidget(self.textEdit)
        # self.statusBar()
 
        # openFile = QAction(QIcon('open.png'), 'Open', self)
        # openFile.setShortcut('Ctrl+O')
        # openFile.setStatusTip('Open new File')
        # openFile.triggered.connect(self.showDialog)
 
        # menubar = self.menuBar()
        # fileMenu = menubar.addMenu('&File')
        # fileMenu.addAction(openFile)      
        # 

        self.lbl0 = QLabel(self)
        self.lbl0.move(20, 20)
        self.lbl0.setText('Input spec file')
        
        self.btn = QPushButton("spec file", self)
        self.btn.move(120, 20)
        self.btn.clicked.connect(self.showDialog1)
        # self.textEdit1 = QTextEdit()
        # self.setCentralWidget(self.textEdit1)
        # self.textEdit1.move(20,50)

        self.lbl1 = QLabel(self)
        self.lbl1.move(220, 20)

        # self.btn = QPushButton("config file", self)
        # self.btn.move(20, 50)
        # self.btn.clicked.connect(self.showDialog2)

        self.lbl2 = QLabel(self)
        self.lbl2.setText('Grating')
        self.lbl2.move(20, 50)

        # self.lbl = QLabel("Ubuntu", self)
 
        self.gratingCombo = QComboBox(self)
        self.gratingCombo.addItem("GI")
        self.gratingCombo.addItem("GV")
        self.gratingCombo.addItem("GU")
        self.gratingCombo.move(120,50)
        self.gratingCombo.activated[str].connect(self.onActivated)

        self.lbl2 = QLabel(self)
        self.lbl2.move(150, 50)

        self.lbl3 = QLabel(self)
        self.lbl3.move(20, 80)
        self.lbl3.setText('Beam Order')
        self.lbl3.adjustSize() 

        self.beamCombo = QComboBox(self)
        self.beamCombo.addItem("A")
        self.beamCombo.addItem("B")
        self.beamCombo.addItem("C")
        self.beamCombo.addItem("D")
        self.beamCombo.addItem("E")

        self.beamCombo.move(120,80)
        self.beamCombo.activated[str].connect(self.onActivated)


        self.lbl_line = QLabel(self)
        self.lbl_line.move(20, 110)
        self.lbl_line.setText('--------------CCD Information--------------')
        self.lbl_line.adjustSize() 


        self.lbl4 = QLabel(self)
        self.lbl4.move(20, 140)
        self.lbl4.setText('PSF(FWHM,")')
        self.lbl4.adjustSize() 

        self.psfText = QLineEdit(self)
        self.psfText.move(150,140)
        self.psfText.setText('0.39')

        # self.lbl41 = QLabel(self)
        # self.lbl41.move(230, 140)
        # self.lbl41.setText('arcsec')
        # self.lbl41.adjustSize() 



        self.lbl5 = QLabel(self)
        self.lbl5.move(20, 170)
        self.lbl5.setText('Readout(e-/pixel)')
        self.lbl5.adjustSize() 

        self.rdText = QLineEdit(self)
        self.rdText.move(150,170)
        self.rdText.setText('5.0')



        self.lbl6 = QLabel(self)
        self.lbl6.move(20, 200)
        self.lbl6.setText('Dark(e-/s/pixel)')
        self.lbl6.adjustSize() 

        self.dkText = QLineEdit(self)
        self.dkText.move(150,200)
        self.dkText.setText('0.02')


        self.lbl7 = QLabel(self)
        self.lbl7.move(20, 230)
        self.lbl7.setText('Pixel size("/pixel)')
        self.lbl7.adjustSize() 

        self.psText = QLineEdit(self)
        self.psText.move(150,230)
        self.psText.setText('0.074')

        self.lbl_line1 = QLabel(self)
        self.lbl_line1.move(20, 260)
        self.lbl_line1.setText('--------------Other Information--------------')
        self.lbl_line1.adjustSize() 


        self.lbl8 = QLabel(self)
        self.lbl8.move(20, 290)
        self.lbl8.setText('Sky background(e-/s/pixel)')
        self.lbl8.adjustSize() 

        self.sbText = QLineEdit(self)
        self.sbText.move(200,290)
        self.sbText.setText('0.3')


        self.lbl9 = QLabel(self)
        self.lbl9.move(20, 320)
        self.lbl9.setText('Exposure Time(s/frame)')
        self.lbl9.adjustSize() 

        self.etText = QLineEdit(self)
        self.etText.move(200,320)
        self.etText.setText('150')


        self.lbl10 = QLabel(self)
        self.lbl10.move(20, 350)
        self.lbl10.setText('Frames')
        self.lbl10.adjustSize() 

        self.frText = QLineEdit(self)
        self.frText.move(200,350)
        self.frText.setText('1')

        self.lbl_line1 = QLabel(self)
        self.lbl_line1.move(20, 380)
        self.lbl_line1.setText('-----------------Source----------------------')
        self.lbl_line1.adjustSize() 

        self.cb1 = QCheckBox('Star', self)
        self.cb1.move(20, 410)
        self.cb1.setChecked(True)

        self.cb2 = QCheckBox('Galaxy', self)
        self.cb2.move(20, 430)
        # self.cb2.setChecked(True)
        # self.cb2.toggle()
        self.cb1.stateChanged.connect(lambda:self.checkButSta(self.cb1.isChecked(),self.cb2))
        self.cb2.stateChanged.connect(lambda:self.checkButSta(self.cb2.isChecked(),self.cb1))

        self.lbl13 = QLabel(self)
        self.lbl13.move(30, 460)
        self.lbl13.setText('Sersic n')
        self.lbl13.adjustSize() 

        self.gsnText = QLineEdit(self)
        # self.gsnText.setPlaceholderText('Sersic n')
        self.gsnText.move(90,456)
        self.gsnText.setText('0.5')

        self.lbl14 = QLabel(self)
        self.lbl14.move(200, 460)
        self.lbl14.setText('re(")')
        self.lbl14.adjustSize() 

        self.greText = QLineEdit(self)
        # self.greText.setPlaceholderText('re')
        self.greText.move(230,456)
        self.greText.setText('0.1')

        self.lbl15 = QLabel(self)
        self.lbl15.move(30, 490)
        self.lbl15.setText('PA(deg)')
        self.lbl15.adjustSize() 

        self.gpaText = QLineEdit(self)
        # self.gsnText.setPlaceholderText('Sersic n')
        self.gpaText.move(90,486)
        self.gpaText.setText('45')

        self.lbl16 = QLabel(self)
        self.lbl16.move(200, 490)
        self.lbl16.setText('q')
        self.lbl16.adjustSize() 

        self.gqText = QLineEdit(self)
        # self.greText.setPlaceholderText('re')
        self.gqText.move(230,486)
        self.gqText.setText('0.1')
        

        self.lbl_line1 = QLabel(self)
        self.lbl_line1.move(20, 520)
        self.lbl_line1.setText('-----------------Output----------------------')
        self.lbl_line1.adjustSize() 



        self.lbl11 = QLabel(self)
        self.lbl11.move(20, 550)
        self.lbl11.setText('output directory')
        
        self.btn = QPushButton("Output", self)
        self.btn.move(120, 550)
        self.btn.clicked.connect(self.showDialog2)

        # self.lbl12 = QLabel(self)
        # self.lbl12.move(220, 20)

        self.lbl_line1 = QLabel(self)
        self.lbl_line1.move(20, 580)
        self.lbl_line1.setText('----------------------------------------------')
        self.lbl_line1.adjustSize() 

        self.simBt = QPushButton('Begin Sim', self)
        self.simBt.setCheckable(True)
        self.simBt.move(20, 610)

        self.simBt.clicked.connect(self.runDialog)
        
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(350, 50, 450, 600))

        self.fig = Figure_Canvas()
        self.LineFigureLayout = QGridLayout(self.groupBox)
        self.LineFigureLayout.addWidget(self.fig)

        

        # self.lbl41 = QLabel(self)
        # self.lbl41.move(230, 110)
        # self.lbl41.setText('arcsec')
        # self.lbl41.adjustSize() 



        
        self.setGeometry(300, 300, 800, 700)
        self.setWindowTitle('1-d spec generator')
        self.show()
    
    def onActivated(self, text):
        # self.lbl.setText(text)
        # self.lbl.adjustSize()
        a = text
 
    def showDialog1(self):
 
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        
        if fname[0]:
            f = open(fname[0], 'r')
 
            with f:
                # data = f.read()
                self.lbl1.setText(fname[0])
                self.lbl1.adjustSize()
                self.sedFn = fname[0]

    def showDialog2(self):
 
        outdir = QFileDialog.getExistingDirectory(self, 'Output', '/home')
        
        if outdir:
            self.btn.setText(outdir)
            self.btn.adjustSize()
            self.outDir = outdir
            
            # self.sedFn = fname[0]

    def checkButSta(self,state, btn):
        if state == True:
            btn.setChecked(False)
        else:
            btn.setChecked(True)


    def runDialog(self):
        
        sedFn = self.sedFn
        grating = self.gratingCombo.currentText()
        beam = self.beamCombo.currentText()

        dataDir = '../data/'
        config = Config(dataDir = dataDir)
        sedFn = dataDir + 'sed/sed_44575.txt'
        fwhm = float(self.psfText.text())
        p_size = float(self.psText.text())
        skybg = float(self.sbText.text())
        dark = float(self.dkText.text())
        readout = float(self.rdText.text())
        t = float(self.etText.text())
        expNum = int(self.frText.text())

        psf = galsim.Gaussian(fwhm=fwhm)
        specG = SpecGenerator(sedFn = sedFn, grating = grating, beam = beam, aper = 2.0, xcenter = 2000,ycenter = 5000, p_size = p_size, psf = psf, skybg = skybg, dark = dark, readout = readout, t = t, expNum = expNum,config = config)
        if self.cb1.isChecked() == True:
            specTab, specImg, img, fluxRa=specG.generateSpec1dforStar()
        else:
            g_sn = float(self.gsnText.text())
            g_re = float(self.greText.text())
            g_pa = float(self.gpaText.text())
            g_q = float(self.gqText.text())
            specTab, specImg, img, fluxRa=specG.generateSpec1dforGal(s_n = g_sn, re = g_re, pa = g_pa,q_ell = g_q,limitfluxratio=0.9)


        spec_orig = np.loadtxt(sedFn)

        self.fig.add_line(specTab['WAVELENGTH'],specTab['FLUX']/fluxRa, spec_orig[:,0], spec_orig[:,1],specImg, grating)

        self.fig.draw()

        fits.writeto(self.outDir + "/specImg.fits",specImg,overwrite=True)
        fits.writeto(self.outDir + "/DImg.fits",img,overwrite=True)
        specTab.write(self.outDir + "/specTab.fits",overwrite=True)



    # def showDialog2(self):
 
    #     fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')
        
    #     if fname[0]:
    #         f = open(fname[0], 'r')
 
    #         with f:
    #             # data = f.read()
    #             self.lbl2.setText(fname[0]) 
    #             self.lbl2.adjustSize() 


if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
