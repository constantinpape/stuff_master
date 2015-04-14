import sys , h5py ; from numpy import float32 , uint8
from PyQt4 . QtCore import QTimer ; from PyQt4 . QtGui import QApplication

import vigra

if __name__ == '__main__':

	# orig	
	path = "data/data_070515/stack.hdf5"
	key = "data"
	
	#gtpath = "/mnt/data/Neuro/knott1000/test/gt_reg.h5"
	#gtkey = "gt_reg"
	
	d = vigra.readHDF5(path, key)
	#gt = vigra.readHDF5(gtpath, gtkey)

	app = QApplication (sys.argv)
	from volumina.api import Viewer
	
	v = Viewer ()
	v.title = " Volumina Demo "
	v.showMaximized ()
	v.addGrayscaleLayer (d , name =" raw data ")
	#v.addRandomColorsLayer(gt, "rcl")
	
	app . exec_ ()


