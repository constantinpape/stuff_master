import sys , h5py ; from numpy import float32 , uint8
from PyQt4 . QtCore import QTimer ; from PyQt4 . QtGui import QApplication

import vigra
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Display data with volumina')
	parser.add_argument('path_data', type = str, nargs = 1, 
				help = 'path to the data')
	parser.add_argument('key', type = str, nargs = 1, 
				help = 'key of the data')

	args = parser.parse_args()
	
	d = vigra.readHDF5( args.path_data[0], args.key[0] )

	app = QApplication (sys.argv)
	from volumina.api import Viewer
	
	v = Viewer ()
	v.title = " Volumina Demo "
	v.showMaximized ()
	v.addGrayscaleLayer (d , name =" raw data ")
	
	app . exec_ ()
