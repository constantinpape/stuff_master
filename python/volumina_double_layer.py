import sys , h5py ; from numpy import float32 , uint8
from PyQt4 . QtCore import QTimer ; from PyQt4 . QtGui import QApplication

import vigra
import numpy as np
import argparse

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Display data with volumina')
	parser.add_argument('path_data', type = str, nargs = 2, 
				help = 'path to the data, first raw data, then overlay')
	parser.add_argument('key', type = str, nargs = 2, 
				help = 'key of the data, overlay')

	args = parser.parse_args()
	
	data = vigra.readHDF5( args.path_data[0], args.key[0] )
	overlay = vigra.readHDF5( args.path_data[1], args.key[1] )
	
	# get data type of the elements of overlay, to determine
	# if we use a grayscale overlay (float32) or a randomcolors overlay (uint) for labels
	mask = []
	for i in range( len(overlay.shape) ):
		mask.append(0)
	mask = tuple(mask)
	data_type = type(overlay[mask])

	app = QApplication (sys.argv)
	from volumina.api import Viewer
	
	v = Viewer ()
	v.title = " Volumina Demo "
	v.showMaximized ()
	v.addGrayscaleLayer(data , name = " raw data ")
	
	if data_type == float32:
		v.addGrayscaleLayer(overlay , name = " overlay ")
	else:
		v.addRandomColorsLayer(overlay, name = " overlay ")
	
	app . exec_ ()