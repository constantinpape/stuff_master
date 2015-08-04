import sys , h5py
from PyQt4 . QtCore import QTimer ; from PyQt4 . QtGui import QApplication

import vigra
import numpy as np
import argparse

from types import *

# plot single data layer
def volumina_single_layer(data):
	app = QApplication (sys.argv)
	from volumina.api import Viewer

	v = Viewer ()
	v.title = " Volumina Demo "
	v.showMaximized ()
	v.addGrayscaleLayer (data , name =" raw data ")

	app . exec_ ()


# plot 2 data layers
def volumina_double_layer(data, overlay):
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

	if data_type == np.float32:
		v.addGrayscaleLayer(overlay , name = " overlay ")
	else:
		v.addRandomColorsLayer(overlay, name = " overlay ")

	app . exec_ ()

# plot n data layers
def volumina_n_layer(data):

    app = QApplication (sys.argv)
    from volumina.api import Viewer

    v = Viewer ()
    v.title = " Volumina Demo "
    v.showMaximized ()

    ind = 0
    for d in data:
    	layer_name = "layer_" + str(ind)
    	# get data type of the elements d, to determine
    	# if we use a grayscale overlay (float32) or a randomcolors overlay (uint) for labels
    	mask = []
    	for i in range( len(d.shape) ):
    		mask.append(0)
    	mask = tuple(mask)
    	data_type = type(d[mask])
    	print data_type

    	if data_type is FloatType or data_type == np.float32 or data_type == np.float64:
    	    v.addGrayscaleLayer(d , name = layer_name)
    	else:
    	    v.addRandomColorsLayer(d.astype(np.uint32), name = layer_name)
    	ind += 1

    app . exec_ ()


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Display data with volumina, assuming HDF5 format,first input all paths then all keys')
	parser.add_argument('input', type = str, nargs = '+',
				help = 'path and keys to the data to be plotted')

	args = parser.parse_args()
	N = len(args.input)

	assert( N % 2 == 0 )

	path_data = args.input[0:N/2]
	key = args.input[N/2:N]

	data = []
	for i in range( N/2 ):
		data.append( vigra.readHDF5( path_data[i], key[i]) )

	if N == 2:
		volumina_single_layer( data[0] )
	elif N == 4:
		volumina_double_layer( data[0], data[1] )
	else:
		volumina_n_layer( data )
