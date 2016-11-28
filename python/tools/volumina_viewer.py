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
def volumina_n_layer(data, labels = None):

    app = QApplication (sys.argv)
    import volumina
    from volumina.api import Viewer

    v = Viewer ()
    v.title = " Volumina Demo "
    v.showMaximized ()

    for ind, d in enumerate(data):
    	layer_name = "layer_" + str(ind)
        if labels is not None:
            layer_name = labels[ind]
    	# get data type of the elements d, to determine
    	# if we use a grayscale overlay (float32) or a randomcolors overlay (uint) for labels
    	data_type = d.dtype

    	if data_type is FloatType or data_type == np.float32 or data_type == np.float64:
    	    v.addGrayscaleLayer(d , name = layer_name)
    	else:
    	    v.addRandomColorsLayer(d.astype(np.uint32), name = layer_name)

    app.exec_()



def streaming_n_layer(data, keys, layer_type, labels = None):
    app = QApplication(sys.argv)

    from volumina.api import Viewer
    from volumina.pixelpipeline.datasources import LazyflowSource

    from lazyflow.graph import Graph
    from lazyflow.operators.ioOperators.opStreamingHdf5Reader import OpStreamingHdf5Reader
    from lazyflow.operators import OpCompressedCache

    v = Viewer()

    v.title = "Streaming Viewer"

    graph = Graph()

    def mkH5source(fname, gname):
        h5file = h5py.File(fname)

        source = OpStreamingHdf5Reader(graph=graph)
        source.Hdf5File.setValue(h5file)
        source.InternalPath.setValue(gname)

        op = OpCompressedCache( parent=None, graph=graph )
        op.BlockShape.setValue( [100, 100, 100] )
        op.Input.connect( source.OutputImage )

        return op.Output

    for i, f in enumerate(data):

        if labels is not None:
            layer_name = labels[i]
        else:
            layer_name = "layer_%i" % (i)

        source, dtype = mkH5source(f, keys[i])

        if layer_type[i] == "seg":
            v.addRandomColorsLayer(source, name = layer_name)
        else:
            v.addGrayscaleLayer(source, name = layer_name)

    v.showMaximized()
    app.exec_()




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
