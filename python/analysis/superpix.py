import sys , h5py
import numpy as np
import vigra
import pylab as plot
import tifffile as tiff
from PyQt4 . QtCore import QTimer ; from PyQt4 . QtGui import QApplication

from slic import slic_superpixel
from slic import slic_superpixel_vigra
from watershed import watershed_superpixel_vigra, watershed_supervoxel_vigra

def view(image):
	plot.figure()
	plot.imshow(image)
	plot.show()
	plot.close()

def view_overlay(probs, segmentation):
	app = QApplication(sys.argv)	
	from volumina.api import Viewer

	v = Viewer()
	v.title = "Segmentation"
	v.showMaximized()
	v.addGrayscaleLayer(probs, name = "probability profile")
	v.addRandomColorsLayer(segmentation, name = "superpixel segmentation")

	app.exec_()


if __name__ == '__main__':
	probs 	= vigra.readHDF5("/home/constantin/Work/data_ssd/data_080515/pedunculus/labeling_2classes.h5","exported_data")
	raw 	= tiff.imread("/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30.tif")
	
	# use superpixel algorithm to segment the image
	# stack 2d segmented images 
	segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) ) 
	#for layer in range(probs.shape[2]):
		#segmentation[:,:,layer] = slic_superpixel(probs[:,:,layer,0], raw[0], 100, 1, True, False)
		#segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer,0])
		#segmentation[:,:,layer] = slic_superpixel_vigra(probs[:,:,layer,0])
	
	segmentation = watershed_supervoxel_vigra(probs[:,:,:,0])
	
	#view(segmentation)
	view_overlay(probs, segmentation)
	path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/"	
	#file = "slic"
	#file = "slic_vigra"
	file = "watershed_vigra"
	#np.save(path + file,segmentation)
