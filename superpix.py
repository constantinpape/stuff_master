import sys , h5py
import numpy as np
import vigra
import pylab as plot
import tifffile as tiff

from slic import slic_superpixel
from slic import slic_superpixel_vigra
from watershed import watershed_superpixel_vigra

def view(image):
	plot.figure()
	plot.imshow(image)
	plot.show()
	plot.close()

def view_overlay(probs, segmentation):
	from PyQt4.QtGui import QApplication

	app = QApplication([])

	from segmentation.labeling.faceLabeling import FaceLabeler

	v = FaceLabeler( np.random.random( (30,30,30)), (2,2,2,2) )

	v.addGrayscaleLayer(probs*255, "hmap" )
	v.addRandomColorsLayer(segmentation*255, "seg")

	v.show()
	app.exec_()


if __name__ == '__main__':
	probs 	= vigra.readHDF5("/home/constantin/Work/data_ssd/data_080515/pedunculus/labeling_2classes.h5","exported_data")
	probs 	= probs.transpose( (2,0,1,3) )
	probs	= probs[:,:,:,0].reshape( (probs.shape[0],probs.shape[1],probs.shape[2]) )
	raw 	= tiff.imread("/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30.tif")
	assert(raw.shape == probs.shape)
	
	# use superpixel algorithm to segment the image
	#segmentation = slic_superpixel(probs[0], raw[0], 100, 5, True, False)
	segmentation = watershed_superpixel_vigra(probs[0])
	segmentation = slic_superpixel_vigra(probs[0])
	
	view(segmentation)
	#view_overlay(probs[0], segmentation)
	
	path = "home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/"
	file = "watershed_vigra"
	#np.save(path + file,segmentation)

