import sys , h5py
import numpy as np
import vigra

from slic import slic_superpixel
from slic import slic_superpixel_vigra
from watershed import watershed_superpixel_vigra, watershed_supervoxel_vigra
from volumina_viewer import volumina_double_layer


if __name__ == '__main__':
	path_probs = "/home/constantin/Work/data_ssd/data_080515/pedunculus/labeling_2classes.h5"
	key_probs  = "exported_data"
	
	path_raw   ="/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30.tif"

	probs 	= vigra.readHDF5(path_probs, key_probs)
	raw 	= vigra.impex.readImage(path_raw)
	
	# use superpixel algorithm to segment the image
	# stack 2d segmented images 
	segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) ) 
	for layer in range(probs.shape[2]):
		#segmentation[:,:,layer] = slic_superpixel(probs[:,:,layer,0], raw[0], 100, 1, True, False)
		segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer,0])
		#segmentation[:,:,layer] = slic_superpixel_vigra(probs[:,:,layer,0],0.1,10)
	
	#segmentation = watershed_supervoxel_vigra(probs[:,:,:,0])
	
	volumina_double_layer(probs, segmentation)
	
	path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/"	
	#name = "slic"
	#name = "slic_vigra"
	name = "watershed_vigra"
	fpath = path + name + ".h5"
	vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )
