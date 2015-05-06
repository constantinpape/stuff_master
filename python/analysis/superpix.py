import sys , h5py
import numpy as np
import vigra

from slic import slic_superpixel
from slic import slic_superpixel_vigra
from watershed import watershed_superpixel_vigra, watershed_supervoxel_vigra
from volumina_viewer import volumina_double_layer
from volumina_viewer import volumina_single_layer


if __name__ == '__main__':
	path_probs = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_combined_Probabilities_sliced.h5"
	key_probs  = "data"

	probs 	= vigra.readHDF5(path_probs, key_probs)

	# use superpixel algorithm to segment the image
	# stack 2d segmented images
	#segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) )
	#for layer in range(probs.shape[2]):
		#offs = 0
		#if layer != 0:
		#	offs = np.max(segmentation[:,:,layer-1])
		#segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer,0],offset = offs
		#segmentation[:,:,layer] = slic_superpixel_vigra(probs[:,:,layer,0],0.1,10,offset = offs)

	segmentation = watershed_supervoxel_vigra(probs[:,:,:,0])

	#volumina_single_layer(probs)

    path = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/superpixel/"

	#name = "slic"
	#name = "slic_vigra"
	#name = "watershed_vigra"
	name = "watershed_voxel"

    fpath = path + name + ".h5"
	vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )
