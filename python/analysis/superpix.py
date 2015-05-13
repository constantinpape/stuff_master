import sys , h5py
import numpy as np
import vigra

from slic import slic_superpixel
from slic import slic_superpixel_vigra
from watershed import watershed_superpixel_vigra, watershed_supervoxel_vigra
from volumina_viewer import volumina_double_layer
from volumina_viewer import volumina_single_layer


if __name__ == '__main__':
    #path_probs = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_combined_Probabilities_sliced.h5"
    #key_probs  = "data"

    path_probs = "/home/constantin/Work/data_ssd/data_080515/pedunculus/pixel_probabilities/combined_autocontext_probs.h5"
    key_probs  = "exported_data"

    probs   = vigra.readHDF5(path_probs, key_probs)
    probs = np.squeeze(probs)
    # exclude slice 6 for this data, which is dark...
    probs = probs[:,:,:,0]
    probs   = np.delete(probs,6,axis = 2)
    # verify that the correct slice was removed
    #volumina_single_layer(probs)

    # use superpixel algorithm to segment the image
    # stack 2d segmented images
    segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) )
    # need offset to keep superpixel of the individual layers seperate!
    offset = 0
    for layer in range(probs.shape[2]):
    	if layer != 0:
    		offset = np.max(segmentation[:,:,layer-1])
    	segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer], offset)
    	#segmentation[:,:,layer] = slic_superpixel_vigra(probs[:,:,layer], 0.05, 15, offset)

    #segmentation = watershed_supervoxel_vigra(probs)

    volumina_double_layer(probs[:,:,4],segmentation[:,:,4])

    path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/"

    #name = "slic_vigra"
    name = "watershed_vigra"
    #name = "watershed_voxel"

    fpath = path + name + ".h5"
    vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )
