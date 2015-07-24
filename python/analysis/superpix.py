import sys , h5py
import numpy as np
import vigra

from slic import slic_superpixel
from slic import slic_superpixel_vigra
from watershed import watershed_superpixel_vigra, watershed_supervoxel_vigra, watersheds_thresholded
from volumina_viewer import volumina_single_layer, volumina_double_layer, volumina_n_layer


def make_superpix_pedunculus():
    path_probs = "/home/constantin/Work/data_ssd/data_080515/pedunculus/pixel_probabilities/probs_final.h5"
    key_probs  = "exported_data"

    probs = vigra.readHDF5(path_probs, key_probs)
    probs = np.squeeze(probs)
    if len(probs.shape) == 4:
        probs = probs[:,:,:,0]
    print probs.shape
    # exclude slice 6 for this data, which is dark...
    probs   = np.delete(probs,6,axis = 2)
    # verify that the correct slice was removed
    volumina_single_layer(probs)

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

    #volumina_double_layer(probs,segmentation)

    path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/"

    #name = "slic_vigra"
    name = "watershed"
    #name = "watershed_voxel"

    fpath = path + name + ".h5"
    vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )


def make_superpix_isbi2012():
    path_probs = "/home/constantin/Work/data_ssd/data_090615/isbi2012/pixel_probabilities/test-probs_final.h5"
    key_probs  = "exported_data"

    probs = vigra.readHDF5(path_probs, key_probs)
    probs = np.squeeze(probs)

    # use superpixel algorithm to segment the image
    # stack 2d segmented images
    segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) )
    # need offset to keep superpixel of the individual layers seperate!
    offset = 0
    for layer in range(probs.shape[2]):
    	if layer != 0:
    		offset = np.max(segmentation[:,:,layer-1])
    	segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer], offset)

    #volumina_double_layer(probs,segmentation)

    path = "/home/constantin/Work/data_ssd/data_090615/isbi2012/superpixel/"

    name = "watershed"

    fpath = path + name + ".h5"
    vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )


def make_superpix_isbi2013():
    path_probs = "/home/constantin/Work/data_ssd/data_150615/isbi2013/pixel_probs/test-probs-nn.h5"
    key_probs  = "exported_data"

    path_raw = "/home/constantin/Work/data_ssd/data_150615/isbi2013/test-input.h5"
    key_raw  = "data"

    probs = vigra.readHDF5(path_probs, key_probs)
    probs = np.squeeze(probs)

    probs = np.array(probs)
    print probs.max(), probs.min()
    probs  = 1. - probs

    raw = vigra.readHDF5(path_raw, key_raw)

    #volumina_n_layer( (raw, probs) )
    #quit()

    # use superpixel algorithm to segment the image
    # stack 2d segmented images
    segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) )
    # need offset to keep superpixel of the individual layers seperate!
    offset = 0
    for layer in range(probs.shape[2]):
    	if layer != 0:
    		offset = np.max(segmentation[:,:,layer-1])
    	segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer], offset)

    volumina_n_layer( (raw, probs, segmentation) )
    #quit()

    path = "/home/constantin/Work/data_ssd/data_150615/isbi2013/superpixel/"

    name = "watershed-test_nn"

    fpath = path + name + ".h5"
    vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )


if __name__ == '__main__':
    make_superpix_isbi2013()
