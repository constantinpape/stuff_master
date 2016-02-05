import sys , h5py
import numpy as np
import vigra

from slic import slic_superpixel
from slic import slic_superpixel_vigra

from watershed import watershed_superpixel_vigra, watershed_supervoxel_vigra
from watershed import watershed_distancetransform_2d, watershed_distancetransform_3d

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
    #probs   = np.delete(probs,6,axis = 2)
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
    	#segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer], offset)
    	#segmentation[:,:,layer] = slic_superpixel_vigra(probs[:,:,layer], 0.05, 15, offset)
        res_wsdt = watershed_distancetransform_2d(probs[:,:,layer], offset)
        segmentation[:,:,layer] = res_wsdt[0]

    #volumina_double_layer(probs,segmentation)
    #quit()

    path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/"

    #name = "slic_vigra"
    name = "watershed_dt"
    #name = "watershed_voxel"

    fpath = path + name + ".h5"
    vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )


def make_superpix_isbi2012():
    path_probs = "/home/constantin/Work/data_ssd/data_090615/isbi2012/pixel_probabilities/probs_train_final.h5"
    #path_unet = "/home/constantin/Work/data_ssd/data_090615/isbi2012/u-net_probs/u-net_probs_test.h5"
    key_probs  = "exported_data"

    probs = vigra.readHDF5(path_probs, key_probs)
    probs = np.squeeze(probs)
    #probs  = 1. - probs

    # use superpixel algorithm to segment the image
    # stack 2d segmented images
    segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) )

    # need offset to keep superpixel of the individual layers seperate!
    offset = 0

    for layer in range(probs.shape[2]):
    	if layer != 0:
    		offset = np.max(segmentation[:,:,layer-1])

        res_wsdt = watershed_distancetransform_2d(probs[:,:,layer], offset)
        segmentation[:,:,layer] = res_wsdt[0]

    #volumina_double_layer(probs,segmentation)
    #quit()

    path = "/home/constantin/Work/data_ssd/data_090615/isbi2012/superpixel/"

    name = "watershed_dt_train"

    fpath = path + name + ".h5"
    vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )


def make_superpix_isbi2013(superpix = True):
    path_probs = "/home/constantin/Work/data_ssd/data_150615/isbi2013/pixel_probs/test-probs-nn.h5"
    key_probs  = "exported_data"

    path_raw = "/home/constantin/Work/data_ssd/data_150615/isbi2013/test-input.h5"
    key_raw  = "data"

    probs = vigra.readHDF5(path_probs, key_probs)
    probs = np.squeeze(probs)

    probs = np.array(probs)
    probs  = 1. - probs

    raw = vigra.readHDF5(path_raw, key_raw)

    #volumina_n_layer( (raw, probs) )
    #quit()

    if superpix:
        # use superpixel algorithm to segment the image
        # stack 2d segmented images
        segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) ,dtype = np.uint32)
        seeds = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) ,dtype = np.uint32)
        weights = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) ,dtype = np.uint32)
        # need offset to keep superpixel of the individual layers seperate!
        offset = 0
        for layer in range(probs.shape[2]):
        	if layer != 0:
        		offset = np.max(segmentation[:,:,layer-1])
        	#segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer], offset)
                res_wsdt = watershed_distancetransform_2d(probs[:,:,layer], offset)
        	segmentation[:,:,layer] = res_wsdt[0]
                seeds[:,:,layer] = res_wsdt[1]
        	weights[:,:,layer] = res_wsdt[2]

        #segmentation[:,:,2] = watershed_distancetransform_2d( probs[:,:,2], 0 )
        volumina_n_layer( (probs, segmentation, seeds, weights) )

    else:
        # use supervoxel algorithm to segment the image
        segmentation = watershed_distancetransform_3d(probs)
        volumina_n_layer( (raw, probs, segmentation) )

    print "Number of superpixels:", segmentation.max()
    #quit()

    path = "/home/constantin/Work/data_ssd/data_150615/isbi2013/superpixel/"

    name = "watershed_nn_dt_supervox_test"

    fpath = path + name + ".h5"
    vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )


def make_superpix_sopnetcomparison():
    path_probs = "/home/constantin/Work/data_ssd/data_110915/sopnet_comparison/pixel_probabilities/probs-final_autocontext.h5"
    key_probs = "data"

    probs = vigra.readHDF5(path_probs, key_probs)

    segmentation = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) ,dtype = np.uint32)
    seeds        = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) ,dtype = np.uint32)
    weights      = np.zeros( (probs.shape[0], probs.shape[1], probs.shape[2]) ,dtype = np.uint32)

    # need offset to keep superpixel of the individual layers seperate!
    offset = 0
    for layer in range(probs.shape[2]):
    	if layer != 0:
    		offset = np.max(segmentation[:,:,layer-1])
    	#segmentation[:,:,layer] = watershed_superpixel_vigra(probs[:,:,layer], offset)
        res_wsdt = watershed_distancetransform_2d(probs[:,:,layer], offset)
    	segmentation[:,:,layer] = res_wsdt[0]
        seeds[:,:,layer] = res_wsdt[1]
    	weights[:,:,layer] = res_wsdt[2]

    #segmentation[:,:,2] = watershed_distancetransform_2d( probs[:,:,2], 0 )

    print "Number of superpixels:", segmentation.max()

    path = "/home/constantin/Work/data_ssd/data_110915/sopnet_comparison/superpixel/"
    name = "watershed_dt_mitooff"
    fpath = path + name + ".h5"

    vigra.impex.writeHDF5(segmentation, fpath, "superpixel" )


# make superpixel for the new bock data
def make_superpix_bock():

    from wsDtSegmentation import wsDtSegmentation

    # sample c
    path_raw   = "/home/constantin/Work/data_hdd/data_131115/Sample_B/raw_data/raw_data_norm_cut.h5"
    key_raw    = "data"
    path_probs = "/home/constantin/Work/data_hdd/data_131115/Sample_B/google_probabilities/probs_xy_cut.h5"
    key_probs  = "exported_data"

    raw   = vigra.readHDF5(path_raw, key_raw)

    probs = vigra.readHDF5(path_probs, key_probs)
    probs = np.array(probs)

    # need to invert the probability maps (need membrane channel!)
    probs = 1. - probs

    #vigra.writeHDF5(probs[:,:,0],"tmp1.h5", "tmp")
    #vigra.writeHDF5(raw[:,:,0],"tmp2.h5", "tmp")
    #quit()

    #probs = vigra.readHDF5("tmp1.h5", "tmp")
    #raw = vigra.readHDF5("tmp2.h5", "tmp")

    #probs = np.expand_dims(probs, axis = 2)
    #raw = np.expand_dims(raw, axis = 2)

    print probs.shape
    print raw.shape

    # visualize the data
    #volumina_double_layer( raw, probs )

    segmentation = np.zeros_like(probs)
    seeds        = np.zeros_like(probs)
    weights      = np.zeros_like(probs)

    # need offset to keep superpixel of the individual layers seperate!
    offset = 0
    for layer in range(probs.shape[2]):

        if layer != 0:
    		offset = np.max(segmentation[:,:,layer-1])

        # syntax: wsDtSegmentation(probs, pmin, minMemSize, minSegSize, sigSeeds, sigWeights)
        res_wsdt = wsDtSegmentation(probs[:,:,layer], 0.1, 20, 100, 0.8, 1.)
        segmentation[:,:,layer] = res_wsdt[0] + offset
        seeds[:,:,layer]        = res_wsdt[1]
        weights[:,:,layer]      = res_wsdt[2]

        # visualize first layer
        #print "Nr of seeds:", np.sum(seeds != 0)
        #volumina_n_layer( [
        #    raw[:,:,layer],
        #    probs[:,:,layer],
        #    weights[:,:,layer],
        #    seeds[:,:,layer].astype(np.uint32),
        #    segmentation[:,:,layer].astype(np.uint32),
        #    segmentation[:,:,layer].astype(np.uint32)] )
        #quit()

    print "Number of superpixels:", segmentation.max()

    path_save = "/home/constantin/Work/data_hdd/data_131115/Sample_B/superpixel/wsdt_seg.h5"

    vigra.impex.writeHDF5(segmentation, path_save, "superpixel" )

    # visualize whole stack
    volumina_n_layer( [raw, probs, segmentation.astype(np.uint32)] )


def make_superpix_from_intepolation(prob_path, prob_key, save_path, anisotropy):
    from wsDtSegmentation import wsDtSegmentation

    pmem = vigra.readHDF5(prob_path, prob_key)

    print pmem.shape
    print anisotropy

    # for some datasets, we have to invert the probabilities
    #probs = 1. - probs

    # interpolate the probability in z - direction
    print "doing spline interpolation"
    pmem_interpol = vigra.sampling.resize(pmem, shape=(pmem.shape[0], pmem.shape[1], anisotropy* pmem.shape[2]))
    pmem_interpol = np.array(pmem_interpol)
    print "Finished interpolation"

    superpix = wsDtSegmentation(pmem_interpol, 0.45, 20, 100, 1.6, 2.)[0]

    superpix = superpix[:,:,::anisotropy]

    #volumina_n_layer( [pmem, superpix.astype(np.uint32)] )

    assert superpix.shape == pmem.shape

    vigra.writeHDF5(superpix, save_path, "superpixel")



if __name__ == '__main__':

    make_superpix_from_intepolation("/home/constantin/Work/data_ssd/data_080515/pedunculus/pixel_probabilities/probs_final.h5",
            "exported_data",
            "/home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/watershed_dt_interpolated.h5",
            9)

    #make_superpix_bock()
    #make_superpix_pedunculus()
    #make_superpix_isbi2012()
    #make_superpix_isbi2013(False)
