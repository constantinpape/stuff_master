import numpy as np
import vigra
from wsDtSegmentation import wsDtSegmentation, remove_wrongly_sized_connected_components
import volumina_viewer

# make 2d superpixel based on watershed on DT
def superpix_2d(probs, thresh, sig_seeds, sig_weights, clean_seeds):
    segmentation = np.zeros_like(probs)
    offset = 0
    for z in xrange(probs.shape[2]):
        wsdt = wsDtSegmentation(probs[:,:,z], thresh, 50, 75, sig_seeds, sig_weights, clean_seeds)
        segmentation[:,:,z] = wsdt
        segmentation[:,:,z] += offset
        offset = np.max(segmentation)
    return segmentation

# make 3d superpixel based on watershed on DT
def superpix_3d(probs, thresh, sig_seeds, sig_weights):
    wsdt_seg = wsDtSegmentation(probs, thresh, 50, 75, sig_seeds, sig_weights, True)
    return wsdt_seg

# make 3d superpixel on interpolated probabilities
def superpix_interpol(probs, thresh, sig_seeds, sig_weights, aniso):
    probs_int = vigra.sampling.resize(probs,
            shape = (probs.shape[0], probs.shape[1], aniso * probs.shape[2]) )
    wsdt_seg = wsDtSegmentation(probs_int, thresh, 50, 75, sig_seeds, sig_weights, True)
    wsdt_seg = wsdt_seg[:,:,::aniso]
    assert wsdt_seg.shape == probs.shape
    return wsdt_seg


# get the signed distance transform of pmap
def getSignedDt(pmap, pmin, minMembraneSize, sigmaNoise):

    # get the thresholded pmap
    binary_membranes = np.zeros_like(pmap, dtype=np.uint8)
    binary_membranes[pmap >= pmin] = 1

    # delete small CCs
    labeled = vigra.analysis.labelImageWithBackground(binary_membranes)
    remove_wrongly_sized_connected_components(labeled, minMembraneSize, in_place=True)

    # use cleaned binary image as mask
    big_membranes_only = np.zeros_like(binary_membranes, dtype = np.float32)
    big_membranes_only[labeled > 0] = 1.

    # perform signed dt on mask
    distance_to_membrane    = vigra.filters.distanceTransform2D(big_membranes_only)
    distance_to_nonmembrane = vigra.filters.distanceTransform2D(big_membranes_only, background=False)
    distance_to_nonmembrane[distance_to_nonmembrane>0] -= 1
    dtSigned = distance_to_membrane - distance_to_nonmembrane
    dtSigned[:] *= -1
    dtSigned[:] -= dtSigned.min()

    if sigmaNoise != 0.:
        dtSigned += np.random.normal(0.0, sigmaNoise, dtSigned.shape)

    return (dtSigned, distance_to_membrane)


def get_2d_disttrafo(probs, thresh):
    dist_trafo = np.zeros_like(probs)
    for z in xrange(probs.shape[2]):
        dist_trafo[:,:,z] = getSignedDt(probs[:,:,z], thresh, 50, 0.)[0]
    return dist_trafo




if __name__ == '__main__':
    isbi13_probs_train = "/home/consti/Work/data_master/isbi2013/probability_maps/train-probs-nn.h5"
    isbi13_probs_test  = "/home/consti/Work/data_master/isbi2013/probability_maps/test-probs-nn.h5"

    ped_probs = "/home/consti/Work/data_master/pedunculus/probs_final.h5"

    #probs = vigra.readHDF5(ped_probs, "exported_data").view(np.ndarray)
    probs = vigra.readHDF5(isbi13_probs_train, "exported_data")

    #volumina_viewer.volumina_single_layer(probs)

    #thresh      = 0.15
    #sig_seeds   = 1.6
    #sig_weights = 2.0
    #clean_seeds = True

    thresh      = 0.3
    sig_seeds   = 1.6
    sig_weights = 2.0
    clean_seeds = True

    superpix = superpix_3d(probs, thresh, sig_seeds, sig_weights)
    vigra.writeHDF5(superpix, "wsdt_3d.h5", "data")

    #volumina_viewer.volumina_n_layer([probs,
    #    superpix.astype(np.uint32),
    #    superpix_noisy.astype(np.uint32),
    #    superpix_noisy_.astype(np.uint32)
    #    ])

