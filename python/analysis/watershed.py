import vigra
import numpy as np
import matplotlib.pyplot as plot
from volumina_viewer import volumina_single_layer
from volumina_viewer import volumina_double_layer
from volumina_viewer import volumina_n_layer

from skneuro.oversegmentation import wsDtSegmentation, wsDtSegmentation2d


def watershed_distancetransform_3d(probs):
    seg = wsDtSegmentation(probs, 0.2, 10, 20, 1.6, 3., cleanCloseSeeds = True)

    return seg


def watershed_distancetransform_2d(probs, offset = 0):
    seg = wsDtSegmentation2d(probs, (0.2, 0.5), 10, 35, 2., 3., cleanCloseSeeds = True).astype(np.uint32)

    if offset != 0:
    	seg += offset * np.ones( seg.shape, dtype = np.uint32 )

    #return seg, res_wsdt[1], res_wsdt[2].astype(np.float32)
    return seg

def watersheds_thresholded(probs, offset = 0):

    hmap = probs.copy()

    disc = vigra.filters.discRankOrderFilter(255. * hmap, 4, 0.1 )

    #volumina_double_layer(hmap, disc)
    #quit()

    thresh = np.ones( disc.shape )
    thresh[disc < .8 * 255] = 0.

    SEEDS = vigra.analysis.labelImageWithBackground(thresh.astype(np.float32))

    #volumina_double_layer(hmap, SEEDS)
    #quit()

    seg_ws, maxRegionLabel = vigra.analysis.watersheds(vigra.filters.gaussianSmoothing(disc,3),
    				neighborhood = 8, seeds = SEEDS.astype(np.uint32) )
    seg_ws = vigra.analysis.labelImage(seg_ws)

    if offset != 0:
    	seg_ws += offset * np.ones( seg_ws.shape )

    volumina_double_layer(hmap, seg_ws)
    quit()

    return seg_ws


# perform the watershed algorithm implemented in vigra
# Adapted from superpixel/watershed/ws.py
# @ param: probs: array w/ image to be segmented
# @ param: offset, labeloffset for the segmentation, in case a stack of images is segmented, default 0
def watershed_superpixel_vigra(probs, offset = 0):
    # in nikos script: substract the mean and take the abs value
    # threshold:

    #hmap = np.abs( probs - np.mean(probs) )
    hmap = probs.copy()

    # threshold the data
    #hmap[hmap < 0.45] = 0.

    # smooth the hmap
    hmap_smooth = vigra.gaussianSmoothing(hmap, 1.5)

    # Hessian of Gaussian
    hessian = vigra.filters.hessianOfGaussian(hmap, sigma = 1.5)
    hessian_ev = vigra.filters.tensorEigenvalues( hessian )

    # combine the filters
    h_ev0 = hessian_ev[:,:,0]
    h_ev1 = hessian_ev[:,:,1]
    combination = 3*np.absolute( h_ev0 ) + 3*np.absolute( h_ev1 ) + hmap_smooth
    #combination = hmap

    # construct a line filter (cf. https://www.spl.harvard.edu/archive/spl-pre2007/pages/papers/yoshi/node3.html)
    #a_0 = 0.5
    #a_1 = 2.0
    #line_filter =  np.multiply( (h_ev0 <= 0), np.exp( - np.divide( np.square(h_ev0), 2*np.square(a_0*h_ev1) ) ) )
    #line_filter += np.multiply( (h_ev0  > 0), np.exp( - np.divide( np.square(h_ev0), 2*np.square(a_1*h_ev1) ) ) )

    #volumina_n_layer( [ np.absolute(hessian_ev[:,:,0]),
    #    np.absolute(hessian_ev[:,:,1]),
    #    combination] )
    #quit()

    # find the local minima
    seeds = vigra.analysis.extendedLocalMinima(combination, neighborhood = 8)
    # find the connected components in the minima and use them as seeds
    SEEDS = vigra.analysis.labelImageWithBackground(seeds)

    # apply the watershed algorithm
    seg_ws, maxRegionLabel = vigra.analysis.watersheds(combination,
    				neighborhood = 8, seeds = SEEDS.astype(np.uint32) )
    # find connected components
    seg_ws = vigra.analysis.labelImage(seg_ws)

    # if we have an offset, add it to the array
    if offset != 0:
    	seg_ws += offset * np.ones( seg_ws.shape )

    #volumina_double_layer(probs, seg_ws)
    #quit()

    return seg_ws

# perform the watershed algorithm implemented in vigra
# Adapted from superpixel/watershed/ws.py
# @ param: probs: array w/ volume to be segmented
def watershed_supervoxel_vigra(probs):

    # compute seeds
    # try different filter for computing the best seeds (= minima)
    # best options so far:
    # for isotropic data (2x2x2 nm): Gaussian Smoothing with sigma = 4.5
    # for anisotropic data: Gaussian Smoothing with sigma = 2

    #sm_probs = np.array(np.abs( probs - 0.5*( np.max(probs) - np.min(probs) ) ), dtype = np.float32 )
    # Gaussian smoothing
    sm_probs = vigra.gaussianSmoothing(probs, (2.5,2.5,0.5) )

    hessian = vigra.filters.hessianOfGaussian(probs, sigma = (2.5,2.5,0.5) )
    hessian_ev = vigra.filters.tensorEigenvalues( hessian )

    print hessian_ev.shape
    volumina_n_layer( [ probs,
        np.absolute(hessian_ev[:,:,:,0]),
        np.absolute(hessian_ev[:,:,:,1]),
        np.absolute(hessian_ev[:,:,:,2]) ] )
    quit()

    # Difference of Gaussians
    #diff = vigra.gaussianSmoothing(probs,2) - sm_probs
    #volumina_single_layer(diff)

    seeds = vigra.analysis.extendedLocalMinima3D(sm_probs, neighborhood = 26)
    SEEDS = vigra.analysis.labelVolumeWithBackground(seeds)
    SEEDS = SEEDS.astype(np.uint32)

    #plot.figure()
    #plot.gray()
    #plot.imshow(SEEDS[:,:,25])
    #plot.show()

    seg_ws, maxRegionLabel = vigra.analysis.watersheds(sm_probs,
    					neighborhood = 6, seeds = SEEDS)

    seg_ws = vigra.analysis.labelVolumeWithBackground(seg_ws)

    #volumina_double_layer(probs, seg_ws)

    return seg_ws

if __name__ == '__main__':
	path_probs = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_Probabilities.h5"
	key_probs  = "exported_data"

	probs 	= vigra.readHDF5(path_probs, key_probs)

	segmentation = watershed_supervoxel_vigra(probs[:,:,:,0])
