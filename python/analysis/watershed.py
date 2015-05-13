import vigra
import numpy as np
import matplotlib.pyplot as plot
from volumina_viewer import volumina_single_layer
from volumina_viewer import volumina_double_layer

def watershed_superpixel():
	pass

# perform the watershed algorithm implemented in vigra
# Adapted from superpixel/watershed/ws.py
# @ param: probs: array w/ image to be segmented
# @ param: offset, labeloffset for the segmentation, in case a stack of images is segmented, default 0
def watershed_superpixel_vigra(probs, offset = 0):
    # in nikos script: substract the mean and take the abs value
    #hmap = np.abs( probs - np.mean(probs) )
    hmap = probs

    # smooth the hmap
    hmap_smooth = vigra.gaussianSmoothing(hmap, 3.5)
    #volumina_single_layer(hmap_smooth)
    # find the local minima
    seeds = vigra.analysis.extendedLocalMinima(hmap_smooth, neighborhood = 8)
    # find the connected components in the minima and use them as seeds
    SEEDS = vigra.analysis.labelImageWithBackground(seeds)

    # apply the watershed algorithm
    seg_ws, maxRegionLabel = vigra.analysis.watersheds(probs,
    				neighborhood = 8, seeds = SEEDS.astype(np.uint32) )
    # find connected components
    seg_ws = vigra.analysis.labelImage(seg_ws)

    # if we have an offset, add it to the array
    if offset != 0:
    	seg_ws += offset * np.ones( seg_ws.shape )

    #volumina_double_layer(probs, seg_ws)

    return seg_ws

# perform the watershed algorithm implemented in vigra
# Adapted from superpixel/watershed/ws.py
# @ param: probs: array w/ volume to be segmented
def watershed_supervoxel_vigra(probs):

    # compute seeds
    # try different filter for computing the best seeds (= minima)
    # best options so far:
    # for isotropic data (2x2x2 nm): Gaussian Smoothing with sigma = 4.5
    # for anisotropic data: Gaussian Smoothing with sigma = 3.2

    #sm_probs = np.array(np.abs( probs - 0.5*( np.max(probs) - np.min(probs) ) ), dtype = np.float32 )
    # Gaussian smoothing
    sm_probs = vigra.gaussianSmoothing(probs, 1.5)
    volumina_single_layer(sm_probs)

    # Difference of Gaussians
    #diff = vigra.gaussianSmoothing(probs,2) - sm_probs
    #volumina_single_layer(diff)

    # Hessian of Gaussian
    #hessian = vigra.filters.hessianOfGaussian(probs, sigma = 2)
    #hessian_ev = vigra.filters.tensorEigenvalues( hessian )
    #volumina_single_layer(hessian_ev)
    #hessian_ev_weighted = np.divide( hessian_ev[:,:,:,0],  (hessian_ev[:,:,:,1] + hessian_ev[:,:,:,2] ) )

    seeds = vigra.analysis.extendedLocalMinima3D(sm_probs, neighborhood = 26)
    SEEDS = vigra.analysis.labelVolumeWithBackground(seeds)
    SEEDS = SEEDS.astype(np.uint32)

    #plot.figure()
    #plot.gray()
    #plot.imshow(SEEDS[:,:,25])
    #plot.show()

    seg_ws, maxRegionLabel = vigra.analysis.watersheds(probs,
    					neighborhood = 6, seeds = SEEDS)

    seg_ws = vigra.analysis.labelVolumeWithBackground(seg_ws)

    volumina_double_layer(probs, seg_ws)

    return seg_ws

if __name__ == '__main__':
	path_probs = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_Probabilities.h5"
	key_probs  = "exported_data"

	probs 	= vigra.readHDF5(path_probs, key_probs)

	segmentation = watershed_supervoxel_vigra(probs[:,:,:,0])
