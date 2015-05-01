import vigra
import numpy as np
import matplotlib.pyplot as plot

def watershed_superpixel():
	pass

# perform the watershed algorithm implemented in vigra
# Adapted from superpixel/watershed/ws.py
def watershed_superpixel_vigra(probs):
	# in nikos script: substract the mean and take the abs value
	# TODO why should we do this ???
	hmap = probs
	#hmap = np.abs( probs - bp.mean(probs) )

	# smooth the hmap
	hmap_smooth = vigra.gaussianSmoothing(hmap, 2.5)	
	# find the local minima
	seeds = vigra.analysis.extendedLocalMinima(hmap_smooth, neighborhood = 8)
	# find the connected components in the minima and use them as seeds
	SEEDS = vigra.analysis.labelImageWithBackground(seeds)
	
	# apply the watershed algorithm
	seg_ws, maxRegionLabel = vigra.analysis.watersheds(hmap_smooth,
					neighborhood = 8, seeds = SEEDS.astype(np.uint32) )
	seg_ws = vigra.analysis.labelImage(seg_ws)
	
	return seg_ws 

def watershed_supervoxel_vigra(probs):

	# compute seeds
	#sm_probs = np.array(np.abs( probs - 0.5*( np.max(probs) - np.min(probs) ) ), dtype = np.float32 )
	sm_probs = vigra.gaussianSmoothing(probs, 4.5)

	#plot.figure()
    	#plot.imshow(sm_probs[:,:,25])
	#plot.colorbar()
	#plot.show()
	
	hessian = vigra.filters.hessianOfGaussian(probs, sigma = 2)
	hessian_ev = vigra.filters.tensorEigenvalues( hessian )
	print hessian_ev.shape
	
	#plot.figure()
    	#plot.imshow(hessian_ev[:,:,25,0])
	#plot.colorbar()
	#plot.show()
	#
	#plot.figure()
    	#plot.imshow(hessian_ev[:,:,25,1])
	#plot.colorbar()
	#plot.show()
	#
	#plot.figure()
    	#plot.imshow(hessian_ev[:,:,25,2])
	#plot.colorbar()
	#plot.show()
	
	#hessian_ev_weighted = np.divide( hessian_ev[:,:,:,0],  (hessian_ev[:,:,:,1] + hessian_ev[:,:,:,2] ) )

	#plot.figure()
    	#plot.imshow( hessian_ev_weighted[:,:,25] )
	#plot.colorbar()
	#plot.show()
	
	seeds = vigra.analysis.extendedLocalMinima3D(sm_probs, neighborhood = 26)
	SEEDS = vigra.analysis.labelVolumeWithBackground(seeds)
	SEEDS = SEEDS.astype(np.uint32)
	
	#plot.figure()
	#plot.gray()
	#plot.imshow(SEEDS[:,:,25])
	#plot.show()


	seg_ws, maxRegionLabel = vigra.analysis.watersheds(probs,
						neighborhood = 6, seeds = SEEDS)

	return seg_ws

if __name__ == '__main__':
	path_probs = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_Probabilities.h5"
	key_probs  = "exported_data"

	probs 	= vigra.readHDF5(path_probs, key_probs)
	
	segmentation = watershed_supervoxel_vigra(probs[:,:,:,0])
