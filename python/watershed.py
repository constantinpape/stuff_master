import numpy as np
import vigra

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
	sm_probs = np.abs( probs - 0.5*( np.max(probs) - np.min(probs) ) )
	sm_probs = sm_probs.astype(np.float32)
	seeds = vigra.analysis.extendedLocalMinima3D(sm_probs, neigborhood = 26)
	SEEDS = vigra.analysis.labelVolumeWithBackground(SEEDS)
	SEEDS = SEEDS.astype(np.uint32)
	seg_ws, maxRegionLabel = vigra.analysis.watersheds(probs,
						neighborhood = 6, seeds = SEEDS)
	return seg_ws

