import numpy as np
import vigra

# make slic superpixels from probabilities and / or raw data
# @params: probabilities, raw = input files
# k = number of clusters / supervoxels
# m = weight of spatial vs. colour distance
def slic_superpixel(probs, raw, k, m, use_prob = True, use_raw = False):
	if(use_prob == False and use_raw == False):
		raise RuntimeError("Neither probability nor greyscale values are set to be used in SLIC algorithm!")
	# init the clustercenters
	N_pix 	= probs.shape[0]*probs.shape[1]
	S	= np.sqrt(N_pix / k)
	# order x, y, prob, raw
	centers = np.zeros( (k, 4) )
	# place initial clusters on lattice with spacing S
	x 	= np.linspace(0, probs.shape[0], np.floor(np.sqrt(k)), endpoint = False )
	y 	= np.linspace(0, probs.shape[1], np.floor(np.sqrt(k)), endpoint = False )
	xv,yv 	= np.meshgrid(x, y, sparse = False, idexing = 'ij')
	c 	= 0
	for i in range(x.shape[0]):
		for j in range(y.shape[0]):
			# TODO set center to lowest gradient in 3x3 neighborhood instead of just assigning it to nearest pixel
			centers[c][0] = xv[i,j]
			centers[c][1] = yv[i,j]
			pix	      = ( np.floor(xv[i,j]), np.floor(yv[i,j]) )
			centers[c][2] = probs[pix]
			centers[c][3] = raw[pix]
			c += 1
	# init pixel labels and distances
	labels 	  = -1*np.ones( (probs.shape[0],probs.shape[1]) )
	distances = float('inf')*np.ones( (probs.shape[0],probs.shape[1]) )
	# execute the slic algorithm
	iter_max  = 50
	err 	  = float('inf')
	thresh    = 1.e-4
	iteration = 0
	while err > thresh and iteration < iter_max:
		for c_indx in range(k):
			center = centers[c_indx]
			x_min = max( int( np.floor(center[0] - S) )   , 0   )
			x_max = min( int( np.ceil(center[0] + S + 1) ), probs.shape[0] )
			y_min = max( int( np.floor(center[1] - S) )   , 0   )
			y_max = min( int( np.ceil(center[1] + S + 1) ), probs.shape[1] )
			for i in range(x_min,x_max):
				for j in range(y_min,y_max):
					# calculate the distance
					dist_space = (i - center[0])**2 + (j - center[1])**2
					dist_colour  = 0.
					if use_prob:
						dist_colour += ( probs[i,j] - center[2] )**2
					if use_raw:
						dist_colour += ( (raw[i,j]  - center[3]) / 255. )**2
					dist = np.sqrt( dist_colour + dist_space / S**2 * m**2)
					if dist < distances[i,j]:
						distances[i,j]  = dist 
						labels[i,j]	= c_indx
		# update cluster centers
		centers_new = np.zeros( (k,4) )
		for c_indx in range(k):
			pix_cluster = np.where(labels == c_indx)
			for i in range(pix_cluster[0].shape[0]):
				pix   = ( pix_cluster[0][i], pix_cluster[1][i] )
				centers_new[c_indx, 0] += pix[0]
				centers_new[c_indx, 1] += pix[1]
				centers_new[c_indx, 2] += probs[pix]
				centers_new[c_indx, 3] += raw[pix]
			centers_new[c_indx] /= pix_cluster[0].shape[0]
		# update residual error
		mask = np.ones(centers_new.shape[1], dtype = bool)
		if use_prob == False:
			mask[2] = False
		elif use_raw == False:
			mask[3] = False
		err = np.linalg.norm(centers[:,mask] - centers_new[:,mask])
		centers = centers_new	
		print "Computing SLIC superpixel:", iteration, "residual:", err
		iteration += 1
	# TODO post processing / unconnected pixel
	return labels



# make slic superpixels from probabilities and / or raw data
# @params: probabilities, raw = input files
# k = number of clusters / supervoxels
# aniso = anisotropy in z direction 
def slic_supervoxel(probabilities, raw, k, m, aniso, use_prob = True, use_raw = False):
	if(use_prob == False and use_raw == False):
		raise RuntimeError("Neither probability nor greyscale values are set to be used in SLIC algorithm!")
	# co

# compute SLIC supepixels with vigra. Adapted from superpixel/slic/vigra/slic_vigr.py.
def slic_superpixel_vigra(probs, slic_weight, diameter):
	# in nikos script the superpixel are calculated on the gaussian gradient magnitude - WHY?
	# calculate the gradient via 1st derivative of gaussian filter
	# ggm = vigra.filters.gaussianGradientMagnitude(probs, 1.)
	
	# calculate the SLIC supepixels
	seg, maxlab = vigra.analysis.slicSuperpixels( probs.astype(np.float32), slic_weight, diameter)
	seg	    = vigra.analysis.labelImage( seg )
	return seg
	
