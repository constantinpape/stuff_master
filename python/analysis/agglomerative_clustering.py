from vigra import numpy as np
import vigra
from IPython import embed
from volumina_viewer import volumina_n_layer

# get a segmentation of the superpixel segmentation via agglomerative clustering
# based on vigranumpy/examples/graph_agglomerative clustering
# only use weight features, dont use node features
def agglomerative_clustering_2d(img, labels, use_gradient = False):
	
	labels = vigra.analysis.labelImage(labels)
	
	imgBig = vigra.resize(img, [img.shape[0]*2-1, img.shape[1]*2-1])
	if use_gradient:
		# compute the gradient on interpolated image
		sigmaGradMag = 2.0
		gradMag = vigra.filters.gaussianGradientMagnitude(imgBig, sigmaGradMag)

	# get 2D grid graph and edgeMap for grid graph
	# from gradMag of interpolated image or from plain image (depending on use_gradient)
	gridGraph = vigra.graphs.gridGraph(img.shape[0:2])
	gridGraphEdgeIndicator = []
	if use_gradient:
		gridGraphEdgeIndicator = vigra.graphs.edgeFeaturesFromInterpolatedImage(gridGraph, gradMag)
	else:
		gridGraphEdgeIndicator = vigra.graphs.edgeFeaturesFromInterpolatedImage(gridGraph, imgBig)

	# get region adjacency graph from super-pixel labels
	rag = vigra.graphs.regionAdjacencyGraph(gridGraph, labels)

	# accumalate edge weights from gradient magintude
	edgeWeights = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)

	# agglomerative clustering
	beta = 0.5
	nodeNumStop = 20
	clustering = vigra.graphs.agglomerativeClustering(graph = rag, edgeWeights = edgeWeights,
					beta = beta, nodeNumStop = nodeNumStop)
	
	res = np.zeros( labels.shape )

	for c in range(len(clustering)):
		res[ np.where( labels== (c-1) ) ] = clustering[c]

	res = vigra.Image(res)
	res = vigra.analysis.labelImage(res)

	return res

def agglomerative_clustering_3d(vol, seg_vol, use_gradient = False):
	
	labels = vigra.analysis.labelVolume(seg_vol)	
	
	volBig = vigra.resize(vol, [ vol.shape[0]*2-1, vol.shape[1]*2-1, vol.shape[2]*2-1 ] )
	if use_gradient:
		sigmaGradMag = 2.0
		gradMag = vigra.filters.gaussianGradientMagnitude(volBig, sigmaGradMag)
	
	# get 3D grid graph and edgeMap for grid graph
	# from gradMag of interpolated vloume or plain volume (depending on use_gradient)
	gridGraph = vigra.graphs.gridGraph(vol.shape[0:2])
	gridGraphEdgeIndicator = []

	if use_gradient:
		gridGraphEdgeIndicator = vigra.graphs.edgeFeaturesFromInterpolatedImage(gridGraph, gradMag)
	else:
		gridGraphEdgeIndicator = vigra.graphs.edgeFeaturesFromInterpolatedImage(gridGraph, volBig)
	
	# get region adjacency graph from super-pixel labels
	rag = vigra.graphs.regionAdjacencyGraph(gridGraph, labels)

	# accumalate edge weights from gradient magintude
	edgeWeights = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)

	# agglomerative clustering
	beta = 0.5
	nodeNumStop = 20
	clustering = vigra.graphs.agglomerativeClustering(graph = rag, edgeWeights = edgeWeights,
					beta = beta, nodeNumStop = nodeNumStop)

	print clustering

		
	
if __name__ == '__main__':
	path_seg = "/home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/"	
	file = "watershed_vigra"
	#file = "slic_vigra"
	path_seg = path_seg + file + ".h5" 
	key_seg = "superpixel"
	
	seg_vol = vigra.readHDF5(path_seg, key_seg)
	
	path_img = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_Probabilities.h5" 
	key_img = "exported_data"
	
	vol = vigra.readHDF5(path_img, key_img)

	# IMPORTANT only use the edge / membrane probability channel !
	vol = vol[:,:,:,0]	

	dim2 = False
	
	if dim2:
		img = vigra.Image(vol[:,:,0])
		seg = vigra.Image(seg_vol[:,:,0]).astype(np.uint8)
		
		res = agglomerative_clustering_2d( img, seg )
		res = vigra.Image(res).astype(np.uint8)
	
		volumina_n_layer( (img, seg, res) )
	else:
		vol = vigra.Volume(vol)
		seg_vol = vigra.Volume(seg_vol)
		
		res = agglomerative_clustering_3d(vol, seg_vol )
	#	res = vigra.Volume(res).astype(np.uint8)
	#	
	#	volumina_n_layer( (vol, seg_vol, res) )
	#
