from vigra import numpy as np
import vigra
from IPython import embed
from volumina_viewer import volumina_n_layer

# get a segmentation of the superpixel segmentation via agglomerative clustering
# based on vigranumpy/examples/graph_agglomerative clustering
def agglomerative_clustering_2d(img, labels):
	labels = vigra.analysis.labelImage(labels)
	# compute the gradient on interpolated image
	imgBig = vigra.resize(img, [img.shape[0]*2-1, img.shape[1]*2-1])
	sigmaGradMag = 2.0
	gradMag = vigra.filters.gaussianGradientMagnitude(imgBig, sigmaGradMag)

	# get 2D gtid graph and edgeMap for grid graph
	# from gradMag of interpolated image
	gridGraph = vigra.graphs.gridGraph(img.shape[0:2])
	gridGraphEdgeIndicator = vigra.graphs.edgeFeaturesFromInterpolatedImage(gridGraph, gradMag)

	# get region adjacency graph from super-pixel labels
	rag = vigra.graphs.regionAdjacencyGraph(gridGraph, labels)

	# accumalate edge weights from gradient magintude
	edgeWeights = rag.accumulateEdgeFeatures(gridGraphEdgeIndicator)

	# accumalate node features from grid graph node map
	# whcih is just a plain image (with channels)
	nodeFeatures = rag.accumulateNodeFeatures(img)

	# agglomerative clustering
	beta = 0.5
	nodeNumStop = 15
	clustering = vigra.graphs.agglomerativeClustering(graph = rag, edgeWeights = edgeWeights,
					beta = beta, nodeFeatures = nodeFeatures,
					nodeNumStop = nodeNumStop)

	print clustering.shape
	print np.unique(labels).shape
	print clustering
	print np.min(labels)
	
	res = np.zeros( labels.shape )

	for c in range(len(clustering)):
		res[ np.where( labels== (c-1) ) ] = clustering[c]

	return res
	
if __name__ == '__main__':
	path_seg = "/home/constantin/Work/data_ssd/data_080515/pedunculus/superpixel/"	
	file = "watershed_vigra"
	#file = "slic_vigra"
	path_seg = path_seg + file + ".h5" 
	key_seg = "superpixel"
	
	seg_vol = vigra.readHDF5(path_seg, key_seg)
	
	path_img = "/home/constantin/Work/data_ssd/data_080515/pedunculus/labeling_2classes.h5"
	key_img = "exported_data"
	
	vol = vigra.readHDF5(path_img, key_img)

	img = vigra.Image(vol[:,:,0,1])
	seg = vigra.Image(seg_vol[:,:,0])
	
	res = agglomerative_clustering_2d( img, seg )

	res = vigra.Image(res)
	
	volumina_n_layer( (img, seg, res) )
