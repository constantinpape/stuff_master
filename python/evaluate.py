import sys , h5py
import numpy as np
import argparse
import tifffile as tiff
import pylab as plot
import vigra
from sklearn.metrics import adjusted_rand_score

# calculate the variation of informaton:
# VI(X,Y) = H(X) = H(Y) - 2*I(X,Y) 
# FIXME only works for binary labels 
def VariationOfInformation(labels_obtained, labels_expected):
	N = labels_obtained.shape[0]
	assert(labels_expected.shape[0] == N)
	# compute the probabilities
	probs_obt   = np.zeros(2)
	probs_exp   = np.zeros(2)
	cross_probs = np.zeros( (2,2) )
	for i in range(N):
		j = int(labels_obtained[i])
		k = int(labels_expected[i])
		probs_obt[j]     += 1
		probs_exp[j]     += 1
		cross_probs[j][k] += 1
	probs_obt /= N
	probs_exp /= N
	cross_probs /= N
	# compute the entropies
	H0 = probs_obt.dot(np.log2(probs_obt))
	H1 = probs_exp.dot(np.log2(probs_exp))
	# compute the mutual information
	I  = 0.
	for j in range(probs_obt.shape[0]):
		for k in range(probs_exp.shape[0]):
			I += cross_probs[j][k] * ( np.log2( cross_probs[j][k] ) - np.log2( probs_obt[j] * probs_exp[k] ) )
	return H0 + H1 - 2*I 

def evaluate_labeling(labels_obtained, labels_exp):
	# calculate the average over all slices
	N = labels_obtained.shape[0]
	rand = 0.
	var = 0.
	labels_obtained = labels_obtained.reshape( (N, labels_obtained.shape[1]*labels_obtained.shape[2], labels_obtained.shape[3]) )
	labels_exp = labels_exp.reshape( (N, labels_exp.shape[1]*labels_exp.shape[2] ) )
	for i in range(N):
		rand += adjusted_rand_score(labels_obtained[i,:,0], labels_exp[i])
		var  += VariationOfInformation(labels_obtained[i,:,0], labels_exp[i])
	rand /= N
	var  /= N
	print 'Evaluating Labeling against ground truth:'
	print 'RandIndex', rand
	print 'VariationOfInformation', var	

def plot_image(image):
	plot.figure()
	plot.imshow(image)
	plot.colorbar()
	plot.show()
	plot.close()

def process_labels_expected(labels):
	class1 = np.max(labels)
	labels[np.where(labels == class1)] = 1
	return labels

def process_labels_probabilities(labels):
	for j in range(labels.shape[3]):
		for i in range(labels.shape[0]):
			thresh = np.max(labels[i]) / 2
			labels[i][np.where(labels[i] < thresh)] = 0 
			labels[i][np.where(labels[i] > thresh)] = 1
	# first layer = membrane classification, second layer = mitochondrium classification
	return labels[:,:,:,1], labels[:,:,:,2]	

def process_labels_segmentation(labels):
	labels_membrane = np.ones( labels.shape )
	labels_mito = np.zeros( labels.shape )
	labels_membrane[np.where(labels == 2)] = 0
	labels_mito[np.where(labels == 3)] = 1
	return labels_membrane, labels_mito

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Evaluate labeling')
	parser.add_argument('path_exp_memb', type = str, nargs = 1, 
				help = 'path to original labeling, membranes')
	parser.add_argument('path_exp_mito', type = str, nargs = 1, 
				help = 'path to original labeling, mitochondriae')
	parser.add_argument('path_obt', type = str, nargs = 1,
				help = 'path to obtained labeling')

	args = parser.parse_args()

	labels_exp_membrane = tiff.imread(args.path_exp_memb[0])
	labels_exp_mito = tiff.imread(args.path_exp_mito[0])

	labels_exp_membrane = process_labels_expected(labels_exp_membrane)
	labels_exp_mito = process_labels_expected(labels_exp_mito)
	
	labels_obt = vigra.readHDF5(args.path_obt[0],"exported_data")
	labels_obt = labels_obt.transpose( (2,0,1,3) )
	
	labels_obt_membrane, labels_obt_mito = process_labels_segmentation(labels_obt)
	
	#plot_image(labels_obt_mito[0,:,:,0])
	#plot_image(labels_obt_membrane[0,:,:,0])

	#plot_image(labels_exp_membrane[0])
	#plot_image(labels_exp_mito[0])

	print "Membrane Labeling"
	evaluate_labeling(labels_obt_membrane, labels_exp_membrane)
	print "Mito Labeling"
	evaluate_labeling(labels_obt_mito, labels_exp_mito)
	

