import sys , h5py
import numpy as np
import argparse
import pylab as plot
import vigra
from skneuro.learning import randIndex
from skneuro.learning import variationOfInformation

def evaluate_labeling(labels_obtained, labels_exp):
	# calculate the average over all slices
	N = labels_obtained.shape[0]
	rand = 0.
	var = 0.
	labels_obtained = labels_obtained.reshape( (N, labels_obtained.shape[1]*labels_obtained.shape[2], labels_obtained.shape[3]) )
	labels_exp = labels_exp.reshape( (N, labels_exp.shape[1]*labels_exp.shape[2] ) )
	for i in range(N):
		rand += randIndex(labels_obtained[i,:,0], labels_exp[i])
		var  += variationOfInformation(labels_obtained[i,:,0], labels_exp[i])
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

	labels_exp_membrane = vigra.impex.readImage(args.path_exp_memb[0])
	labels_exp_mito = vigra.impex.readImgage((args.path_exp_mito[0])

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


