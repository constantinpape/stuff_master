import numpy as np

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
