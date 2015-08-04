import numpy as np

# metrics of the ISBI2013 challenge
# adapted from FIJI script
def f_score_metrics(seg1, seg2):
    assert seg1.shape == seg2.shape
    assert len(seg1.shape) == 1

    print "Compute F-score"

    print "Background labelsize, gt segmentation"
    print np.where(seg1 == 0)[0].size
    print "Background labelsize, mc segmentation (should be zero!)"
    print np.where(seg2 == 0)[0].size

    n = float(seg1.shape[0])

    # get max id of both segmentations
    n_labels_1 = np.max(seg1)
    n_labels_2 = np.max(seg2)

    # init contingency matrix
    p_ij = np.zeros( (n_labels_1 + 1, n_labels_2 + 1) )

    print "Computing contingency matrix"
    for i in range(seg1.shape[0]):
        lab1 = seg1[i]
        lab2 = seg2[i]
        p_ij[lab1,lab2] += 1

    # sum of rows
    a_i = np.zeros( p_ij.shape[0] )
    # skip zeroth entry, because it is background label
    for i in range(1, p_ij.shape[0]):
        for j in range(0, p_ij.shape[1]):
            a_i[i] += p_ij[i,j]

    # sum of columns
    b_j = np.zeros( p_ij.shape[1])
    for j in range( 1, p_ij.shape[1] ):
        for i in range( 1, p_ij.shape[0] ):
            b_j[j] += p_ij[i,j]

    p_i0 = np.zeros(p_ij.shape[0])
    aux  = 0.
    for i in range(1, p_ij.shape[0]):
        p_i0[i] = p_ij[i,0]
        aux += p_i0[i]

    # sum of squares of sum of rows
    sumA = 0.
    for i in range(a_i.shape[0]):
        sumA += a_i[i] * a_i[i]

    # sum of squares of sum of cols
    sumB = 0.
    for j in range(b_j.shape[0]):
        sumB += b_j[j] * b_j[j]

    sumB += aux / n

    sumAB = 0.
    for i in range(1, p_ij.shape[0]):
        for j in range(1, p_ij.shape[1]):
            sumAB += p_ij[i,j] * p_ij[i,j]

    sumAB += aux / n

    precision   = sumAB / sumB
    recall      = sumAB / sumA
    randind     = 1.0 - (sumA + sumB - 2.*sumAB) / (n*n)

    print sumA, sumB, sumAB
    print precision, recall

    f_score     = 2 * precision * recall / (recall + precision)

    return randind, f_score
