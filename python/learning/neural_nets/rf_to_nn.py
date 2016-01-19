# implement RF to NN conversion
# algorithm from J. Welbl, "Casting Random Forests as Artificial Neural Networks (and Profiting from It)

import numpy as np
import vigra
import cPickle as pickle
import scipy.sparse as sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as tree_vis

def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def fit_rf(save_path):
    path_feats  = "/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/features/ffeats/ffeat_bert_1_True.h5"
    path_labels = "/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/gt_face_segid1.h5"

    key_feats   = "data"
    key_labels  = "gt_face"

    X = np.nan_to_num( np.array( vigra.readHDF5(path_feats, key_feats) ) )
    Y = np.squeeze( vigra.readHDF5(path_labels, key_labels) )

    N = X.shape[0]
    assert N == Y.shape[0]

    X = scale_features(X)

    rf = RandomForestClassifier(n_estimators = 10)
    rf.fit(X,Y)

    with open(save_path, 'w') as f:
        pickle.dump(rf, f)


def fit_dt(save_path):
    path_feats  = "/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/features/ffeats/ffeat_bert_1_True.h5"
    path_labels = "/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/gt_face_segid1.h5"

    key_feats   = "data"
    key_labels  = "gt_face"

    X = np.nan_to_num( np.array( vigra.readHDF5(path_feats, key_feats) ) )
    Y = np.squeeze( vigra.readHDF5(path_labels, key_labels) )

    N = X.shape[0]
    assert N == Y.shape[0]

    X = scale_features(X)

    dt = DecisionTreeClassifier()
    dt.fit(X,Y)

    with open(save_path, 'w') as f:
        pickle.dump(dt, f)


def inspect_trees(rf):
    #from inspect import getmembers
    #from StringIO import StringIO
    #out = StringIO()

    for dt in rf.estimators_:
        #print getmembers(dt.tree_)
        #out = tree_vis.export_graphviz(dt, out_file = out)
        #print out
        #quit()
        tree = dt.tree_
        print tree.value[0]
        # wtih this info transformation to nn should be possible
        #tree_to_nn(tree)
        quit()


# return the leaf nodes, that can be reached from node i
# returns two arrays, one for the nodes to the left, the other for the nodes to the right
def get_leaf_nodes(tree, i):
    i_l = tree.children_left[i]
    i_r = tree.children_right[i]

    # find all leafs to the left and to the right
    # via depth first search
    leaves_found = []
    for root in (i_l, i_r):
        S = [root]
        visited = []
        leaves = []
        while S:
            node = S.pop()
            if not node in visited:
                visited.append(node)

                n_l = tree.children_left[i]
                n_r = tree.children_left[i]

                # check whether node is terminal
                if n_l < 0:
                    assert n_r < 0
                    leaves.append(node)
                else:
                    S.append(n_l)
                    S.append(n_r)
        leaves_found.append(leaves)
    assert len(leaves_found) == 2
    return leaves_found[0], leaves_found[1]


# FIXME
# get dict with depth for all the nodes in the tree
def get_depth_dict(tree):
    depth_dict = {0 : 0}
    # start breadth first search from root to reach all the leaves
    S = [0]
    visited = []
    while S:
        node = S.pop()
        if not node in visited:
            visited.append(node)

            n_l = tree.children_left[node]
            n_r = tree.children_right[node]

            # check whether node is terminal
            if n_l < 0:
                assert n_r < 0, "If node is terminal, both children should be non-existing!"
                continue
            else:
                S.append(n_l)
                S.append(n_r)
                depth_dict[n_l] = depth_dict[node] + 1
                depth_dict[n_r] = depth_dict[node] + 1

    return depth_dict


def tree_to_nn_params(tree, n_feats, str_01 = 1., str_12 = 1., str_23 = 1.):

    n_nodes = len(tree.children_right)

    nodes_to_layers = []
    nodes_l1 = {}
    nodes_l2 = {}
    i_l1 = 0
    i_l2 = 0
    for i in range(n_nodes):
        # check whether this node is a leaf, if it is:
        # -> layer 2
        # else:
        # -> layer 1
        if tree.children_right[i] > 0:
            assert tree.children_left[i] > 0
            nodes_to_layers.append(1)
            nodes_l1[i] = i_l1
            i_l1 += 1
        else:
            assert tree.children_left[i] < 0
            nodes_to_layers.append(2)
            nodes_l2[i] = i_l2
            i_l2 += 1

    assert len(nodes_to_layers) == n_nodes
    n_inner = np.sum( np.array(nodes_to_layers) == 1 )
    n_leaf  = np.sum( np.array(nodes_to_layers) == 2 )
    assert n_inner + n_leaf == n_nodes
    assert len(nodes_l1.keys()) == n_inner
    assert len(nodes_l2.keys()) == n_leaf

    # get the depth for all the leaf nodes
    depth_dict = get_depth_dict(tree)
    assert len( depth_dict.keys() ) == n_nodes, str( len(depth_dict.keys() ) ) + " , " + str(n_nodes)

    # initialize all the weights for our 2 - hidden - layer - nn
    weights_01 = sparse.lil_matrix( (n_feats, n_inner) )
    biases_1   = sparse.lil_matrix( (n_inner,1) )
    weights_12 = sparse.lil_matrix( (n_inner, n_leaf) )
    biases_2   = sparse.lil_matrix( (n_leaf,1) )
    weights_23 = sparse.lil_matrix( (n_leaf, 2) )
    biases_3   = sparse.lil_matrix( (2,1) )

    for i in range(n_nodes):
        # check whether we are in first or second layer
        if nodes_to_layers[i] == 1:
            split_feat = tree.feature[i]
            threshold  = tree.threshold[i]
            # determine feature to neuron weight
            weights_01[ split_feat, nodes_l1[i] ] = str_01
            biases_1[ nodes_l1[i] ] = - str_01 * threshold
            # determine l1 to l2 weights
            # need to walk the tree and find all leave nodes that can be reached!
            leaf_nodes_l, leaf_nodes_r = get_leaf_nodes(tree, i)
            for l in leaf_nodes_l:
                weights_12[i, nodes_l2[l] ] = - str_12
            for l in leaf_nodes_r:
                weights_12[i, nodes_l2[l] ] = str_12
        else:
            # second layer:
            # determine the bias:
            biases_2[ nodes_l2[i] ] = -str_12 * (depth_dict[i] - 1)
            # weights l2 to output
            frequencies = tree.value[i]
            # majority vote (implementation in original paper)
            # find the class this node votes for
            c = np.argmax(frequencies)
            assert c in (0,1)
            weights_23[ nodes_l2[i], c] = str_23
            # probabalistic vote (closer to sklearn voting mechanism)
            #n_samples = float( frequencies[0] + frequencies[1] )
            #weights_23[ nodes_l2[1], 0] = str_23 * frequencies[0] / n_samples
            #weights_23[ nodes_l2[1], 1] = str_23 * frequencies[1] / n_samples

    # check that the weights have the right number of non-zero entries
    assert weights_01.nnz == n_inner, str(weights_01.nnz) + " , " + str(n_inner)
    # number of nonzero weighs in 12 is not that obvious!
    #assert weights_12.nnz() == n_inner, str(weights_01.nnz()) + " , " + str(n_inner)
    assert weights_23.nnz == n_leaf, str(weights_23.nnz) + " , " + str(n_leaf)

    return (weights_01, weights_12, weights_23, biases_1, biases_2, biases_3)


def build_nn(nn_params):
    pass


if __name__ == '__main__':

    save_path = './../models/dt_ped_full.pkl'
    #fit_dt(save_path)
    with open(save_path, 'r') as f:
        dt = pickle.load(f)
    # magic number of features
    nn_params = tree_to_nn_params(dt.tree_, 413)
    build_nn(nn_params)
