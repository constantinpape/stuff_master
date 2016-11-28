# implement RF to NN conversion
# algorithm from J. Welbl, "Casting Random Forests as Artificial Neural Networks (and Profiting from It)

import numpy as np
import vigra
import cPickle as pickle
import scipy.sparse as sparse
import os

import caffe
from caffe import layers as L
from caffe import params as P

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as tree_vis

from get_data import get_data_ped_full

def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def fit_rf(save_path):

    X,Y = get_data_ped_full()

    X = scale_features(X)

    rf = RandomForestClassifier(n_estimators = 10)
    rf.fit(X,Y)

    with open(save_path, 'w') as f:
        pickle.dump(rf, f)


def fit_dt(save_path):

    X,Y = get_data_ped_full()

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


def tree_to_nn_params(tree, n_feats, proto_path, str_01 = 1000., str_12 = 1000., str_23 = 0.1):

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
    weights_01 = sparse.lil_matrix( (n_inner, n_feats)  )
    biases_1   = sparse.lil_matrix( (n_inner,1) )
    weights_12 = sparse.lil_matrix( (n_leaf, n_inner) )
    biases_2   = sparse.lil_matrix( (n_leaf,1) )
    weights_23 = sparse.lil_matrix( (2, n_leaf) )
    biases_3   = sparse.lil_matrix( (2,1) )

    for i in range(n_nodes):
        # check whether we are in first or second layer
        if nodes_to_layers[i] == 1:
            split_feat = tree.feature[i]
            threshold  = tree.threshold[i]
            # determine feature to neuron weight
            weights_01[ nodes_l1[i], split_feat ] = str_01
            biases_1[ nodes_l1[i] ] = - str_01 * threshold
            # determine l1 to l2 weights
            # need to walk the tree and find all leave nodes that can be reached!
            leaf_nodes_l, leaf_nodes_r = get_leaf_nodes(tree, i)
            for l in leaf_nodes_l:
                weights_12[ nodes_l2[l], i ] = - str_12
            for l in leaf_nodes_r:
                weights_12[ nodes_l2[l], i ] = str_12
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
            weights_23[ c, nodes_l2[i]] = str_23
            # probabalistic vote (closer to sklearn voting mechanism)
            #n_samples = float( frequencies[0] + frequencies[1] )
            #weights_23[ nodes_l2[1], 0] = str_23 * frequencies[0] / n_samples
            #weights_23[ nodes_l2[1], 1] = str_23 * frequencies[1] / n_samples

    # check that the weights have the right number of non-zero entries
    assert weights_01.nnz == n_inner, str(weights_01.nnz) + " , " + str(n_inner)
    # number of nonzero weighs in 12 is not that obvious!
    #assert weights_12.nnz() == n_inner, str(weights_01.nnz()) + " , " + str(n_inner)
    assert weights_23.nnz == n_leaf, str(weights_23.nnz) + " , " + str(n_leaf)

    p = to_prototex(proto_path, [n_feats, n_inner, n_leaf, 2], 1 )
    print p
    quit()

    return (weights_01, weights_12, weights_23, biases_1, biases_2, biases_3)


# write the prototex file for the given architecture
# (only 2 hidden layers!)
#TODO make sure details of ntwk topology (type of the layers / non-linearity!)
def to_prototex(filename, nn_architecture, batchsize):

    n = caffe.NetSpec()
    # data layer
    n.data, n.label = L.DummyData(shape = [dict(dim = [batchsize, nn_architecture[0], 1, 1]),
            dict(dim = [batchsize, 1, 1, 1] )])
    # first hidden layer, corresponding to inner nodes
    n.fc1 = L.TanH(n.data, num_output = nn_architecture[1])
    # second hidden layer, corresponding to leaf nodes
    n.fc2 = L.Sigmoid(n.fc1, num_output = nn_architecture[2])
    # two class output layer
    n.out = L.Sigmoid(n.fc2, num_output = nn_architecure[3])

    return n.to_proto()

    #with open(filename, 'w') as f:
    #    net_name = os.path.split(filename)[1].split('.')[0]

    #    f.write("name: " + "\"" + str(net_name) + "\"" + "\n" )
    #    f.write("\n")

    #    # input layer
    #    f.write("input: \"data\"" + "\n")
    #    f.write("input_dim: 1" + "\n")
    #    f.write("input_dim: " + str(nn_architecture[0]) + "\n")
    #    f.write("input_dim: 1" + "\n")
    #    f.write("input_dim: 1" + "\n")

    #    # first hidden layer, corresponding to inner nodes
    #    f.write("\n")
    #    f.write("layers {" + "\n")
    #    f.write("  " + "bottom: \"data\"" + "\n")
    #    f.write("  " + "top: \"fc1\"" + "\n")
    #    f.write("  " + "name: \"fc1\"" + "\n")
    #    #f.write("  " + "type: \"TanH\"" + "\n")
    #    f.write("  " + "type: INNER_PRODUCT" + "\n")
    #    f.write("  " + "inner_product_param {" + "\n")
    #    f.write("    " + "num_output: " + str(nn_architecture[1]) + "\n")
    #    f.write("  " + "}" + "\n")
    #    f.write("}" + "\n")

    #    # second hidden layer, corresponding to leaf nodes
    #    f.write("layers {" + "\n")
    #    f.write("  " + "bottom: \"fc1\"" + "\n")
    #    f.write("  " + "top: \"fc2\"" + "\n")
    #    f.write("  " + "name: \"fc2\"" + "\n")
    #    #f.write("  " + "type: \"Sigmoid\"" + "\n")
    #    f.write("  " + "type: INNER_PRODUCT" + "\n")
    #    f.write("  " + "inner_product_param {" + "\n")
    #    f.write("    " + "num_output: " + str(nn_architecture[2]) + "\n")
    #    f.write("  " + "}" + "\n")
    #    f.write("}" + "\n")

    #    # two class output layer
    #    f.write("layers {" + "\n")
    #    f.write("  " + "bottom: \"fc2\"" + "\n")
    #    f.write("  " + "top: \"out\"" + "\n")
    #    f.write("  " + "name: \"out\"" + "\n")
    #    #f.write("  " + "type: \"Sigmoid\"" + "\n")
    #    f.write("  " + "type: INNER_PRODUCT" + "\n")
    #    f.write("  " + "inner_product_param {" + "\n")
    #    f.write("    " + "num_output: " + str(nn_architecture[3]) + "\n")
    #    f.write("  " + "}" + "\n")
    #    f.write("}" + "\n")


def init_caffe_nn(prototex_path, weights_path, nn_params):

    assert len(nn_params) == 6

    # init
    caffe.set_mode_cpu()

    nn_net = caffe.Net(prototex_path, caffe.TRAIN)

    # set weights and biases for the layers
    nn_net.params["fc1"][0].data[...] = nn_params[0].todense()
    nn_net.params["fc1"][1].data[...] = np.squeeze( nn_params[3].todense() )

    nn_net.params["fc2"][0].data[...] = nn_params[1].todense()
    nn_net.params["fc2"][1].data[...] = np.squeeze( nn_params[4].todense() )

    nn_net.params["out"][0].data[...] = nn_params[2].todense()
    nn_net.params["out"][1].data[...] = np.squeeze( nn_params[5].todense() )

    nn_net.save(weights_path)


def predict_with_nn(proto_path, model_path, X):
    caffe.set_mode_cpu()

    net = caffe.Classifier(proto_path,
            model_path,
            image_dims = (413,1,1)
            ) # what else do we need?

    X = X[:,:,np.newaxis,np.newaxis]
    pred = net.predict(X)
    pred = np.argmax(pred, axis = 1)

    assert pred.shape[0] == X.shape[0], str(pred.shape) + " , " + str(X.shape)

    return pred


if __name__ == '__main__':

    save_path = './../models/dt_ped_full.pkl'
    #fit_dt(save_path)
    with open(save_path, 'r') as f:
        dt = pickle.load(f)

    proto_path = './../models/rfnet_test.prototex'
    weights_path = './../models/rfnet_test.caffemodel'

    # TODO get magic number of features from somewhere
    nn_params = tree_to_nn_params(dt.tree_, 413, proto_path)
    init_caffe_nn(proto_path, weights_path, nn_params)

    # predict with the dt and the nn
    X, Y = get_data_ped_full()
    X = scale_features(X)
    reduce_to = 10000
    X = X[:reduce_to]

    dt_pred = dt.predict(X)
    dt_acc  = np.sum( np.equal(dt_pred, Y[:reduce_to]) ) / float(Y[:reduce_to].shape[0])

    nn_pred = predict_with_nn(proto_path, weights_path, X)
    nn_acc  = np.sum( np.equal(nn_pred, Y[:reduce_to]) ) / float(Y[:reduce_to].shape[0])

    print "Accuracy of dt:", dt_acc
    print "Accuracy of nn:", nn_acc

