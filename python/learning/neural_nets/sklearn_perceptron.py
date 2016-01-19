import vigra
import numpy as np
import cPickle as pickle
import time

from sklearn.neural_network   import MLPClassifier
from sklearn.cross_validation import KFold

def train_mlp(X, Y, save_path, nn_params = None):

    print "Start Learning Net"

    clf = MLPClassifier( algorithm = 'l-bfgs',
            alpha = 1e-5,
            hidden_layer_sizes = (500,2),
            random_state = 1)
    clf.fit(X,Y)

    with open(save_path, 'w') as f:
        pickle.dump(clf,f)


def mlp_cv(X,Y, test_algo = True, test_architectures = False):

    res_dict = {}
    if test_algo:
        print "Crossvalidation to determine best optimization algorithm"
        # iterate over optimization algorithms
        for algo in ('sgd', 'adam', 'l-bfgs'):
            print algo
            # for sgd, we try:
            if algo == 'sgd':
                # constant does not converge!
                for learn_rate in ('invscaling', 'adaptive'):
                    print learn_rate
                    kfold = KFold(X.shape[0], n_folds = 10)

                    mlp = MLPClassifier( algorithm = algo,
                            learning_rate = learn_rate,
                            hidden_layer_sizes = (500,2),
                            random_state = 1)

                    train_times    = []
                    train_accuracy = []
                    test_accuracy  = []

                    for train, test in kfold:
                        t_tr = time.time()
                        mlp.fit( X[train], Y[train] )
                        train_times.append( time.time() - t_tr )
                        acc_train = np.sum( np.equal( mlp.predict( X[train]), Y[train] ) ) / float(X[train].shape[0])
                        acc_test  = np.sum( np.equal( mlp.predict( X[test]), Y[test] ) ) / float(X[test].shape[0])
                        train_accuracy.append( acc_train )
                        test_accuracy.append(  acc_test )

                    res_dict[algo + "_" + learn_rate] = (np.mean(train_accuracy), np.std(train_accuracy),
                                                         np.mean( test_accuracy), np.std( test_accuracy),
                                                         np.mean(train_times), np.std(train_times))
            else:
                kfold = KFold(X.shape[0], n_folds = 10)

                mlp = MLPClassifier( algorithm = algo,
                        hidden_layer_sizes = (500,2),
                        random_state = 1)

                train_times    = []
                train_accuracy = []
                test_accuracy  = []

                for train, test in kfold:
                    t_tr = time.time()
                    mlp.fit( X[train], Y[train] )
                    train_times.append( time.time() - t_tr )
                    acc_train = np.sum( np.equal( mlp.predict( X[train]), Y[train] ) ) / float(X[train].shape[0])
                    acc_test  = np.sum( np.equal( mlp.predict( X[test]), Y[test] ) ) / float(X[test].shape[0])
                    train_accuracy.append( acc_train )
                    test_accuracy.append(  acc_test )

                res_dict[algo] = (np.mean(train_accuracy), np.std(train_accuracy),
                                  np.mean(test_accuracy), np.std(test_accuracy),
                                  np.mean(train_times), np.std(train_times))


    with open('./../results/res_nncv_algo.pkl', 'w') as f:
        pickle.dump(res_dict,f)


def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)


def simple_nn_test(X,Y):
    # make train / test split

    indices = np.random.permutation(N)
    split = int(0.6*N)

    X_tr = X[:split]
    Y_tr = Y[:split]
    X_te = X[split:]
    Y_te = Y[split:]

    save_path = './../models/nn.pkl'
    train_mlp(X_tr, Y_tr, save_path)
    with open(save_path, 'r') as f:
        mlp = pickle.load(f)

    prediction = mlp.predict(X_te)

    print "Correct Classification:", np.sum( np.equal( prediction, Y_te) ) / float(Y_te.shape[0])


if __name__ == '__main__':
    # paths on notebook
    #path_feats  = "/home/consti/Work/data_master/cache/cached_datasets/pedunculus/features/ffeats/ffeat_bert_0_True.h5"
    #path_labels = "/home/consti/Work/data_master/cache/cached_datasets/pedunculus/gt_face_segid0.h5"

    # paths  on henharrier
    path_feats  = "/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/features/ffeats/ffeat_bert_1_True.h5"
    path_labels = "/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/gt_face_segid1.h5"

    key_feats   = "data"
    key_labels  = "gt_face"

    X = np.nan_to_num( np.array( vigra.readHDF5(path_feats, key_feats) ) )
    Y = np.squeeze( vigra.readHDF5(path_labels, key_labels) )

    N = X.shape[0]
    assert N == Y.shape[0]

    X = scale_features(X)

    mlp_cv(X,Y)

