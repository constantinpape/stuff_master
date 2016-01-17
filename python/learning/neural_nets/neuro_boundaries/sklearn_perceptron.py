import vigra
import numpy as np

from sklearn.neural_network import MLPClassifier

def train_mlp(X, Y, nn_params = None):

    print "Start Learning Net"

    clf = MLPClassifier( algorithm = 'l-bfgs',
            alpha = 1e-5,
            hidden_layer_sizes = (500,400,200,20,2),
            random_state = 1)
    clf.fit(X,Y)

    return clf

def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

if __name__ == '__main__':
    path_feats  = "/home/consti/Work/data_master/cache/cached_datasets/pedunculus/features/ffeats/ffeat_bert_0_True.h5"
    key_feats   = "data"
    path_labels = "/home/consti/Work/data_master/cache/cached_datasets/pedunculus/gt_face_segid0.h5"

    key_labels  = "gt_face"

    X = np.nan_to_num( np.array( vigra.readHDF5(path_feats, key_feats) ) )
    Y = np.squeeze( vigra.readHDF5(path_labels, key_labels) )

    N = X.shape[0]
    assert N == Y.shape[0]

    X = scale_features(X)

    # make train / test split

    indices = np.random.permutation(N)
    split = int(0.6*N)

    X_tr = X[:split]
    Y_tr = Y[:split]
    X_te = X[split:]
    Y_te = Y[split:]

    mlp = train_mlp(X_tr, Y_tr)
    prediction = mlp.predict(X_te)

    print "Correct Classification:", np.sum(prediction == Y_te) / float(Y_te.shape[0])

