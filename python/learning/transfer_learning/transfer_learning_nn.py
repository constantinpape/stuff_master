import vigra
import numpy as np
import cPickle as pickle

from sklearn.neural_network import MLPClassifier
from transfer_learning_experiments import get_source, get_target

def scale_features(X):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def train_on_source(X,Y):

    print "Start Learning Net on source"

    clf = MLPClassifier( algorithm = 'l-bfgs',
            alpha = 1e-5,
            hidden_layer_sizes = (500,2),
            random_state = 1,
            warm_start = 1,
            max_iter = 400)

    clf.fit(X,Y)
    #new_loss = 0
    #old_loss = 10000
    #for step in range(200):
    #    clf.fit(X,Y)
    #    new_loss = clf.loss_
    #    # stop training, if improvement is small
    #    improvement = abs(new_loss - old_loss)
    #    print "Step:", step, "Loss:", new_loss, "Improvement:", improvement
    #    if improvement < 1.e-5:
    #        print "Training converged!"
    #        break
    #    old_loss = new_loss
    print "Pretrained CLF on Source with num_iter:", clf.n_iter_
    return clf


def tune_on_target(clf, X, Y):
    print "Finetuune Net on target"

    clf.fit(X,Y)
    #new_loss = 0
    #old_loss = 10000
    #for step in range(200):
    #    clf.fit(X,Y)
    #    new_loss = clf.loss_
    #    # stop training, if improvement is small
    #    improvement = abs(new_loss - old_loss)
    #    print "Step:", step, "Loss:", new_loss, "Improvement:", improvement
    #    if improvement < 1.e-5:
    #        print "Training converged!"
    #        break
    #    old_loss = new_loss
    print "Refined MLP on Target with num_iter:", clf.n_iter_
    return clf


def get_target_baseline(X_t_tr, X_t_te, Y_t_tr, Y_t_te):
    mlp = MLPClassifier( algorithm = 'l-bfgs',
            alpha = 1e-5,
            hidden_layer_sizes = (500,2),
            random_state = 1,
            max_iter = 200)

    mlp.fit(X_t_tr,Y_t_tr)

    print "Training on target only with n_iter = ", mlp.n_iter_

    prediction = mlp.predict(X_t_te)

    accuracy = np.sum( np.equal(prediction, Y_t_te) ) / float( Y_t_te.shape[0] )
    return accuracy


def get_source_baseline(mlp, X_t_tr, X_t_te, Y_t_tr, Y_t_te):

    prediction = mlp.predict(X_t_te)

    accuracy = np.sum( np.equal(prediction, Y_t_te) ) / float( Y_t_te.shape[0] )
    return accuracy


def transfer_learning(mlp, X_t_tr, X_t_te, Y_t_tr, Y_t_te):

    #finetune on train
    mlp = tune_on_target(mlp, X_t_tr, Y_t_tr)

    prediction = mlp.predict(X_t_te)

    accuracy = np.sum( np.equal(prediction, Y_t_te) ) / float( Y_t_te.shape[0] )
    return accuracy


# TODO everything with CV
if __name__ == '__main__':
    # take care, labels for target should be from manual labeling, while labels for source should be gt labels
    (X_s, Y_s) = get_source("pedunculus")
    (X_t, Y_t) = get_target("sopnetcompare_train")

    # make train / tes_split on target
    split = int( 0.6*X_t.shape[0] )
    split_indices = np.random.permutation(X_t.shape[0])
    X_t_tr = X_t[split_indices][:split]
    Y_t_tr = Y_t[split_indices][:split]

    X_t_te = X_t[split_indices][split:]
    Y_t_te = Y_t[split_indices][split:]

    mlp = train_on_source(X_s,Y_s)

    acc_s  = get_source_baseline(mlp, X_t_tr, X_t_te, Y_t_tr, Y_t_te)
    acc_st = transfer_learning(mlp, X_t_tr, X_t_te, Y_t_tr, Y_t_te)
    acc_t  = get_target_baseline(X_t_tr, X_t_te, Y_t_tr, Y_t_te)

    print
    print "NN Transfer Learning results"
    print

    print "Source Baseline:", acc_s
    print "Target Baseline:", acc_t
    print "Transfer Learning:", acc_st
