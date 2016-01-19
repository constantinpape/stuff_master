import numpy as np
import vigra

from exp import exp
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from transfer_learning_experiments import get_source, get_target

# TODO Balancing and/or Weighting ?
# learn a random forest on concetenation of source and target domain
def naive_transfer(feats_source, labels_source, feats_target, labels_target):

    n_s = feats_source.shape[0]
    assert n_s == labels_source.shape[0]
    n_t = feats_target.shape[0]
    assert n_t == labels_target.shape[0]

    assert feats_target.shape[1] == feats_sorce.shape[1]

    print "Learning RF on", n_source, "labels from the source domain"
    print "and on", n_target, "labels from the source domain"

    # TODO introduce option for balancing here !

    # TODO order should not matter, right?!
    feats_combined = np.concatenate( [feats_source, feats_target] )
    labels_combined = np.concatenate( [labels_source, labels_target] )

    #  learn RF
    # TODO introduce option for weighting here
    rf = RandomForestClassifier(n_jobs = 8)
    rf.fit( feats_combined, labels_combined.ravel() )

    return rf


def eval_prediction(pred, gt):
    assert pred.shape == gt.shape, str(pred.shape) + " " + str(gt.shape)
    return np.sum( pred == gt ) / float(gt.shape[0])


# train a random forest on:
# concatentation of source + target
# source only
# target only
# and predict on target, evaluate with adapted 10fold CV (learn on 9 / 10 folds, predict on target domain that was not trained on)
def compare(
        labels_s,
        labels_t,
        feats_s,
        feats_t,
        use_weights = False,
        use_noise = False,
        use_balancing = False):


    print feats_s.shape, feats_t.shape

    # add noise to the source domain features
    if use_noise:
        noise_factor = 100.
        for j in range(feats_t.shape[1]):
            range_j = np.max(feats_s[:,j]) - np.min(feats_s[:,j])
            sig_j = max( range_j / noise_factor, 0.01 )
            feats_s[:,j] += np.random.normal(0.0, sig_j, size = feats_s.shape[0])

    # balance
    if use_balancing:
    # permute source features and labels
        perm_s = np.random.permutation(feats_s.shape[0])
        feats_s = feats_s[perm_s][:feats_t.shape[0]]
        labels_s = labels_s[perm_s][:feats_t.shape[0]]

    feats_st = np.concatenate([feats_s,  feats_t])
    labels_st = np.concatenate([labels_s, labels_t])

    acc_st = []
    acc_s  = []
    acc_t  = []

    # for finding the taret instances, that were not trained on
    indices_s = np.arange(feats_s.shape[0])
    indices_t = np.arange(feats_s.shape[0], feats_s.shape[0] + feats_t.shape[0])

    print "training on source + target"
    for _ in range(10):
        kf_st = KFold(feats_s.shape[0] + feats_t.shape[0], n_folds = 10, shuffle = True)
        for train, test in kf_st:
            not_trained_t = np.squeeze( np.intersect1d(test, indices_t) )
            print "Not trained on:", not_trained_t.shape[0], "/", labels_t.shape[0]
            rf = RandomForestClassifier(n_jobs = 8)
            if use_weights:
                # find instances in training fold, that belong to source and target
                trained_s = np.intersect1d(train, indices_s)
                trained_t = np.intersect1d(train, indices_t)
                n_s = trained_s.shape[0]
                n_t = trained_t.shape[0]
                assert n_s + n_t == train.shape[0]
                # TODO this should be done more efficient!!!
                weights = np.zeros(train.shape[0])
                for i in range(train.shape[0]):
                    train_index = train[i]
                    if train_index in trained_s:
                        weights[i] = float(n_t) / train.shape[0]
                    elif train_index in trained_t:
                        weights[i] = float(n_s) / train.shape[0]
                    else:
                        print "This shouldnt happen"
                        quit()
                rf.fit( feats_st[train], labels_st[train].ravel(), weights )
            else:
                rf.fit( feats_st[train], labels_st[train].ravel() )
            acc = eval_prediction( rf.predict(feats_st[not_trained_t]), labels_st[not_trained_t].ravel() )
            print acc
            acc_st.append( acc )


    #print "Training on source only"
    #for _ in range(10):
    #    kf_s  = KFold(feats_s.shape[0], n_folds = 10, shuffle = True)
    #    for train, test in kf_s:
    #        rf = RandomForestClassifier(n_jobs = 8)
    #        rf.fit( feats_s[train], labels_s[train].ravel() )
    #        acc = eval_prediction( rf.predict(feats_t), labels_t.ravel())
    #        print acc
    #        acc_s.append( eval_prediction( rf.predict(feats_t), labels_t.ravel()) )

    #print "Training on target only"
    #for _ in range(10):
    #    kf_t  = KFold(feats_t.shape[0], n_folds = 10, shuffle = True)
    #    for train, test in kf_t:
    #        rf = RandomForestClassifier(n_jobs = 8)
    #        rf.fit( feats_t[train], labels_t[train].ravel() )
    #        acc = eval_prediction( rf.predict(feats_t[test]), labels_t[test].ravel() )
    #        print acc
    #        acc_t.append( acc )

    print "Training on source + target, weighting =", use_weights, ":"
    print np.mean(acc_st), "+-", np.std(acc_st)

    print "Training on source only:"
    print np.mean(acc_s), "+-", np.std(acc_s)

    print "Training on target only:"
    print np.mean(acc_t), "+-", np.std(acc_t)


def transfer_rf_cv(
        labels_s,
        labels_t,
        feats_s,
        feats_t,
        use_weights = False,
        use_noise = False):

    assert use_weights or use_noise

    if use_noise:
        assert not use_weights
        optimize_over = (10., 20., 50., 75., 100., 150.)

    if use_weights:
        optimize_over = ( 2., 3., 5., 10., 15., 25.)

    acc_glob = []
    std_glob = []
    for opt in optimize_over:
        # add noise to the source domain features
        if use_noise:
            for j in range(feats_t.shape[1]):
                range_j = np.max(feats_s[:,j]) - np.min(feats_s[:,j])
                sig_j = max( range_j / opt, 0.001 )
                feats_s[:,j] += np.random.normal(0.0, sig_j, size = feats_s.shape[0])

        feats_st = np.concatenate([feats_s,  feats_t])
        labels_st = np.concatenate([labels_s, labels_t])

        acc_opt = []

        # for finding the taret instances, that were not trained on
        indices_s = np.arange(feats_s.shape[0])
        indices_t = np.arange(feats_s.shape[0], feats_s.shape[0] + feats_t.shape[0])

        kf_st = KFold(feats_s.shape[0] + feats_t.shape[0], n_folds = 10, shuffle = True)
        for train, test in kf_st:
            not_trained_t = np.squeeze( np.intersect1d(test, indices_t) )
            print "Not trained on:", not_trained_t.shape[0], "/", labels_t.shape[0]
            rf = RandomForestClassifier(n_jobs = 8)
            if use_weights:
                # find instances in training fold, that belong to source and target
                trained_s = np.intersect1d(train, indices_s)
                trained_t = np.intersect1d(train, indices_t)
                n_s = trained_s.shape[0]
                n_t = trained_t.shape[0]
                assert n_s + n_t == train.shape[0]
                # TODO this should be done more efficient!!!
                weights = np.zeros(train.shape[0])
                for i in range(train.shape[0]):
                    train_index = train[i]
                    if train_index in trained_s:
                        weights[i] = 1.
                    elif train_index in trained_t:
                        weights[i] = opt
                    else:
                        print "This shouldnt happen"
                        quit()
                rf.fit( feats_st[train], labels_st[train].ravel(), weights )
            else:
                rf.fit( feats_st[train], labels_st[train].ravel() )
            acc = eval_prediction( rf.predict(feats_st[not_trained_t]), labels_st[not_trained_t].ravel() )
            print acc
            acc_opt.append( acc )

        acc_glob.append( np.mean(acc_opt))
        std_glob.append( np.std(acc_opt) )

    print "CV-results for optimizing"
    if use_weights:
        print "weights"
    else:
        print "noise"
    print optimize_over
    print acc_glob
    print std_glob



if __name__ == '__main__':
    # take care, labels for target should be from manual labeling, while labels for source should be gt labels
    (X_s, Y_s) = get_source("pedunculus")
    (X_t, Y_t) = get_target("sopnetcompare_train")

    transfer_rf_cv(Y_s, Y_t, X_s, X_t, use_weights = False, use_noise = True)
