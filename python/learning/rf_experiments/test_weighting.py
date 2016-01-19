import numpy as np
import vigra
import cPickle as pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold

def test_rf_weighting(X,Y):
    res_dict = {}
    kf = KFold(X.shape[0], n_folds = 10)
    for weighting in (None, "balanced", "balanced_subsample"):
        print weighting
        rf = RandomForestClassifier(n_estimators = 255, n_jobs = 8, class_weight = weighting)
        test_accuracy = []
        for train, test in kf:
            rf.fit(X[train], Y[train])
            acc = np.sum( np.equal( rf.predict(X[test]), Y[test] ) ) / float(Y[test].shape[0])
            test_accuracy.append(acc)

        if weighting == None:
            res_dict["no_weighting"] = (np.mean(test_accuracy), np.std(test_accuracy) )
        else:
            res_dict[weighting] = (np.mean(test_accuracy), np.std(test_accuracy) )

    with open("./../results/res_rfcv_weighting.pkl", 'w') as f:
        pickle.dump(res_dict, f)



if __name__ == '__main__':
    # paths  on henharrier
    path_feats  = "/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/features/ffeats/ffeat_bert_1_True.h5"
    path_labels = "/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/gt_face_segid1.h5"

    key_feats   = "data"
    key_labels  = "gt_face"

    X = np.nan_to_num( np.array( vigra.readHDF5(path_feats, key_feats) ) )
    Y = np.squeeze( vigra.readHDF5(path_labels, key_labels) )

    test_rf_weighting(X,Y)
