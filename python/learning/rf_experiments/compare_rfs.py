import numpy as np
import vigra
from precompute_features import compute_ilastik_2dfeatures
import os
import cPickle as pickle

from sklearn.ensemble import RandomForestClassifier

def load_precomputed_feats(save_path, shape):

    assert len(shape) == 3

    file_names = []
    with open( os.path.join(save_path, "feat_file_list.txt"), "r" ) as files:
        for name in files:
            file_names.append(name[:-1])
    n_feats = len( file_names )
    feats = np.zeros( (shape[0], shape[1], shape[2], n_feats) )

    for i in range(n_feats):
        feat_path = file_names[i]
        feats[:,:,:,i] = vigra.readHDF5(feat_path, "data")

    return feats

def load_feats_and_gt_pedunculus():

    #raw_data = vigra.readHDF5(
    #        "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_sliced.h5",
    #        "data"
    #        )

    gt = vigra.readVolume(
            "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401_pedunculus_membrane_labeling.tif")
    gt = np.squeeze(gt)

    # delete black slice
    gt = np.delete(gt, 6, axis = 2)

    gt[gt == 0.] = 1
    gt[gt == 255.] = 0
    gt = gt.astype(np.uint32)

    save_path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/features"
    #compute_ilastik_2dfeatures(raw_data, save_path)

    feats_path = os.path.join( save_path, "all_features.h5")
    # make sure that features are computed!
    #feats = load_precomputed_feats(save_path, raw_data.shape)
    #vigra.writeHDF5(feats, feats_path, "data")

    feats = vigra.readHDF5(feats_path, "data")

    return (feats, gt)

def eval_pixerror(pred, gt, thresh = 0.5):
    assert pred.shape == gt.shape, str(pred.shape) + " , " + str(gt.shape)
    pred = np.array(pred)
    gt   = np.array(gt)
    correct = pred == gt
    return np.sum(correct) / float(gt.size)

def eval_rf_from_gt(X_tr, Y_tr, X_te, Y_te):
    print "Learning RF on GT"
    #rf = RandomForestClassifier(n_estimators = 15, n_jobs = 6)
    # TODO think about balancing the labels for training!
    #rf.fit(X_tr.reshape( (X_tr.shape[0] * X_tr.shape[1] * X_tr.shape[2], X_tr.shape[3] ) ), Y_tr.ravel() )
    ## pickle
    #with open("./../models/rf_gt.pkl", 'w') as f:
    #    pickle.dump(rf, f)
    with open("./../models/rf_gt.pkl", 'r') as f:
        rf = pickle.load(f)

    print "Predicting"
    pmap_train = rf.predict( X_tr.reshape( (X_tr.shape[0] * X_tr.shape[1] * X_tr.shape[2], X_tr.shape[3] ) ) )
    pmap_test = rf.predict(  X_te.reshape( (X_te.shape[0] * X_te.shape[1] * X_te.shape[2], X_te.shape[3] ) ) )

    # evaluate
    # TODO segementation quality measures after thresholding: RI, VI
    #pmap_train = pmap_train.reshape( X_tr.shape[0], X_tr.shape[1], X_tr.shape[2] )
    #pmap_test  = pmap_test.reshape(  X_te.shape[0], X_te.shape[1], X_te.shape[2] )
    # FIXME something in the reshaping goes wrong!

    train_pix_acc = eval_pixerror(pmap_train, Y_tr.ravel())
    test_pix_acc  = eval_pixerror(pmap_test,  Y_te.ravel())

    print train_pix_acc, test_pix_acc



def eval_rf_from_ilastiklabels(X_tr, Y_tr, X_te, Y_te, labels_path):
    pass


def eval_implicit_rf_from(X_tr, Y_tr, X_te, Y_te, use_gt = True):
    pass


if __name__ == '__main__':

    (feats, gt) = load_feats_and_gt_pedunculus()

    # make train_test split
    X_tr  = feats[:,:,:15,:]
    Y_tr  = gt[:,:,:15]
    X_te  = feats[:,:,15:,:]
    Y_te  = gt[:,:,15:]

    eval_rf_from_gt(X_tr, Y_tr, X_te, Y_te)
