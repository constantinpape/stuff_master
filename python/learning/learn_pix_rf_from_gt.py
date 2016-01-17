import numpy as np
import vigra

from sklearn.ensemble import RandomForestClassifier as rf

from volumina_viewer import volumina_n_layer

def learn_rf(dat_train, lbl_train, estimators = 255, min_samples_leaf = 1):
    rf_pix = rf(n_estimators = estimators, min_samples_leaf = min_samples_leaf, n_jobs = -1)
    rf_pix.fit(
            dat_train.astype(np.float32),
            lbl_train.astype(np.uint8  )
            )
    return rf_pix

def predict_probability_map(rf_pix, dat_test):
    prob_map = rf_pix.predict_proba(dat_test)
    return prob_map


if __name__ == '__main__':

    raw_train_p = "/home/constantin/Work/data_ssd/data_010915/INI/data_norm.h5"
    raw_train_k = "data"

    raw_train = vigra.readHDF5(raw_train_p, raw_train_k)

    #feat_train_p =
    #feat_train_k =

    lbl_train_p = "/home/constantin/Work/data_ssd/data_010915/INI/membrane_labels.h5"
    lbl_train_k = "data"

    lbl_train = vigra.readHDF5(lbl_train_p, lbl_train_k).astype(np.uint32)

    #print np.unique( lbl_train )
    #print lbl_train[340,398,0]

    mem_label = 168

    lbl_train[np.where(lbl_train != mem_label)] = 0

    #volumina_n_layer([raw_train, lbl_train])

    dat_train = raw_train.flatten()
    lbl       = lbl_train.flatten()

    dat_train = np.expand_dims(dat_train, axis = 1)

    print dat_train.shape
    print lbl.shape

    rf_pix    = learn_rf( dat_train, lbl )
    probs = predict_probability_map(rf_pix, dat_train)

    probs = probs[:,0]
    probs = probs.reshape( (512,512,30) )

    vigra.writeHDF5(probs, "probs.h5", "probs")

    volumina_n_layer([raw_train, probs, lbl_train])
