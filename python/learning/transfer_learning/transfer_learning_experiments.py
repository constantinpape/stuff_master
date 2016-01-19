import numpy as np
import vigra

# load source feats and labels from gt labeling
def get_source(ds_str = "pedunculus"):
    assert ds_str in ("pedunculus",), ds_str # TODO more datasets!!!
    print "Loading Features and Labels for:", ds_str

    labelpath = '/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/gt_face_segid1.h5'
    ffeatpath = '/home/constantin/Work/data_hdd/cache/cached_datasets/pedunculus/features/ffeats/ffeat_bert_1_True.h5'

    feats  = np.nan_to_num( vigra.readHDF5(ffeatpath, 'data') )
    labels = np.squeeze( vigra.readHDF5(labelpath, 'gt_face') )

    assert feats.shape[0] == labels.shape[0]

    return (feats, labels)


# load target feats and labels from manual labeling
def get_target(ds_str = "sopnetcompare_train"):

    assert ds_str in ("sopnetcompare_train",), ds_str # TODO more datasets!!!
    print "Loading Features and Labels for:", ds_str

    labelpath = '/home/constantin/Work/data_ssd/data_110915/sopnet_comparison/facelabs/facelabs_mitooff.h5'
    ffeatpath = '/home/constantin/Work/data_hdd/cache/cached_datasets/sopnetcompare_train/features/ffeats/ffeat_bert_0_True.h5'

    feats = np.nan_to_num( vigra.readHDF5(ffeatpath, 'data') )

    import h5py
    lab_file = h5py.File(labelpath)
    key = lab_file.keys()[0]
    lab_file.close()

    labels = np.array( vigra.readHDF5(labelpath, key) )

    feats = feats[labels != 0.5]
    labels = labels[labels != 0.5]
    labels = labels[:,np.newaxis]

    assert all(np.unique(labels) == np.array([0, 1]))
    assert labels.shape[0] == feats.shape[0]

    labels = np.squeeze(labels)

    return (feats, labels)
