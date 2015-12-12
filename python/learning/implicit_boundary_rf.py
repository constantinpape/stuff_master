from sklearn.ensemble import RandomForestClassifier
import numpy as np
import vigra


class implicit_boundary_rf(object):

    def __init__(self, feature_path, feature_key, init_labels_path, labels_key, gt_path = None, gt_key = None):
        # load the features and initial labels
        self.X = np.array( vigra.readHDF5(feature_path, feature_key)    )
        self.Y = np.array( vigra.readHDF5(init_labels_path, labels_key) )
        assert self.X.shape[0] == self.Y.shape[0] and self.X.shape[1] == self.Y.shape[1] and self.X.shape[2] == self.Y.shape[2]
        assert np.unique(self.Y).shape[0] == 3, "We should only have 2 classes + non_labeled"
        # get the volume dimensions
        assert len( self.X.shape ) == 4, "Data should have three spatial dimensions and feature channels"
        self.shape = self.X.shape
        # train the initial rf
        self.rf = RandomForestClassifier(n_estimators = 15, n_jobs = 4)
        labeled = self.Y != 0.5
        X_train = self.X[labeled]
        Y_train = self.Y[labeled]
        self.rf.fit( X_train.reshape( X_train.shape[0] * X_train.shape[1] * X_train.shape[2], self.shape[4]  ), Y_train.ravel() )
        # get the first probability map
        self.pmap = self.predict_proba( self.X )
        # check whether we use thresholding or external gt:
        self.use_thresholding = False
        if gt_path != None:
            assert gt_key != None
            self.ref_segmentation = np.array(vigra.readHDF5(gt_path, gt_key))
        else:
            self.use_thresholding = True

    # TODO
    def __find_minmaxpoint(self, start_point, end_point, z):
        pass


    def __get_segments_thresholding(self, threshold):
        binary = self.pmap
        binary[self.pmap >= threshold] = 0
        binary[self.pmap < threshold]  = 1
        self.ref_segmentation = vigra.analysis.labelVolumeWithBackground(binary)


    def __find_new_labels(self, n_labels):

        unlabeled = self.Y == 0.5

        if self.use_thresholding:
            __get_segments_thresholding( 0.5 )

        for i in range(n_labels):
            # sample points
            # iterate over z the slice (anisotropic data!)
            z = i % self.shape[2]
            # TODO better sampling heuristic ?!
            point_is_old = True
            while point_is_old:

                start_point = ( np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1]) )
                end_point   = ( np.random.randint(0, self.shape[0]), np.random.randint(0, self.shape[1]) )

                seg_start = self.ref_segmentation[start_point[0], start_point[1], z]
                seg_end   = self.ref_segmentation[end_point[0], end_point[1], z]

                # make sure that sampled points are not on ignore label
                if seg_start == 0 or seg_end == 0:
                    continue

                # find the new label point
                new_label_point = __find_minmaxpoint(start_point, end_point, z)

                # check whether this is a truly new label
                if unlabeled[new_label_point[0], new_label_point[1], z] == True:
                    point_is_old = False
                    if seg_start == seg_end:
                        new_label = 0
                    else:
                        new_label = 1

                    self.Y[new_label_point[0], new_label_point[1], z] = new_label


    def __fit_iteration(self):
        self.__find_new_labels(1000)
        X_train = X_train[labeled]
        Y_train = Y[labeled]
        self.rf.fit( X_train.reshape( X_train.shape[0] * X_train.shape[1] * X_train.shape[2], self.shape[4]  ), Y_train.ravel() )
        self.pmap = self.rf.predict_proba(self.X)


    #def fit(self):
    #    pass


    def predict_proba(self, test_X):
        assert len( test_X.shape ) == 4, "Data should have three spatial dimensions and feature channels"
        assert test_X.shape[3] == self.shape[3]
        pmap = rf.predict_proba( test_X.reshape( (test_X.shape[0] * test_X.shape[1] * test_X.shape[2], 2) ) )
        # make sure, that '1' is the boundary channel !
        pmap = pmap[:,1]
        return pmap.reshape( (test_X.shape[0], test_X.shape[1], test_X.shape[[2]) )
