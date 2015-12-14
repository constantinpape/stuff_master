import numpy as np
import vigra
import os

from volumina_viewer import volumina_n_layer, volumina_single_layer

# comput the standatd ilastik features in 2d,
# however dont compte the diff of gaussians, because it is more or less the same than laplacian of gaussian
# TODO check if feature are already present
def compute_ilastik_2dfeatures(
        raw,
        save_path,
        sigmas = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0) ):

    assert len(raw.shape) == 3
    #raw = raw.astype(np.float32)

    feature_list = ( "gaussianSmoothing",
                   "laplacianOfGaussian",
                   "gaussianGradientMagnitude",
                   "structureTensorEigenvalues",
                   "hessianOfGaussianEigenvalues")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    list_path    = os.path.join(save_path, "feat_file_list.txt")
    if os.path.exists(list_path):
        feature_file = open(list_path, 'a')
    else:
        feature_file = open(list_path, 'w')

    for feat in feature_list:

        print "Computing", feat

        for sig in sigmas:

            # for sig = 0.3 only compute gaussian smoothing
            if np.isclose( sig, 0.3 ) and feat != "gaussianSmoothing":
                continue

            print "for sigma =", sig

            if feat == "structureTensorEigenvalues" or feat == "hessianOfGaussianEigenvalues":
                feat_array = np.zeros( ( raw.shape[0], raw.shape[1], raw.shape[2], 2 ) )
            else:
                feat_array = np.zeros( ( raw.shape[0], raw.shape[1], raw.shape[2] ) )

            for z in range(raw.shape[2]):

                # for the structure tensor we need two scales
                # do outer scale = 3 * inner scale for now
                # TODO Thorsten does 2 loops, one for outer = 2 * inner, one for outer = 4 * inner
                # probably should do this too
                if feat == "structureTensorEigenvalues":
                    eval_str = "vigra.filters." + feat + "(raw[:,:,z], sig, 3*sig)"

                else:
                    eval_str = "vigra.filters." + feat + "(raw[:,:,z], sig)"

                feat_array[:,:,z] = eval(eval_str)

            assert len(feat_array.shape) <= 4

            if len(feat_array.shape) == 4:
                for chan in range(feat_array.shape[3]):
                    path = os.path.join(save_path, feat + "_" + str(sig) + "_channel_" + str(chan)  + ".h5")
                    vigra.writeHDF5(feat_array[:,:,:,chan].astype(np.float32), path, "data")
                    feature_file.write( path + '\n' )

            else:
                path = os.path.join(save_path, feat + "_" + str(sig) + ".h5")
                vigra.writeHDF5(feat_array.astype(np.float32), path, "data")
                feature_file.write( path + '\n' )

# compute nonlinear diffusion filter
# TODO understand this filter and look for good parameter values
def compute_nonlinear_diffusion(raw, save_path, edge_threshold, scale):

    print "Computing NonLinearDiffusion"

    list_path    = save_path + "feats"
    if os.path.exists(list_path):
        feature_file = open(list_path, 'a')
    else:
        feature_file = open(list_path, 'w')

    feat_array = np.zeros(raw.shape)

    for z in range(raw.shape[2]):
        feat_array[:,:,z] = vigra.filters.nonlinearDiffusion(raw[:,:,z], edge_threshold, scale)

    path = save_path + "NonLinearDiffusion" + ".h5"
    vigra.writeHDF5(feat_array.astype(np.float32), path, "data")
    #feature_file.write( path + '\n' )


# compute disc rank order filter
# TODO understand this filter and look for good parameter values
def compute_disc_rank_order(raw, save_path):

    print "Computing DiscRankOrder"

    list_path    = save_path + "feats"
    if os.path.exists(list_path):
        feature_file = open(list_path, 'a')
    else:
        feature_file = open(list_path, 'w')

    feat_array = np.zeros(raw.shape)

    for z in range(raw.shape[2]):
        feat_array[:,:,z] = vigra.filters.discRankOrderFilter(raw[:,:,z])

    path = save_path + "DiscRankOrder" + ".h5"
    vigra.writeHDF5(feat_array.astype(np.float32), path, "data")
    #feature_file.write( path + '\n' )


def compute_features_isbi2012():
    raw_path = "/home/constantin/Work/data_ssd/data_090615/isbi2012/train-volume.h5"
    raw_key = "data"

    raw = vigra.readHDF5(raw_path, raw_key)

    sigmas = ( 0.3, 0.7, 1.0, 1.6, 3.5, 5.0 )

    feature_path = "/home/constantin/Work/data_ssd/data_090615/isbi2012/features/train-"

    compute_ilastik_2dfeatures(raw, feature_path, sigmas)
    #compute_nonlinear_diffusion(raw, feature_path, 0.2, 10.0)


if __name__ == '__main__':
    compute_features_isbi2012()

