import numpy as np
import vigra
from wsDtSegmentation import wsDtSegmentation
import volumina_viewer

# make 2d superpixel based on watershed on DT
def superpix_2d(probs, thresh, sig_seeds, sig_weights):
    segmentation = np.zeros_like(probs)
    offset = 0
    for z in xrange(probs.shape[2]):
        wsdt = wsDtSegmentation(probs[:,:,z], thresh, 50, 75, sig_seeds, sig_weights, True)
        segmentation[:,:,z] = wsdt
        segmentation[:,:,z] += offset
        offset = np.max(segmentation)
    return segmentation


# make 3d superpixel based on watershed on DT
def superpix_3d(probs, thresh, sig_seeds, sig_weights):
    wsdt_seg = wsDtSegmentation(probs, thresh, 50, 75, sig_seeds, sig_weights, True)
    return wsdt_seg


# make 3d superpixel on interpolated probabilities
def superpix_interpol(probs, thresh, sig_seeds, sig_weights, aniso):
    probs_int = vigra.sampling.resize(probs,
            shape = (probs.shape[0], probs.shape[1], aniso * probs.shape[2]) )
    wsdt_seg = wsDtSegmentation(probs_int, thresh, 50, 75, sig_seeds, sig_weights, True)
    wsdt_seg = wsdt_seg[:,:,::aniso]
    assert wsdt_seg.shape == probs.shape
    return wsdt_seg

# make superpixel with anisotropic weights
def superpix_aniso(probs, thresh, sig_seeds, sig_weights, aniso):
    wsdt_seg = wsDtSegmentation(probs, thresh, 50, 75, sig_seeds, sig_weights, True, aniso_fac = aniso)
    return wsdt_seg

def fix_mess_isbi():
    prob_path = "/home/consti/Work/data_master/isbi2013/train-probs_nn.h5"
    probs = vigra.readHDF5(prob_path, "exported_data")
    probs = np.array(probs)
    probs = 1 - probs

    thresh      = 0.55
    sig_seeds   = 1.6
    sig_weights = 2.0

    seg_interpol = superpix_interpol(probs, thresh, sig_seeds, sig_weights, 5)

    save_interpol = "/home/consti/Work/data_master/isbi2013/seg/watershedinterpol_"
    save_interpol += str(thresh) + "_" + str(sig_seeds) + "_" + str(sig_weights) + ".h5"
    vigra.writeHDF5(seg_interpol,
            save_interpol,
            "superpixel")

    thresh      = 0.45
    sig_seeds   = 1.6
    sig_weights = 2.0

    seg_interpol = superpix_interpol(probs, thresh, sig_seeds, sig_weights, 5)

    save_interpol = "/home/consti/Work/data_master/isbi2013/seg/watershedinterpol_"
    save_interpol += str(thresh) + "_" + str(sig_seeds) + "_" + str(sig_weights) + ".h5"
    vigra.writeHDF5(seg_interpol,
            save_interpol,
            "superpixel")


if __name__ == '__main__':
    #prob_path = "/home/consti/Work/data_master/pedunculus/probs_final.h5"
    prob_path = "/home/consti/Work/data_master/isbi2013/train-probs_nn.h5"
    probs = vigra.readHDF5(prob_path, "exported_data")
    probs = np.array(probs)
    probs = 1 - probs

    thresh      = 0.55
    sig_seeds   = 1.6
    sig_weights = 2.0

    seg_2d = superpix_2d(probs, thresh, sig_seeds, sig_weights)
    print "2d done"

    save_2d = "/home/consti/Work/data_master/isbi2013/seg/watershed2d_" + str(thresh)
    save_2d += "_" + str(sig_seeds) + "_" + str(sig_weights) + ".h5"
    vigra.writeHDF5(seg_2d,
            save_2d,
            "superpixel")

    #seg_3d = superpix_3d(probs, thresh, sig_seeds, sig_weights)
    #print "3d done"

    #save_3d = "/home/consti/Work/data_master/isbi2013/seg/watershed3d_" + str(thresh)
    #save_3d += "_" + str(sig_seeds) + "_" + str(sig_weights) + ".h5"
    #vigra.writeHDF5(seg_3d,
    #        save_3d,
    #        "superpixel")

    seg_interpol = superpix_interpol(probs, thresh, sig_seeds, sig_weights, 5)
    print "interpol done"


    save_interpol = "/home/consti/Work/data_master/isbi2013/seg/watershedinterpol_"
    save_interpol += str(thresh) + "_" + str(sig_seeds) + "_" + str(sig_weights) + ".h5"
    vigra.writeHDF5(seg_interpol,
            save_interpol,
            "superpixel")
