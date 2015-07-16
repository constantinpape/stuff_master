import vigra
import numpy as np

import cPickle as pickle

from volumina_viewer import volumina_n_layer, volumina_single_layer
import matplotlib.pyplot as plot

from graph_class import my_graph

# preprocessing to prevent the superpixel in the grondtruth from spilling out
def preprocess_for_bgsmoothing_pedunculus( labeling ):
    labeling_copy = np.zeros( labeling.shape )
    labeling_copy[np.where(labeling == 0)] = 254.
    labeling_copy[np.where(labeling == 255)] = 0.

    labeling_smooth = vigra.gaussianSmoothing(labeling_copy.astype(np.float32) , sigma = (2.2,2.2,0.1) )
    labeling_smooth[labeling_smooth > 50] = 255
    labeling_smooth[labeling_smooth <= 50] = 0

    # smooth the edges, where superpixel are spilling out...
    labeling[405:,445:,3][np.where(labeling_smooth[405:,445:,3] == 0)] = 255
    labeling[405:,445:,3][np.where(labeling_smooth[405:,445:,3] == 255)] = 0

    labeling[450:,315:,6][np.where(labeling_smooth[450:,315:,6] == 0)] = 255
    labeling[450:,315:,6][np.where(labeling_smooth[450:,315:,6] == 255)] = 0

    labeling[390:410,470:,9][np.where(labeling_smooth[390:410,470:,9] == 0)] = 255
    labeling[390:410,470:,9][np.where(labeling_smooth[390:410,470:,9] == 255)] = 0

    labeling[415:,415:,10][np.where(labeling_smooth[415:,415:,10] == 0)] = 255
    labeling[415:,415:,10][np.where(labeling_smooth[415:,415:,10] == 255)] = 0

    labeling[370:,415:,11][np.where(labeling_smooth[370:,415:,11] == 0)] = 255
    labeling[370:,415:,11][np.where(labeling_smooth[370:,415:,11] == 255)] = 0

    labeling[:50,460:,11][np.where(labeling_smooth[:50,460:,11] == 0)] = 255
    labeling[:50,460:,11][np.where(labeling_smooth[:50,460:,11] == 255)] = 0

    labeling[180:400,470:,13][np.where(labeling_smooth[180:400,470:,13] == 0)] = 255
    labeling[180:400,470:,13][np.where(labeling_smooth[180:400,470:,13] == 255)] = 0

    labeling[475:,150:165,18][np.where(labeling_smooth[475:,150:165,18] == 0)] = 255
    labeling[475:,150:165,18][np.where(labeling_smooth[475:,150:165,18] == 255)] = 0

    labeling[395:430,300:360,21][np.where(labeling_smooth[395:430,300:360,21] == 0)] = 255
    labeling[395:430,300:360,21][np.where(labeling_smooth[395:430,300:360,21] == 255)] = 0

    labeling[480:,365:395,24][np.where(labeling_smooth[480:,365:395,24] == 0)] = 255
    labeling[480:,365:395,24][np.where(labeling_smooth[480:,365:395,24] == 255)] = 0

    labeling[140:160,485:,25][np.where(labeling_smooth[140:160,485:,25] == 0)] = 255
    labeling[140:160,485:,25][np.where(labeling_smooth[140:160,485:,25] == 255)] = 0

    labeling[470:,360:,27][np.where(labeling_smooth[470:,360:,27] == 0)] = 255
    labeling[470:,360:,27][np.where(labeling_smooth[470:,360:,27] == 255)] = 0

    labeling[460:,350:,28][np.where(labeling_smooth[460:,350:,28] == 0)] = 255
    labeling[460:,350:,28][np.where(labeling_smooth[460:,350:,28] == 255)] = 0

    return labeling


# smooth away the background in the gt labeling with watershed growing
def smooth_background( labeling ):

    offset = 0
    labeling_ret = np.zeros( labeling.shape )
    for z in range(labeling.shape[2]):
        Seeds = vigra.analysis.labelImageWithBackground(labeling[:,:,z])
        smoothed, maxRegionLabel = vigra.analysis.watersheds(
                labeling[:,:,z].astype(np.float32),
                    neighborhood = 8,
                    seeds = Seeds.astype(np.uint32) )
        smoothed = vigra.analysis.labelImage(smoothed)
        labeling_ret[:,:,z] =  smoothed
        labeling_ret[:,:,z] += offset
        offset = labeling_ret[:,:,z].max()

    return labeling_ret


# get connected components of the groundtruth
def get_gt_2d( labeling):
    gt = np.zeros(labeling.shape)
    offset = 0
    for z in range(labeling.shape[2]):
        gt[:,:,z] = vigra.analysis.labelImageWithBackground(labeling[:,:,z])
        gt[:,:,z][gt[:,:,z]!=0] += offset
        offset = gt[:,:,z].max()
    max_label = gt.max()
    offset = np.zeros(labeling.shape)
    for z in range(labeling.shape[2]):
        offset[:,:,z] = z*max_label
        print z*max_label
        gt[:,:,z] = vigra.analysis.labelImageWithBackground(labeling[:,:,z])
    gt[gt!=0] += offset[gt!=0]
    gt = vigra.analysis.labelVolumeWithBackground( gt.astype(np.uint32) )
    return gt


# reconstruct the connectivity of the ground truth from own graph-model
# FIXME does not work yet
def graph_to_gt(labeling, raw):

    #segmentation = get_gt_2d(labeling)
    segmentation = np.load( "tmp/seg.npy" )

    #graph = my_graph(segmentation, raw)
    #pickle.dump(graph, open("tmp/graph.pkl","wb"))
    #quit()

    graph = pickle.load( open("tmp/graph.pkl", "r") )
    print type(graph)

    energies = graph.get_edge_energies()
    connectivity = graph.get_connectivity(energies)

    #neurons = graph.get_neurons(connectivity)
    #gt = graph.nrns_to_segmentation(neurons)

    return gt

# gt for data_090515/pedunculus
def gt_pedunculus():
    labels_path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401_pedunculus_membrane_labeling.tif"
    raw_path    = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_sliced.h5"

    labels = vigra.readVolume(labels_path)

    labels = np.squeeze(labels)
    labels = np.delete(labels, 6, axis = 2)

    raw = vigra.readHDF5(raw_path, "data")

    labels = preprocess_for_bgsmoothing_pedunculus(labels)
    gt = smooth_background(labels)

    volumina_n_layer( (raw, labels, gt) )

    gt_path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/ground_truth_seg.h5"
    vigra.writeHDF5(gt, gt_path, "gt")

def gt_isbi2012():
    labels_path = "/home/constantin/Work/data_ssd/data_090615/isbi2012/train-labels.h5"
    raw_path    = "/home/constantin/Work/data_ssd/data_090615/isbi2012/train-volume.h5"

    labels      = vigra.readHDF5(labels_path, "labels")
    raw         = vigra.readHDF5(raw_path, "data")

    gt          = smooth_background(labels)

    #volumina_n_layer( (raw, labels, gt) )

    gt_path = "/home/constantin/Work/data_ssd/data_090615/isbi2012/groundtruth/ground_truth_seg.h5"
    vigra.writeHDF5(gt, gt_path, "gt")


if __name__ == '__main__':
    gt_isbi2012()
