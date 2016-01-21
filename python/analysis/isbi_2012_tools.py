import vigra
import numpy as np
import volumina_viewer
import os

from vigra import graphs


def seg_to_binary(seg_path, seg_key):
    seg = np.array( vigra.readHDF5(seg_path, seg_key), dtype = np.uint32 )
    print seg.shape
    binary = np.ones(seg.shape, dtype = np.uint8) * 255

    for z in range(seg.shape[2]):
        grid = graphs.gridGraph(seg.shape[0:2])
        rag  = graphs.regionAdjacencyGraph(grid, seg[:,:,z])
        for edge in rag.edgeIds():
            edge_loc = rag.edgeCoordinates(edge)

            edge_loc_dn = np.floor(edge_loc).astype(np.int)
            edge_loc_up = np.ceil(edge_loc).astype(np.int)

            edge_loc_dn = ([x for x in edge_loc_dn[:,0]],
                           [y for y in edge_loc_dn[:,1]],
                           [z for _ in range(edge_loc.shape[0])])
            edge_loc_up = ([x for x in edge_loc_up[:,0]],
                           [y for y in edge_loc_up[:,1]],
                           [z for _ in range(edge_loc.shape[0])])

            binary[edge_loc_dn] = 0
            binary[edge_loc_up] = 0

    #volumina_viewer.volumina_n_layer( [seg, binary.astype(np.float)] )
    return binary


def layer_gt(labels_path, gt_path, gt_key):
    labels = np.squeeze( vigra.impex.readVolume(labels_path) )
    labels = np.array(labels)
    gt_layerwise = np.zeros_like(labels, dtype = np.uint32)
    for z in range(labels.shape[2]):
        gt_layerwise[:,:,z] = vigra.analysis.labelImageWithBackground(labels[:,:,z])

    #volumina_viewer.volumina_n_layer( [gt_layerwise, labels] )
    vigra.writeHDF5(gt_layerwise, gt_path, gt_key)



# TODO implement the evaluation measures from the challenge
def eval_2d(seg_path, seg_key, gt_path, gt_key):
    seg = np.array( vigra.readHDF5(seg_path, seg_key) )
    gt  = np.array( vigra.readHDF5(gt_path, gt_key)   )

    assert seg.shape == gt.shape

    from skneuro.learning import randIndex
    from skneuro.learning import variationOfInformation
    from error_measures   import compute_fscore

    ris = []
    vis = []
    fss = []

    # evaluate layerwise
    for z in range(seg.shape[2]):
        gt_z_flat = gt[:,:,z].flatten()
        seg_z_flat = seg[:,:,z].flatten()
        # compute fscore, ri and vi
        fs = compute_fscore(         gt_z_flat.astype(np.uint32), seg_z_flat.astype(np.uint32) )[1]
        ri = randIndex(              gt_z_flat.astype(np.uint32), seg_z_flat.astype(np.uint32),  ignoreDefaultLabel = True)
        vi = variationOfInformation( gt_z_flat.astype(np.uint32), seg_z_flat.astype(np.uint32),  ignoreDefaultLabel = True)

        ris.append(ri)
        vis.append(vi)
        fss.append(fs)

    print "Mean RandIndex:", np.mean(ris), "Layerwise Deviation:", np.std(ris)
    print "Mean VoI:", np.mean(vis), "Layerwise Deviation:", np.std(vis)
    print "Mean F-Score:", np.mean(fss), "Layerwise Deviation:", np.std(fss)


if __name__ == '__main__':

    # make layerwise groundtruth segmentation from labels
    labels_path = "/home/constantin/Work/data_hdd/data_090615/isbi2012/train-labels.tif"
    gt_path     = "/home/constantin/Work/data_hdd/data_090615/isbi2012/processed/groundtruth/gt_layerwise.h5"
    gt_key      = "gt"
    #layer_gt(labels_path, gt_path, gt_key)

    # compare result on training block to layerwise gt
    seg_path_train = "/home/constantin/Work/data_hdd/cache/cached_multicuts/isbi2012_train/multicut_from_manuallabels/['isbi2012_train', 'isbi2012_train', '_home_constantin_Work_data_hdd_data_090615_isbi2012_processed_facelabels_isbi2012_train_labs_h5', '1453371148_74', 'ffeat_bert', '0', '0', 'True', 'z', '16_0', 'ccc3d', '0_001'][1].h5"
    seg_key_train = os.path.split(seg_path_train)[1][:-3]
    print seg_key_train
    eval_2d(seg_path_train, seg_key_train, gt_path, gt_key)

    # make binary segmenation from mc seg for submission
    seg_path_test = "/home/constantin/Work/data_hdd/cache/cached_multicuts/isbi2012_train/multicut_from_manuallabels/['isbi2012_train', 'isbi2012_test', '_home_constantin_Work_data_hdd_data_090615_isbi2012_processed_facelabels_isbi2012_train_labs_h5', '1453371148_74', 'ffeat_bert', '0', '0', 'True', 'z', '16_0', 'ccc3d', '0_001'][1].h5"
    seg_key_test = os.path.split(seg_path_test)[1][:-3]
    #binary = seg_to_binary(seg_path_test, seg_key_test)
    #vigra.writeHDF5(binary, "/home/constantin/Work/data_ssd/isbi2012_submits/submit1.h5", "submit1")
