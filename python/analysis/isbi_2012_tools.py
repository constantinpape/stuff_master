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

    #volumina_viewer.volumina_n_layer( [seg.astype(np.uint32), gt.astype(np.uint32)])
    #quit()

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


def compare_test_probs():
    path_raw = "/home/constantin/Work/data_hdd/data_090615/isbi2012/processed/test-volume.h5"
    key_raw  = "data"

    path_ilastik = "/home/constantin/Work/data_hdd/data_090615/isbi2012/processed/pixel_probabilities/test-probs_final.h5"
    key_ilastik  = "exported_data"

    path_nasim   = "/home/constantin/Work/data_hdd/data_090615/isbi2012/processed/external_probability_maps/nasims_probs/probs_test_nasim.h5"
    key_nasim = "data"

    path_unet = "/home/constantin/Work/data_hdd/data_090615/isbi2012/processed/external_probability_maps/u-net_probs/u-net_probs_test_original.h5"
    key_unet = "exported_data"

    raw = vigra.readHDF5(path_raw, key_raw)
    ila = vigra.readHDF5(path_ilastik, key_ilastik)
    nas = vigra.readHDF5(path_nasim, key_nasim)
    une = vigra.readHDF5(path_unet, key_unet)

    volumina_viewer.volumina_n_layer( [raw, ila, nas, une] )


#
def get_weight_map_train():
    labels_path = "/home/constantin/Work/data_hdd/data_090615/isbi2012/train-labels.tif"

    labels = np.squeeze( vigra.impex.readVolume(labels_path) )
    labels = np.array(labels)

    n_edge = np.sum(labels == 0)
    n_cell = np.sum(labels == 255)

    n_tot = float(n_edge + n_cell)
    assert n_tot == labels.size

    #class_prior = np.zeros_like(labels)

    # is this (inverse) weighting correct?
    #class_prior[labels == 0] = n_cell / n_tot
    #class_prior[labels == 0] = n_edge / n_tot

    weights_1 = np.zeros_like(labels)
    weights_2 = np.zeros_like(labels)
    weights_4 = np.zeros_like(labels)

    weights_1[labels == 0] = 10.
    weights_2[labels == 0] = 10.
    weights_4[labels == 0] = 10.

    for z in range(labels.shape[2]):
        weights_1[:,:,z] = vigra.filters.gaussianSmoothing(weights_1[:,:,z], 1.0)
        weights_2[:,:,z] = vigra.filters.gaussianSmoothing(weights_2[:,:,z], 2.0)
        weights_4[:,:,z] = vigra.filters.gaussianSmoothing(weights_4[:,:,z], 4.0)

    #weights_1 += class_prior
    #weights_2 += class_prior
    #weights_4 += class_prior

    #volumina_viewer.volumina_n_layer( [labels, class_prior, weights_1, weights_2, weights_4])
    #quit()

    vigra.writeHDF5(weights_1,
            "/home/constantin/Work/data_hdd/data_090615/isbi2012/processed/external_probability_maps/weight_maps/weights_sig1.h5",
            "weights")
    vigra.writeHDF5(weights_2,
            "/home/constantin/Work/data_hdd/data_090615/isbi2012/processed/external_probability_maps/weight_maps/weights_sig2.h5",
            "weights")
    vigra.writeHDF5(weights_4,
            "/home/constantin/Work/data_hdd/data_090615/isbi2012/processed/external_probability_maps/weight_maps/weights_sig4.h5",
            "weights")


if __name__ == '__main__':

    #gt_path_train   = "/home/constantin/Work/data_hdd/cache/cached_datasets/isbi2012_train_train/gt_reg_ignore.h5"
    #gt_path_val     = "/home/constantin/Work/data_hdd/cache/cached_datasets/isbi2012_train_val/gt_reg_ignore.h5"
    #gt_key      = "gt_reg_ignore"

    #res_path_train = "/home/constantin/Work/data_hdd/cache/cached_multicuts/isbi2012_train_train/multicut_from_manuallabels/807262cb65f6d6a7e57c086929861b7f.h5"
    #res_key_train = os.path.split(res_path_train)[1][:-3]

    #res_path_val = "/home/constantin/Work/data_hdd/cache/cached_multicuts/isbi2012_train_train/multicut_from_manuallabels/c4ebe69029de81c4083209a7792dbe19.h5"
    #res_key_val = os.path.split(res_path_val)[1][:-3]

    #print "Results: Train:"
    #eval_2d(res_path_train, res_key_train, gt_path_train, gt_key)
    #print "Results: Validation"
    #eval_2d(res_path_val, res_key_val, gt_path_val, gt_key)


    # compare result on training block to layerwise gt
    #seg_path_train = "/home/constantin/Work/data_hdd/cache/cached_multicuts/isbi2012_train/multicut_from_manuallabels/['isbi2012_train', 'isbi2012_train', '_home_constantin_Work_data_hdd_data_090615_isbi2012_processed_facelabels_isbi2012_train_labs_h5', '1453371148_74', 'ffeat_bert', '0', '0', 'True', 'z', '16_0', 'ccc3d', '0_001'][1].h5"
    #gt_path = "/home/constantin/Work/data_hdd/cache/cached_datasets/isbi2012_train/gt_reg_ignore.h5"
    #gt_key = "gt_reg_ignore"
    #seg_key_train = os.path.split(seg_path_train)[1][:-3]
    #eval_2d(seg_path_train, seg_key_train, gt_path, gt_key)

    ## make binary segmenation from mc seg for submission
    seg_path_test = "/home/constantin/Work/data_ssd/isbi2012_submits/submit3_segmentation.h5"
    seg_key_test = "submit"
    binary = seg_to_binary(seg_path_test, seg_key_test)
    vigra.writeHDF5(binary, "/home/constantin/Work/data_ssd/isbi2012_submits/submit3.h5", "submit")
