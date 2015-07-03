import vigra
import numpy as np
from volumina_viewer import volumina_n_layer

# project background labels of the pixelwise labeling back to the groundtruth
def project_gt(pixwise_labels, gt_segmentation):
    assert pixwise_labels.shape == gt_segmentation.shape
    gt_segmentation[np.where(pixwise_labels == 0)] = 0
    return gt_segmentation

if __name__ == '__main__':
    labels_path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401_pedunculus_membrane_labeling.tif"
    gt_path     = "/home/constantin/Work/data_ssd/data_080515/pedunculus/gt_mc.h5"
    raw_path    = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_sliced.h5"

    labels = vigra.readVolume(labels_path)
    labels = np.squeeze(labels)
    labels = np.delete(labels, 6, axis = 2)

    gt = vigra.readHDF5(gt_path, "gt")
    raw = vigra.readHDF5(raw_path, "data")

    gt = project_gt(labels, gt)

    save_path     = "/home/constantin/Work/data_ssd/data_080515/pedunculus/gt_mc_bkg.h5"

    volumina_n_layer( (raw, gt, labels) )

    vigra.writeHDF5(gt, save_path, "gt")
