import numpy as np
import vigra
import os

from volumina_viewer import volumina_n_layer

def read_in_dojo(path):
    nums = []
    for f in os.listdir(path):
        nums.append( int(f[2:]) )

    nums = np.sort(nums)

    gt = np.zeros( ( 1024, 1024, nums.shape[0] ) )
    for num in nums:
        string = "z=" + str(num).zfill(8)
        num_path = os.path.join(path, string)
        file_names = os.listdir( num_path  )
        for f in file_names:

            file_path = os.path.join(num_path, f)
            dat = vigra.readHDF5(file_path, "dataset_1")
            split =  f.split(",")
            y_coord = int( split[0][-1] )
            x_coord = int( split[1].split(".")[0][-1] )

            if x_coord == 0 and y_coord == 0:
                gt[:512,:512,num] = dat.transpose()
            elif x_coord == 0 and y_coord == 1:
                gt[:512:,512:,num] = dat.transpose()
            elif x_coord == 1 and y_coord == 0:
                gt[512:,:512,num] = dat.transpose()
            elif x_coord == 1 and y_coord == 1:
                gt[512:,512:,num] = dat.transpose()
            else:
                print "Somethingswrong with the coordinates:"
                print f
                print x_coord, y_coord

    return gt


if __name__ == '__main__':
    #path = "/tmp/tmpwvtDlG/ids/tiles/w=00000001"
    path = "/tmp/tmpzfO7mI/ids/tiles/w=00000000"

    gt_corrected = read_in_dojo(path)

    gt_old_path = "/home/constantin/Work/data_ssd/data_110915/sopnet_comparison/gt_stack2.h5"

    raw_path    = "/home/constantin/Work/data_ssd/data_110915/sopnet_comparison/raw_stack2_norm.h5"

    #raw = vigra.readHDF5(raw_path, "data")
    #gt_old = vigra.readHDF5(gt_old_path, "gt")

    #volumina_n_layer( [raw, gt_old.astype(np.uint32), gt_corrected.astype(np.uint32)])

    save_path = "/home/constantin/Work/data_ssd/data_110915/sopnet_comparison/gt_stack2_corrected.h5"

    vigra.writeHDF5(gt_corrected, save_path, "gt")
