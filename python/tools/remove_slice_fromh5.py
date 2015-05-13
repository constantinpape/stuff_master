import sys
import vigra
import numpy as np
if __name__ == '__main__':

    path_in = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30.h5"
    path_out = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_sliced.h5"
    key = "data"

    data_in = vigra.readHDF5(path_in,key)

    slice_to_remove = 6
    data_in = np.delete(data_in, slice_to_remove, axis = 2)

    vigra.writeHDF5(data_in, path_out, key)
