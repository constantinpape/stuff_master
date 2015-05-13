import sys
import vigra
import numpy as np

if __name__ == '__main__':
    path_in = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30.tif"
    data_in = vigra.impex.readVolume(path_in)
    print data_in.shape
    data_in = np.squeeze(data_in)


    file_out = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30.h5"
    label  = "data"
    vigra.writeHDF5(data_in, file_out, label)

