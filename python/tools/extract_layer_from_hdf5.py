import numpy as np
import h5py
import vigra

def extract_layer(f_in, key_in, f_out, key_out, layer = 0):
    data = vigra.readHDF5(f_in, key_in)
    data_new = data[:,:,:,layer]
    out_file = h5py.File(f_in, 'w')
    out_file.create_dataset(key_out, data = data_new)

if __name__ == '__main__':
    f_in = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_combined_Probabilities_sliced.h5"
    key_in = "data"

    f_out = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_Probabilities_sliced.h5"
    key_out = "exported_data"
    extract_layer(f_in, key_in,f_out, key_out, 0)
