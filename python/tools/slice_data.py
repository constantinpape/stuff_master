import sys , h5py
import vigra
import numpy as np

if __name__ == '__main__':

	path_in = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub.h5"
	path_out = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_sliced.h5"
        key = "data"

        #path_in = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/superpixel/watershed_voxel.h5"
        #path_out = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/superpixel/watershed_voxel_sliced.h5"
        #key = "superpixel"

	data_in = vigra.readHDF5(path_in,key)
	f_out = h5py.File(path_out,"w")

	dset = f_out.create_dataset(key, (1000,1000,1000), dtype = 'f', chunks = True  )

	dset[:,:,:] = data_in[0:1000,0:1000,0:1000]


