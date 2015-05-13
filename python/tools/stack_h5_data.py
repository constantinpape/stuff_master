import numpy as np
import h5py
import vigra

# stack the HDF5 data by adding the layers in a new file
# TODO fix magic numbers....
def stack_h5_data_memory(data0, data1, out_path):

    # squeeze the data
    data0 = np.squeeze(data0)
    data1 = np.squeeze(data1)

    #assert that data has the same shape
    assert( data0.shape[0:3] == data1.shape[0:3] )

    # normalize the data
    data0 /= ( np.max(data0) - np.min(data0) )
    data1 /= ( np.max(data1) - np.min(data1) )

    # magic layer numbers
    layers0 = 1
    layers1 = 6

    new_shape = (data0.shape[0],data0.shape[1],data0.shape[2], layers0 + layers1)

    # have to do it htis way due to limited main memory
    f_out = h5py.File(out_path,"w")
    dset  = f_out.create_dataset("exported_data", new_shape, dtype = 'f', chunks = True )

    dset[:,:,:,0] = data0
    for layer in range(layers1):
	dset[:,:,:,1+layer] = data1[:,:,:,layer]
	print layer

def stack_h5_data(data0, data1, out_path):
    # squeeze the data
    data0 = np.squeeze(data0)
    data1 = np.squeeze(data1)
    data0 = np.expand_dims(data0, axis = 3)

    #assert that data has the same shape
    assert( data0.shape[0:3] == data1.shape[0:3] )

    # normalize the data
    data0 /= ( np.max(data0) - np.min(data0) )
    data1 /= ( np.max(data1) - np.min(data1) )

    data_new = np.concatenate( (data0,data1), axis = 3 )

    f_out = h5py.File(out_path,"w")
    dset  = f_out.create_dataset("exported_data", data = data_new, chunks = True )



if __name__ == '__main__':
    # TODO add argparsing
    #in0  = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub.h5"
    #key0 = "data"
    #in1  = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_probs_thorsten.h5"
    #key1 = "exported_data"
    #out  = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_combined.h5"

    in0  = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30.h5"
    key0 = "data"
    in1  = "/home/constantin/Work/data_ssd/data_080515/pedunculus/semantic_labeling_probs.h5"
    key1 = "exported_data"
    out  = "/home/constantin/Work/data_ssd/data_080515/pedunculus/combined.h5"

    data0 = vigra.readHDF5(in0, key0 ).astype(np.float32)
    data1 = vigra.readHDF5(in1, key1 ).astype(np.float32)

    stack_h5_data(data0, data1, out)
