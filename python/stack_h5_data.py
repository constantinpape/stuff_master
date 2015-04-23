import numpy as np
import h5py
import vigra
from volumina_double_layer import volumina_double_layer 

# stack the HDF5 data by adding the layers in a new file
# TODO fix magic numbers....
def stack_h5_data(data0, data1, out_path):
	
	#assert that data has the same shape
	assert( data0.shape[0:3] == data1.shape[0:3] )
	
	#layers0 = data0.shape[-1]
	#layers1 = data1.shape[-1]

	layers0 = 1
	layers1 = 6

	new_shape = (data0.shape[0],data0.shape[1],data0.shape[2], layers0 + layers1) 
	
	f_out = h5py.File(out_path,"w")
	dset  = f_out.create_dataset("exported_data", new_shape, dtype = 'f', chunks = True )
	print dset.shape

	for layer in range(layers0):
		dset[:,:,:,layer] = data0	
		print layer
	for layer in range(layers1):
		dset[:,:,:,layers0+layer] = data1[:,:,:,0,layer]	
		print layer

if __name__ == '__main__':
	# TODO add argparsing
	in0  = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub.h5" 
	key0 = "data"
	in1  = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_probs_thorsten.h5" 
	key1 =	"exported_data"
	out  = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_combined.h5"
	
	data0 = vigra.readHDF5(in0, key0 )
	data1 = vigra.readHDF5(in1, key1 )
	
	shape = data0.shape[0:3]

	#data0 = data0.reshape( (shape[0],shape[1],shape[2],1) )
	#data1 = data1.reshape( (shape[0],shape[1],shape[2],6) )

	#volumina_double_layer(data0, data1)

	stack_h5_data(data0, data1, out)
