import numpy as np
import ilp
import h5py

def extract_labels(in_path, out_folder, data_nr):
    mILP = ilp.ILP(in_path, out_folder)
    # get the labels from the dataset
    blocks, block_slices = mILP.get_labels(data_nr)
    # get the shape of the dataset
    shape = mILP.get_data(data_nr).shape
    shape = ( shape[0] , shape[1], shape[2] )
    print shape
    # open new h5 file to store the extracted labels
    out_file = out_folder + "/2label_labels.h5"
    f_out = h5py.File(out_file,"w")
    dset  = f_out.create_dataset("labels", shape, dtype = 'i', chunks = True )
    assert( len(blocks) == len(block_slices) )
    # copy the extracted labes to the new dataset
    for b in range( len(blocks) ):
    	blk_slice = block_slices[b]
    	blk_slice = blk_slice.strip('[]')
    	blk_slice = blk_slice.split(',')
    	bounds 	  = []
    	for i in range( len(blk_slice) ):
    		numbers = blk_slice[i].split(':')
    		for n in numbers:
    			bounds.append(int(n))
        #print bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]
        #print blocks[b][:,:,:,0].transpose( (1,2,0) ).shape
        # BE AWARE OF TRANSPOSING!
        dset[  bounds[2]:bounds[3], bounds[4]:bounds[5], bounds[0]:bounds[1] ] = blocks[b][:,:,:,0].transpose( (1,2,0) )

if __name__ == '__main__':
    #in_path    = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/MyProject_sub.ilp"
    #out_folder = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm"

    in_path    = "/home/constantin/Work/data_ssd/data_080515/pedunculus/2label.ilp"
    out_folder = "/home/constantin/Work/data_ssd/data_080515/pedunculus"

    extract_labels(in_path, out_folder, 0)


