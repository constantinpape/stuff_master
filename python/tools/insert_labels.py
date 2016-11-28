import numpy as np
import h5py
import vigra

def slicingToString(slicing):
    """Convert the given slicing into a string of the form
    '[0:1,2:3,4:5]'

    """
    strSlicing = '['
    for s in slicing:
        strSlicing += str(s.start)
        strSlicing += ':'
        strSlicing += str(s.stop)
        strSlicing += ','

    strSlicing = strSlicing[:-1] # Drop the last comma
    strSlicing += ']'
    return strSlicing


# the new file should already exist and have the labels added, but not drawn (if drawn they will be deleted)

def transfer(labelfile, new_project_filepath):
    labelimage = vigra.readHDF5(labelfile, "exported_data")
    if len(labelimage.shape) != 4:
        labelimage = np.expand_dims( labelimage, axis = 3)
    #labelbinary = (labelimage>0).astype(np.uint8)

    with h5py.File(new_project_filepath, 'a') as project_file:
    	if 'PixelClassification/LabelSets' in project_file:
    		# start from scratch: delete all previous labels
    		del project_file['PixelClassification/LabelSets']
    	labelset_group = project_file.create_group('PixelClassification/LabelSets')

    	print "big block shape", labelimage.shape
    	label_group_name = 'labels000'
    	label_group = labelset_group.create_group(label_group_name)
    	dataset = label_group.create_dataset( "block0000", data = labelimage)
    	print dataset.shape

    	slicing = [slice(0, stop) for stop in dataset.shape]
    	print slicing, slicingToString(slicing), np.sum(dataset)
    	dataset.attrs["blockSlice"] = slicingToString(slicing)


if __name__ == '__main__':
    labelfile        = "/home/consti/Work/Neurocut/more_isbi_exp/binary_labs_squeezed.h5"
    new_project_file = "/home/consti/Work/Neurocut/more_isbi_exp/ilastik_projects/binary_labeling_on_raw.ilp"
    transfer(labelfile, new_project_file)
