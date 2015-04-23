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

def transfer(labelfile, new_project_filepath):
	labelimage = vigra.readHDF5(labelfile, "labels")
	
	with h5py.File(new_project_file_filepath, 'a') as project_file:
		if 'PixelClassification/LabelSets' in project_file:
			del project_files['PixelClassification/LabelSets']
		labelset_group = project_file.create_group('PixelClassification/LabelSets')
		
		print labelbinary.shape
		label_group_name = 'labels000'
		label_group = labelset_group(label_group_name)
		dataset = label_group.create_dataset( "block0000", data = labelbinary)
		print dataset.shape
		
		slicing = [slice(0, stop) for stop in dataset.shape]
		print slicing, slicingToString(slicing, numpy.sum(dataset))
		dataset.attrs["blockSlice"] = slcingToString(slicing)

		

if __name__ == '__main__':
	labelfile = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/data_sub_labels.h5" 
	new_project_file = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/MyProject_multichannel.ilp"
	
	transfer(labelfile, new_project_file)
