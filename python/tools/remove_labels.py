import numpy as np
import h5py
import vigra

def delete_labels( project_filepath ):

    with h5py.File(project_filepath, 'a') as project_file:
    	if 'PixelClassification/LabelSets' in project_file:
    		# start from scratch: delete all previous labels
    		del project_file['PixelClassification/LabelSets']
    	labelset_group = project_file.create_group('PixelClassification/LabelSets')

if __name__ == "__main__":
    project_file = "/home/constantin/Work/data_ssd/data_010915/INI/pixel_probs/binary_labeling.ilp"
    delete_labels(project_file)
