import numpy as np
import ilp
import h5py

def predict_ilastik(in_path, out_folder, data_nr, ilastik_cmd):
	mILP = ilp.ILP(in_path, out_folder)
	mILP.predict_all_datasets(ilastik_cmd)

if __name__ == "__main__":
	in_path =  "/home/constantin/Work/data_ssd/data_090515/2x2x2nm/MyProject_multichannel.ilp"
	out_path = "/home/constantin/Work/data_ssd/data_090515/2x2x2nm"
	ilastik_cmd = "/home/constantin/Work/programs/ilastik/ilastik-1.1.5-Linux/run_ilastik.sh"
	predict_ilastik(in_path, out_path, 0, ilastik_cmd)

