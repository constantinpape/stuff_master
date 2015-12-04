import numpy as np
import vigra

from volumina_viewer import volumina_double_layer

def normalize(dat):
    dat = np.squeeze(dat)
    dat = np.array(dat)
    #dat = dat.transpose((1,0,2))
    means = np.mean(dat, axis = (0,1) )
    assert means.shape[0] == dat.shape[2]
    for z in range(dat.shape[2]):
        dat[:,:,z] -= means[z]
    dat -= dat.min()
    dat /= dat.max()
    return dat


if __name__ == '__main__':
    #f = "/home/constantin/raw_stack1.h5"
    f = "/media/constantin/4c03279b-1283-477d-a03e-440898f78d6f/constantin_projects/data/data_131115/Sample_B/raw_data/raw.h5"
    dat = vigra.readHDF5(f,"data")
    dat_norm = normalize(dat)

    #volumina_double_layer( dat, dat_norm )

    save_f = "/media/constantin/4c03279b-1283-477d-a03e-440898f78d6f/constantin_projects/data/data_131115/Sample_B/raw_data_norm.h5"
    vigra.writeHDF5(dat_norm, save_f, "data")
