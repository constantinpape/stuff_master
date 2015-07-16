import numpy as np
import vigra

from volumina_viewer import volumina_double_layer

def normalize(f):
    dat = vigra.impex.readVolume(f)
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
    f = "/home/constantin/Work/data_ssd/data_090615/isbi2012/test-volume.tif"
    dat_norm = normalize(f)

    #dat = to_dat(f)
    #volumina_double_layer( dat, dat_norm )

    save_f = "/home/constantin/Work/data_ssd/data_090615/isbi2012/test-volume.h5"
    vigra.writeHDF5(dat_norm, save_f, "data")
