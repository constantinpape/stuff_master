import numpy as np
import vigra

def save_weights_as_h5():

    e_opengm = vigra.readHDF5("edge_weights_opengm.h5", "energy")

    #f = open("ccc3dprobs.txt", 'r')

    #e_ccc3d = []
    #for line in f:
    #    num = float(line[:-1])
    #    e_ccc3d.append(num)

    #e_ccc3d = np.array(e_ccc3d)

    e_ccc3d = np.loadtxt("ccc3dweights.txt")

    print "opengm:"
    print e_opengm.shape

    print "ccc3d:"
    print e_ccc3d.shape

    vigra.writeHDF5(e_ccc3d, "edge_weights_ccc3d.h5", "energy")


if __name__ == '__main__':
    save_weights_as_h5()
