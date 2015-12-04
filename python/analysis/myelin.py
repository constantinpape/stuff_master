import vigra
import numpy as np
import volumina_viewer


def get_myelin_cc(myelin):

    myelin_probs = myelin.copy()

    myelin_probs = myelin_probs[:,:,:,0]
    # thresholding for the train data
    #myelin_probs[myelin_probs < 0.55] = 0.
    # thresholding for the test data
    myelin_probs[myelin_probs < 0.45] = 0.
    myelin_probs[myelin_probs != 0.] = 1

    myelin_cc = np.zeros_like( myelin_probs )

    segs_per_slice = np.zeros(myelin.shape[2])
    for z in range(myelin.shape[2]):

        # slices with 2 myelin segments
        if z in range(19) or z is 32 or z in range(36,63):
            segs_per_slice[z] = 2
        # slices with 4 myelin segments
        elif z in range(33,36):
            segs_per_slice[z] = 3
        # default: 1 segment per slice
        else:
            segs_per_slice[z] = 1

    for z in range(myelin.shape[2]):

        print z

        myelin_cc_z = vigra.analysis.labelImageWithBackground( myelin_probs[:,:,z] )

        cc_ids = np.unique(myelin_cc_z)

        seg_sizes = np.zeros( cc_ids.shape[0] )

        for id in cc_ids[1:]:
            size = np.sum( myelin_cc_z == id )
            seg_sizes[id] = size

        max_segs = np.argsort( seg_sizes )[::-1]

        for i in range( int(segs_per_slice[z]) ):
            # for slice 62, continue, because there is a large segment due to wron staining
            if z is 62 and i is 0:
                continue
            seg = cc_ids[ max_segs[i] ]
            #print "Max Segments:"
            #print seg
            myelin_cc[:,:,z][ myelin_cc_z == seg ] = 1.

    myelin_cc_closed = np.zeros_like(myelin_cc)

    for z in range( myelin_cc.shape[2] ):
        myelin_cc_closed[:,:,z] = vigra.filters.discClosing( myelin_cc[:,:,z].astype(np.uint8), 12 )

    # routine for isbi train
    #for z in range(raw.shape[2]):

    #    myelin_cc_z = vigra.analysis.labelImageWithBackground( myelin_probs[:,:,z] )

    #    cc_ids = np.unique(myelin_cc_z)

    #    thresh = 750

    #    for id in cc_ids[1:]:
    #        size = np.sum( myelin_cc_z == id )
    #        #print size

    #        if size > thresh:
    #            myelin_cc[:,:,z][ myelin_cc_z == id ] = 1.

    #myelin_cc_closed = np.zeros_like(myelin_cc)

    #for z in range( myelin_cc.shape[2] ):

    #    # slices that need larger closing radius (for train data)
    #    if z in (74,78):
    #        print z, "WHOOOSA"
    #        myelin_cc_closed[:,:,z] = vigra.filters.discClosing( myelin_cc[:,:,z].astype(np.uint8), 13 )
    #    elif z in (88,89):
    #        print z, "WHOOOSA"
    #        myelin_cc_closed[:,:,z] = vigra.filters.discClosing( myelin_cc[:,:,z].astype(np.uint8), 18 )
    #    else:
    #        print z, "BLOBBBSA"
    #        myelin_cc_closed[:,:,z] = vigra.filters.discClosing( myelin_cc[:,:,z].astype(np.uint8), 4 )

    return myelin_cc_closed


def get_myelin_segments(myelin_cc, myelin):

    myelin_segments = np.zeros_like( myelin_cc )
    myelin_probs = myelin.copy()
    myelin_probs = myelin[:,:,:,0]

    myelin[ myelin_probs < 0.55 ] = 0.

    offset = 0

    for z in range( myelin_cc.shape[2] ):

        print z

        seedmap = np.zeros( (myelin_cc.shape[0], myelin_cc.shape[1]) )
        seedmap[ myelin_cc[:,:,z] == 0 ] = 1
        seedmap = vigra.analysis.labelImageWithBackground(seedmap.astype(np.uint32))

        hmap = vigra.filters.gaussianSmoothing( myelin_probs[:,:,z], 3. )

        ws, max_label = vigra.analysis.watershedsNew(
                hmap.astype(np.float32),
                neighborhood = 8,
                seeds = seedmap.astype(np.uint32)  )

        ws += offset

        # find too big segments (= background) and set them to zero
        thresh = myelin_cc.shape[0] * myelin_cc.shape[1] / 3
        for i in np.unique(ws):

            area = np.sum( ws == i )

            # check whether the corner (0,1023) is inside the seg, then it is bg (test)
            is_origin = False
            # only for the first 16 slides!
            if z in range(16):
                pix = np.where( ws == i)
                for p in range(pix[0].shape[0]):
                    if pix[0][p] == 0 and pix[1][p] == 1023:
                        is_origin = True
            if area > thresh or is_origin:
                print "bg_seg,", is_origin
                ws[ ws == i ] = 0

        offset = np.max(ws) + 1

        # Look, where original myelin segments are not covered
        additional_myelin = np.zeros_like( ws )
        additional_myelin[ np.logical_and( ws == 0, myelin_cc[:,:,z] == 1  ) ] = 1
        additional_myelin = vigra.filters.discOpening(additional_myelin.astype(np.uint8), 10)

        if np.sum( additional_myelin == 1) > 0:
            additional_cc = vigra.analysis.labelImageWithBackground(additional_myelin)
            for id in np.unique(additional_cc)[1:]:
                #if np.sum(additional_cc == id) > 2000:
                ws[additional_cc == id] = offset
                offset += 1

        myelin_segments[:,:,z] = ws

    return myelin_segments


def project_onto_superpix(superpix, myelin_segments):
    assert superpix.shape == myelin_segments.shape

    max_id = np.max(superpix)

    superpix_projected = np.zeros_like(superpix)

    for z in range(superpix.shape[2]):

        no_myelin = np.where( myelin_segments[:,:,z] == 0 )
        myelin    = np.where( myelin_segments[:,:,z] != 0 )

        superpix_projected[:,:,z][ no_myelin ] = superpix[:,:,z][ no_myelin]
        superpix_projected[:,:,z][myelin]      = myelin_segments[:,:,z][myelin] + max_id

    superpix_projected = vigra.analysis.labelVolume(superpix_projected.astype(np.uint32))
    return superpix_projected



if __name__ == '__main__':

    raw = vigra.readHDF5("/home/constantin/Work/data_ssd/data_150615/isbi2013/test-input.h5", "data")
    myelin = vigra.readHDF5("/home/constantin/Work/data_ssd/data_150615/isbi2013/pixel_probs/myelin_probs_test3.h5", "exported_data")
    myelin = np.array(myelin)

    #myelin_cc = get_myelin_cc(myelin)
    #vigra.writeHDF5(myelin_cc, "tmp.h5", "tmp")

    #volumina_viewer.volumina_n_layer( [raw, myelin, myelin_cc.astype(np.uint32)] )

    myelin_cc = vigra.readHDF5("tmp.h5", "tmp")

    #myelin_segments = get_myelin_segments(myelin_cc, myelin)

    #vigra.writeHDF5(myelin_segments, "my_segs.h5", "tmp")

    #volumina_viewer.volumina_n_layer( [raw, myelin_cc, myelin_segments.astype(np.uint32)] )

    myelin_segments = vigra.readHDF5("my_segs.h5", "tmp")

    superpix = vigra.readHDF5(
            "/home/constantin/Work/data_ssd/data_150615/isbi2013/superpixel/watershed-test_nn_dt.h5",
            "superpixel")

    superpix_projected = project_onto_superpix(superpix, myelin_segments)

    vigra.writeHDF5(
            superpix_projected,
            "/home/constantin/Work/data_ssd/data_150615/isbi2013/superpixel/myelin_test.h5",
            "superpixel")

    volumina_viewer.volumina_n_layer( [raw, superpix.astype(np.uint32), superpix_projected.astype(np.uint32)] )
