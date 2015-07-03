import numpy as np
import vigra

#path_raw = "/home/constantin/Work/data_ssd/data_080515/pedunculus/150401pedunculus_middle_512x512_first30_sliced.h5"
#key_raw  = "data"
#
#path_seg = "/home/constantin/Work/data_ssd/data_080515/pedunculus/res_mc/post_processed/mc_2d_pmem_seg.npy"
##path_seg = "/home/constantin/Work/data_ssd/data_080515/pedunculus/res_mc/post_processed/mc_2d_pmem_mitoon.npy"
#
#path_gt = "/home/constantin/Work/data_ssd/data_080515/pedunculus/gt_mc.h5"
#key_gt  = "gt"
#
#raw = vigra.readHDF5(path_raw, key_raw)
#seg = np.load(path_seg)
#gt = vigra.readHDF5(path_gt, key_gt)
#
#assert raw.shape == seg.shape
#assert gt.shape == seg.shape
#
#shape = list(gt.shape)
#shape.append(3)
#shape = tuple( shape )
##print shape
#
##vol_write = np.zeros( shape )
#
##vol_write[:,:,:,0] = raw
##vol_write[:,:,:,1] = seg
##vol_write[:,:,:,2] = gt
#
##assert vol_write.shape == shape
#
#print np.unique(seg).size
#print seg.max()
#
#print "Making the ids sequential!"
## make the ids sequential
#i = 1
#for id in np.unique(seg):
#    if id != i:
#        seg[np.where(seg==id)] = i
#    i += 1
#
#
##seg = seg.astype( np.float32 )
##seg /= seg.max()
##seg *= 255.0
#
#print np.unique(seg).size
#print seg.max()
#
#save_path = "/home/constantin/Work/data_ssd/data_080515/pedunculus/res_mc/post_processed/final_mitooff.tif"
#
#vigra.impex.writeVolume( seg, save_path, ''  )

path = "/home/constantin/Desktop/results_for_fred/export_Z=00.png"

#
#for i in range(29):
#    path_im = path + str(i).zfill(2) + ".png"
#    im = vigra.impex.readImage(path_im)
#    vol_save[:,:,:,i] = im.astype(np.uint8)

vol_save = vigra.readVolume(path)

print vol_save.shape

save_path = "/home/constantin/Desktop/results_for_fred/result_mitooff.tif"

vigra.impex.writeVolume( vol_save, save_path, '', dtype = 'UINT16', compression = '' )

