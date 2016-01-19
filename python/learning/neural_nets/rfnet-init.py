from __future__ import division
import caffe
import numpy as np

from pdb import set_trace


# init
caffe.set_mode_cpu()

rf_model = 'rfnet.prototxt'
rf_weights = 'rfnet.caffemodel' # where result is going to go
rf_net = caffe.Net(rf_model, caffe.TRAIN)

#rf_net.params["fc1"][0].data[...] =

#bias
#rf_net.params["fc1"][1].data[...] =


rf_net.save(rf_weights)
