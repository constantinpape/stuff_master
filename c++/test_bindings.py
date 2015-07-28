import numpy as np

import error_measures

error_measures.blob()

segA = np.array([1,1,2,3,4,0,5,7])
segB = np.array([1,3,2,19,4,0,5,6])

print error_measures.compute_fscore(segA.astype(np.uint32), segB.astype(np.uint32))
