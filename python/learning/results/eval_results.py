import numpy as np
import cPickle as pickle

def eval_rf_weighting():
    path = "./res_rfcv_weighting.pkl"
    with open(path,'r') as f:
        res = pickle.load(f)
    for key in res.keys():
        print "Results for", key
        print "Test-accuracy:", res[key][0], "+-", res[key][1]

def eval_nn_solver():
    path = "./res_nncv_algo.pkl"
    with open(path,'r') as f:
        res = pickle.load(f)
    for key in res.keys():
        print "Results for", key
        print "Train-accuracy:", res[key][0], "+-", res[key][1]
        print "Test-accuracy:",  res[key][2], "+-", res[key][3]
        print "Train-time:",     res[key][4], "+-", res[key][5]

def eval_nn_architecture():
    path = "./res_nncv_architecture.pkl"
    with open(path,'r') as f:
        res = pickle.load(f)
    for key in res.keys():
        print "Results for", key
        print "Train-accuracy:", res[key][0], "+-", res[key][1]
        print "Test-accuracy:",  res[key][2], "+-", res[key][3]
        print "Train-time:",     res[key][4], "+-", res[key][5]


if __name__ == '__main__':
    eval_nn_architecture()
