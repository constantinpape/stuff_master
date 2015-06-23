#from ccc3d.pyMulticut import pyMulticut
import opengm
import numpy as np

def run_mc_ccc3d():
    pass
    #mc = PyMulticut(oneB.T, twoB.T, faceProbs.astype(np.float64), g.maxLabel(3))
    #verbose=True
    #mc.setBias(bias)
    #mc.setMaxLocalIterations(0)
    #nThreads = 8
    #mc.setNumberOfOptimizerThreads(nThreads)
    #mc.setNumberOfPathSearchThreads(nThreads)
    #mc.setProcessNonLocalConstraints(True)
    #mc.setChordalityCheck(True)
    #mc.setAddEqualityConstraints(True)
    #mc.setOptimizerVerbose(verbose)
    #mc.setSelectSegmentsOfInterest(False)
    ##mc.setUseRepairHeuristic(True)
    #mc.setUseRepairHeuristic(False) #use CPLEX's internal heuristics, as we add user constraints!!!!
    #mc.setAbsoluteGap(1E-3)
    #mc.setRelativeGap(1E-3)
    #mc.setDoubleEndedPathSearch(True)
    #mc.optimize()

    #states = mc.currentStates()


def run_mc_opengm(segmentation, edges, energies):

    n_seg = np.max(segmentation)

    states = np.ones(n_seg)
    gm = opengm.gm(states)

    print "AAA"
    "pairwise"
    potts_shape = [ 2, 2]
    potts = opengm.pottsFunctions(potts_shape,
        np.array([0.0]),
        np.array(energies)
        )
    print "AAA"

    fids_p = gm.addFunctions(potts)
    gm.addFactors(fids_p, edges)
    gm_path = "/tmp/gm.h5"
    opengm.saveGm(gm, gm_path)
    print "AAA"

    "parameters"
    # wf = "(TTC)(MTC)(IC)(CC-IFD,TTC-I)" # default workflow
    #wf = "(IC)(TTC-I,CC-I)" # propper workflow
    # wf = "(TTC)(TTC,CC)" # lp relaxation
    param = opengm.InfParam()#workflow=wf)
    print "---inference---"
    print " starting time:", time.strftime("%H:%M:%S"), ";", time.strftime("%d/%m/%Y")
    print "..."
    inf = opengm.inference.Multicut(gm, parameter=param)
    inf.infer()
    print " end time:", time.strftime("%H:%M:%S"), ";", time.strftime("%d/%m/%Y")
    res_node = inf.arg()
    res_edge = inf.getEdgeLabeling()
    res_seg = inf.getSegmentation()
    print res_node.shape, np.unique(res_node)
    print res_edge.shape, np.unique(res_edge)
    print res_seg.shape, np.unique(res_seg)
    quit()
