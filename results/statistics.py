import numpy as np

from results import results_crossvalidation_full_gt, read_res_dict


def average_over_feats(res_dict, feat_list):

    avg_2d_l = np.zeros( len( feat_list) )
    err_2d_l = np.zeros( len( feat_list) )
    avg_3d_l = np.zeros( len( feat_list) )
    err_3d_l = np.zeros( len( feat_list) )

    i = 0
    for feat in res_dict.keys():
        if feat in feat_list:
            avg_2d_l[i] = res_dict[feat][0]
            err_2d_l[i] = res_dict[feat][1]
            avg_3d_l[i] = res_dict[feat][2]
            err_3d_l[i] = res_dict[feat][3]

            i += 1

    assert i == len(feat_list), str(i) + " , " + str( len(feat_list) )


    avg_2d = np.mean(avg_2d_l)
    err_2d = 0.
    for x in err_2d_l:
        err_2d += x**2
    err_2d = np.sqrt(err_2d)

    avg_3d = np.mean(avg_3d_l)
    err_3d = 0.
    for x in err_3d_l:
        err_3d += x**2
    err_3d = np.sqrt(err_3d)

    return (avg_2d, err_2d, avg_3d, err_3d)


def eval_ds(ds, feat_list):

    res = results_crossvalidation_full_gt()[ds]

    (avg_2d, err_2d, avg_3d, err_3d) = average_over_feats(res, feat_list)

    print "Averaging crossvalidation results for ", ds
    print "over", feat_list
    print "2d - Accuracy:", avg_2d, "+-", err_2d
    print "3d - Accuracy:", avg_3d, "+-", err_3d


def plot_feat_curve(ds, feat_list, plot_3d = True):
    import matplotlib.pyplot as plt

    path = "./cross_validation/crossvalidationresults_curves_" + ds
    path_2d = path + "_True"

    if ds is not "sopnetcompare":
        path_3d = path + "_False"
    else:
        path_3d = None

    res_dict = read_res_dict(path_2d, path_3d)

    x = [10, 100, 1000, 10000, 50000, 75000, 100000]

    ax = plt.subplot()

    for feat in feat_list:
        res_2d = res_dict[feat + "_2d"]
        res_3d = res_dict[feat + "_3d"]

        acc_2d = res_2d[0]
        err_2d = res_2d[1]

        plt.errorbar(x, acc_2d, yerr=err_2d, label=feat + "_2d")
        if res_3d is not False and plot_3d:
            acc_3d = res_3d[0]
            err_3d = res_3d[1]
            plt.errorbar(x, acc_3d, yerr=err_3d, label=feat + "_3d")

    plt.legend(loc=4)

    ax.set_xscale("log", nonposx='clip')
    # ax.set_yscale("log", nonposy='clip')

    plt.xlabel('number of training examples')#, fontsize=18)
    plt.ylabel('accuracy')#, fontsize=16)

    plt.grid(True)

    #plt.savefig('/mnt/homes/nkrasows/phd/src/microtubuli/images/plots/stack2_acc_vs_n.png', dpi=1000)
    plt.show()


def plot_feat_curve_anisocompare(ds, feat_list_iso, feat_list_aniso):
    import matplotlib.pyplot as plt

    path_iso = "./cross_validation/crossvalidationresults_curves_" + ds + "_True"
    path_aniso = "./cross_validation/crossvalidationresults_curves_anisotropic" + ds + "_True"

    res_dict_iso = read_res_dict(path_iso)
    res_dict_aniso = read_res_dict(path_aniso)

    x1 = [10, 100, 1000, 10000, 50000, 75000, 100000]
    x2 = [10, 100, 1000, 10000, 50000]
    if ds == "sopnetcompare":
        x2 = [10, 100, 1000, 10000, 30000]

    ax = plt.subplot()

    for feat in feat_list_iso:
        res_iso    = res_dict_iso[feat + "_2d"]

        acc_2d = res_iso[0]
        err_2d = res_iso[1]

        plt.errorbar(x1, acc_2d, yerr=err_2d, label=feat + "_2d")

    for feat in feat_list_aniso:
        res_aniso    = res_dict_aniso[feat + "_2d"]

        acc_2d = res_aniso[0]
        err_2d = res_aniso[1]

        plt.errorbar(x2, acc_2d, yerr=err_2d, label=feat + "_2d")

    plt.legend(loc=4)

    ax.set_xscale("log", nonposx='clip')
    # ax.set_yscale("log", nonposy='clip')

    plt.xlabel('number of training examples')#, fontsize=18)
    plt.ylabel('accuracy')#, fontsize=16)

    plt.grid(True)

    plt.show()


def plot_feat_curve_anisofeats(ds, feat_list):
    import matplotlib.pyplot as plt

    path_2d           = "./cross_validation/crossvalidationresults_curves_" + ds + "_True"
    path_3d           = "./cross_validation/crossvalidationresults_curves_" + ds + "_False"
    path_aniso_2      = "./cross_validation/crossvalidationresults_curves_" + ds + "_aniso2"
    path_aniso_native = "./cross_validation/crossvalidationresults_curves_" + ds + "_aniso_native"

    if ds != "sopnetcompare":
        res_dict                = read_res_dict(path_2d, path_3d)
    else:
        res_dict                = read_res_dict(path_2d)
    res_dict_aniso2         = read_res_dict(path_aniso_2)
    res_dict_aniso_native   = read_res_dict(path_aniso_native)

    x1 = [10, 100, 1000, 10000, 50000, 75000, 100000]

    ax = plt.subplot()

    for feat in feat_list:

        res_2d = res_dict[feat + "_2d"]
        acc_2d = res_2d[0]
        err_2d = res_2d[1]
        plt.errorbar(x1, acc_2d, yerr = err_2d, label = feat + "_2d")

        if ds != "sopnetcompare":
            res_3d = res_dict[feat + "_3d"]
            acc_3d = res_3d[0]
            err_3d = res_3d[1]
            plt.errorbar(x1, acc_3d, yerr = err_3d, label = feat + "_3d")

        res_aniso2 = res_dict_aniso2[feat + "_2d"]
        acc_aniso2 = res_aniso2[0]
        err_aniso2 = res_aniso2[1]
        plt.errorbar(x1, acc_aniso2, yerr = err_aniso2, label = feat + "_aniso2")

        res_aniso_native = res_dict_aniso_native[feat + "_2d"]
        acc_aniso_native = res_aniso_native[0]
        err_aniso_native = res_aniso_native[1]
        plt.errorbar(x1, acc_aniso_native, yerr = err_aniso_native, label = feat + "_anisonative")

    plt.legend(loc=4)

    ax.set_xscale("log", nonposx='clip')
    # ax.set_yscale("log", nonposy='clip')

    plt.xlabel('number of training examples')#, fontsize=18)
    plt.ylabel('accuracy')#, fontsize=16)

    plt.grid(True)

    plt.show()



if __name__ == '__main__':

    feats = ["ffeat_ert"]
    plot_feat_curve_anisofeats("isbi2013_nn_train", feats)

    #ffeat_list = ["ffeat_brt"]#, "ffeat_ert"]
    #ffeat_list_aniso = ["('ffeat_brt', 'ffeat_brt2')"]
    #plot_feat_curve_anisocompare("sopnetcompare", ffeat_list, ffeat_list_aniso)

    #feats_simple  = ("b", "e", "be")
    #feats_complex = ("brt", "ert", "bert")
    #feats_extra   = ("brt2", "ert2", "bert2")

    #eval_ds("isbi2013", feats_extra)
