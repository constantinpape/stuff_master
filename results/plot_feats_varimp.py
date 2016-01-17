import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import os

def plot_featcurve(res_path, save_path):

    with open(res_path, 'r') as f:
        results = pickle.load(f)

    title = os.path.split(res_path)[1][:-4]
    save_file = os.path.join(save_path, title) + ".svg"

    n_feats = []
    accs    = []
    stds    = []
    for key in results.keys():
        feats = results[key][0]
        acc_l = results[key][1]
        acc   = np.mean(acc_l)
        std   = np.std(acc_l)

        n_feats.append(feats.shape[0])
        accs.append(acc)
        stds.append(std)

    n_feats = np.array(n_feats)
    accs = np.array(accs)
    stds = np.array(stds)

    # sort after length of features
    sort = np.argsort(n_feats)

    n_feats = n_feats[sort]
    accs = accs[sort]
    stds = stds[sort]

    plt.subplot(2,1,1)
    plt.title(title)
    plt.plot( n_feats, accs )
    plt.subplot(2,1,2)
    plt.errorbar( n_feats, accs, yerr = stds)
    plt.xlabel("Num Feats")
    plt.ylabel("RF Accuracy")
    plt.savefig(save_file, format = 'svg')
    plt.close()

if __name__ == '__main__':
    #res_path = "./cross_validation/feature_num_eval/filter_ICAP_pedunculus_bert2.pkl"
    #res_path = "./cross_validation/feature_num_eval/varimp_pedunculus_bert2_True.pkl"
    save_path = "./cross_validation/feature_num_eval/eval"
    #plot_featcurve(res_path, save_path)

    path = "./cross_validation/feature_num_eval"

    for f in os.listdir("./cross_validation/feature_num_eval"):
        fpath = os.path.join(path, f)
        if os.path.isfile(fpath):
            print "plotting"
            print fpath
            plot_featcurve(fpath, save_path)
