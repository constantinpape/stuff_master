import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt


def plot_featcurve(res_path):

    with open(res_path, 'r') as f:
        results = pickle.load(f)

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
    plt.plot( n_feats, accs )
    plt.subplot(2,1,2)
    plt.errorbar( n_feats, accs, yerr = stds)
    plt.xlabel("Num Feats")
    plt.ylabel("RF Accuracy")
    plt.show()

if __name__ == '__main__':
    res_path = "./cross_validation/feature_num_eval/wrapper_test_ds_bert2_True.pkl"

    plot_featcurve(res_path)
