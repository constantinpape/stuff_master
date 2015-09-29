import numpy as np

import matplotlib.pyplot as plot

from read_in_txt import results_from_txt, results_and_errors_from_txt


def plot_xy(x, y, x_label = " " , y_label = " "):
    plot.scatter(x, y)
    plot.xlabel( x_label )
    plot.ylabel( y_label )
    plot.show()


def plot_xy_errors(x, y, y_errs, x_label = " ", y_label = " "):
    plot.errorbar(x, y, yerr = y_errs)
    plot.xlabel( x_label )
    plot.ylabel( y_label )
    plot.show()





if __name__ == '__main__':
    #path = "results_isbi2013/eval_num_labels_isbi2013_nn_train_ffeats=ffeat_bert2_use_weighted_faces=False_use_2d_features=False_-5673437082304587008.txt"
    path = "results_isbi2013/eval_num_labels_isbi2013_nn_train_ffeats=ffeat_bert2_use_weighted_faces=True_use_2d_features=False_-5673437082304587008.txt"

    res = results_and_errors_from_txt(path, skip_lines = 5, skip_errors = 2)

    X = res["N_labels"]
    Y = res["FSCORE"]
    Y_errs = res["FSCORE_err"]

    plot_xy_errors(X, Y, Y_errs, "num_labels", "fscore")

