import numpy as np

def get_file_len( f_name ):

    with open(f_name, 'r') as f:

        for i, l in enumerate(f):
            pass

    return i + 1



def results_from_txt(f_name, skip_lines = 0):

    print "Reading in results from file:", f_name

    num_lines = get_file_len(f_name) - skip_lines - 1

    print "Length of file about to read in:", num_lines

    with open(f_name, 'r') as f:

        if skip_lines != 0:
            for i in range(skip_lines):
                f.readline()

        # interpret first line as keys
        keys = f.readline().split()

        print "Keys in file:", keys

        return_dict = dict()

        for k in keys:
            return_dict[k] = []

        for l in range(num_lines):
            vals = f.readline().split()
            assert len(vals) == len(keys)
            for i in range(len(keys)):
                k = keys[i]
                return_dict[k].append( float(vals[i]) )

    return return_dict


def results_and_errors_from_txt(f_name, skip_lines = 0, skip_errors = 0):

    print "Reading in results and errors file:", f_name

    num_lines = get_file_len(f_name) - skip_lines - 1

    print "Length of file about to read in:", num_lines

    assert num_lines % 2 == 0, "Number of lines has to be even, when there are errors in second line"

    with open(f_name, 'r') as f:

        if skip_lines != 0:
            for i in range(skip_lines):
                f.readline()

        # interpret first line as keys
        keys = f.readline().split()
        err_keys = []

        print "Keys in file:", keys

        return_dict = dict()

        ii = 0
        for k in keys:

            return_dict[k] = []

            if ii >= skip_errors:
                err_key = k + "_err"
                err_keys.append(err_key)
                return_dict[err_key] = []

            ii += 1

        print "Added errorkeys:", err_keys

        for l in range(num_lines / 2):

            vals = f.readline().split()
            assert len(vals) == len(keys)
            for i in range(len(keys)):
                k = keys[i]
                return_dict[k].append( float(vals[i]) )
            err_vals = f.readline().split()
            err_vals = err_vals[1:]

            assert len(err_vals) == len(err_keys)
            for i in range( len(err_keys) ):
                k = err_keys[i]
                return_dict[k].append( float(err_vals[i]) )

    return return_dict




if __name__ == '__main__':
    path = "results_isbi2013/eval_num_labels_isbi2013_nn_train_ffeats=ffeat_bert2_use_weighted_faces=False_use_2d_features=False_-5673437082304587008.txt"

    res = results_and_errors_from_txt(path, skip_lines = 5, skip_errors = 2)

    print res


