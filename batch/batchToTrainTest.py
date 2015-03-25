"""
build entire (X,y) to (X_train,y_train), (X_test, y_test) sets
"""

import logging, os, sys, pickle
sys.path.append("../")
import numpy as np
from feelit import utils
from feelit.features import dump
from scipy.sparse import csr_matrix

def create_gid_to_lid_map(y):
    ## build global idx --> local idx mapping
    idx_map, lid, prev = {}, 0, y[0]
    for i,current in enumerate(y):
        if prev and prev != current:
            prev, lid = current, 0
        idx_map[i] = lid
        lid += 1
    return idx_map

def slice_arr_by_class(idx_map, each_class):
    if "<" in each_class:
        th = int(each_class.replace("<",""))
        to_delete = [gidx for gidx, lidx in idx_map.iteritems() if lidx >= th]
    elif ">" in each_class: # e.g., >800, including 800
        th = int(each_class.replace(">",""))
        to_delete = [gidx for gidx, lidx in idx_map.iteritems() if lidx < th]
    else:
        return False
    return to_delete  

def resample_by_delete_idx(arr, to_delete, axis=0): 
    """
    axis: 0 or 1
        set 0 to delete by row
        set 1 to delete by col
    """
    return np.delete(arr, to_delete, axis=axis)

if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        feature_names = sys.argv[1:]
    else:
        print 'usage: python %s <feature_names>' 
        exit(-1)

    for feature_name in feature_names:

        ## load
        npz_path = "../exp/data/from_mongo/"+feature_name+".Xy.npz"
        print 'loading from',npz_path
        data = np.load(npz_path)
        
        print 'transform X to Dense'
        X = utils.toDense( data['X'] )
        print 'get X', X.shape

        y = data['y']
        print 'get y', y.shape


        ## build the gid --> lid mapping
        gid_to_lid = create_gid_to_lid_map(y)

        ## generate the index to be deleted in training/testing
        delete_testing = slice_arr_by_class(gid_to_lid, "<800")
        delete_training = slice_arr_by_class(gid_to_lid, ">800")

        # slice (X,y) to train/test
        print 'slicing training set'
        X_train = resample_by_delete_idx(X, delete_testing)
        y_train = resample_by_delete_idx(y, delete_testing)

        print 'slicing testing set'
        X_test = resample_by_delete_idx(X, delete_training)
        y_test = resample_by_delete_idx(y, delete_training)

        print 'got train:%d, test:%d samples' % (y_train.shape[0], y_test.shape[0])

        ## save
        train_path = "../exp/data/from_mongo/"+feature_name+".Xy"+".train"+".npz"
        test_path = "../exp/data/from_mongo/"+feature_name+".Xy"+".test"+".npz"

        # make sparse matrix
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)

        print ' > dumping training to', train_path
        dump(train_path, X=X_train, y=y_train)
        print ' > dumping testing to', test_path
        dump(test_path, X=X_test, y=y_test)

        print '='*10,'done','='*10

