###########################################
#           batchBuildTestKernel          #
###########################################
# build kernels for testing
# steps:
#   1. input
#       rgba_gist+rgba_phog.Xy.test.npz
#   2. specify positive emotion
#       
#   2. split
#       X_tr, X_dev and y_tr, y_dev
#   3. build
#       K_tr, K_dev
#   4. save
###########################################

import sys, os, pickle
import numpy as np
sys.path.append("../")
from feelit.kernel import RBF
from feelit import utils

def help():
    print "usage: python batchBuildTestKernel.py [root] [feature] [begin:end]"
    print 
    print "  e.g: python batchBuildTestKernel.py ../exp/data rgba_gist+rgba_phog 0  10 "
    print "       python batchBuildTestKernel.py ../exp/data rgba_gist+rgba_phog 10 20 "
    print "       python batchBuildTestKernel.py ../exp/data rgba_gist+rgba_phog 20 30 "
    print "       python batchBuildTestKernel.py ../exp/data rgba_gist+rgba_phog 30 40 "
    print "file required:"
    print "  `emap.pkl`"   
    exit(-1)

if __name__ == '__main__':
    
    if len(sys.argv) < 5: help()

    root, feature, begin, end = map(lambda x:x.strip(), sys.argv[1:])
    begin, end = int(begin), int(end)

    ## rgba_gist+rgba_phog.Xy.test.npz
    test_npz_fn = "%s.Xy.test.npz" % (feature)
    test_npz_path = os.path.join(root, test_npz_fn)

    out_subdir = "%s/Ky" % (feature)
    out_dir = os.path.join(root, out_subdir)

    try:
        emap = pickle.load(open('../exp/data/emap.pkl'))
    except:
        help()

    to_process_emotions = sorted(emap.keys())[begin:end]

    ## load train/dev index files
    ## train_idx: a list containing 1440 index (int)
    ## dev_idx: a list containing 160 index (int)
    try:
        dev_idx = pickle.load(open('../exp/data/dev_binary_idx.pkl'))
    except:
        help()

    print 'emotions to be processed:'
    print '\n'.join(['\t'+x for x in to_process_emotions])
    print '='*50

    test = RBF(verbose=True)
    # X: 8000x1640, y: 8000
    test.load(test_npz_path)
    # X_te, y_te = test.X, test.y
    X_te = test.X

    for emotion in to_process_emotions:

        print 'processing', emotion

        # transform y_te [multiple emotions] into [binary emotions]
        # Counter({'_happy': 7800, 'happy': 200})
        y_te = np.array([ e if e == emotion else '_'+emotion for e in test.y])


        train = RBF(verbose=True)
        train_npz_path = '../exp/train/%s/Xy/%s.Xy.%s.train.npz' % (feature, feature, emotion)
        # X: 1600x1640, y: 1600
        train.load(train_npz_path)
        # X_tr: (1440x1640), y_tr:1440
        X_tr, y_tr = utils.RandomSample((train.X, train.y), delete_index=dev_idx)

        ## build
        rbf = RBF(verbose=True)
        K_te = rbf.build( (X_tr, X_te) )

        ## save
        ## rgba_gist+rgba_phog.Xy.happy.train.npz
        feature, xy, emotion, dtype, ext = npz_fn.split('.')
        out_test_fn = "%s.Ky.%s.test.npz" % (feature, emotion)
        
        rbf.save(os.path.join(out_dir, out_test_fn), K_tr=K_te, y_tr=y_te)
