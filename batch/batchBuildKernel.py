## split data

import sys, os, pickle
sys.path.append("../")
from feelit.kernel import RBF
from feelit import utils

def help():
    print "usage: python batchBuildKernel.py [root] [feature] [begin:end]"
    print 
    print "  e.g: python batchBuildKernel.py ../exp/train rgba_gist+rgba_phog 0  10 "
    print "       python batchBuildKernel.py ../exp/train rgba_gist+rgba_phog 10 20 "
    print "       python batchBuildKernel.py ../exp/train rgba_gist+rgba_phog 20 30 "
    print "       python batchBuildKernel.py ../exp/train rgba_gist+rgba_phog 30 40 "
    print 
    print "file required:"
    print "  `train_binary_idx.pkl` and `dev_binary_idx.pkl`"
    print 
    print "  if the size of samples is 1600, and the train/dev is set as 9:1, i.e., (1440:160)"
    print "  then use the following code segments to produce these pickle files:"
    print 
    print "    >> from feelit.utils import random_idx"
    print "    >> train_idx, dev_idx = random_idx(1600, 1440)"
    print "    >> import pickle"
    print "    >> pickle.dump(train_idx, open('train_binary_idx.pkl', 'w'))"
    print "    >> pickle.dump(dev_idx, open('dev_binary_idx.pkl', 'w'))"
    exit(-1)

if __name__ == '__main__':
    
    if len(sys.argv) < 5: help()

    root, feature, begin, end = map(lambda x:x.strip(), sys.argv[1:])
    begin, end = int(begin), int(end)

    in_subdir  = "%s/Xy" % (feature)
    out_subdir = "%s/Ky" % (feature)

    in_dir = os.path.join(root, in_subdir)
    out_dir = os.path.join(root, out_subdir)

    npzs = filter(lambda x:x.endswith('.npz'), os.listdir(in_dir))
    to_process = sorted(npzs)[begin:end]

    print 'files to be processed:'
    print '\n'.join(['\t'+x for x in to_process])
    print '='*50

    ## load train/dev index files
    ## train_idx: a list containing 1440 index (int)
    ## dev_idx: a list containing 160 index (int)
    try:
        train_idx, dev_idx = pickle.load(open('../exp/data/train_binary_idx.pkl')), pickle.load(open('../exp/data/dev_binary_idx.pkl'))
    except:
        help()

    for npz_fn in to_process:

        ## npz_fn: rgba_gist+rgba_phog.Xy.happy.train.npz
        print 'processing', npz_fn

        rbf = RBF(verbose=True)
        rbf.load(os.path.join(in_dir, npz_fn))

        ## devide X,y into (X_train, y_train) and (X_dev, y_dev)
        # get dev by deleting the indexes of train
        X_dev, y_dev = utils.RandomSample((rbf.X, rbf.y), delete_index=train_idx)
        # get train by deleting the indexes of dev
        X_tr, y_tr = utils.RandomSample((rbf.X, rbf.y), delete_index=dev_idx)

        ## build
        K_tr, K_dev = rbf.build( (X_tr, X_tr), (X_tr, X_dev) )

        ## save
        ## rgba_gist+rgba_phog.Xy.happy.train.npz
        feature, xy, emotion, dtype, ext = npz_fn.split('.')
        out_train_fn = "%s.Ky.%s.train.npz" % (feature, emotion)
        out_dev_fn = "%s.Ky.%s.dev.npz" % (feature, emotion)
        
        rbf.save(os.path.join(out_dir, out_train_fn), K_tr=K_tr,   y_tr=y_tr   )
        rbf.save(os.path.join(out_dir, out_dev_fn),   K_dev=K_dev, y_dev=y_dev ) 
