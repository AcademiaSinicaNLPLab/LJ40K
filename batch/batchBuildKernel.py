## split data

import sys, os
sys.path.append("../")
from feelit.kernel import RBF
from feelit.utils import devide

def help():
    print "usage: python batchBuildKernel.py [root] [feature] [begin:end]"
    print 
    print "  e.g: python batchBuildKernel.py ../exp/train rgba_gist+rgba_phog 0  10 "
    print "       python batchBuildKernel.py ../exp/train rgba_gist+rgba_phog 10 20 "
    print "       python batchBuildKernel.py ../exp/train rgba_gist+rgba_phog 20 30 "
    print "       python batchBuildKernel.py ../exp/train rgba_gist+rgba_phog 30 40 "
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

    for npz_fn in to_process:

        rbf = RBF(verbose=True)
        rbf.load(os.path.join(in_dir, npz_fn))

        ## devide data --> train/dev
        X_tr, X_dev = devide(rbf.X, 0.9)
        y_tr, y_dev = devide(rbf.y, 0.9)

        ## build
        K_tr, K_dev = rbf.build( (X_tr, X_tr), (X_tr, X_dev) )


        ## save
        out_train_fn = "%s.Ky.%s.train.npz" % (feature, emotion)
        out_dev_fn = "%s.Ky.%s.dev.npz" % (feature, emotion)
        
        rbf.save(os.path.join(out_dir, out_train_fn), K_tr=K_tr,   y_tr=y_tr   )
        rbf.save(os.path.join(out_dir, out_dev_fn),   K_dev=K_dev, y_dev=y_dev ) 
   