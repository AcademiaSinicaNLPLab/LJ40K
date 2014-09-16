
__support_kernels__ = ['linear', 'polynomial', 'rbf', 'sigmoid']

import sys, os
sys.path.append("../")
import numpy as np
from feelit import utils
from feelit.features import LIBSVM, dump
classifier = "LIBSVM"   ## static value, don't modify it
classtype = "binary"    ## static value, don't modify it
emotions = utils.LJ40K

kernel = "rbf"
prob = True

def usage():
    msg = """
    usage:
        python batchLIBSVMTestBinary.py <feature_name>
    """
    print msg

if __name__ == '__main__':
    
    if len(sys.argv) < 2:
        usage()
        exit(-1)

    if len(sys.argv) >= 2:
    
        feature_name = sys.argv[1]
        
        if len(sys.argv) == 4:
            _from = int(sys.argv[2])
            _to = int(sys.argv[3])
            emotions = emotions[_from:_to]
    else:
        print 'usage: python %s <feature_name> [from, to]' % (__file__)
        exit()


    key = ["classifier", "kernel", "classtype", "prob"]
    val = [classifier, kernel, classtype, prob]
    arg = '_'.join(map(lambda a: '='.join([a[0], str(a[1])]), sorted(zip(key, val), key=lambda x:x[0])))

    print zip(key, val)
    print 'will test', len(emotions), 'emotions: ', emotions, 'go?', raw_input()



    print '>>> processing', feature_name

    ## load text_TFIDF.Xy.test.npz
    npz_path = "../data/"+feature_name+".Xy.test.npz"

    print ' > loading',npz_path
    svm = LIBSVM(verbose=True)
    svm.load_test(npz_path, scaling=False)

    model_root = os.path.join("../models/", feature_name+'.'+arg+".models")
    results_root = os.path.join("../results/", feature_name+'.'+arg+".results")

    for i, label in enumerate(emotions):

        print '>>> processing %s (%d/%d)' % (label, i+1, len(emotions) )
        model_path = os.path.join(model_root, feature_name+"."+label+".train.model")
        
        print ' > loading', label, 'model'
        svm.load_model(model_path)

        if not prob: options = "-q"
        else: options = "-b 1 -q"

        print ' > predicting', label
        svm.predict(target=label, param=options)

        print ' > dumping', label, 'results'
        results_path = os.path.join(results_root, feature_name+"."+label+".results")

        dump(results_path, results=svm.p_vals, classes=svm.classes_, answers=svm.y_test)
            
