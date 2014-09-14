

import sys, os
sys.path.append("../")
import numpy as np
from feelit import utils
from feelit.features import LIBSVM, dump

emotions = utils.LJ40K
classifier = "LIBSVM" ## static value, don't modify it
classtype = "binary"
kernel = "linear"
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

    ## e.g., ['text_TFIDF', 'image_rgba_gist']
    feature_names = sys.argv[1:]

    key = ["classifier", "kernel", "classtype", "prob"]
    val = [classifier, kernel, classtype, prob]
    arg = '_'.join(map(lambda a: '='.join([a[0], str(a[1])]), sorted(zip(key, val), key=lambda x:x[0])))

    for feature_name in feature_names:

        print '>>> processing', feature_name

        ## load text_TFIDF.Xy.test.npz
        npz_path = "../data/"+feature_name+".Xy.test.npz"

        print ' > loading',npz_path
        svm = LIBSVM(verbose=True)
        svm.load_test(npz_path)

        model_root = os.path.join("../models/", feature_name+'.'+arg+".models")
        results_root = os.path.join("../results/", feature_name+'.'+arg+".results")

        for label in utils.LJ40K:

            print '>>> processing', label
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
            
