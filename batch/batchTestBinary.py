

import sys, os
sys.path.append("../")
import numpy as np
from feelit import utils
import pickle
from feelit.features import dump


def relabel(y, label): return [label if _y == label else "_"+label for _y in y ]

def binary_labeling(y, positive): return [positive if _y == positive else "_"+positive for _y in y ]

if __name__ == '__main__':
    
    # feature_name = "image_rgba_gist"
    if len(sys.argv) > 1:
        feature_names = sys.argv[1:]
    else:
        feature_names = ["image_rgba_phog"]

    
    classifier = "SVM"
    kernel = "rbf"
    classtype = "binary"
    prob = True

    key = ["classifier", "kernel", "classtype", "prob"]
    val = [classifier, kernel, classtype, prob]

    arg = '_'.join(map(lambda a: '='.join([a[0], str(a[1])]), sorted(zip(key, val), key=lambda x:x[0])))

    for feature_name in feature_names:

        print '>>> processing', feature_name

        ## load text_TFIDF.Xy.test.npz
        npz_path = "../data/"+feature_name+".Xy.test.npz"

        print ' > loading',npz_path

        data = np.load(npz_path)

        # slice to train/test
        # print ' > X to Dense'
        X_test = utils.toDense( data['X'] )
        print ' > get X_test', X_test.shape

        y_test = data['y']
        print ' > get y_test', y_test.shape

        model_root = os.path.join("../models/", feature_name+'.'+arg+".models")
        results_root = os.path.join("../results/", feature_name+'.'+arg+".results")


        for label in utils.LJ40K:

            print '>>> processing', label

            model_path = os.path.join(model_root, feature_name+"."+label+".train.model")
            
            print ' > loading', label, 'model'
            clf = pickle.load(open(model_path))

            print ' > predicting', label
            if prob:
                result = clf.predict_proba(X_test)
            else:
                result = clf.predict(X_test)

            print ' > dumping', label, 'results'
            results_path = os.path.join(results_root, feature_name+"."+label+".results")

            dump(results_path, results=result, classes=clf.classes_, answers=y_test)
            
