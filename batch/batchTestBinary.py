_support_ = ['SVM', 'SGD', 'GaussianNB']

from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append("../")
import numpy as np
from feelit import utils
import pickle
from feelit.features import dump
classtype = "binary"    ## static value, don't modify it

classifier = "SGD"
kernel = "linear"    ## don't care if the classifier ends with "NB"
prob = True

emotions = utils.LJ40K

def usage():
    msg = """
    usage:
        python batchTestBinary.py <feature_name>
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

    # feature_name: "image_rgba_gist"
    # feature_names = sys.argv[1:]

    if not classifier.endswith("NB"): # a SVM-like classifier
        key = ["classifier", "kernel", "classtype", "prob"]
        val = [classifier, kernel, classtype, prob]
    else:
        key = ["classifier", "classtype", "prob"]
        val = [classifier, classtype, prob]

    print zip(key, val)
    print 'will test', len(emotions), 'emotions: ', emotions, 'go?', raw_input()

    arg = '_'.join(map(lambda a: '='.join([a[0], str(a[1])]), sorted(zip(key, val), key=lambda x:x[0])))

    # print 'would run test using "%s", go?' % arg, raw_input()

    # for feature_name in feature_names:

    print '>>> processing', feature_name

    ## load text_TFIDF.Xy.test.npz
    npz_path = "../data/"+feature_name+".Xy.test.npz"

    print ' > loading',npz_path

    data = np.load(npz_path)

    # slice to train/test and scaling
    scaler = StandardScaler()
    X_test = scaler.fit_transform(utils.toDense( data['X'] ))

    
    print ' > get X_test', X_test.shape

    y_test = data['y']
    print ' > get y_test', y_test.shape

    model_root = os.path.join("../models/", feature_name+'.'+arg+".models")
    results_root = os.path.join("../results/", feature_name+'.'+arg+".results")


    for i, label in enumerate(emotions):

        print '>>> processing %s (%d/%d)' % (label, i+1, len(emotions) )
        model_path = os.path.join(model_root, feature_name+"."+label+".model")
        
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
            
