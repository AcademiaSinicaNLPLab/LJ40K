
import sys
import numpy as np
from feelit import utils
from feelit.features import Learning
emotions = utils.LJ40K

def evals(y_te, y_predict, emotion):

    y_te_ = [1 if a == emotion else 0 for a in y_te]
    y_predict_ = [0 if a.startswith('_') else 1 for a in y_predict]

    Y = zip(y_te_, y_predict_)

    FP = len([ 1 for a,b in Y if a == 0 and b == 1 ])
    TN = len([ 1 for a,b in Y if a == 0 and b == 0 ])
    TP = len([ 1 for a,b in Y if a == 1 and b == 1 ])
    FN = len([ 1 for a,b in Y if a == 1 and b == 0 ])
    accu = (TP + TN/39) / float(FP/39 + TN/39 + TP + FN)

    return accu

def usage():
    print '[feature] [emition_id]'
    print 'rgba_gist+rgba_phog 1'
    exit()

if __name__ == '__main__':

    if len(sys.argv) != 3:
        usage()

    feature, eid = sys.argv[1:]
    emotion = emotions[int(eid)]

    ## init
    l = Learning(verbose=False)

    ## ================ training ================ ##

    ## load train
    l.load(path="exp/train/%s/Xy/%s.Xy.%s.train.npz" % (feature, feature, emotion))

    ## train
    l.train(classifier="SVM", kernel="rbf", prob=False)

    ## ================= testing ================= ##

    ## load test data
    test_data = np.load('exp/data/%s.Xy.test.npz' % (feature))
    # y_te
    # array(['accomplished', 'accomplished', 'accomplished', ..., 'tired',
    #        'tired', 'tired'],
    #       dtype='|S13')
    X_te, y_te = test_data['X'], test_data['y']

    ## predict
    # y_predict
    # array([u'_sad', u'_sad', u'sad', ..., u'_sad', u'_sad', u'_sad'],
    #       dtype='<U4')
    y_predict = l.clf.predict(X_te)

    ## eval
    accuracy = evals(y_te, y_predict, emotion)

    print emotion, '\t', accuracy

