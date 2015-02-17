import sys
import logging
sys.path.append('..')
from feelit import utils
from feelit.features import Learning
from feelit.features import DataPreprocessor
from sklearn.cross_validation import KFold
import numpy as np


if __name__ == '__main__':

    logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.DEBUG)

    #features = ['rgb_gist', 'rgb_phog', 'rgba_gist', 'rgba_phog']
    ###################################
    features = ['rgba_gist']
    paths = ['/home/doug919/projects/github_repo/LJ40K/images/programs/output/npzs/rgba_gist/rgba_gist_1000.npz']

    dp = DataPreprocessor(logger=logging)
    dp.loads(features, paths)
    X_train, y_train, feature_name = dp.fuse()

    emotions = utils.LJ40K

    for emotion in emotions:

        logging.info('emotion = %s' % emotion)

        learner = Learning(logger=logging)

        yb_train = dp.get_binary_y_by_emotion(y_train, emotion)
        learner.set(X_train, yb_train, feature_name)


        kfolder = KFold(n=utils.getArrayN(X_train) , n_folds=10, shuffle=True)

        ######################
        Cs = [10,100,300,1000,3000,10000,30000]
        gammas = [0.0001,0.0003,0.001,0.003,0.01,0.1]

        scores = {}
        for c in Cs:
            for gamma in gammas:
                score = learner.kfold(kfolder, classifier='SVM', kernel='rbf', prob=False, C=c, scaling=True, gamma=gamma)
                scores.update({(c, gamma): score})

        best_C, best_gamma = max(scores.iteritems(), key=operator.itemgetter(1))[0]

        import pdb; pdb.set_trace()
