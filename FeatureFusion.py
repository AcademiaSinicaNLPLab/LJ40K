# -*- coding: utf-8 -*-

import numpy as np
import pickle
import pymongo
import logging

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm


class FeatureFusion(object):
    """
    A scikit-learn wrapper: fuse several features from mongodb
    """
    def __init__(self, **kwargs):

        loglevel = logging.DEBUG if 'verbose' in kwargs and kwargs['verbose'] == True else logging.INFO
        logging.basicConfig(format='[%(levelname)s] %(message)s', level=loglevel)

        ## config mongodb
        mongo_addr = 'doraemon.iis.sinica.edu.tw' if 'mongo_addr' not in kwargs else kwargs['mongo_addr']
        db_name = 'LJ40K' if 'db_name' not in kwargs else kwargs['db_name']

        self._db = pymongo.Connection(mongo_addr)[db_name]

        self.feature_names = set()

        self._feature = []
        self._target = []

        ## emoID --> emotion
        # self._load_emoID_map(**kwargs)

        self.emoID_map = {e:i for i, e in enumerate(sorted(map(lambda x:x['emotion'], self._db["emotions"].find({ 'label': db_name }))))}

        self.kfold_results = []

    # def setClassifer(self):
    #     self.clf = svm.SVC()
        
    def _getCollectionName(self, feature_name, prefix="features"):
        return '.'.join([prefix, feature_name])

    def add(self, feature_name):
        """
        feature_name    :   e.g., TFIDF, pattern_emotion, keyword, ... ,etc.
        """
        collection_name = self._getCollectionName(feature_name)

        if collection_name not in self._db.collection_names():
            raise Exception('cannot find collection %s in %s' % (collection_name, self._db.name))
        else:
            if feature_name in self.feature_names:
                logging.info('feature %s already exists' % (feature_name))
            else:
                self.feature_names.add(feature_name)
                logging.info('feature %s added' % (feature_name))

    def fuse(self, **kwargs):
        """
        !! could cause heavy memory usage !!
        """
        for feature_name in self.feature_names:

            ## TFIDF --> features.TFIDF
            collection_name = self._getCollectionName(feature_name)
            co = self._db[collection_name]

            logging.info('extracting %s features from %s' % (feature_name, co.full_name) )


            _count = co.count()
            batch_size = _count/25 if 'batch_size' not in kwargs else kwargs['batch_size']

            ## fetch mongodb
            for i, mdoc in enumerate(co.find().batch_size(batch_size)):

                if type(mdoc['feature']) == dict:
                    pass

                ## transform to dictionary
                elif type(mdoc['feature']) == list:
                    mdoc['feature'] = { f[0]:f[1] for f in mdoc['feature'] }

                else:
                    raise TypeError('make sure the feature format is either <dict> or <list> in mongodb')

                
                emoID = self.emoID_map[mdoc['emotion']] if 'emoID' not in mdoc else mdoc['emoID']

                ## store all features in a list
                self._feature.append(mdoc['feature'])
                self._target.append(emoID)

                logging.debug('mongo doc %d/%d fetched' % ( i+1, _count))

        ## vectorize
        vec = DictVectorizer()
        self.X = vec.fit_transform(self._feature)
        self.y = np.array(self._target)

    def nFold(self, **kwargs):
        """
        alias for kFold() function
        """
        self.kFold(**kwargs)

    def kFold(self, **kwargs):
        """
        kwargs:
            n_folds :   default 10
            shuffle :   True
        """

        # config n-fold verification
        n_folds = 10 if 'n_folds' not in kwargs else kwargs['n_folds']
        shuffle = True if 'shuffle' not in kwargs else kwargs['shuffle']
        kf = KFold( len(self._feature), n_folds=n_folds, shuffle=shuffle )

        loss = 'modified_huber' if 'loss' not in kwargs else kwargs['loss']
        penalty = 'elasticnet' if 'penalty' not in kwargs else kwargs['penalty']

        ## setup a Scaler
        with_mean = False if 'with_mean' not in kwargs else kwargs['with_mean']
        scaler = StandardScaler(with_mean=False)

        for (i, (train_index, test_index)) in enumerate(kf):
            logging.info('cross validation round %d' % (i+1))

            X_train, X_test, y_train, y_test = self.X[train_index], self.X[test_index], self.y[train_index], self.y[test_index]
            
            #scale data for SGD performance
            logging.debug('scaling')
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            
            ## train
            # logging.info('training with SGDClassifier')
            # clf = SGDClassifier(loss=loss, penalty=penalty, shuffle=shuffle)

            logging.info('training with svm.SVC classifier')

            clf = svm.SVC()
            ## libsvm
            ## cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
            ## gamma : set gamma in kernel function (default 1/num_features)
            ## epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
            ## cachesize : set cache memory size in MB (default 100)
            ## tolerance : set tolerance of termination criterion (default 0.001)
            ## shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)
            ## probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)

            # SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
            #     gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
            #     shrinking=True, tol=0.001, verbose=False)
            clf.fit(X_train, y_train)

            ## scoring
            score = clf.score(X_test, y_test)
            logging.info('score %.3f' % (score))

            ## predict
            logging.info('predicting')
            result = clf.predict(X_test)

            self.kfold_results.append( (i+1, y_test, result) )

    def save(self, root="."):

        ## generate fused feature name
        fused_feature_name = '+'.join(sorted(list(self.feature_names)))

        for ith, y_test, result in self.kfold_results:

            out_fn = "%s.fold-%d.result" % (fused_feature_name, ith)

            with open(out_fn, 'w') as fw:

                fw.write( ' '.join( map(lambda x:str(x), y_test) ) )
                fw.write('\n')

                fw.write( ' '.join( map(lambda x:str(x), result) ) )
                fw.write('\n')

if __name__ == '__main__':

    from FeatureFusion import FeatureFusion

    F = FeatureFusion(mongo_addr="doraemon.iis.sinica.edu.tw", db_name="LJ40K", verbose=True)

    F.add(feature_name="TFIDF")

    F.fuse()

    F.kFold(n_folds=10, shuffle=True, loss="modified_huber", penalty="elasticnet")

    F.save()

            

